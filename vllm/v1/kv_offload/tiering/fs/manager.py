# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
import threading
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import override

from vllm.distributed.kv_events import MEDIUM_FS
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.capacity import (
    AdmissionStatus,
    FileSystemCapacityManager,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    load_block,
    make_temp_path,
    store_block,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class _FileSystemMetrics:
    CAPACITY_BYTES = "vllm:kv_offload_fs_capacity_bytes"
    ACCOUNTED_BYTES = "vllm:kv_offload_fs_accounted_bytes"
    RESERVED_BYTES = "vllm:kv_offload_fs_reserved_bytes"
    EVICTIONS = "vllm:kv_offload_fs_evictions"
    EVICTED_BYTES = "vllm:kv_offload_fs_evicted_bytes"
    CAPACITY_SKIPS = "vllm:kv_offload_fs_capacity_skips"
    EVICTION_FAILURES = "vllm:kv_offload_fs_eviction_failures"


class _StoreJobRecord:
    """Thread-safe per-key commit bookkeeping for one store job."""

    def __init__(self, keys: list[OffloadKey]) -> None:
        self.keys = keys
        self._committed_indices: set[int] = set()
        self._lock = threading.Lock()

    def record_commit(self, index: int) -> None:
        with self._lock:
            self._committed_indices.add(index)

    def committed_keys(self) -> list[OffloadKey]:
        with self._lock:
            indices = sorted(self._committed_indices)
        return [self.keys[index] for index in indices]


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        return self._tier._capacity.contains_many(paths)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    medium: ClassVar[str] = MEDIUM_FS

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        max_bytes: int,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            max_bytes: Positive hard capacity ceiling for cache payload bytes.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer number of bytes")
        self.max_bytes = max_bytes
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Per-key commit state for in-flight store jobs. Kept only when
        # events are enabled; failed jobs never publish partial commits.
        self._store_job_keys: dict[JobId, _StoreJobRecord] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._capacity = FileSystemCapacityManager(
            namespace_root=self.file_mapper.get_data_dir_path(),
            max_bytes=self.max_bytes,
            expected_file_size=self._block_size,
        )
        # Capacity counters are cumulative. Start from zero so restart
        # shrink evictions are visible on the first stats poll.
        self._last_capacity_counters = (0, 0, 0, 0, 0)

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        record: _StoreJobRecord | None = None
        if self.events is not None:
            record = _StoreJobRecord(list(job_metadata.keys))
            self._store_job_keys[job_metadata.job_id] = record

        tasks = (
            functools.partial(
                self._store_one,
                self.file_mapper.get_file_name(key),
                int(bid) * self._block_size,
                record,
                index,
            )
            for index, (key, bid) in enumerate(
                zip(job_metadata.keys, job_metadata.block_ids)
            )
        )
        self._pool.enqueue_store(
            job_metadata.job_id,
            len(job_metadata.keys),
            tasks,
        )

    def _store_one(
        self,
        final_path: str,
        offset: int,
        record: _StoreJobRecord | None,
        key_index: int,
    ) -> None:
        admission = self._capacity.admit_write(
            final_path,
            self._block_size,
        )
        if admission.status is not AdmissionStatus.RESERVED:
            # Already-present, duplicate-inflight, oversized, and capacity
            # rejection are normal cache-write skips.
            return

        reservation = admission.reservation
        assert reservation is not None
        tmp_path = make_temp_path(final_path)

        try:
            store_block(
                final_path,
                tmp_path,
                self._primary_kv_view,
                offset,
                self._block_size,
            )
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                self._capacity.abort_write(reservation)
            except OSError:
                # Physical temp presence cannot be disproven, so retain the
                # full reservation conservatively.
                self._capacity.retain_orphan_temp(
                    reservation,
                    tmp_path,
                )
            else:
                self._capacity.abort_write(reservation)
            raise
        else:
            self._capacity.commit_write(reservation)
            if record is not None:
                record.record_commit(key_index)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        tasks = (
            functools.partial(
                self._load_one,
                self.file_mapper.get_file_name(key),
                int(bid) * self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(
            job_metadata.job_id,
            len(job_metadata.keys),
            tasks,
        )

    def _load_one(
        self,
        final_path: str,
        offset: int,
    ) -> None:
        pin = self._capacity.pin_for_read(final_path)
        if pin is None:
            raise FileNotFoundError(
                f"filesystem cache entry is no longer committed: {final_path}"
            )

        try:
            load_block(
                final_path,
                self._primary_kv_view,
                offset,
                self._block_size,
            )
        except Exception:
            self._capacity.release_read(
                pin,
                invalidate=True,
            )
            raise
        else:
            self._capacity.release_read(pin)

    @override
    def touch(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        self._capacity.touch(paths)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = []
        for job_id, success in self._pool.get_finished():
            if self.events is not None:
                record = self._store_job_keys.pop(job_id, None)
                if success and record is not None:
                    committed_keys = record.committed_keys()
                    if committed_keys:
                        self.events.append(
                            OffloadingEvent(
                                keys=committed_keys,
                                medium=self.medium,
                                removed=False,
                                locality=self.locality,
                            )
                        )
            results.append(JobResult(job_id=job_id, success=success))
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @classmethod
    @override
    def build_metric_definitions(
        cls,
        extra_config: dict[str, Any],
    ) -> dict[str, OffloadingMetricMetadata]:
        return {
            _FileSystemMetrics.CAPACITY_BYTES: OffloadingGaugeMetadata(
                documentation=(
                    "Configured logical hard capacity of a filesystem "
                    "KV-cache tier, in bytes."
                ),
                labelnames=("tier",),
            ),
            _FileSystemMetrics.ACCOUNTED_BYTES: OffloadingGaugeMetadata(
                documentation=(
                    "Committed filesystem KV-cache bytes currently "
                    "accounted against the logical capacity."
                ),
                labelnames=("tier",),
            ),
            _FileSystemMetrics.RESERVED_BYTES: OffloadingGaugeMetadata(
                documentation=(
                    "In-flight or conservatively retained filesystem "
                    "KV-cache bytes reserved against the logical capacity."
                ),
                labelnames=("tier",),
            ),
            _FileSystemMetrics.EVICTIONS: OffloadingCounterMetadata(
                documentation=(
                    "Number of filesystem KV-cache entries evicted to "
                    "enforce the logical capacity."
                ),
                labelnames=("tier",),
            ),
            _FileSystemMetrics.EVICTED_BYTES: OffloadingCounterMetadata(
                documentation=(
                    "Filesystem KV-cache bytes evicted to enforce the logical capacity."
                ),
                labelnames=("tier",),
            ),
            _FileSystemMetrics.CAPACITY_SKIPS: OffloadingCounterMetadata(
                documentation=(
                    "Number of filesystem KV-cache writes skipped by "
                    "logical capacity admission."
                ),
                labelnames=("tier", "reason"),
            ),
            _FileSystemMetrics.EVICTION_FAILURES: OffloadingCounterMetadata(
                documentation=(
                    "Number of filesystem KV-cache eviction unlink failures."
                ),
                labelnames=("tier",),
            ),
        }

    @override
    def get_stats(self) -> "OffloadingConnectorStats":
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        snapshot = self._capacity.snapshot()
        stats = OffloadingConnectorStats()
        tier_label = (self.instance_id,)

        stats.set_gauge(
            _FileSystemMetrics.CAPACITY_BYTES,
            snapshot.max_bytes,
            tier_label,
        )
        stats.set_gauge(
            _FileSystemMetrics.ACCOUNTED_BYTES,
            snapshot.accounted_bytes,
            tier_label,
        )
        stats.set_gauge(
            _FileSystemMetrics.RESERVED_BYTES,
            snapshot.reserved_bytes,
            tier_label,
        )

        current_counters = (
            snapshot.eviction_count,
            snapshot.evicted_bytes,
            snapshot.oversized_skip_count,
            snapshot.capacity_skip_count,
            snapshot.eviction_failure_count,
        )
        previous_counters = self._last_capacity_counters

        deltas = tuple(
            max(0, current - previous)
            for current, previous in zip(
                current_counters,
                previous_counters,
            )
        )

        if deltas[0] > 0:
            stats.increase_counter(
                _FileSystemMetrics.EVICTIONS,
                deltas[0],
                tier_label,
            )
        if deltas[1] > 0:
            stats.increase_counter(
                _FileSystemMetrics.EVICTED_BYTES,
                deltas[1],
                tier_label,
            )
        if deltas[2] > 0:
            stats.increase_counter(
                _FileSystemMetrics.CAPACITY_SKIPS,
                deltas[2],
                (self.instance_id, "oversized"),
            )
        if deltas[3] > 0:
            stats.increase_counter(
                _FileSystemMetrics.CAPACITY_SKIPS,
                deltas[3],
                (
                    self.instance_id,
                    "no_evictable_capacity",
                ),
            )
        if deltas[4] > 0:
            stats.increase_counter(
                _FileSystemMetrics.EVICTION_FAILURES,
                deltas[4],
                tier_label,
            )

        self._last_capacity_counters = current_counters
        return stats

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Stop lookup work first, then cancel/join the I/O pool before
        releasing filesystem capacity ownership. Queued tasks never own
        reservations because admission happens only after a worker starts.
        """
        self._lookup_manager.shutdown()
        try:
            self._pool.shutdown(wait=True)
        finally:
            # close() retries orphan-temp cleanup before releasing the
            # namespace ownership lock.
            self._capacity.close()

        snapshot = self._capacity.snapshot()

        if snapshot.pending_write_count:
            raise AssertionError(
                "pending filesystem capacity reservation survived "
                f"shutdown: count={snapshot.pending_write_count}, "
                f"reserved_bytes={snapshot.reserved_bytes}"
            )

        if snapshot.orphan_temp_count:
            logger.warning(
                "filesystem KV cache shutdown retained %d orphan temp "
                "reservation(s), reserved_bytes=%d; restart recovery "
                "remains responsible for cleanup or fail-fast",
                snapshot.orphan_temp_count,
                snapshot.reserved_bytes,
            )
