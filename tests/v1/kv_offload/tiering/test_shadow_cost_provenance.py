# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="attr-defined"

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_stubs() -> tuple[ModuleType, ModuleType, type, type]:
    for name in (
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading",
        "vllm.v1",
        "vllm.v1.kv_offload",
        "vllm.v1.kv_offload.cpu",
        "vllm.v1.kv_offload.tiering",
    ):
        if name not in sys.modules:
            _package(name)

    numpy = ModuleType("numpy")
    numpy.ndarray = list
    numpy.int64 = int
    numpy.array = lambda values, dtype=None: list(values)
    sys.modules["numpy"] = numpy

    torch = ModuleType("torch")
    torch.Tensor = object
    sys.modules["torch"] = torch

    logger_module = ModuleType("vllm.logger")

    class _Logger:
        def warning(self, *args, **kwargs) -> None:
            pass

        def info(self, *args, **kwargs) -> None:
            pass

        def error(self, *args, **kwargs) -> None:
            pass

        def debug(self, *args, **kwargs) -> None:
            pass

        def exception(self, *args, **kwargs) -> None:
            pass

    logger_module.init_logger = lambda name: _Logger()
    sys.modules["vllm.logger"] = logger_module

    config_module = ModuleType("vllm.v1.kv_offload.config")
    config_module.OffloadingConfig = object
    sys.modules["vllm.v1.kv_offload.config"] = config_module

    cost_model = _load(
        "vllm.v1.kv_offload.cost_model",
        _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "cost_model.py",
    )
    base = _load(
        "vllm.v1.kv_offload.base",
        _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "base.py",
    )

    try:
        import typing_extensions  # noqa: F401
    except ModuleNotFoundError:
        typing_extensions = ModuleType("typing_extensions")
        typing_extensions.override = lambda function: function
        sys.modules["typing_extensions"] = typing_extensions

    metrics_module = ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics"
    )

    class _ConnectorMetricName:
        COST_OBSERVATIONS = "vllm:kv_offload_cost_observations"
        COST_RUNTIME_SCALE = "vllm:kv_offload_cost_runtime_scale"

    class _Stats:
        def __init__(self) -> None:
            self.histograms: list[tuple[object, object, object]] = []
            self.counters: list[tuple[object, object, object]] = []
            self.gauges: list[tuple[object, object, object]] = []

        def observe_histogram(self, name, value, labelvalues=()) -> None:
            self.histograms.append((name, value, labelvalues))

        def increase_counter(self, name, value=1, labelvalues=()) -> None:
            self.counters.append((name, value, labelvalues))

        def set_gauge(self, name, value, labelvalues=()) -> None:
            self.gauges.append((name, value, labelvalues))

        def is_empty(self) -> bool:
            return not (self.histograms or self.counters or self.gauges)

        def aggregate(self, other):
            self.histograms.extend(other.histograms)
            self.counters.extend(other.counters)
            self.gauges.extend(other.gauges)
            return self

    metrics_module.OffloadingConnectorStats = _Stats
    metrics_module._ConnectorMetricName = _ConnectorMetricName
    sys.modules[metrics_module.__name__] = metrics_module

    cpu_common = ModuleType("vllm.v1.kv_offload.cpu.common")

    class _CPULoadStoreSpec(base.LoadStoreSpec):  # type: ignore[name-defined]
        def __init__(self, block_ids) -> None:
            self.block_ids = list(block_ids)

    cpu_common.CPULoadStoreSpec = _CPULoadStoreSpec
    sys.modules[cpu_common.__name__] = cpu_common

    cpu_manager = ModuleType("vllm.v1.kv_offload.cpu.manager")

    class _CPUOffloadingManager:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    cpu_manager.CPUOffloadingManager = _CPUOffloadingManager
    sys.modules[cpu_manager.__name__] = cpu_manager

    shared_region = ModuleType("vllm.v1.kv_offload.cpu.shared_offload_region")
    shared_region.SharedOffloadRegion = type("SharedOffloadRegion", (), {})
    sys.modules[shared_region.__name__] = shared_region

    tiering_base = ModuleType("vllm.v1.kv_offload.tiering.base")
    tiering_base.JobId = int

    @dataclass
    class _JobMetadata:
        job_id: int
        keys: object
        block_ids: object
        is_promotion: bool
        req_context: object

    @dataclass
    class _JobResult:
        job_id: int
        success: bool

    tiering_base.JobMetadata = _JobMetadata
    tiering_base.JobResult = _JobResult
    tiering_base.ParentManager = object
    tiering_base.SecondaryTierManager = object
    tiering_base.TieringOffloadingMetrics = type(
        "TieringOffloadingMetrics",
        (),
        {
            "LOOKUP_SYNC_DELAY": "lookup_sync",
            "LOOKUP_ASYNC_DELAY": "lookup_async",
        },
    )
    sys.modules[tiering_base.__name__] = tiering_base

    _load(
        "vllm.v1.kv_offload.tiering.manager",
        _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "tiering" / "manager.py",
    )
    return base, cost_model, _CPULoadStoreSpec, _JobResult


_BASE, _COST_MODEL, CPULoadStoreSpec, JobResult = _install_stubs()
_MANAGER_MODULE = sys.modules["vllm.v1.kv_offload.tiering.manager"]
_METRICS_MODULE = sys.modules[
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics"
]
TieringOffloadingManager = _MANAGER_MODULE.TieringOffloadingManager
LookupResult = _BASE.LookupResult
ReqContext = _BASE.ReqContext
RequestOffloadingContext = _BASE.RequestOffloadingContext
make_offload_key = _BASE.make_offload_key
LoadProvenance = _COST_MODEL.LoadProvenance
OffloadCostModel = _COST_MODEL.OffloadCostModel
_ConnectorMetricName = _METRICS_MODULE._ConnectorMetricName


class FakePrimary:
    def __init__(self) -> None:
        self.lookup_results: list[object] = []
        self.prepare_write_result = None
        self.completed_writes: list[tuple[object, bool]] = []
        self.reset_count = 0

    def lookup(self, key, req_context):
        assert self.lookup_results
        return self.lookup_results.pop(0)

    def prepare_write(self, keys, req_context):
        return self.prepare_write_result

    def complete_write(self, keys, req_context, success=True) -> None:
        self.completed_writes.append((tuple(keys), success))

    def on_request_finished(self, req_context) -> None:
        pass

    def reset_cache(self) -> None:
        self.reset_count += 1

    def get_stats(self):
        return None

    def touch(self, keys, req_context) -> None:
        pass

    def prepare_load(self, keys, req_context):
        return SimpleNamespace(keys=tuple(keys))

    def complete_load(self, keys, req_context) -> None:
        pass

    def shutdown(self) -> None:
        pass


class FakeSecondary:
    tier_type = "fs"

    def __init__(self) -> None:
        self.lookup_results: list[object] = []
        self.submitted_loads: list[object] = []
        self.finished_jobs: list[object] = []

    def lookup(self, key, req_context):
        assert self.lookup_results
        return self.lookup_results.pop(0)

    def on_new_request(self, req_context):
        return RequestOffloadingContext()

    def submit_load(self, job_metadata) -> None:
        self.submitted_loads.append(job_metadata)

    def get_finished_jobs(self):
        jobs = self.finished_jobs
        self.finished_jobs = []
        return jobs

    def on_request_finished(self, req_context) -> None:
        pass

    def drain_jobs(self) -> None:
        pass

    def has_pending_work(self) -> bool:
        return False

    def take_events(self):
        return ()

    def touch(self, keys, req_context) -> None:
        pass

    def serve_external_requests(self, parent) -> None:
        pass

    def on_schedule_end(self, context) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_stats(self):
        return None


PROFILE = {
    "cache_cost_model": {
        "mode": "shadow",
        "ewma_alpha": 0.2,
        "profile": {
            "recompute_ms": {64: 100.0},
            "tiers": {
                "cpu_primary": {"restore_ms": {64: 20.0}},
                "filesystem": {
                    "restore_ms": {64: 80.0},
                    "promotion_ms": {64: 100.0},
                },
            },
        },
    }
}


def make_manager():
    model = OffloadCostModel.from_extra_config(PROFILE)
    assert model is not None
    primary = FakePrimary()
    secondary = FakeSecondary()
    manager = TieringOffloadingManager(
        primary,
        [secondary],
        cost_model=model,
        secondary_tier_keys=("filesystem",),
        tokens_per_chunk_by_group=(64,),
    )
    return manager, primary, secondary, model


def prepared_write(key):
    return SimpleNamespace(
        store_spec=CPULoadStoreSpec([7]),
        keys_to_store=[key],
        evicted_keys=[],
    )


def test_direct_primary_hit_reports_idempotent_cpu_provenance() -> None:
    manager, primary, _, _ = make_manager()
    ctx = ReqContext("r")
    key = make_offload_key(b"cpu", 0)
    manager.on_new_request(ctx)
    primary.lookup_results = [LookupResult.HIT]

    assert manager.lookup(key, ctx) is LookupResult.HIT
    first = manager.get_load_provenance([key], ctx, 64)
    second = manager.get_load_provenance([key], ctx, 64)

    assert first == second
    assert first is not None
    assert first.source == "cpu_primary"
    assert first.external_tokens == 64
    assert first.secondary_promoted_tokens == 0
    assert first.confidence == "high"


def test_secondary_source_survives_later_primary_hit() -> None:
    manager, primary, secondary, _ = make_manager()
    ctx = ReqContext("r")
    key = make_offload_key(b"fs", 0)
    manager.on_new_request(ctx)
    primary.lookup_results = [LookupResult.MISS, LookupResult.HIT]
    primary.prepare_write_result = prepared_write(key)
    secondary.lookup_results = [LookupResult.HIT]

    assert manager.lookup(key, ctx) is LookupResult.RETRY
    assert manager.lookup(key, ctx) is LookupResult.HIT

    provenance = manager.get_load_provenance([key], ctx, 64)
    assert provenance is not None
    assert provenance.source == "secondary:filesystem"
    assert provenance.secondary_promoted_tokens == 64
    assert provenance.confidence == "high"


def test_failed_promotion_is_not_marked_as_secondary_restore() -> None:
    manager, primary, secondary, _ = make_manager()
    ctx = ReqContext("r")
    key = make_offload_key(b"full", 0)
    manager.on_new_request(ctx)
    primary.lookup_results = [LookupResult.MISS]
    primary.prepare_write_result = None
    secondary.lookup_results = [LookupResult.HIT]

    assert manager.lookup(key, ctx) is LookupResult.MISS
    assert manager.get_load_provenance([key], ctx, 64) is None


def test_mixed_cpu_and_secondary_prefix_is_low_confidence() -> None:
    manager, primary, secondary, _ = make_manager()
    ctx = ReqContext("r")
    cpu_key = make_offload_key(b"cpu", 0)
    fs_key = make_offload_key(b"fs", 0)
    manager.on_new_request(ctx)
    primary.lookup_results = [LookupResult.HIT, LookupResult.MISS]
    primary.prepare_write_result = prepared_write(fs_key)
    secondary.lookup_results = [LookupResult.HIT]

    assert manager.lookup(cpu_key, ctx) is LookupResult.HIT
    assert manager.lookup(fs_key, ctx) is LookupResult.RETRY

    provenance = manager.get_load_provenance([cpu_key, fs_key], ctx, 128)
    assert provenance is not None
    assert provenance.source == "mixed"
    assert provenance.sources == ("cpu_primary", "secondary:filesystem")
    assert provenance.secondary_promoted_tokens is None
    assert provenance.confidence == "low"


def test_reset_cache_clears_active_request_cost_provenance() -> None:
    manager, primary, _, _ = make_manager()
    ctx = ReqContext("r")
    key = make_offload_key(b"cpu", 0)
    manager.on_new_request(ctx)
    primary.lookup_results = [LookupResult.HIT]
    assert manager.lookup(key, ctx) is LookupResult.HIT
    assert manager.get_load_provenance([key], ctx, 64) is not None

    manager.reset_cache()

    assert primary.reset_count == 1
    assert manager.get_load_provenance([key], ctx, 64) is None


def test_successful_promotion_updates_secondary_runtime_scale() -> None:
    manager, primary, secondary, model = make_manager()
    ctx = ReqContext("r")
    key = make_offload_key(b"observe", 0)
    manager.on_new_request(ctx)
    primary.lookup_results = [LookupResult.MISS]
    primary.prepare_write_result = prepared_write(key)
    secondary.lookup_results = [LookupResult.HIT]

    now = [1.0]
    original_monotonic = _MANAGER_MODULE.time.monotonic
    _MANAGER_MODULE.time.monotonic = lambda: now[0]
    try:
        assert manager.lookup(key, ctx) is LookupResult.RETRY
        manager._flush_pending_promotions()
        assert len(secondary.submitted_loads) == 1

        job = secondary.submitted_loads[0]
        now[0] = 1.2
        secondary.finished_jobs = [JobResult(job.job_id, True)]
        manager._process_finished_jobs()
    finally:
        _MANAGER_MODULE.time.monotonic = original_monotonic

    provenance = manager.get_load_provenance([key], ctx, 64)
    assert provenance is not None
    assert provenance.lookup_async_seconds is not None
    assert abs(provenance.lookup_async_seconds - 0.2) < 1e-9

    decision = model.shadow_decide(provenance)
    assert decision is not None
    assert abs(decision.runtime_scale - 1.2) < 1e-9

    stats = manager.get_stats()
    assert stats is not None
    assert (
        _ConnectorMetricName.COST_OBSERVATIONS,
        1,
        ("secondary:filesystem",),
    ) in stats.counters
    assert (
        _ConnectorMetricName.COST_RUNTIME_SCALE,
        1.2,
        ("secondary:filesystem", "64"),
    ) in stats.gauges


def test_failed_completed_promotion_removes_provenance_and_skips_ewma() -> None:
    manager, primary, secondary, model = make_manager()
    ctx = ReqContext("r")
    key = make_offload_key(b"failed", 0)
    manager.on_new_request(ctx)
    primary.lookup_results = [LookupResult.MISS]
    primary.prepare_write_result = prepared_write(key)
    secondary.lookup_results = [LookupResult.HIT]

    now = [1.0]
    original_monotonic = _MANAGER_MODULE.time.monotonic
    _MANAGER_MODULE.time.monotonic = lambda: now[0]
    try:
        assert manager.lookup(key, ctx) is LookupResult.RETRY
        manager._flush_pending_promotions()
        job = secondary.submitted_loads[0]
        now[0] = 1.2
        secondary.finished_jobs = [JobResult(job.job_id, False)]
        manager._process_finished_jobs()
    finally:
        _MANAGER_MODULE.time.monotonic = original_monotonic

    assert manager.get_load_provenance([key], ctx, 64) is None
    unchanged = model.shadow_decide(
        LoadProvenance(
            source="secondary:filesystem",
            external_tokens=64,
            secondary_promoted_tokens=64,
            sources=("secondary:filesystem",),
            confidence="high",
        )
    )
    assert unchanged is not None
    assert unchanged.runtime_scale == 1.0
