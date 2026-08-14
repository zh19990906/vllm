# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import stat
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import regex as re

logger = logging.getLogger(__name__)

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_GROUP_DIR_RE = re.compile(r"^([0-9a-f]{2})_g[0-9]+$")
_TEMP_FILENAME_RE = re.compile(r"^([0-9a-f]+)\.bin_[0-9]+\.tmp$")
_CAPACITY_LOCK_FILENAME = ".capacity.lock"


class EntryState(Enum):
    COMMITTED = "committed"
    EVICTING = "evicting"
    INVALID = "invalid"


class AdmissionStatus(Enum):
    RESERVED = "reserved"
    ALREADY_PRESENT = "already_present"
    DUPLICATE_INFLIGHT = "duplicate_inflight"
    OVERSIZED = "oversized"
    CAPACITY = "capacity"


@dataclass(slots=True)
class EntryRecord:
    path: str
    size: int
    recency: int
    readers: int
    state: EntryState
    generation: int


@dataclass(slots=True)
class WriteReservation:
    token: int
    path: str
    size: int
    replaced_generation: int | None
    active: bool = True


@dataclass(frozen=True, slots=True)
class ReadPin:
    path: str
    generation: int


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    status: AdmissionStatus
    reservation: WriteReservation | None = None


@dataclass(frozen=True, slots=True)
class CapacitySnapshot:
    max_bytes: int
    accounted_bytes: int
    reserved_bytes: int
    entry_count: int
    pending_write_count: int
    orphan_temp_count: int
    oversized_skip_count: int
    capacity_skip_count: int
    eviction_failure_count: int
    eviction_count: int
    evicted_bytes: int


class FileSystemCapacityManager:
    def __init__(
        self,
        namespace_root: str,
        max_bytes: int,
        expected_file_size: int | None,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise ValueError("max_bytes must be a positive integer")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")

        if expected_file_size is not None and (
            isinstance(expected_file_size, bool)
            or not isinstance(expected_file_size, int)
            or expected_file_size <= 0
        ):
            raise ValueError("expected_file_size must be a positive integer or None")

        self.namespace_root = os.path.abspath(namespace_root)
        self.max_bytes = max_bytes
        self.expected_file_size = expected_file_size

        Path(self.namespace_root).mkdir(parents=True, exist_ok=True)
        self._lock_fd: int | None = None

        # Required lock order:
        # admission_lock -> metadata_lock.
        self._metadata_lock = threading.Lock()
        self._admission_lock = threading.Lock()

        self._entries: dict[str, EntryRecord] = {}
        self._pending_writes: dict[str, WriteReservation] = {}
        self._orphan_temps: dict[str, WriteReservation] = {}

        self._accounted_bytes = 0
        self._reserved_bytes = 0
        self._clock = 0
        self._generation = 0
        self._reservation_token = 0

        self._oversized_skip_count = 0
        self._capacity_skip_count = 0
        self._eviction_failure_count = 0
        self._eviction_count = 0
        self._evicted_bytes = 0

        try:
            self._acquire_namespace_lock()
            self._warn_if_physical_space_below_logical_limit()
            self._recover_existing_finals()
        except BaseException:
            self._release_namespace_lock()
            raise

    def __enter__(self) -> FileSystemCapacityManager:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        try:
            with self._admission_lock:
                self._reap_orphan_temps()
        finally:
            self._release_namespace_lock()

    def snapshot(self) -> CapacitySnapshot:
        with self._metadata_lock:
            return CapacitySnapshot(
                max_bytes=self.max_bytes,
                accounted_bytes=self._accounted_bytes,
                reserved_bytes=self._reserved_bytes,
                entry_count=len(self._entries),
                pending_write_count=len(self._pending_writes),
                orphan_temp_count=len(self._orphan_temps),
                oversized_skip_count=self._oversized_skip_count,
                capacity_skip_count=self._capacity_skip_count,
                eviction_failure_count=self._eviction_failure_count,
                eviction_count=self._eviction_count,
                evicted_bytes=self._evicted_bytes,
            )

    def contains(self, path: str) -> bool:
        normalized = self._normalize_path(path)
        with self._metadata_lock:
            entry = self._entries.get(normalized)
            return entry is not None and entry.state is EntryState.COMMITTED

    def contains_many(self, paths: list[str]) -> list[bool]:
        normalized = [self._normalize_path(path) for path in paths]
        with self._metadata_lock:
            return [
                (
                    (entry := self._entries.get(path)) is not None
                    and entry.state is EntryState.COMMITTED
                )
                for path in normalized
            ]

    def touch(self, paths: Iterable[str]) -> None:
        normalized = [self._normalize_path(path) for path in paths]
        with self._metadata_lock:
            for path in normalized:
                entry = self._entries.get(path)
                if entry is None or entry.state is not EntryState.COMMITTED:
                    continue
                self._clock += 1
                entry.recency = self._clock

    def pin_for_read(self, path: str) -> ReadPin | None:
        normalized = self._normalize_path(path)
        with self._metadata_lock:
            entry = self._entries.get(normalized)
            if entry is None or entry.state is not EntryState.COMMITTED:
                return None
            entry.readers += 1
            return ReadPin(
                path=entry.path,
                generation=entry.generation,
            )

    def release_read(
        self,
        pin: ReadPin,
        *,
        invalidate: bool = False,
    ) -> None:
        # Use the global lock order even when the common path only updates
        # metadata, because INVALID cleanup may require unlink.
        with self._admission_lock:
            with self._metadata_lock:
                entry = self._entries.get(pin.path)
                if entry is None or entry.generation != pin.generation:
                    raise ValueError("read pin is stale")
                if entry.readers <= 0:
                    raise ValueError("read pin is already released")

                if invalidate:
                    entry.state = EntryState.INVALID

                entry.readers -= 1
                should_cleanup = (
                    entry.state is EntryState.INVALID and entry.readers == 0
                )

            if not should_cleanup:
                return

            try:
                os.unlink(pin.path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning(
                    "failed to remove invalid filesystem KV cache entry %s",
                    pin.path,
                    exc_info=True,
                )
                return

            with self._metadata_lock:
                entry = self._entries.get(pin.path)
                if (
                    entry is None
                    or entry.generation != pin.generation
                    or entry.state is not EntryState.INVALID
                    or entry.readers != 0
                ):
                    return

                del self._entries[pin.path]
                self._accounted_bytes -= entry.size
                self._assert_invariants_locked()

    def admit_write(
        self,
        path: str,
        size: int,
        *,
        replace: bool = False,
    ) -> AdmissionResult:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("size must be a positive integer")

        normalized = self._normalize_path(path)

        with self._admission_lock:
            self._reap_orphan_temps()

            with self._metadata_lock:
                existing = self._entries.get(normalized)

                if (
                    existing is not None
                    and existing.state is EntryState.COMMITTED
                    and not replace
                ):
                    if existing.size != size:
                        raise ValueError(
                            "existing committed entry size does not match "
                            "requested write size"
                        )
                    self._clock += 1
                    existing.recency = self._clock
                    return AdmissionResult(status=AdmissionStatus.ALREADY_PRESENT)

                if normalized in self._pending_writes:
                    return AdmissionResult(status=AdmissionStatus.DUPLICATE_INFLIGHT)

                if existing is not None and existing.state is not EntryState.COMMITTED:
                    self._capacity_skip_count += 1
                    return AdmissionResult(status=AdmissionStatus.CAPACITY)

                if size > self.max_bytes:
                    self._oversized_skip_count += 1
                    return AdmissionResult(status=AdmissionStatus.OVERSIZED)

                replaced_generation = (
                    existing.generation if replace and existing is not None else None
                )

            if not self._ensure_capacity_for_write(
                normalized,
                size,
            ):
                with self._metadata_lock:
                    self._capacity_skip_count += 1
                return AdmissionResult(status=AdmissionStatus.CAPACITY)

            with self._metadata_lock:
                # admission_lock excludes concurrent byte-changing
                # transitions, so capacity cannot race between reclaim
                # and reservation installation.
                self._reservation_token += 1
                reservation = WriteReservation(
                    token=self._reservation_token,
                    path=normalized,
                    size=size,
                    replaced_generation=replaced_generation,
                )
                self._pending_writes[normalized] = reservation
                self._reserved_bytes += size

                self._assert_invariants_locked()

                return AdmissionResult(
                    status=AdmissionStatus.RESERVED,
                    reservation=reservation,
                )

    def commit_write(
        self,
        reservation: WriteReservation,
        final_size: int | None = None,
    ) -> None:
        with self._admission_lock:
            if final_size is None:
                final_size = os.path.getsize(reservation.path)

            if (
                isinstance(final_size, bool)
                or not isinstance(final_size, int)
                or final_size <= 0
            ):
                raise ValueError("final_size must be a positive integer")

            if final_size != reservation.size:
                raise ValueError("committed file size does not match reserved size")

            with self._metadata_lock:
                if not reservation.active:
                    raise ValueError("reservation is no longer active")

                pending = self._pending_writes.get(reservation.path)
                if pending is None or pending.token != reservation.token:
                    raise ValueError("reservation is not pending")

                existing = self._entries.get(reservation.path)
                old_size = 0

                if reservation.replaced_generation is not None:
                    if (
                        existing is None
                        or existing.state is not EntryState.COMMITTED
                        or existing.generation != reservation.replaced_generation
                    ):
                        raise ValueError("replacement target changed before commit")
                    old_size = existing.size
                elif existing is not None:
                    raise ValueError(
                        "new-write reservation unexpectedly has an existing entry"
                    )

                del self._pending_writes[reservation.path]
                self._reserved_bytes -= reservation.size
                self._accounted_bytes -= old_size

                self._generation += 1
                self._clock += 1
                self._entries[reservation.path] = EntryRecord(
                    path=reservation.path,
                    size=final_size,
                    recency=self._clock,
                    readers=0,
                    state=EntryState.COMMITTED,
                    generation=self._generation,
                )
                self._accounted_bytes += final_size
                reservation.active = False

                self._assert_invariants_locked()

    def abort_write(self, reservation: WriteReservation) -> None:
        with self._admission_lock, self._metadata_lock:
            if not reservation.active:
                return

            pending = self._pending_writes.get(reservation.path)
            if pending is None or pending.token != reservation.token:
                raise ValueError("reservation is not pending")

            del self._pending_writes[reservation.path]
            self._reserved_bytes -= reservation.size
            reservation.active = False

            self._assert_invariants_locked()

    def retain_orphan_temp(
        self,
        reservation: WriteReservation,
        temp_path: str,
    ) -> None:
        normalized_temp = self._normalize_path(temp_path)

        with self._admission_lock, self._metadata_lock:
            if not reservation.active:
                raise ValueError("reservation is no longer active")

            pending = self._pending_writes.get(reservation.path)
            if pending is None or pending.token != reservation.token:
                raise ValueError("reservation is not pending")

            if normalized_temp in self._orphan_temps:
                raise ValueError("orphan temp path is already tracked")

            del self._pending_writes[reservation.path]
            self._orphan_temps[normalized_temp] = reservation

            # The reservation charge remains in _reserved_bytes until
            # deletion of the temp is confirmed.
            reservation.active = False

            self._assert_invariants_locked()

    def _reap_orphan_temps(self) -> None:
        # admission_lock must be held. Actual unlink happens outside
        # metadata_lock.
        with self._metadata_lock:
            orphans = list(self._orphan_temps.items())

        for temp_path, reservation in orphans:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning(
                    "failed to reap filesystem KV cache temp %s",
                    temp_path,
                    exc_info=True,
                )
                continue

            with self._metadata_lock:
                current = self._orphan_temps.get(temp_path)
                if current is None or current.token != reservation.token:
                    continue

                del self._orphan_temps[temp_path]
                self._reserved_bytes -= current.size
                self._assert_invariants_locked()

    def _ensure_capacity_for_write(
        self,
        incoming_path: str,
        size: int,
    ) -> bool:
        # admission_lock must be held by the caller.
        excluded: set[str] = {incoming_path}

        while True:
            with self._metadata_lock:
                if (
                    self._accounted_bytes + self._reserved_bytes + size
                    <= self.max_bytes
                ):
                    return True

                candidates = [
                    entry
                    for entry in self._entries.values()
                    if (
                        entry.path not in excluded
                        and entry.state is EntryState.COMMITTED
                        and entry.readers == 0
                    )
                ]
                if not candidates:
                    return False

                victim = min(
                    candidates,
                    key=lambda entry: (
                        entry.recency,
                        entry.path,
                    ),
                )
                victim.state = EntryState.EVICTING
                victim_generation = victim.generation
                victim_size = victim.size
                victim_path = victim.path

            try:
                os.unlink(victim_path)
            except FileNotFoundError:
                pass
            except OSError:
                with self._metadata_lock:
                    current = self._entries.get(victim_path)
                    if (
                        current is not None
                        and current.generation == victim_generation
                        and current.state is EntryState.EVICTING
                    ):
                        current.state = EntryState.COMMITTED
                    self._eviction_failure_count += 1
                    self._assert_invariants_locked()

                excluded.add(victim_path)
                logger.warning(
                    "failed to evict filesystem KV cache entry %s",
                    victim_path,
                    exc_info=True,
                )
                continue

            with self._metadata_lock:
                current = self._entries.get(victim_path)
                if (
                    current is None
                    or current.generation != victim_generation
                    or current.state is not EntryState.EVICTING
                ):
                    raise RuntimeError(
                        "filesystem KV cache eviction victim changed unexpectedly"
                    )

                del self._entries[victim_path]
                self._accounted_bytes -= victim_size
                self._eviction_count += 1
                self._evicted_bytes += victim_size
                self._assert_invariants_locked()

    def _warn_if_physical_space_below_logical_limit(self) -> None:
        try:
            free_bytes = shutil.disk_usage(self.namespace_root).free
        except OSError:
            logger.warning(
                "failed to query physical free space for filesystem KV "
                "cache namespace %s; logical max_bytes remains %d",
                self.namespace_root,
                self.max_bytes,
                exc_info=True,
            )
            return

        if free_bytes < self.max_bytes:
            logger.warning(
                "filesystem KV cache logical max_bytes=%d exceeds current "
                "physical free space=%d for %s; max_bytes remains the "
                "configured logical ceiling and physical ENOSPC/EDQUOT "
                "may occur earlier",
                self.max_bytes,
                free_bytes,
                self.namespace_root,
            )

    def _acquire_namespace_lock(self) -> None:
        if self._lock_fd is not None:
            raise RuntimeError("capacity lock is already held")

        lock_path = os.path.join(
            self.namespace_root,
            _CAPACITY_LOCK_FILENAME,
        )
        flags = os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)

        try:
            fd = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise RuntimeError(
                f"failed to open filesystem KV cache capacity lock: {lock_path}"
            ) from exc

        try:
            fcntl.flock(
                fd,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as exc:
            os.close(fd)
            raise RuntimeError(
                "filesystem KV cache namespace is already owned "
                f"or its capacity lock cannot be acquired: "
                f"{self.namespace_root}"
            ) from exc

        self._lock_fd = fd

    def _release_namespace_lock(self) -> None:
        fd = self._lock_fd
        if fd is None:
            return

        self._lock_fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _recover_existing_finals(self) -> None:
        recovered: list[tuple[int, str, int]] = []
        root = Path(self.namespace_root)
        lock_path = root / _CAPACITY_LOCK_FILENAME
        directories = [root]

        while directories:
            directory = directories.pop()

            try:
                with os.scandir(directory) as iterator:
                    children = sorted(
                        iterator,
                        key=lambda child: child.name,
                    )
            except OSError as exc:
                raise RuntimeError(
                    f"failed to scan filesystem KV cache namespace: {directory}"
                ) from exc

            for child in children:
                path = Path(child.path)

                if path == lock_path:
                    continue

                try:
                    metadata = child.stat(follow_symlinks=False)
                except OSError as exc:
                    raise RuntimeError(
                        f"failed to inspect filesystem KV cache artifact: {path}"
                    ) from exc

                mode = metadata.st_mode

                if stat.S_ISDIR(mode):
                    directories.append(path)
                    continue

                if stat.S_ISLNK(mode):
                    raise RuntimeError(
                        "unknown symlink artifact in filesystem KV cache "
                        f"namespace: {path}"
                    )

                if not stat.S_ISREG(mode):
                    raise RuntimeError(
                        "unknown special artifact in filesystem KV cache "
                        f"namespace: {path}"
                    )

                if self._is_recognized_temp(path):
                    self._startup_unlink(
                        path,
                        operation="temporary artifact cleanup",
                    )
                    continue

                if not self._is_managed_final(path):
                    raise RuntimeError(f"unknown filesystem KV cache artifact: {path}")

                size = metadata.st_size
                if (
                    self.expected_file_size is not None
                    and size != self.expected_file_size
                ):
                    self._startup_unlink(
                        path,
                        operation=(
                            "corrupt final size cleanup "
                            f"(expected {self.expected_file_size}, "
                            f"found {size})"
                        ),
                    )
                    continue

                recovered.append(
                    (
                        metadata.st_mtime_ns,
                        str(path),
                        size,
                    )
                )

        # Restart recency is deterministic but intentionally approximate:
        # oldest mtime first, stable path tie-break second.
        recovered.sort(key=lambda item: (item[0], item[1]))

        total = sum(size for _, _, size in recovered)
        first_retained = 0

        while total > self.max_bytes:
            _, victim_path, victim_size = recovered[first_retained]
            self._startup_unlink(
                Path(victim_path),
                operation="restart capacity shrink eviction",
            )
            self._eviction_count += 1
            self._evicted_bytes += victim_size
            total -= victim_size
            first_retained += 1

        for _, recovered_path, size in recovered[first_retained:]:
            self._generation += 1
            self._clock += 1
            self._entries[recovered_path] = EntryRecord(
                path=recovered_path,
                size=size,
                recency=self._clock,
                readers=0,
                state=EntryState.COMMITTED,
                generation=self._generation,
            )
            self._accounted_bytes += size

        self._assert_invariants_locked()

    def _startup_unlink(
        self,
        path: Path,
        *,
        operation: str,
    ) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            # Absence is the state we needed to confirm.
            return
        except OSError as exc:
            raise RuntimeError(
                f"filesystem KV cache {operation} failed: {path}"
            ) from exc

    def _is_recognized_temp(self, path: Path) -> bool:
        match = _TEMP_FILENAME_RE.fullmatch(path.name)
        if match is None:
            return False

        final_path = path.with_name(f"{match.group(1)}.bin")
        return self._is_managed_final(final_path)

    def _is_managed_final(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.namespace_root)
        except ValueError:
            return False

        if len(relative.parts) != 3:
            return False

        first, group_dir, filename = relative.parts
        if len(first) != 3 or _HEX_RE.fullmatch(first) is None:
            return False

        group_match = _GROUP_DIR_RE.fullmatch(group_dir)
        if group_match is None:
            return False

        file_path = Path(filename)
        if file_path.suffix != ".bin":
            return False

        hash_hex = file_path.stem
        if len(hash_hex) < 5 or _HEX_RE.fullmatch(hash_hex) is None:
            return False

        return hash_hex[:3] == first and hash_hex[3:5] == group_match.group(1)

    def _normalize_path(self, path: str) -> str:
        normalized = os.path.abspath(path)
        try:
            common = os.path.commonpath([self.namespace_root, normalized])
        except ValueError as exc:
            raise ValueError("managed path must be inside namespace_root") from exc

        if common != self.namespace_root:
            raise ValueError("managed path must be inside namespace_root")

        return normalized

    def _assert_invariants_locked(self) -> None:
        assert self._accounted_bytes >= 0
        assert self._reserved_bytes >= 0
        assert self._accounted_bytes + self._reserved_bytes <= self.max_bytes
