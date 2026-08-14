# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic real-filesystem validation for Issue #31."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Direct invocation sets sys.path[0] to benchmarks/cache. Put the
# repository root first so this smoke always validates the current
# worktree rather than an installed vLLM package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from vllm.v1.kv_offload.tiering.fs.capacity import (  # noqa: E402
    AdmissionStatus,
    FileSystemCapacityManager,
)

SCHEMA_VERSION = 1
BLOCK_SIZE = 4096
MAX_BLOCKS = 3


def _managed_path(root: Path, value: int) -> Path:
    block_hash = f"{value:016x}"
    return root / block_hash[:3] / f"{block_hash[3:5]}_g0" / f"{block_hash}.bin"


def _write_final(path: Path, size: int, fill: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([fill]) * size)


def _payload_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*.bin") if path.is_file())


def _temp_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*.tmp") if path.is_file())


def run_smoke(root: Path) -> dict[str, Any]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if any(root.iterdir()):
        raise ValueError(f"smoke root must be an owned empty directory: {root}")

    max_bytes = MAX_BLOCKS * BLOCK_SIZE
    paths = [_managed_path(root, i) for i in range(1, 9)]

    peak_accounted = 0
    peak_reserved = 0
    peak_combined = 0

    def observe(cap: FileSystemCapacityManager) -> None:
        nonlocal peak_accounted, peak_reserved, peak_combined
        snap = cap.snapshot()
        combined = snap.accounted_bytes + snap.reserved_bytes
        if combined > cap.max_bytes:
            raise AssertionError(
                f"logical hard capacity exceeded: {combined} > {cap.max_bytes}"
            )
        peak_accounted = max(
            peak_accounted,
            snap.accounted_bytes,
        )
        peak_reserved = max(
            peak_reserved,
            snap.reserved_bytes,
        )
        peak_combined = max(peak_combined, combined)

    runtime_eviction_observed = False
    ownership_conflict_rejected = False
    temp_peak_observed = False

    runtime_eviction_count = 0
    runtime_evicted_bytes = 0
    oversized_skips = 0
    no_evictable_skips = 0

    with FileSystemCapacityManager(
        namespace_root=str(root),
        max_bytes=max_bytes,
        expected_file_size=BLOCK_SIZE,
    ) as cap:
        # Fill the logical capacity with three real final files.
        for index, path in enumerate(paths[:3], start=1):
            admission = cap.admit_write(str(path), BLOCK_SIZE)
            if admission.status is not AdmissionStatus.RESERVED:
                raise AssertionError(admission.status)
            _write_final(path, BLOCK_SIZE, index)
            cap.commit_write(admission.reservation)
            observe(cap)

        # A live second owner must be rejected.
        try:
            FileSystemCapacityManager(
                namespace_root=str(root),
                max_bytes=max_bytes,
                expected_file_size=BLOCK_SIZE,
            )
        except RuntimeError:
            ownership_conflict_rejected = True
        else:
            raise AssertionError("second filesystem capacity owner was accepted")

        # Runtime LRU: touch path[0], so path[1] is the oldest victim.
        cap.touch([str(paths[0])])
        admission = cap.admit_write(
            str(paths[3]),
            BLOCK_SIZE,
        )
        if admission.status is not AdmissionStatus.RESERVED:
            raise AssertionError(admission.status)

        if paths[1].exists():
            raise AssertionError("runtime LRU did not unlink the expected oldest entry")
        if not paths[0].exists() or not paths[2].exists():
            raise AssertionError("runtime LRU removed a non-victim entry")

        _write_final(paths[3], BLOCK_SIZE, 4)
        cap.commit_write(admission.reservation)
        observe(cap)

        snap = cap.snapshot()
        runtime_eviction_observed = snap.eviction_count >= 1
        runtime_eviction_count = snap.eviction_count
        runtime_evicted_bytes = snap.evicted_bytes

        # Oversized admission is a normal capacity skip.
        oversized = cap.admit_write(
            str(paths[7]),
            max_bytes + 1,
        )
        if oversized.status is not AdmissionStatus.OVERSIZED:
            raise AssertionError(oversized.status)

        # Pin every committed entry: normal-size admission has no
        # evictable victim and must skip without reserving.
        committed = [path for path in (paths[0], paths[2], paths[3]) if path.exists()]
        pins = [cap.pin_for_read(str(path)) for path in committed]
        if any(pin is None for pin in pins):
            raise AssertionError("failed to pin committed entry")

        try:
            blocked = cap.admit_write(
                str(paths[4]),
                BLOCK_SIZE,
            )
            if blocked.status is not AdmissionStatus.CAPACITY:
                raise AssertionError(blocked.status)
            if blocked.reservation is not None:
                raise AssertionError("capacity skip unexpectedly owns reservation")
        finally:
            for pin in pins:
                cap.release_read(pin)

        observe(cap)

        # Obtain two concurrent reservations. Admission evicts two old
        # finals, so accounted + reserved stays exactly within max.
        reservations = []
        temp_paths = []

        for sequence, path in enumerate(
            (paths[4], paths[5]),
            start=1001,
        ):
            admission = cap.admit_write(str(path), BLOCK_SIZE)
            if admission.status is not AdmissionStatus.RESERVED:
                raise AssertionError(admission.status)

            reservations.append(admission.reservation)
            temp_path = Path(f"{path}_{sequence}.tmp")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(b"t" * BLOCK_SIZE)
            temp_paths.append(temp_path)
            observe(cap)

        peak_snap = cap.snapshot()
        physical_temp_bytes = _temp_bytes(root)
        physical_final_bytes = _payload_bytes(root)

        temp_peak_observed = (
            physical_temp_bytes == 2 * BLOCK_SIZE
            and peak_snap.reserved_bytes == 2 * BLOCK_SIZE
            and physical_final_bytes == BLOCK_SIZE
            and (peak_snap.accounted_bytes + peak_snap.reserved_bytes == max_bytes)
        )
        if not temp_peak_observed:
            raise AssertionError("real temp-file capacity peak was not observed")

        # Atomically publish each temp and transfer reservation to
        # committed accounting.
        for reservation, temp_path in zip(
            reservations,
            temp_paths,
        ):
            os.replace(temp_path, reservation.path)
            cap.commit_write(reservation)
            observe(cap)

        if _temp_bytes(root) != 0:
            raise AssertionError("stale temp remained after commit")

        runtime_snap = cap.snapshot()
        runtime_eviction_count = runtime_snap.eviction_count
        runtime_evicted_bytes = runtime_snap.evicted_bytes
        oversized_skips = runtime_snap.oversized_skip_count
        no_evictable_skips = runtime_snap.capacity_skip_count

    # Restart with the same max: accounting must be rebuilt from finals.
    with FileSystemCapacityManager(
        namespace_root=str(root),
        max_bytes=max_bytes,
        expected_file_size=BLOCK_SIZE,
    ) as recovered:
        restart_snap = recovered.snapshot()
        restart_recovered_bytes = restart_snap.accounted_bytes
        restart_recovery_ok = (
            restart_recovered_bytes == _payload_bytes(root) == max_bytes
        )

    # Restart with a smaller max: constructor must synchronously shrink.
    smaller_max = 2 * BLOCK_SIZE
    with FileSystemCapacityManager(
        namespace_root=str(root),
        max_bytes=smaller_max,
        expected_file_size=BLOCK_SIZE,
    ) as smaller:
        smaller_snap = smaller.snapshot()
        startup_shrink_ok = (
            smaller_snap.accounted_bytes == smaller_max
            and _payload_bytes(root) == smaller_max
            and smaller_snap.eviction_count >= 1
            and smaller_snap.evicted_bytes >= BLOCK_SIZE
        )
        startup_eviction_count = smaller_snap.eviction_count
        startup_evicted_bytes = smaller_snap.evicted_bytes

    disk = shutil.disk_usage(root)
    final_payload_bytes = _payload_bytes(root)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "filesystem_provenance": "filesystem",
        "root": str(root),
        "max_bytes": max_bytes,
        "block_size": BLOCK_SIZE,
        "peak_accounted_bytes": peak_accounted,
        "peak_reserved_bytes": peak_reserved,
        "peak_accounted_plus_reserved_bytes": peak_combined,
        "temp_peak_observed": temp_peak_observed,
        "runtime_eviction_observed": runtime_eviction_observed,
        "eviction_count": runtime_eviction_count,
        "evicted_bytes": runtime_evicted_bytes,
        "capacity_skips": {
            "oversized": oversized_skips,
            "no_evictable_capacity": no_evictable_skips,
        },
        "restart_recovered_bytes": restart_recovered_bytes,
        "restart_recovery_ok": restart_recovery_ok,
        "startup_shrink_ok": startup_shrink_ok,
        "startup_eviction_count": startup_eviction_count,
        "startup_evicted_bytes": startup_evicted_bytes,
        "ownership_conflict_rejected": ownership_conflict_rejected,
        "final_payload_apparent_bytes": final_payload_bytes,
        "physical_filesystem": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }

    if not runtime_eviction_observed:
        raise AssertionError("runtime eviction was not observed")
    if not restart_recovery_ok:
        raise AssertionError("restart accounting recovery failed")
    if not startup_shrink_ok:
        raise AssertionError("startup shrink failed")
    if not ownership_conflict_rejected:
        raise AssertionError("ownership conflict was not rejected")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = run_smoke(args.root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(
        "issue31_fs_capacity_smoke=OK "
        f"peak_combined={result['peak_accounted_plus_reserved_bytes']} "
        f"max_bytes={result['max_bytes']}"
    )


if __name__ == "__main__":
    main()
