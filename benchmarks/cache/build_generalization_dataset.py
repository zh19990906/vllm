# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _record_pair_key(
    record: dict[str, Any],
) -> tuple[int, int, str]:
    return (
        int(record["prompt_tokens"]),
        int(record["concurrency"]),
        str(record["request_rate"]),
    )


def _load_eviction_restore_records(
    run_dir: Path,
    *,
    expected_cache_mode: str,
) -> list[dict[str, Any]]:
    path = run_dir / "scenario-results.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    matches = [
        record
        for record in records
        if record.get("workload_kind") == "eviction-restore"
        and record.get("cache_mode") == expected_cache_mode
    ]

    if not matches:
        raise ValueError(
            "expected at least one eviction-restore "
            f"{expected_cache_mode} record in {run_dir}"
        )

    by_key: dict[
        tuple[int, int, str],
        dict[str, Any],
    ] = {}

    for record in matches:
        key = _record_pair_key(record)
        if key in by_key:
            raise ValueError(
                "duplicate eviction-restore pair key "
                f"{key!r} for {expected_cache_mode} "
                f"in {run_dir}"
            )
        by_key[key] = record

    return [by_key[key] for key in sorted(by_key)]


def _resolve_metadata_path(run_dir: Path, record: dict[str, Any]) -> Path:
    path = Path(str(record["workload_metadata"]))
    if not path.is_absolute():
        path = run_dir / path
    return path


def _workload_hashes(
    run_dir: Path,
    record: dict[str, Any],
) -> tuple[str, str]:
    metadata = _load_json(_resolve_metadata_path(run_dir, record))
    files = metadata["files"]
    return (
        str(files["measure"]["sha256"]),
        str(files["populate"]["sha256"]),
    )


def _metric_delta_sum(
    record: dict[str, Any],
    *,
    base_name: str,
    required_label_fragment: str | None = None,
) -> float:
    delta = record["normalized"]["prometheus"]["delta"] or {}
    total = 0.0

    for key, item in delta.items():
        if key.split("{", 1)[0] != base_name:
            continue
        if required_label_fragment is not None and required_label_fragment not in key:
            continue

        value = item.get("value")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) > 0.0
        ):
            total += float(value)

    return total


def _external_kv_tokens(record: dict[str, Any]) -> int:
    value = _metric_delta_sum(
        record,
        base_name="vllm:prompt_tokens_by_source",
        required_label_fragment='source="external_kv_transfer"',
    )
    return int(value)


def _cpu_to_gpu_bytes(record: dict[str, Any]) -> int:
    directional = _metric_delta_sum(
        record,
        base_name="vllm:kv_offload_size_sum",
        required_label_fragment='transfer_type="CPU_to_GPU"',
    )
    if directional > 0:
        return int(directional)

    # Compatibility with older/synthetic artifacts that exposed
    # load-specific byte counters without a transfer_type label.
    return int(
        _metric_delta_sum(
            record,
            base_name="vllm:kv_offload_load_bytes",
        )
    )


def _kv_offload_allocation_failures(record: dict[str, Any]) -> int:
    return int(
        _metric_delta_sum(
            record,
            base_name="vllm:kv_offload_allocation_failure",
        )
    )


def _cpu_to_gpu_transfers(record: dict[str, Any]) -> int:
    directional = _metric_delta_sum(
        record,
        base_name="vllm:kv_offload_size_count",
        required_label_fragment='transfer_type="CPU_to_GPU"',
    )
    if directional > 0:
        return int(directional)

    # Compatibility with older/synthetic artifacts that exposed
    # load-specific counters without a transfer_type label.
    return int(
        _metric_delta_sum(
            record,
            base_name="vllm:kv_offload_load_size_count",
        )
    )


def _tiered_fs_async_lookup_evidence(
    record: dict[str, Any],
) -> tuple[int, float]:
    item = record["normalized"]["cache"]["tiering_lookup_async_delay_seconds"]

    raw_count = item.get("count")
    raw_sum = item.get("sum")

    count = (
        int(raw_count)
        if isinstance(raw_count, (int, float)) and not isinstance(raw_count, bool)
        else 0
    )
    total_seconds = (
        float(raw_sum)
        if isinstance(raw_sum, (int, float)) and not isinstance(raw_sum, bool)
        else 0.0
    )
    return count, total_seconds


def _ttft(
    record: dict[str, Any],
    *,
    percentile: str,
) -> float:
    value = record["normalized"]["benchmark"]["ttft_ms"][percentile]
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric TTFT {percentile} for {record['case_id']}")
    return float(value)


def _selected_gpu_index(
    manifest: dict[str, Any],
) -> int:
    env = manifest.get("config", {}).get("server", {}).get("env", {})
    if not isinstance(env, dict):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly one numeric GPU index"
        )

    raw_visible = env.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is None:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be explicitly set to exactly "
            "one numeric GPU index"
        )

    visible = str(raw_visible).strip()
    if not visible.isdigit():
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly one numeric GPU index"
        )

    return int(visible)


def _selected_gpu_uuid(
    run_dir: Path,
    *,
    selected_index: int,
) -> str:
    environment = _load_json(run_dir / "environment.json")
    inventory = environment.get("gpu_inventory")

    if not isinstance(inventory, dict):
        raise ValueError("GPU inventory is missing from environment artifact")
    if inventory.get("status") != "available":
        raise ValueError("GPU inventory is unavailable in environment artifact")

    stdout = inventory.get("stdout")
    if not isinstance(stdout, str):
        raise ValueError("GPU inventory stdout is missing")

    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue

        try:
            physical_index = int(parts[0])
        except ValueError:
            continue

        if physical_index != selected_index:
            continue

        gpu_uuid = parts[1].strip()
        if not gpu_uuid:
            raise ValueError(f"selected GPU index {selected_index} has no GPU UUID")
        return gpu_uuid

    raise ValueError(
        f"selected GPU index {selected_index} not found in environment inventory"
    )


def _assert_same_case(
    recompute: dict[str, Any],
    restore: dict[str, Any],
) -> None:
    for key in (
        "prompt_tokens",
        "concurrency",
        "request_rate",
        "model_id",
        "tensor_parallel_size",
    ):
        if recompute[key] != restore[key]:
            raise ValueError(
                f"paired run mismatch for {key}: {recompute[key]!r} != {restore[key]!r}"
            )


def _excluded_sample(
    *,
    source: str,
    restore_record: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "case_id": restore_record.get("case_id"),
        "requested_tokens": restore_record.get("prompt_tokens"),
        "reason": reason,
    }


def _build_sample(
    *,
    source: str,
    recompute_run: Path,
    restore_run: Path,
    recompute_record: dict[str, Any],
    restore_record: dict[str, Any],
    percentile: str,
    requests_per_case: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    _assert_same_case(recompute_record, restore_record)

    if restore_record.get("status") != "completed":
        return None, _excluded_sample(
            source=source,
            restore_record=restore_record,
            reason="restore_record_not_completed",
        )

    recompute_hashes = _workload_hashes(
        recompute_run,
        recompute_record,
    )
    restore_hashes = _workload_hashes(
        restore_run,
        restore_record,
    )
    if recompute_hashes != restore_hashes:
        raise ValueError(
            f"paired workload SHA mismatch for {source}: "
            f"{recompute_hashes!r} != {restore_hashes!r}"
        )

    external_total = _external_kv_tokens(restore_record)
    if external_total <= 0:
        return None, _excluded_sample(
            source=source,
            restore_record=restore_record,
            reason="no_external_kv_tokens",
        )

    if external_total % requests_per_case != 0:
        raise ValueError("external KV tokens not divisible by requests_per_case")

    if _kv_offload_allocation_failures(restore_record) > 0:
        return None, _excluded_sample(
            source=source,
            restore_record=restore_record,
            reason="kv_offload_allocation_failure",
        )

    transfers = _cpu_to_gpu_transfers(restore_record)
    transfer_bytes = _cpu_to_gpu_bytes(restore_record)
    if transfers <= 0 or transfer_bytes <= 0:
        return None, _excluded_sample(
            source=source,
            restore_record=restore_record,
            reason="missing_cpu_to_gpu_transfer_evidence",
        )

    evidence: dict[str, Any] = {
        "cpu_to_gpu_transfers": transfers,
        "cpu_to_gpu_bytes": transfer_bytes,
    }

    if source == "secondary:filesystem":
        async_count, async_sum = _tiered_fs_async_lookup_evidence(restore_record)
        if async_count <= 0 or async_sum <= 0.0:
            return None, _excluded_sample(
                source=source,
                restore_record=restore_record,
                reason="missing_tiered_fs_async_lookup_evidence",
            )

        evidence["tiered_fs_async_lookups"] = async_count
        evidence["tiered_fs_async_lookup_seconds"] = async_sum

    sample = {
        "source": source,
        "requested_tokens": int(recompute_record["prompt_tokens"]),
        "external_kv_tokens_total": external_total,
        "external_kv_tokens_per_request": (external_total // requests_per_case),
        "latency_ms": {
            "recompute": {
                percentile: _ttft(
                    recompute_record,
                    percentile=percentile,
                ),
            },
            "restore": {
                percentile: _ttft(
                    restore_record,
                    percentile=percentile,
                ),
            },
        },
        "workload": {
            "measure_sha256": recompute_hashes[0],
            "populate_sha256": recompute_hashes[1],
        },
        "transfer_evidence": evidence,
    }
    return sample, None


def _manifest_requests_per_case(
    manifest: dict[str, Any],
) -> int:
    try:
        value = int(manifest["config"]["workload"]["requests_per_case"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid requests_per_case in run manifest") from error

    if value <= 0:
        raise ValueError("requests_per_case must be positive in run manifest")
    return value


def build_generalization_dataset(
    *,
    condition_id: str,
    recompute_run: Path,
    cpu_run: Path,
    filesystem_run: Path,
    percentile: str = "p95",
) -> dict[str, Any]:
    recompute_run = Path(recompute_run)
    cpu_run = Path(cpu_run)
    filesystem_run = Path(filesystem_run)

    recompute_manifest = _load_json(recompute_run / "manifest.json")
    cpu_manifest = _load_json(cpu_run / "manifest.json")
    filesystem_manifest = _load_json(filesystem_run / "manifest.json")

    request_counts = {
        "recompute": _manifest_requests_per_case(recompute_manifest),
        "cpu_primary": _manifest_requests_per_case(cpu_manifest),
        "secondary:filesystem": _manifest_requests_per_case(filesystem_manifest),
    }
    if len(set(request_counts.values())) != 1:
        raise ValueError(
            f"requests_per_case mismatch across manifests: {request_counts}"
        )
    requests_per_case = next(iter(request_counts.values()))

    recompute_records = _load_eviction_restore_records(
        recompute_run,
        expected_cache_mode="no-cache",
    )
    cpu_records = _load_eviction_restore_records(
        cpu_run,
        expected_cache_mode="cpu-offload",
    )
    filesystem_records = _load_eviction_restore_records(
        filesystem_run,
        expected_cache_mode="tiered-fs",
    )

    recompute_by_key = {
        _record_pair_key(record): record for record in recompute_records
    }
    cpu_by_key = {_record_pair_key(record): record for record in cpu_records}
    filesystem_by_key = {
        _record_pair_key(record): record for record in filesystem_records
    }

    recompute_keys = set(recompute_by_key)
    if set(cpu_by_key) != recompute_keys or set(filesystem_by_key) != recompute_keys:
        raise ValueError(
            "case identity mismatch across eviction-restore "
            "pair keys (prompt_tokens, concurrency, request_rate): "
            f"recompute={sorted(recompute_by_key)}, "
            f"cpu_primary={sorted(cpu_by_key)}, "
            f"secondary:filesystem={sorted(filesystem_by_key)}"
        )

    first_recompute = recompute_records[0]
    for record in recompute_records:
        if record.get("status") != "completed":
            raise ValueError("recompute record must be completed")

        for field in (
            "model_id",
            "concurrency",
            "request_rate",
            "tensor_parallel_size",
        ):
            if record.get(field) != first_recompute.get(field):
                raise ValueError(
                    "case identity mismatch across recompute "
                    f"records for {field}: "
                    f"{first_recompute.get(field)!r} != "
                    f"{record.get(field)!r}"
                )

    selected_indexes = {
        "recompute": _selected_gpu_index(recompute_manifest),
        "cpu_primary": _selected_gpu_index(cpu_manifest),
        "secondary:filesystem": _selected_gpu_index(filesystem_manifest),
    }

    selected_gpu_uuids = {
        "recompute": _selected_gpu_uuid(
            recompute_run,
            selected_index=selected_indexes["recompute"],
        ),
        "cpu_primary": _selected_gpu_uuid(
            cpu_run,
            selected_index=selected_indexes["cpu_primary"],
        ),
        "secondary:filesystem": _selected_gpu_uuid(
            filesystem_run,
            selected_index=selected_indexes["secondary:filesystem"],
        ),
    }

    if len(set(selected_indexes.values())) != 1:
        raise ValueError(
            f"paired runs use different selected GPU indexes: {selected_indexes}"
        )
    gpu_index = next(iter(selected_indexes.values()))

    gpu_uuids = set(selected_gpu_uuids.values())
    if len(gpu_uuids) != 1:
        raise ValueError(f"paired runs use different GPU UUIDs: {sorted(gpu_uuids)}")
    gpu_uuid = next(iter(gpu_uuids))

    config = recompute_manifest["config"]

    samples: list[dict[str, Any]] = []
    excluded_samples: list[dict[str, Any]] = []

    for pair_key in sorted(recompute_by_key):
        recompute = recompute_by_key[pair_key]

        for source, restore_run, restore_by_key in (
            ("cpu_primary", cpu_run, cpu_by_key),
            (
                "secondary:filesystem",
                filesystem_run,
                filesystem_by_key,
            ),
        ):
            sample, exclusion = _build_sample(
                source=source,
                recompute_run=recompute_run,
                restore_run=restore_run,
                recompute_record=recompute,
                restore_record=restore_by_key[pair_key],
                percentile=percentile,
                requests_per_case=requests_per_case,
            )

            if sample is not None:
                samples.append(sample)
            if exclusion is not None:
                excluded_samples.append(exclusion)

    return {
        "schema_version": 1,
        "issue": 15,
        "condition": {
            "id": condition_id,
            "model": str(config["model"]["id"]),
            "served_model": str(config["model"]["served_name"]),
            "concurrency": int(first_recompute["concurrency"]),
            "request_rate": first_recompute["request_rate"],
            "requests_per_case": requests_per_case,
            "tensor_parallel_size": int(config["parallelism"]["tensor_parallel_size"]),
            "gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
            "environment_artifact": str(recompute_run / "environment.json"),
            "run_directories": {
                "recompute": str(recompute_run),
                "cpu_primary": str(cpu_run),
                "secondary:filesystem": str(filesystem_run),
            },
        },
        "samples": samples,
        "excluded_samples": excluded_samples,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an Issue #15 generalization condition dataset "
            "from cache benchmark run-suite artifacts."
        )
    )
    parser.add_argument(
        "--condition-id",
        required=True,
    )
    parser.add_argument(
        "--recompute-run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cpu-run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--filesystem-run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--percentile",
        choices=("p50", "p95", "p99"),
        default="p95",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    result = build_generalization_dataset(
        condition_id=args.condition_id,
        recompute_run=args.recompute_run,
        cpu_run=args.cpu_run,
        filesystem_run=args.filesystem_run,
        percentile=args.percentile,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "condition={condition} accepted={accepted} excluded={excluded}".format(
            condition=result["condition"]["id"],
            accepted=len(result["samples"]),
            excluded=len(result["excluded_samples"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
