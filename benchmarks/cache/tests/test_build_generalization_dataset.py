# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "base_commit": "test-commit",
        "config": {
            "schema_version": 1,
            "model": {
                "id": "/mnt/model/Qwen2.5-14B-Instruct",
                "served_name": "qwen2.5-14b",
                "dtype": "auto",
                "max_model_len": 32768,
                "trust_remote_code": False,
            },
            "parallelism": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            "server": {
                "host": "127.0.0.1",
                "port": 8100,
                "startup_timeout_seconds": 900,
                "shutdown_timeout_seconds": 60,
                "extra_args": [],
                "env": {
                    "CUDA_VISIBLE_DEVICES": "0",
                },
            },
            "cache": {
                "gpu_memory_utilization": 0.9,
                "cpu_bytes_to_use": 8589934592,
                "offload_block_size": 64,
                "eviction_policy": "lru",
                "filesystem": {
                    "enabled": False,
                    "root_dir": "/tmp/vllm-kv-cache",
                    "read_threads": 32,
                    "write_threads": 16,
                },
            },
            "workload": {
                "seed": 1,
                "tokenizer": "/mnt/model/Qwen2.5-14B-Instruct",
                "prompt_tokens": [256],
                "output_tokens": 1,
                "concurrency": [1],
                "request_rate": ["inf"],
                "requests_per_case": 8,
                "pressure_fill_requests": 0,
                "pressure_fill_tokens": 65536,
                "shared_prefix_ratios": [0.0],
                "warmup_requests": 0,
                "token_length_tolerance": 2,
            },
            "results": {
                "root_dir": "/tmp/cache-results",
                "keep_server_logs": True,
                "fail_fast": False,
            },
        },
        "config_fingerprint": "fixture",
        "selected_case_ids": [],
        "dry_run": False,
        "created_at": "2026-08-11T00:00:00+00:00",
        "cases": [],
    }


def _environment() -> dict:
    return {
        "gpu_inventory": {
            "status": "available",
            "command": [
                "nvidia-smi",
                ("--query-gpu=index,uuid,name,memory.total,driver_version"),
                "--format=csv,noheader",
            ],
            "stdout": (
                "0, GPU-test-0, NVIDIA RTX PRO 5000, "
                "73415 MiB, 580.126.09\n"
                "1, GPU-test-1, NVIDIA RTX PRO 5000, "
                "73415 MiB, 580.126.09"
            ),
        },
        "git_commit": {
            "status": "available",
            "stdout": "test-commit",
        },
    }


def _metadata(
    path: Path,
    *,
    cache_mode: str,
    measure_sha: str = "same-measure-sha",
    populate_sha: str = "same-populate-sha",
) -> None:
    _write_json(
        path,
        {
            "case_id": f"{cache_mode}-case",
            "cache_mode": cache_mode,
            "workload_kind": "eviction-restore",
            "prompt_tokens": 256,
            "prefix_ratio": 0.0,
            "concurrency": 1,
            "request_rate": "inf",
            "repetition": 1,
            "generator_seed": 123,
            "pressure_fill_requests": 0,
            "pressure_fill_tokens": 65536,
            "derived_pressure_fill_requests": 256,
            "files": {
                "measure": {
                    "path": str(path.parent / "measure.jsonl"),
                    "sha256": measure_sha,
                    "requested_token_length": 256,
                    "observed_token_lengths": [256] * 8,
                    "rows": 8,
                },
                "populate": {
                    "path": str(path.parent / "populate.jsonl"),
                    "sha256": populate_sha,
                    "requested_token_length": 256,
                    "observed_token_lengths": [256] * 264,
                    "rows": 264,
                },
            },
        },
    )


def _delta(
    *,
    external_tokens: int = 0,
    load_bytes: int = 0,
    load_count: int = 0,
) -> dict:
    return {
        (
            "vllm:prompt_tokens_by_source{"
            'engine="0",model_name="qwen2.5-14b",'
            'source="external_kv_transfer"}'
        ): {
            "value": external_tokens,
            "reason": None,
        },
        'vllm:kv_offload_load_bytes{engine="0"}': {
            "value": load_bytes,
            "reason": None,
        },
        'vllm:kv_offload_load_size_count{engine="0"}': {
            "value": load_count,
            "reason": None,
        },
    }


def _record(
    *,
    cache_mode: str,
    metadata_path: Path,
    p95_ttft_ms: float,
    external_tokens: int = 0,
    load_bytes: int = 0,
    load_count: int = 0,
    tiered_async_count: int = 0,
) -> dict:
    return {
        "case_id": f"{cache_mode}-case",
        "cache_mode": cache_mode,
        "workload_kind": "eviction-restore",
        "prompt_tokens": 256,
        "prefix_ratio": 0.0,
        "concurrency": 1,
        "request_rate": "inf",
        "repetition": 1,
        "model_id": "/mnt/model/Qwen2.5-14B-Instruct",
        "model_revision": None,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "commands": {},
        "logs": {},
        "status": "completed",
        "started_at": "2026-08-11T00:00:00+00:00",
        "ended_at": "2026-08-11T00:00:01+00:00",
        "workload_metadata": str(metadata_path),
        "normalized": {
            "benchmark": {
                "completed": 8,
                "failed": 0,
                "ttft_ms": {
                    "mean": p95_ttft_ms,
                    "median": p95_ttft_ms,
                    "p50": p95_ttft_ms,
                    "p95": p95_ttft_ms,
                    "p99": p95_ttft_ms,
                },
            },
            "cache": {
                "tiering_lookup_async_delay_seconds": {
                    "value": (0.001 if tiered_async_count else None),
                    "sum": (0.008 if tiered_async_count else None),
                    "count": (tiered_async_count if tiered_async_count else None),
                    "reason": (None if tiered_async_count else "metric_not_exposed"),
                },
            },
            "prometheus": {
                "before": {},
                "after": {},
                "delta": _delta(
                    external_tokens=external_tokens,
                    load_bytes=load_bytes,
                    load_count=load_count,
                ),
                "unavailable_reason": None,
            },
        },
        "resources": {},
        "command_result": {},
    }


def _write_run(
    root: Path,
    *,
    cache_mode: str,
    p95_ttft_ms: float,
    external_tokens: int = 0,
    load_bytes: int = 0,
    load_count: int = 0,
    tiered_async_count: int = 0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    if cache_mode == "tiered-fs":
        manifest["config"]["cache"]["filesystem"]["enabled"] = True
        manifest["config"]["cache"]["cpu_bytes_to_use"] = 2147483648

    _write_json(root / "manifest.json", manifest)
    _write_json(root / "environment.json", _environment())

    metadata_path = root / "case" / "metadata.json"
    _metadata(metadata_path, cache_mode=cache_mode)

    record = _record(
        cache_mode=cache_mode,
        metadata_path=metadata_path,
        p95_ttft_ms=p95_ttft_ms,
        external_tokens=external_tokens,
        load_bytes=load_bytes,
        load_count=load_count,
        tiered_async_count=tiered_async_count,
    )
    (root / "scenario-results.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


class GeneralizationDatasetBuilderTests(unittest.TestCase):
    def test_builds_condition_from_real_run_suite_shapes(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            recompute_run = _write_run(
                base / "recompute",
                cache_mode="no-cache",
                p95_ttft_ms=30.0,
            )
            cpu_run = _write_run(
                base / "cpu",
                cache_mode="cpu-offload",
                p95_ttft_ms=24.0,
                external_tokens=1856,
                load_bytes=106430464,
                load_count=8,
            )
            filesystem_run = _write_run(
                base / "filesystem",
                cache_mode="tiered-fs",
                p95_ttft_ms=38.0,
                external_tokens=1856,
                load_bytes=106430464,
                load_count=8,
                tiered_async_count=8,
            )

            result = build_generalization_dataset(
                condition_id="c-model",
                recompute_run=recompute_run,
                cpu_run=cpu_run,
                filesystem_run=filesystem_run,
                percentile="p95",
            )

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["issue"], 15)

        condition = result["condition"]
        self.assertEqual(condition["id"], "c-model")
        self.assertEqual(
            condition["model"],
            "/mnt/model/Qwen2.5-14B-Instruct",
        )
        self.assertEqual(condition["served_model"], "qwen2.5-14b")
        self.assertEqual(condition["concurrency"], 1)
        self.assertEqual(condition["request_rate"], "inf")
        self.assertEqual(condition["requests_per_case"], 8)
        self.assertEqual(condition["tensor_parallel_size"], 1)
        self.assertEqual(condition["gpu_uuid"], "GPU-test-0")
        self.assertEqual(
            set(condition["run_directories"]),
            {
                "recompute",
                "cpu_primary",
                "secondary:filesystem",
            },
        )

        self.assertEqual(len(result["samples"]), 2)
        samples = {sample["source"]: sample for sample in result["samples"]}

        cpu = samples["cpu_primary"]
        self.assertEqual(cpu["requested_tokens"], 256)
        self.assertEqual(cpu["external_kv_tokens_total"], 1856)
        self.assertEqual(cpu["external_kv_tokens_per_request"], 232)
        self.assertEqual(cpu["latency_ms"]["recompute"]["p95"], 30.0)
        self.assertEqual(cpu["latency_ms"]["restore"]["p95"], 24.0)
        self.assertEqual(
            cpu["transfer_evidence"]["cpu_to_gpu_transfers"],
            8,
        )
        self.assertEqual(
            cpu["transfer_evidence"]["cpu_to_gpu_bytes"],
            106430464,
        )

        filesystem = samples["secondary:filesystem"]
        self.assertEqual(
            filesystem["external_kv_tokens_per_request"],
            232,
        )
        self.assertEqual(
            filesystem["transfer_evidence"]["tiered_fs_async_lookups"],
            8,
        )

        for sample in samples.values():
            self.assertEqual(
                sample["workload"]["measure_sha256"],
                "same-measure-sha",
            )
            self.assertEqual(
                sample["workload"]["populate_sha256"],
                "same-populate-sha",
            )

        self.assertEqual(result["excluded_samples"], [])


if __name__ == "__main__":
    unittest.main()


def _rewrite_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)


def _rewrite_only_record(run_dir: Path, mutate) -> None:
    path = run_dir / "scenario-results.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    mutate(rows[0])
    path.write_text(
        json.dumps(rows[0], sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_three_runs(base: Path) -> tuple[Path, Path, Path]:
    recompute = _write_run(
        base / "recompute",
        cache_mode="no-cache",
        p95_ttft_ms=30.0,
    )
    cpu = _write_run(
        base / "cpu",
        cache_mode="cpu-offload",
        p95_ttft_ms=24.0,
        external_tokens=1856,
        load_bytes=106430464,
        load_count=8,
    )
    filesystem = _write_run(
        base / "filesystem",
        cache_mode="tiered-fs",
        p95_ttft_ms=38.0,
        external_tokens=1856,
        load_bytes=106430464,
        load_count=8,
        tiered_async_count=8,
    )
    return recompute, cpu, filesystem


class GeneralizationDatasetValidationTests(unittest.TestCase):
    def test_rejects_mismatched_workload_sha(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            _rewrite_json(
                cpu / "case" / "metadata.json",
                lambda payload: payload["files"]["measure"].update(
                    {"sha256": "different-measure-sha"}
                ),
            )

            with self.assertRaisesRegex(ValueError, "workload SHA"):
                build_generalization_dataset(
                    condition_id="c-model",
                    recompute_run=recompute,
                    cpu_run=cpu,
                    filesystem_run=filesystem,
                    percentile="p95",
                )

    def test_rejects_different_gpu_uuid(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            def change_gpu(payload: dict) -> None:
                stdout = payload["gpu_inventory"]["stdout"]
                payload["gpu_inventory"]["stdout"] = stdout.replace(
                    "GPU-test-0",
                    "GPU-other-0",
                    1,
                )

            _rewrite_json(cpu / "environment.json", change_gpu)

            with self.assertRaisesRegex(ValueError, "different GPU UUID"):
                build_generalization_dataset(
                    condition_id="c-model",
                    recompute_run=recompute,
                    cpu_run=cpu,
                    filesystem_run=filesystem,
                    percentile="p95",
                )

    def test_rejects_case_identity_mismatch(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            _rewrite_only_record(
                cpu,
                lambda record: record.update({"concurrency": 4}),
            )

            with self.assertRaisesRegex(ValueError, "concurrency"):
                build_generalization_dataset(
                    condition_id="c-model",
                    recompute_run=recompute,
                    cpu_run=cpu,
                    filesystem_run=filesystem,
                    percentile="p95",
                )

    def test_zero_external_kv_excludes_cpu_sample(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            _rewrite_only_record(
                cpu,
                lambda record: record["normalized"]["prometheus"]["delta"].update(
                    _delta(
                        external_tokens=0,
                        load_bytes=106430464,
                        load_count=8,
                    )
                ),
            )

            result = build_generalization_dataset(
                condition_id="c-model",
                recompute_run=recompute,
                cpu_run=cpu,
                filesystem_run=filesystem,
                percentile="p95",
            )

        self.assertEqual(
            [row["source"] for row in result["samples"]],
            ["secondary:filesystem"],
        )
        self.assertEqual(len(result["excluded_samples"]), 1)
        self.assertEqual(
            result["excluded_samples"][0]["source"],
            "cpu_primary",
        )
        self.assertEqual(
            result["excluded_samples"][0]["reason"],
            "no_external_kv_tokens",
        )

    def test_missing_cpu_transfer_evidence_excludes_cpu_sample(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            _rewrite_only_record(
                cpu,
                lambda record: record["normalized"]["prometheus"]["delta"].update(
                    _delta(
                        external_tokens=1856,
                        load_bytes=0,
                        load_count=0,
                    )
                ),
            )

            result = build_generalization_dataset(
                condition_id="c-model",
                recompute_run=recompute,
                cpu_run=cpu,
                filesystem_run=filesystem,
                percentile="p95",
            )

        self.assertEqual(
            [row["source"] for row in result["samples"]],
            ["secondary:filesystem"],
        )
        self.assertEqual(
            result["excluded_samples"][0]["reason"],
            "missing_cpu_to_gpu_transfer_evidence",
        )

    def test_missing_async_lookup_excludes_filesystem_sample(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            def remove_async(record: dict) -> None:
                item = record["normalized"]["cache"][
                    "tiering_lookup_async_delay_seconds"
                ]
                item.update(
                    {
                        "value": None,
                        "sum": None,
                        "count": None,
                        "reason": "metric_not_exposed",
                    }
                )

            _rewrite_only_record(filesystem, remove_async)

            result = build_generalization_dataset(
                condition_id="c-model",
                recompute_run=recompute,
                cpu_run=cpu,
                filesystem_run=filesystem,
                percentile="p95",
            )

        self.assertEqual(
            [row["source"] for row in result["samples"]],
            ["cpu_primary"],
        )
        self.assertEqual(
            result["excluded_samples"][0]["source"],
            "secondary:filesystem",
        )
        self.assertEqual(
            result["excluded_samples"][0]["reason"],
            "missing_tiered_fs_async_lookup_evidence",
        )

    def test_incomplete_restore_record_is_explicitly_excluded(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            _rewrite_only_record(
                cpu,
                lambda record: record.update(
                    {
                        "status": "benchmark_error",
                        "error": {
                            "stage": "benchmark",
                            "type": "BenchmarkExecutionError",
                            "message": "fixture failure",
                            "retryable": True,
                        },
                    }
                ),
            )

            result = build_generalization_dataset(
                condition_id="c-model",
                recompute_run=recompute,
                cpu_run=cpu,
                filesystem_run=filesystem,
                percentile="p95",
            )

        self.assertEqual(
            [row["source"] for row in result["samples"]],
            ["secondary:filesystem"],
        )
        self.assertEqual(
            result["excluded_samples"][0]["source"],
            "cpu_primary",
        )
        self.assertEqual(
            result["excluded_samples"][0]["reason"],
            "restore_record_not_completed",
        )

    def test_requests_per_case_is_not_hardcoded_to_eight(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            for run in (recompute, cpu, filesystem):
                _rewrite_json(
                    run / "manifest.json",
                    lambda payload: payload["config"]["workload"].update(
                        {"requests_per_case": 3}
                    ),
                )

            for run in (cpu, filesystem):

                def update_transfer(record: dict) -> None:
                    delta = record["normalized"]["prometheus"]["delta"]
                    delta.update(
                        _delta(
                            external_tokens=699,
                            load_bytes=40083456,
                            load_count=3,
                        )
                    )

                _rewrite_only_record(run, update_transfer)

            result = build_generalization_dataset(
                condition_id="c-model",
                recompute_run=recompute,
                cpu_run=cpu,
                filesystem_run=filesystem,
                percentile="p95",
            )

        self.assertEqual(
            result["condition"]["requests_per_case"],
            3,
        )
        self.assertEqual(
            {row["external_kv_tokens_per_request"] for row in result["samples"]},
            {233},
        )


class GeneralizationDatasetFinalContractTests(unittest.TestCase):
    def test_rejects_requests_per_case_manifest_mismatch(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            _rewrite_json(
                cpu / "manifest.json",
                lambda payload: payload["config"]["workload"].update(
                    {"requests_per_case": 9}
                ),
            )

            with self.assertRaisesRegex(ValueError, "requests_per_case"):
                build_generalization_dataset(
                    condition_id="c-model",
                    recompute_run=recompute,
                    cpu_run=cpu,
                    filesystem_run=filesystem,
                    percentile="p95",
                )

    def test_requires_explicit_cuda_visibility(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            def remove_visibility(payload: dict) -> None:
                payload["config"]["server"]["env"].pop("CUDA_VISIBLE_DEVICES")

            _rewrite_json(cpu / "manifest.json", remove_visibility)

            with self.assertRaisesRegex(
                ValueError,
                "CUDA_VISIBLE_DEVICES",
            ):
                build_generalization_dataset(
                    condition_id="c-model",
                    recompute_run=recompute,
                    cpu_run=cpu,
                    filesystem_run=filesystem,
                    percentile="p95",
                )

    def test_requires_exactly_one_numeric_cuda_index(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            _rewrite_json(
                cpu / "manifest.json",
                lambda payload: payload["config"]["server"]["env"].update(
                    {"CUDA_VISIBLE_DEVICES": "0,1"}
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "exactly one numeric",
            ):
                build_generalization_dataset(
                    condition_id="c-model",
                    recompute_run=recompute,
                    cpu_run=cpu,
                    filesystem_run=filesystem,
                    percentile="p95",
                )

    def test_rejects_selected_gpu_index_missing_from_inventory(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            _rewrite_json(
                cpu / "manifest.json",
                lambda payload: payload["config"]["server"]["env"].update(
                    {"CUDA_VISIBLE_DEVICES": "9"}
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "selected GPU index 9",
            ):
                build_generalization_dataset(
                    condition_id="c-model",
                    recompute_run=recompute,
                    cpu_run=cpu,
                    filesystem_run=filesystem,
                    percentile="p95",
                )

    def test_rejects_inventory_row_without_selected_gpu_uuid(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            def remove_uuid(payload: dict) -> None:
                stdout = payload["gpu_inventory"]["stdout"]
                payload["gpu_inventory"]["stdout"] = stdout.replace(
                    "0, GPU-test-0,",
                    "0, ,",
                )

            for run in (recompute, cpu, filesystem):
                _rewrite_json(
                    run / "environment.json",
                    remove_uuid,
                )

            with self.assertRaisesRegex(ValueError, "GPU UUID"):
                build_generalization_dataset(
                    condition_id="c-model",
                    recompute_run=recompute,
                    cpu_run=cpu,
                    filesystem_run=filesystem,
                    percentile="p95",
                )

    def test_filesystem_requires_positive_async_count_and_sum(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            def zero_async_sum(record: dict) -> None:
                record["normalized"]["cache"][
                    "tiering_lookup_async_delay_seconds"
                ].update(
                    {
                        "value": 0.001,
                        "sum": 0.0,
                        "count": 8,
                        "reason": None,
                    }
                )

            _rewrite_only_record(filesystem, zero_async_sum)

            result = build_generalization_dataset(
                condition_id="c-model",
                recompute_run=recompute,
                cpu_run=cpu,
                filesystem_run=filesystem,
                percentile="p95",
            )

        self.assertEqual(
            [row["source"] for row in result["samples"]],
            ["cpu_primary"],
        )
        self.assertEqual(
            result["excluded_samples"][0]["reason"],
            "missing_tiered_fs_async_lookup_evidence",
        )

    def test_non_divisible_external_tokens_hard_fail(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            _rewrite_only_record(
                cpu,
                lambda record: record["normalized"]["prometheus"]["delta"].update(
                    _delta(
                        external_tokens=1857,
                        load_bytes=106430464,
                        load_count=8,
                    )
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "external KV tokens not divisible by requests_per_case",
            ):
                build_generalization_dataset(
                    condition_id="c-model",
                    recompute_run=recompute,
                    cpu_run=cpu,
                    filesystem_run=filesystem,
                    percentile="p95",
                )

    def test_prometheus_evidence_sums_positive_values_only(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            def add_negative_samples(record: dict) -> None:
                delta = record["normalized"]["prometheus"]["delta"]
                delta.update(
                    {
                        (
                            "vllm:prompt_tokens_by_source{"
                            'engine="1",model_name="qwen2.5-14b",'
                            'source="external_kv_transfer"}'
                        ): {
                            "value": -8,
                            "reason": None,
                        },
                        ('vllm:kv_offload_load_bytes{engine="1"}'): {
                            "value": -1024,
                            "reason": None,
                        },
                        ('vllm:kv_offload_load_size_count{engine="1"}'): {
                            "value": -1,
                            "reason": None,
                        },
                    }
                )

            _rewrite_only_record(cpu, add_negative_samples)

            result = build_generalization_dataset(
                condition_id="c-model",
                recompute_run=recompute,
                cpu_run=cpu,
                filesystem_run=filesystem,
                percentile="p95",
            )

        cpu_sample = next(
            row for row in result["samples"] if row["source"] == "cpu_primary"
        )
        self.assertEqual(
            cpu_sample["external_kv_tokens_total"],
            1856,
        )
        self.assertEqual(
            cpu_sample["external_kv_tokens_per_request"],
            232,
        )
        self.assertEqual(
            cpu_sample["transfer_evidence"]["cpu_to_gpu_transfers"],
            8,
        )
        self.assertEqual(
            cpu_sample["transfer_evidence"]["cpu_to_gpu_bytes"],
            106430464,
        )

    def test_condition_preserves_selected_gpu_index(self) -> None:
        from benchmarks.cache.build_generalization_dataset import (
            build_generalization_dataset,
        )

        with TemporaryDirectory() as tmp:
            recompute, cpu, filesystem = _build_three_runs(Path(tmp))

            result = build_generalization_dataset(
                condition_id="c-model",
                recompute_run=recompute,
                cpu_run=cpu,
                filesystem_run=filesystem,
                percentile="p95",
            )

        self.assertEqual(result["condition"]["gpu_index"], 0)
        self.assertEqual(
            result["condition"]["gpu_uuid"],
            "GPU-test-0",
        )

    def test_cli_writes_loader_compatible_condition_json(self) -> None:
        from benchmarks.cache.build_generalization_dataset import main
        from benchmarks.cache.cost_model_generalization import (
            load_generalization_condition,
        )

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            recompute, cpu, filesystem = _build_three_runs(base)
            output = base / "condition.json"

            rc = main(
                [
                    "--condition-id",
                    "cli-condition",
                    "--recompute-run",
                    str(recompute),
                    "--cpu-run",
                    str(cpu),
                    "--filesystem-run",
                    str(filesystem),
                    "--percentile",
                    "p95",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output.is_file())

            payload = json.loads(output.read_text(encoding="utf-8"))
            loaded = load_generalization_condition(output)

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["issue"], 15)
        self.assertEqual(
            payload["condition"]["id"],
            "cli-condition",
        )
        self.assertEqual(loaded.condition_id, "cli-condition")
        self.assertEqual(
            len(loaded.dataset.decision_samples),
            2,
        )
