# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import csv
from pathlib import Path

from benchmarks.cache.report import (
    append_result,
    build_summary_rows,
    load_results,
    rebuild_reports,
)


def synthetic_record(
    cache_mode: str,
    *,
    prompt_tokens: int = 1024,
    p95_ttft: float | None = 100.0,
    status: str = "completed",
) -> dict:
    return {
        "case_id": f"{cache_mode}-{prompt_tokens}",
        "status": status,
        "cache_mode": cache_mode,
        "workload_kind": "warm-exact-prefix",
        "prompt_tokens": prompt_tokens,
        "prefix_ratio": 0.0,
        "concurrency": 8,
        "request_rate": "inf",
        "repetition": 0,
        "model_id": "model",
        "model_revision": None,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "normalized": {
            "benchmark": {
                "request_throughput": 10.0,
                "ttft_ms": {"p95": p95_ttft},
                "tpot_ms": {"p95": 5.0},
            },
            "cache": {
                "prefix_cache_hits_tokens": {
                    "value": None,
                    "reason": "metric_not_exposed",
                }
            },
        },
        "commands": {"server": ["vllm", "serve"], "measure": ["vllm", "bench"]},
        "logs": {"server_stdout": "raw/a/server.stdout.log"},
    }


def test_jsonl_is_immediately_recoverable(tmp_path: Path) -> None:
    path = tmp_path / "scenario-results.jsonl"
    append_result(path, {"case_id": "a", "status": "completed"})
    append_result(path, {"case_id": "b", "status": "benchmark_error"})
    assert [row["case_id"] for row in load_results(path)] == ["a", "b"]


def test_report_compares_only_matching_workloads() -> None:
    rows = build_summary_rows(
        [
            synthetic_record("no-cache", prompt_tokens=1024, p95_ttft=100),
            synthetic_record("gpu-apc", prompt_tokens=1024, p95_ttft=50),
            synthetic_record("cpu-offload", prompt_tokens=4096, p95_ttft=40),
        ]
    )
    cpu_row = next(row for row in rows if row["cache_mode"] == "cpu-offload")
    assert cpu_row["p95_ttft_delta_vs_no_cache_pct"] is None
    assert cpu_row["comparison_reason"] == "matching_baseline_not_found"


def test_ttft_delta_and_improvement_have_opposite_signs() -> None:
    rows = build_summary_rows(
        [
            synthetic_record("no-cache", p95_ttft=100),
            synthetic_record("gpu-apc", p95_ttft=50),
        ]
    )
    gpu = next(row for row in rows if row["cache_mode"] == "gpu-apc")
    assert gpu["p95_ttft_delta_vs_no_cache_pct"] == -50.0
    assert gpu["p95_ttft_improvement_vs_no_cache_pct"] == 50.0


def test_rebuild_reports_writes_csv_and_markdown(tmp_path: Path) -> None:
    append_result(tmp_path / "scenario-results.jsonl", synthetic_record("no-cache"))
    append_result(
        tmp_path / "scenario-results.jsonl", synthetic_record("gpu-apc", p95_ttft=50)
    )
    (tmp_path / "environment.json").write_text(
        '{"gpu_inventory":{"status":"available","stdout":"GPU"}}\n',
        encoding="utf-8",
    )

    rebuild_reports(tmp_path)

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Environment summary" in report
    assert "Completed: 2" in report
    assert "Population runs are setup only" in report
    assert "metric_not_exposed" in report
    assert "vllm serve" in report
    assert "server.stdout.log" in report
