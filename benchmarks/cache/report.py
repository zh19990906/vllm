# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

COMPARISON_FIELDS = (
    "workload_kind",
    "prompt_tokens",
    "prefix_ratio",
    "concurrency",
    "request_rate",
    "repetition",
    "model_id",
    "model_revision",
    "tensor_parallel_size",
    "pipeline_parallel_size",
)

SUMMARY_FIELDS = (
    "case_id",
    "status",
    "cache_mode",
    *COMPARISON_FIELDS,
    "completed_requests",
    "failed_requests",
    "request_throughput",
    "p95_ttft_ms",
    "p95_tpot_ms",
    "prefix_cache_hits_tokens",
    "prefix_cache_hits_reason",
    "p95_ttft_delta_vs_no_cache_pct",
    "p95_ttft_improvement_vs_no_cache_pct",
    "p95_ttft_delta_vs_gpu_apc_pct",
    "p95_ttft_improvement_vs_gpu_apc_pct",
    "comparison_reason",
)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def append_result(path: Path, record: Mapping[str, Any]) -> None:
    """Append one durable JSON object to the scenario result journal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load the append-only result journal in original record order."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSONL record at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record at line {line_number} is not an object")
        records.append(record)
    return records


def _nested(record: Mapping[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _comparison_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in COMPARISON_FIELDS)


def _percent_delta(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return round((candidate - baseline) / baseline * 100.0, 12)


def _percent_improvement(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return round((baseline - candidate) / baseline * 100.0, 12)


def _valid_ttft_record(record: Mapping[str, Any]) -> bool:
    return (
        record.get("status") == "completed"
        and _number(_nested(record, "normalized", "benchmark", "ttft_ms", "p95"))
        is not None
    )


def _baseline_maps(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[tuple[Any, ...], Mapping[str, Any]]]:
    maps: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {
        "no-cache": {},
        "gpu-apc": {},
    }
    for record in records:
        mode = record.get("cache_mode")
        if mode in maps and _valid_ttft_record(record):
            maps[str(mode)].setdefault(_comparison_key(record), record)
    return maps


def build_summary_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten result records and calculate workload-safe baseline deltas."""
    record_list = list(records)
    baselines = _baseline_maps(record_list)
    rows: list[dict[str, Any]] = []

    for record in record_list:
        candidate_ttft = _number(
            _nested(record, "normalized", "benchmark", "ttft_ms", "p95")
        )
        prefix_metric = _nested(
            record, "normalized", "cache", "prefix_cache_hits_tokens"
        )
        if not isinstance(prefix_metric, Mapping):
            prefix_metric = {}

        row: dict[str, Any] = {
            "case_id": record.get("case_id"),
            "status": record.get("status"),
            "cache_mode": record.get("cache_mode"),
            **{field: record.get(field) for field in COMPARISON_FIELDS},
            "completed_requests": _number(
                _nested(record, "normalized", "benchmark", "completed")
            ),
            "failed_requests": _number(
                _nested(record, "normalized", "benchmark", "failed")
            ),
            "request_throughput": _number(
                _nested(record, "normalized", "benchmark", "request_throughput")
            ),
            "p95_ttft_ms": candidate_ttft,
            "p95_tpot_ms": _number(
                _nested(record, "normalized", "benchmark", "tpot_ms", "p95")
            ),
            "prefix_cache_hits_tokens": _number(prefix_metric.get("value")),
            "prefix_cache_hits_reason": prefix_metric.get("reason"),
            "p95_ttft_delta_vs_no_cache_pct": None,
            "p95_ttft_improvement_vs_no_cache_pct": None,
            "p95_ttft_delta_vs_gpu_apc_pct": None,
            "p95_ttft_improvement_vs_gpu_apc_pct": None,
            "comparison_reason": None,
        }

        if record.get("status") != "completed":
            row["comparison_reason"] = "record_not_completed"
            rows.append(row)
            continue
        if candidate_ttft is None:
            row["comparison_reason"] = "candidate_metric_missing"
            rows.append(row)
            continue

        key = _comparison_key(record)
        found_baseline = False
        zero_baseline = False
        for mode, suffix in (("no-cache", "no_cache"), ("gpu-apc", "gpu_apc")):
            baseline_record = baselines[mode].get(key)
            if baseline_record is None:
                continue
            baseline_ttft = _number(
                _nested(
                    baseline_record,
                    "normalized",
                    "benchmark",
                    "ttft_ms",
                    "p95",
                )
            )
            if baseline_ttft is None:
                continue
            found_baseline = True
            if float(baseline_ttft) == 0.0:
                zero_baseline = True
                continue
            row[f"p95_ttft_delta_vs_{suffix}_pct"] = _percent_delta(
                float(candidate_ttft), float(baseline_ttft)
            )
            row[f"p95_ttft_improvement_vs_{suffix}_pct"] = _percent_improvement(
                float(candidate_ttft), float(baseline_ttft)
            )

        if not found_baseline:
            row["comparison_reason"] = "matching_baseline_not_found"
        elif zero_baseline and all(
            row[field] is None
            for field in (
                "p95_ttft_delta_vs_no_cache_pct",
                "p95_ttft_delta_vs_gpu_apc_pct",
            )
        ):
            row["comparison_reason"] = "baseline_metric_is_zero"
        rows.append(row)

    return rows


def write_summary_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write summary rows atomically using a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(SUMMARY_FIELDS), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _format_metric(value: Any, *, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        rendered = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        rendered = str(value)
    return f"{rendered}{suffix}"


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _command_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command) if command else "unavailable"


def write_markdown_report(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    *,
    environment: Mapping[str, Any] | None = None,
) -> None:
    """Write a human-readable benchmark report atomically."""
    record_list = list(records)
    row_list = list(rows)
    completed = sum(record.get("status") == "completed" for record in record_list)
    failed = len(record_list) - completed

    lines = [
        "# Cache Benchmark Report",
        "",
        "Population runs are setup only and are excluded from measured comparisons. "
        "Restart-persistence rows are post-restart measurements.",
        "",
        "## Environment summary",
        "",
    ]
    if environment:
        for key, value in sorted(environment.items()):
            if isinstance(value, Mapping):
                status = value.get("status", "unknown")
                detail = value.get("stdout") or value.get("reason") or ""
                lines.append(
                    f"- **{_markdown_escape(key)}**: {_markdown_escape(status)}"
                    + (f" — `{_markdown_escape(detail)}`" if detail else "")
                )
            else:
                lines.append(
                    f"- **{_markdown_escape(key)}**: `{_markdown_escape(value)}`"
                )
    else:
        lines.append("- Environment evidence unavailable.")

    lines.extend(
        [
            "",
            "## Run status",
            "",
            f"- Completed: {completed}",
            f"- Failed: {failed}",
            "",
            "## Results",
            "",
            "| Case | Status | Cache mode | Workload | Prompt | Concurrency | "
            "P95 TTFT (ms) | Req/s | TTFT improvement vs no-cache | "
            "TTFT improvement vs GPU APC | Cache metric evidence |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in row_list:
        missing_reason = row.get("prefix_cache_hits_reason") or "available"
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(row.get("case_id")),
                    _markdown_escape(row.get("status")),
                    _markdown_escape(row.get("cache_mode")),
                    _markdown_escape(row.get("workload_kind")),
                    _format_metric(row.get("prompt_tokens")),
                    _format_metric(row.get("concurrency")),
                    _format_metric(row.get("p95_ttft_ms")),
                    _format_metric(row.get("request_throughput")),
                    _format_metric(
                        row.get("p95_ttft_improvement_vs_no_cache_pct"), suffix="%"
                    ),
                    _format_metric(
                        row.get("p95_ttft_improvement_vs_gpu_apc_pct"), suffix="%"
                    ),
                    _markdown_escape(missing_reason),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Commands and evidence", ""])
    for record in record_list:
        lines.append(f"### `{_markdown_escape(record.get('case_id'))}`")
        commands = (
            record.get("commands")
            if isinstance(record.get("commands"), Mapping)
            else {}
        )
        logs = record.get("logs") if isinstance(record.get("logs"), Mapping) else {}
        lines.append(
            f"- Server: `{_markdown_escape(_command_text(commands.get('server')))}`"
        )
        if commands.get("populate"):
            lines.append(
                "- Population setup: "
                f"`{_markdown_escape(_command_text(commands.get('populate')))}`"
            )
        lines.append(
            "- Measurement: "
            f"`{_markdown_escape(_command_text(commands.get('measure')))}`"
        )
        if logs:
            for name, log_path in sorted(logs.items()):
                lines.append(
                    f"- {_markdown_escape(name)}: `{_markdown_escape(log_path)}`"
                )
        comparison_reason = next(
            (
                row.get("comparison_reason")
                for row in row_list
                if row.get("case_id") == record.get("case_id")
            ),
            None,
        )
        if comparison_reason:
            lines.append(f"- Comparison note: `{_markdown_escape(comparison_reason)}`")
        lines.append("")

    _atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def rebuild_reports(run_dir: Path) -> None:
    """Regenerate CSV and Markdown from the durable result journal."""
    records = load_results(run_dir / "scenario-results.jsonl")
    rows = build_summary_rows(records)
    environment_path = run_dir / "environment.json"
    environment: Mapping[str, Any] | None = None
    if environment_path.exists():
        loaded = json.loads(environment_path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            environment = loaded
    write_summary_csv(run_dir / "summary.csv", rows)
    write_markdown_report(run_dir / "report.md", records, rows, environment=environment)
