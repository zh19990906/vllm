# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests
from prometheus_client.parser import text_string_to_metric_families

SELECTED_METRICS = {
    "cpu_cache_usage_perc": "vllm:kv_offload_cpu_cache_usage_perc",
    "cpu_cache_write_usage_perc": "vllm:kv_offload_cpu_cache_write_usage_perc",
    "cpu_cache_read_usage_perc": "vllm:kv_offload_cpu_cache_read_usage_perc",
    "cpu_allocation_size": "vllm:kv_offload_cpu_allocation_size",
    "stores_skipped": "vllm:kv_offload_stores_skipped",
    "tiering_lookup_sync_delay_seconds": (
        "vllm:kv_offload_tiering_lookup_sync_delay_seconds"
    ),
    "tiering_lookup_async_delay_seconds": (
        "vllm:kv_offload_tiering_lookup_async_delay_seconds"
    ),
}
PREFIX_HIT_CANDIDATES = (
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_hit_tokens_total",
)
PREFIX_QUERY_CANDIDATES = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_query_tokens_total",
)


@dataclass(frozen=True, slots=True)
class PrometheusSnapshot:
    fetched_at: str
    samples: dict[str, float]
    raw_text: str


@dataclass(frozen=True, slots=True)
class ResourceSample:
    timestamp: float
    process_tree_rss_bytes: float | None
    system_used_memory_bytes: float | None
    system_available_memory_bytes: float | None
    gpu_used_memory_mib: float | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    rendered = ",".join(
        f"{key}={json.dumps(str(value), ensure_ascii=True)}"
        for key, value in sorted(labels.items())
    )
    return f"{name}{{{rendered}}}"


def _base_sample_name(key: str) -> str:
    return key.split("{", 1)[0]


def parse_prometheus_text(text: str) -> PrometheusSnapshot:
    """Parse all vLLM samples into a stable, label-aware flat mapping."""
    samples: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            sample_name = sample.name
            if family.type == "counter" and sample_name == f"{family.name}_total":
                sample_name = family.name
            if not sample_name.startswith("vllm:"):
                continue
            samples[_sample_key(sample_name, dict(sample.labels))] = float(sample.value)
    return PrometheusSnapshot(fetched_at=_utc_now(), samples=samples, raw_text=text)


def fetch_prometheus_snapshot(
    base_url: str, *, timeout_seconds: float = 5.0
) -> PrometheusSnapshot:
    response = requests.get(
        f"{base_url.rstrip('/')}/metrics", timeout=(timeout_seconds, timeout_seconds)
    )
    response.raise_for_status()
    return parse_prometheus_text(response.text)


def compute_prometheus_delta(
    before: PrometheusSnapshot, after: PrometheusSnapshot
) -> dict[str, dict[str, float | str | None]]:
    """Compute label-preserving deltas for samples present in both snapshots."""
    delta: dict[str, dict[str, float | str | None]] = {}
    for key in sorted(set(before.samples) | set(after.samples)):
        if key not in before.samples:
            delta[key] = {"value": None, "reason": "metric_missing_before"}
        elif key not in after.samples:
            delta[key] = {"value": None, "reason": "metric_missing_after"}
        else:
            value = round(after.samples[key] - before.samples[key], 12)
            delta[key] = {"value": value, "reason": None}
    return delta


def _aggregate_candidate(
    snapshot: PrometheusSnapshot, candidates: tuple[str, ...]
) -> tuple[float | None, str | None]:
    bases = {_base_sample_name(key) for key in snapshot.samples}
    for candidate in candidates:
        if candidate in bases:
            return (
                sum(
                    value
                    for key, value in snapshot.samples.items()
                    if _base_sample_name(key) == candidate
                ),
                candidate,
            )
    for candidate in candidates:
        matching = [name for name in bases if name.endswith(candidate)]
        if matching:
            return (
                sum(
                    value
                    for key, value in snapshot.samples.items()
                    if _base_sample_name(key) in matching
                ),
                sorted(matching)[0],
            )
    return None, None


def _prefix_delta(
    before: PrometheusSnapshot | None,
    after: PrometheusSnapshot | None,
    candidates: tuple[str, ...],
) -> dict[str, float | str | None]:
    if before is None or after is None:
        return {"value": None, "reason": "metric_not_exposed"}
    before_value, before_name = _aggregate_candidate(before, candidates)
    after_value, after_name = _aggregate_candidate(after, candidates)
    if before_value is None or after_value is None:
        return {"value": None, "reason": "metric_not_exposed"}
    return {
        "value": round(after_value - before_value, 12),
        "reason": None,
        "sample_name": after_name or before_name,
    }


def _selected_delta(
    base_name: str,
    delta: dict[str, dict[str, float | str | None]],
) -> dict[str, float | str | None]:
    exact_values = [
        item["value"]
        for key, item in delta.items()
        if _base_sample_name(key) == base_name and item["value"] is not None
    ]
    if exact_values:
        return {"value": float(sum(exact_values)), "reason": None}

    sum_values = [
        item["value"]
        for key, item in delta.items()
        if _base_sample_name(key) == f"{base_name}_sum" and item["value"] is not None
    ]
    count_values = [
        item["value"]
        for key, item in delta.items()
        if _base_sample_name(key) == f"{base_name}_count" and item["value"] is not None
    ]
    if sum_values or count_values:
        total_sum = float(sum(sum_values)) if sum_values else None
        total_count = float(sum(count_values)) if count_values else None
        mean = (
            total_sum / total_count
            if total_sum is not None and total_count not in (None, 0.0)
            else None
        )
        return {
            "value": mean,
            "sum": total_sum,
            "count": total_count,
            "reason": None,
        }
    return {"value": None, "reason": "metric_not_exposed"}


def _load_native_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValueError("native result list must contain exactly one object")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("native result must be an object or one-element object list")
    return payload


def _latency_group(payload: dict, name: str) -> dict[str, float | None]:
    def value(key: str) -> float | None:
        raw = payload.get(key)
        return float(raw) if isinstance(raw, (int, float)) else None

    return {
        "mean": value(f"mean_{name}_ms"),
        "median": value(f"median_{name}_ms"),
        "p50": value(f"p50_{name}_ms"),
        "p95": value(f"p95_{name}_ms"),
        "p99": value(f"p99_{name}_ms"),
    }


def normalize_native_result(
    path: Path,
    *,
    before: PrometheusSnapshot | None = None,
    after: PrometheusSnapshot | None = None,
) -> dict:
    """Normalize native benchmark JSON and optional cache metric snapshots."""
    payload = _load_native_payload(path)

    def number(key: str, cast: type[int] | type[float] = float):
        raw = payload.get(key)
        if not isinstance(raw, (int, float)):
            return None
        return cast(raw)

    benchmark = {
        "completed": number("completed", int),
        "failed": number("failed", int),
        "duration_seconds": number("duration"),
        "request_throughput": number("request_throughput"),
        "output_token_throughput": number("output_throughput"),
        "total_token_throughput": number("total_token_throughput"),
        "ttft_ms": _latency_group(payload, "ttft"),
        "tpot_ms": _latency_group(payload, "tpot"),
        "itl_ms": _latency_group(payload, "itl"),
        "e2el_ms": _latency_group(payload, "e2el"),
    }

    delta = (
        compute_prometheus_delta(before, after)
        if before is not None and after is not None
        else {}
    )
    cache: dict[str, object] = {
        "prefix_cache_hits_tokens": _prefix_delta(before, after, PREFIX_HIT_CANDIDATES),
        "prefix_cache_query_tokens": _prefix_delta(
            before, after, PREFIX_QUERY_CANDIDATES
        ),
    }
    for friendly_name, sample_name in SELECTED_METRICS.items():
        cache[friendly_name] = _selected_delta(sample_name, delta)

    prometheus = {
        "before": before.samples if before is not None else None,
        "after": after.samples if after is not None else None,
        "delta": delta if delta else None,
        "unavailable_reason": (
            None
            if before is not None and after is not None
            else "metrics_not_collected"
        ),
    }
    return {"benchmark": benchmark, "cache": cache, "prometheus": prometheus}


RssSampler = Callable[[int], int]
MemorySampler = Callable[[], tuple[int, int]]
GpuSampler = Callable[[set[int]], int]


def _process_tree_pids(pid: int) -> set[int]:
    process = psutil.Process(pid)
    return {pid, *(child.pid for child in process.children(recursive=True))}


def _default_rss_sampler(pid: int) -> int:
    pids = _process_tree_pids(pid)
    total = 0
    for process_id in pids:
        try:
            total += psutil.Process(process_id).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _default_memory_sampler() -> tuple[int, int]:
    memory = psutil.virtual_memory()
    return int(memory.used), int(memory.available)


def _default_gpu_sampler(pids: set[int]) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    total = 0
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            process_id, used_memory = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if process_id in pids:
            total += used_memory
    return total


class ResourceSampler:
    """Collect process, host-memory, and GPU-memory samples in a small thread."""

    def __init__(
        self,
        pid: int,
        *,
        interval_seconds: float = 1.0,
        rss_sampler: RssSampler = _default_rss_sampler,
        memory_sampler: MemorySampler = _default_memory_sampler,
        gpu_sampler: GpuSampler = _default_gpu_sampler,
    ) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self._rss_sampler = rss_sampler
        self._memory_sampler = memory_sampler
        self._gpu_sampler = gpu_sampler
        self._samples: list[ResourceSample] = []
        self._reasons: dict[str, str] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> ResourceSample:
        rss: float | None
        used: float | None
        available: float | None
        gpu: float | None
        try:
            rss = float(self._rss_sampler(self.pid))
        except Exception as error:
            rss = None
            self._reasons.setdefault(
                "process_tree_rss_bytes", f"rss_sampler_unavailable: {error}"
            )
        try:
            used_raw, available_raw = self._memory_sampler()
            used, available = float(used_raw), float(available_raw)
        except Exception as error:
            used = available = None
            reason = f"memory_sampler_unavailable: {error}"
            self._reasons.setdefault("system_used_memory_bytes", reason)
            self._reasons.setdefault("system_available_memory_bytes", reason)
        try:
            pids = _process_tree_pids(self.pid)
        except (psutil.Error, OSError):
            pids = {self.pid}
        try:
            gpu = float(self._gpu_sampler(pids))
        except Exception as error:
            gpu = None
            self._reasons.setdefault(
                "gpu_used_memory_mib", f"gpu_sampler_unavailable: {error}"
            )
        sample = ResourceSample(
            timestamp=time.time(),
            process_tree_rss_bytes=rss,
            system_used_memory_bytes=used,
            system_available_memory_bytes=available,
            gpu_used_memory_mib=gpu,
        )
        self._samples.append(sample)
        return sample

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("resource sampler is already running")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, dict[str, float | int | str | None]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        return self.summary()

    def summary(self) -> dict[str, dict[str, float | int | str | None]]:
        result: dict[str, dict[str, float | int | str | None]] = {}
        for field_name in (
            "process_tree_rss_bytes",
            "system_used_memory_bytes",
            "system_available_memory_bytes",
            "gpu_used_memory_mib",
        ):
            values = [
                float(value)
                for sample in self._samples
                if (value := getattr(sample, field_name)) is not None
            ]
            result[field_name] = {
                "peak": max(values) if values else None,
                "mean": statistics.fmean(values) if values else None,
                "final": values[-1] if values else None,
                "samples": len(values),
                "reason": self._reasons.get(field_name),
            }
        return result


_ENVIRONMENT_COMMANDS: dict[str, list[str]] = {
    "vllm_version": ["vllm", "--version"],
    "python_version": ["python", "--version"],
    "git_commit": ["git", "rev-parse", "HEAD"],
    "git_status": ["git", "status", "--porcelain"],
    "gpu_inventory": [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ],
    "gpu_topology": ["nvidia-smi", "topo", "-m"],
    "cpu_inventory": ["lscpu", "--json"],
    "numa_topology": ["numactl", "--hardware"],
}


def collect_environment_evidence() -> dict[str, dict[str, object]]:
    """Collect optional host evidence without making unavailable tools fatal."""
    evidence: dict[str, dict[str, object]] = {}
    for name, command in _ENVIRONMENT_COMMANDS.items():
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            evidence[name] = {"status": "unavailable", "reason": str(error)}
            continue
        if completed.returncode != 0:
            evidence[name] = {
                "status": "unavailable",
                "reason": completed.stderr.strip()
                or f"command exited with {completed.returncode}",
            }
            continue
        evidence[name] = {
            "status": "available",
            "command": command,
            "stdout": completed.stdout.strip(),
        }
    return evidence
