# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.cache.metrics import (
    ResourceSampler,
    compute_prometheus_delta,
    normalize_native_result,
    parse_prometheus_text,
)


def test_prometheus_counter_and_histogram_delta() -> None:
    before = parse_prometheus_text(
        """
# TYPE vllm:kv_offload_stores_skipped counter
vllm:kv_offload_stores_skipped 2
vllm:kv_offload_tiering_lookup_sync_delay_seconds_count 3
vllm:kv_offload_tiering_lookup_sync_delay_seconds_sum 0.6
"""
    )
    after = parse_prometheus_text(
        """
vllm:kv_offload_stores_skipped 5
vllm:kv_offload_tiering_lookup_sync_delay_seconds_count 7
vllm:kv_offload_tiering_lookup_sync_delay_seconds_sum 1.8
"""
    )
    delta = compute_prometheus_delta(before, after)
    assert delta["vllm:kv_offload_stores_skipped"]["value"] == 3
    assert (
        delta["vllm:kv_offload_tiering_lookup_sync_delay_seconds_count"]["value"] == 4
    )
    assert (
        delta["vllm:kv_offload_tiering_lookup_sync_delay_seconds_sum"]["value"] == 1.2
    )


def test_prometheus_labels_are_flattened_deterministically() -> None:
    snapshot = parse_prometheus_text(
        'vllm:prefix_cache_hits_total{model="m",rank="0"} 4\n'
    )
    assert snapshot.samples == {'vllm:prefix_cache_hits_total{model="m",rank="0"}': 4.0}


def test_normalize_native_result_preserves_null_reason(tmp_path: Path) -> None:
    native = tmp_path / "native-result.json"
    native.write_text(
        json.dumps(
            {
                "completed": 8,
                "failed": 0,
                "request_throughput": 2.5,
                "p50_ttft_ms": 20.0,
                "p95_ttft_ms": 30.0,
                "p99_ttft_ms": 40.0,
                "p50_tpot_ms": 4.0,
                "p95_tpot_ms": 5.0,
                "p99_tpot_ms": 6.0,
            }
        ),
        encoding="utf-8",
    )
    normalized = normalize_native_result(native)
    assert normalized["benchmark"]["ttft_ms"]["p95"] == 30.0
    assert normalized["cache"]["prefix_cache_hits_tokens"]["value"] is None
    assert (
        normalized["cache"]["prefix_cache_hits_tokens"]["reason"]
        == "metric_not_exposed"
    )


def test_normalize_native_result_accepts_one_element_list(tmp_path: Path) -> None:
    native = tmp_path / "native-result.json"
    native.write_text(json.dumps([{"completed": 2, "failed": 0}]), encoding="utf-8")
    assert normalize_native_result(native)["benchmark"]["completed"] == 2


def test_normalize_native_result_uses_prometheus_prefix_delta(tmp_path: Path) -> None:
    native = tmp_path / "native-result.json"
    native.write_text(json.dumps({"completed": 1}), encoding="utf-8")
    before = parse_prometheus_text("vllm:prefix_cache_hit_tokens_total 10\n")
    after = parse_prometheus_text("vllm:prefix_cache_hit_tokens_total 42\n")
    normalized = normalize_native_result(native, before=before, after=after)
    assert normalized["cache"]["prefix_cache_hits_tokens"]["value"] == 32
    assert normalized["cache"]["prefix_cache_hits_tokens"]["reason"] is None


def test_resource_sampler_records_peak_mean_and_final() -> None:
    rss_values = iter([100, 300, 200])
    memory_values = iter([(1000, 9000), (1100, 8900), (1200, 8800)])
    gpu_values = iter([10, 30, 20])
    sampler = ResourceSampler(
        pid=123,
        rss_sampler=lambda pid: next(rss_values),
        memory_sampler=lambda: next(memory_values),
        gpu_sampler=lambda pids: next(gpu_values),
    )
    sampler.sample_once()
    sampler.sample_once()
    sampler.sample_once()
    summary = sampler.summary()
    assert summary["process_tree_rss_bytes"] == {
        "peak": 300.0,
        "mean": 200.0,
        "final": 200.0,
        "samples": 3,
        "reason": None,
    }
    assert summary["gpu_used_memory_mib"]["peak"] == 30.0


def test_resource_sampler_records_gpu_unavailable() -> None:
    def unavailable(_pids: set[int]) -> int:
        raise FileNotFoundError("nvidia-smi")

    sampler = ResourceSampler(
        pid=123,
        rss_sampler=lambda pid: 100,
        memory_sampler=lambda: (1000, 9000),
        gpu_sampler=unavailable,
    )
    sampler.sample_once()
    summary = sampler.summary()
    assert summary["gpu_used_memory_mib"]["final"] is None
    assert "gpu_sampler_unavailable" in summary["gpu_used_memory_mib"]["reason"]
