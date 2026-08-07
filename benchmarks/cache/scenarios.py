from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, TypeVar

from benchmarks.cache.config import SuiteConfig

RequestRate = str | float
WorkloadKind = Literal[
    "cold-unique",
    "warm-exact-prefix",
    "eviction-restore",
    "shared-prefix",
    "mixed-prefix",
    "restart-persistence",
]


class CacheMode(str, Enum):
    NO_CACHE = "no-cache"
    GPU_APC = "gpu-apc"
    CPU_OFFLOAD = "cpu-offload"
    TIERED_FS = "tiered-fs"


@dataclass(frozen=True, slots=True)
class ExecutionCase:
    case_id: str
    cache_mode: CacheMode
    workload_kind: WorkloadKind
    prompt_tokens: int
    prefix_ratio: float
    concurrency: int
    request_rate: RequestRate
    repetition: int
    result_dir: Path
    filesystem_cache_dir: Path | None


_T = TypeVar("_T")


def _unique(values: list[_T]) -> list[_T]:
    result: list[_T] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _request_rate_label(value: RequestRate) -> str:
    if value == "inf":
        return "inf"
    return format(float(value), "g").replace(".", "p")


def _case_id(
    cache_mode: CacheMode,
    workload_kind: WorkloadKind,
    prompt_tokens: int,
    prefix_ratio: float,
    concurrency: int,
    request_rate: RequestRate,
    repetition: int,
) -> str:
    identity = {
        "cache_mode": cache_mode.value,
        "concurrency": concurrency,
        "prefix_ratio": prefix_ratio,
        "prompt_tokens": prompt_tokens,
        "repetition": repetition,
        "request_rate": request_rate,
        "workload_kind": workload_kind,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return (
        f"{cache_mode.value}__{workload_kind}__p{prompt_tokens}"
        f"__r{prefix_ratio:.3f}__c{concurrency}"
        f"__q{_request_rate_label(request_rate)}__{digest}"
    )


def build_execution_cases(config: SuiteConfig, run_dir: Path) -> list[ExecutionCase]:
    """Expand a suite configuration into deterministic, side-effect-free cases."""
    cache_modes = [
        CacheMode.NO_CACHE,
        CacheMode.GPU_APC,
        CacheMode.CPU_OFFLOAD,
    ]
    if config.cache.filesystem.enabled:
        cache_modes.append(CacheMode.TIERED_FS)

    prompt_tokens = _unique(config.workload.prompt_tokens)
    concurrencies = _unique(config.workload.concurrency)
    request_rates = _unique(config.workload.request_rate)
    shared_ratios = [
        ratio
        for ratio in _unique(config.workload.shared_prefix_ratios)
        if ratio > 0.0
    ]
    pressure_enabled = (
        config.workload.pressure_fill_requests > 0
        or config.workload.pressure_fill_tokens > 0
    )

    run_path = run_dir.expanduser().resolve()
    filesystem_namespace = config.cache.filesystem.root_dir / run_path.name
    cases: list[ExecutionCase] = []

    for cache_mode in cache_modes:
        workload_ratios: list[tuple[WorkloadKind, list[float]]] = [
            ("cold-unique", [0.0]),
            ("warm-exact-prefix", [0.0]),
            ("shared-prefix", shared_ratios),
            ("mixed-prefix", [0.0]),
        ]
        if pressure_enabled:
            workload_ratios.append(("eviction-restore", [0.0]))
        if cache_mode is CacheMode.TIERED_FS:
            workload_ratios.append(("restart-persistence", [0.0]))

        for workload_kind, ratios in workload_ratios:
            for prompt_length in prompt_tokens:
                for ratio in ratios:
                    for concurrency in concurrencies:
                        for request_rate in request_rates:
                            repetition = 0
                            case_id = _case_id(
                                cache_mode,
                                workload_kind,
                                prompt_length,
                                ratio,
                                concurrency,
                                request_rate,
                                repetition,
                            )
                            filesystem_cache_dir = (
                                filesystem_namespace / case_id
                                if cache_mode is CacheMode.TIERED_FS
                                else None
                            )
                            cases.append(
                                ExecutionCase(
                                    case_id=case_id,
                                    cache_mode=cache_mode,
                                    workload_kind=workload_kind,
                                    prompt_tokens=prompt_length,
                                    prefix_ratio=ratio,
                                    concurrency=concurrency,
                                    request_rate=request_rate,
                                    repetition=repetition,
                                    result_dir=run_path / "raw" / case_id,
                                    filesystem_cache_dir=filesystem_cache_dir,
                                )
                            )
    return cases


def _offloading_config(case: ExecutionCase, config: SuiteConfig) -> dict:
    extra: dict = {
        "cpu_bytes_to_use": config.cache.cpu_bytes_to_use,
        "block_size": config.cache.offload_block_size,
        "eviction_policy": config.cache.eviction_policy,
    }
    if case.cache_mode is CacheMode.TIERED_FS:
        if case.filesystem_cache_dir is None:
            raise ValueError("tiered filesystem case requires a cache directory")
        extra.update(
            {
                "spec_name": "TieringOffloadingSpec",
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": str(case.filesystem_cache_dir),
                        "n_read_threads": config.cache.filesystem.read_threads,
                        "n_write_threads": config.cache.filesystem.write_threads,
                        "locality": "LOCAL",
                    }
                ],
            }
        )
    return {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
    }


def build_server_command(case: ExecutionCase, config: SuiteConfig) -> list[str]:
    """Build the exact native vLLM server command for one execution case."""
    command = [
        "vllm",
        "serve",
        config.model.id,
        "--served-model-name",
        config.model.served_name,
        "--host",
        config.server.host,
        "--port",
        str(config.server.port),
        "--dtype",
        config.model.dtype,
        "--max-model-len",
        str(config.model.max_model_len),
        "--tensor-parallel-size",
        str(config.parallelism.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(config.parallelism.pipeline_parallel_size),
        "--gpu-memory-utilization",
        str(config.cache.gpu_memory_utilization),
    ]
    if config.model.trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(config.server.extra_args)

    if case.cache_mode is CacheMode.NO_CACHE:
        command.append("--no-enable-prefix-caching")
    else:
        command.append("--enable-prefix-caching")

    if case.cache_mode in (CacheMode.CPU_OFFLOAD, CacheMode.TIERED_FS):
        transfer_config = json.dumps(
            _offloading_config(case, config),
            sort_keys=True,
            separators=(",", ":"),
        )
        command.extend(["--kv-transfer-config", transfer_config])
    return command


def build_server_environment(
    case: ExecutionCase, config: SuiteConfig
) -> dict[str, str]:
    """Build the process environment, stabilizing hashes for shared FS keys."""
    environment = dict(os.environ)
    environment.update(config.server.env)
    if case.cache_mode is CacheMode.TIERED_FS:
        environment["PYTHONHASHSEED"] = "0"
    return environment
