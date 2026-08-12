# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def valid_config_dict(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "model": {
            "id": "/models/example",
            "served_name": "example",
            "dtype": "auto",
            "max_model_len": 32768,
            "trust_remote_code": False,
        },
        "parallelism": {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
        },
        "server": {
            "host": "127.0.0.1",
            "port": 8100,
            "startup_timeout_seconds": 900,
            "shutdown_timeout_seconds": 60,
            "extra_args": [],
            "env": {},
        },
        "cache": {
            "gpu_memory_utilization": 0.9,
            "cpu_bytes_to_use": 68719476736,
            "offload_block_size": 64,
            "eviction_policy": "lru",
            "filesystem": {
                "enabled": True,
                "root_dir": str(tmp_path / "kv"),
                "max_bytes": 1099511627776,
                "read_threads": 32,
                "write_threads": 16,
            },
        },
        "workload": {
            "seed": 1,
            "tokenizer": "/models/example",
            "prompt_tokens": [1024],
            "output_tokens": 128,
            "concurrency": [1, 8],
            "request_rate": ["inf", 4.0],
            "requests_per_case": 8,
            "shared_prefix_ratios": [0.0, 0.5, 0.9],
            "warmup_requests": 2,
            "token_length_tolerance": 2,
        },
        "results": {
            "root_dir": str(tmp_path / "results"),
            "keep_server_logs": True,
            "fail_fast": False,
        },
    }


@pytest.fixture
def suite_config(valid_config_dict: dict):
    from benchmarks.cache.config import SuiteConfig

    return SuiteConfig.model_validate(deepcopy(valid_config_dict))


def _case_fixture(suite_config, tmp_path: Path, workload_kind: str, prefix_ratio=0.0):
    from benchmarks.cache.scenarios import build_execution_cases

    run_dir = suite_config.results.root_dir / "fixture-run"
    cases = build_execution_cases(suite_config, run_dir)
    return next(
        case
        for case in cases
        if case.workload_kind == workload_kind
        and case.prefix_ratio == prefix_ratio
        and case.prompt_tokens == suite_config.workload.prompt_tokens[0]
        and case.concurrency == suite_config.workload.concurrency[0]
        and case.request_rate == suite_config.workload.request_rate[0]
    )


@pytest.fixture
def cold_case(suite_config, tmp_path: Path):
    return _case_fixture(suite_config, tmp_path, "cold-unique")


@pytest.fixture
def warm_exact_case(suite_config, tmp_path: Path):
    return _case_fixture(suite_config, tmp_path, "warm-exact-prefix")


@pytest.fixture
def shared_prefix_case(suite_config, tmp_path: Path):
    return _case_fixture(suite_config, tmp_path, "shared-prefix", 0.5)


@pytest.fixture
def mixed_prefix_case(suite_config, tmp_path: Path):
    return _case_fixture(suite_config, tmp_path, "mixed-prefix")


@pytest.fixture
def restart_case(suite_config, tmp_path: Path):
    return _case_fixture(suite_config, tmp_path, "restart-persistence")
