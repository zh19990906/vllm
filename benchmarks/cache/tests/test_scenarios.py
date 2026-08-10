# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.cache.scenarios import (
    CacheMode,
    build_execution_cases,
    build_server_command,
    build_server_environment,
)


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _representative_cases(suite_config, tmp_path: Path):
    cases = build_execution_cases(suite_config, tmp_path)
    return {
        case.cache_mode: case
        for case in cases
        if case.workload_kind == "cold-unique"
        and case.prompt_tokens == suite_config.workload.prompt_tokens[0]
        and case.concurrency == suite_config.workload.concurrency[0]
        and case.request_rate == suite_config.workload.request_rate[0]
    }


def test_server_commands_cover_four_cache_modes(suite_config, tmp_path: Path) -> None:
    representative = _representative_cases(suite_config, tmp_path)

    no_cache = build_server_command(representative[CacheMode.NO_CACHE], suite_config)
    assert "--no-enable-prefix-caching" in no_cache
    assert "--kv-transfer-config" not in no_cache

    gpu = build_server_command(representative[CacheMode.GPU_APC], suite_config)
    assert "--enable-prefix-caching" in gpu
    assert "--kv-transfer-config" not in gpu

    cpu = build_server_command(representative[CacheMode.CPU_OFFLOAD], suite_config)
    cpu_cfg = json.loads(_arg_value(cpu, "--kv-transfer-config"))
    assert cpu_cfg["kv_connector"] == "OffloadingConnector"
    assert cpu_cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == 68719476736
    assert "spec_name" not in cpu_cfg["kv_connector_extra_config"]

    tiered = build_server_command(representative[CacheMode.TIERED_FS], suite_config)
    tiered_cfg = json.loads(_arg_value(tiered, "--kv-transfer-config"))
    extra = tiered_cfg["kv_connector_extra_config"]
    assert extra["spec_name"] == "TieringOffloadingSpec"
    assert extra["secondary_tiers"][0]["type"] == "fs"
    assert extra["secondary_tiers"][0]["locality"] == "LOCAL"


def test_case_ids_are_deterministic_and_unique(suite_config, tmp_path: Path) -> None:
    first = build_execution_cases(suite_config, tmp_path)
    second = build_execution_cases(suite_config, tmp_path)
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert len({case.case_id for case in first}) == len(first)


def test_tiered_cases_use_proposed_per_case_directories(
    suite_config, tmp_path: Path
) -> None:
    cases = build_execution_cases(suite_config, tmp_path)
    tiered = next(case for case in cases if case.cache_mode is CacheMode.TIERED_FS)
    assert tiered.filesystem_cache_dir is not None
    assert tiered.filesystem_cache_dir.is_relative_to(
        suite_config.cache.filesystem.root_dir
    )
    assert not tiered.filesystem_cache_dir.exists()
    assert not tiered.result_dir.exists()


def test_restart_persistence_exists_only_for_tiered_fs(
    suite_config, tmp_path: Path
) -> None:
    cases = build_execution_cases(suite_config, tmp_path)
    restart_modes = {
        case.cache_mode for case in cases if case.workload_kind == "restart-persistence"
    }
    assert restart_modes == {CacheMode.TIERED_FS}


def test_shared_prefix_expands_only_positive_ratios(
    suite_config, tmp_path: Path
) -> None:
    cases = build_execution_cases(suite_config, tmp_path)
    ratios = {
        case.prefix_ratio
        for case in cases
        if case.cache_mode is CacheMode.GPU_APC
        and case.workload_kind == "shared-prefix"
        and case.prompt_tokens == suite_config.workload.prompt_tokens[0]
        and case.concurrency == suite_config.workload.concurrency[0]
        and case.request_rate == suite_config.workload.request_rate[0]
    }
    assert ratios == {0.5, 0.9}


def test_tiered_environment_forces_stable_hash_seed(
    suite_config, tmp_path: Path
) -> None:
    representative = _representative_cases(suite_config, tmp_path)
    tiered_env = build_server_environment(
        representative[CacheMode.TIERED_FS], suite_config
    )
    assert tiered_env["PYTHONHASHSEED"] == "0"
