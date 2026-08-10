# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from benchmarks.cache.config import SuiteConfig
from benchmarks.cache.scenarios import CacheMode, build_execution_cases
from benchmarks.cache.workload import generate_workload


class FakeTokenizer:
    all_special_ids: list[int] = []
    vocab_size = 10000

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(part) for part in text.split()] if text else []

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _pressure_config(valid_config_dict: dict, fill_requests: int = 5) -> SuiteConfig:
    raw = deepcopy(valid_config_dict)
    raw["workload"]["pressure_fill_requests"] = fill_requests
    return SuiteConfig.model_validate(raw)


def _pressure_cases(config: SuiteConfig):
    return [
        case
        for case in build_execution_cases(
            config, config.results.root_dir / "pressure-run"
        )
        if case.workload_kind == "eviction-restore"
        and case.prompt_tokens == config.workload.prompt_tokens[0]
        and case.concurrency == config.workload.concurrency[0]
        and case.request_rate == config.workload.request_rate[0]
    ]


def test_pressure_workload_is_opt_in(suite_config) -> None:
    cases = build_execution_cases(
        suite_config, suite_config.results.root_dir / "default-run"
    )
    assert all(case.workload_kind != "eviction-restore" for case in cases)


def test_pressure_workload_expands_all_cache_modes(valid_config_dict: dict) -> None:
    config = _pressure_config(valid_config_dict)
    cases = _pressure_cases(config)
    assert {case.cache_mode for case in cases} == set(CacheMode)


def test_pressure_population_puts_fillers_after_victims(
    valid_config_dict: dict,
) -> None:
    config = _pressure_config(valid_config_dict, fill_requests=5)
    case = next(
        case
        for case in _pressure_cases(config)
        if case.cache_mode is CacheMode.NO_CACHE
    )

    artifacts = generate_workload(case, config, FakeTokenizer())
    assert artifacts.populate_path is not None

    measure_rows = _rows(artifacts.measure_path)
    populate_rows = _rows(artifacts.populate_path)
    victim_count = config.workload.requests_per_case

    assert len(measure_rows) == victim_count
    assert len(populate_rows) == victim_count + 5
    assert populate_rows[:victim_count] == measure_rows

    victim_prompts = {row["prompt"] for row in measure_rows}
    filler_prompts = {row["prompt"] for row in populate_rows[victim_count:]}
    assert len(victim_prompts) == victim_count
    assert len(filler_prompts) == 5
    assert victim_prompts.isdisjoint(filler_prompts)

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["pressure_fill_requests"] == 5


def test_pressure_workload_is_identical_across_cache_modes(
    valid_config_dict: dict,
) -> None:
    config = _pressure_config(valid_config_dict, fill_requests=5)
    artifacts = [
        generate_workload(case, config, FakeTokenizer())
        for case in _pressure_cases(config)
    ]

    assert len({artifact.measure_path.read_bytes() for artifact in artifacts}) == 1
    assert (
        len(
            {
                artifact.populate_path.read_bytes()
                for artifact in artifacts
                if artifact.populate_path is not None
            }
        )
        == 1
    )
