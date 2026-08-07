# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from copy import deepcopy

import pytest

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


def _token_pressure_config(
    valid_config_dict: dict,
    *,
    prompt_tokens: list[int] | None = None,
) -> SuiteConfig:
    raw = deepcopy(valid_config_dict)
    raw["workload"]["pressure_fill_requests"] = 0
    raw["workload"]["pressure_fill_tokens"] = 65536
    if prompt_tokens is not None:
        raw["workload"]["prompt_tokens"] = prompt_tokens
    return SuiteConfig.model_validate(raw)


def _eviction_cases(config: SuiteConfig):
    return [
        case
        for case in build_execution_cases(
            config, config.results.root_dir / "token-pressure-run"
        )
        if case.workload_kind == "eviction-restore"
        and case.concurrency == config.workload.concurrency[0]
        and case.request_rate == config.workload.request_rate[0]
    ]


def test_pressure_fill_tokens_defaults_to_zero(valid_config_dict: dict) -> None:
    config = SuiteConfig.model_validate(deepcopy(valid_config_dict))
    assert config.workload.pressure_fill_tokens == 0


def test_pressure_fill_tokens_is_accepted(valid_config_dict: dict) -> None:
    config = _token_pressure_config(valid_config_dict)
    assert config.workload.pressure_fill_requests == 0
    assert config.workload.pressure_fill_tokens == 65536


def test_pressure_modes_are_mutually_exclusive(valid_config_dict: dict) -> None:
    raw = deepcopy(valid_config_dict)
    raw["workload"]["pressure_fill_requests"] = 64
    raw["workload"]["pressure_fill_tokens"] = 65536
    with pytest.raises(ValueError, match="at most one"):
        SuiteConfig.model_validate(raw)


def test_token_pressure_enables_eviction_restore_cases(valid_config_dict: dict) -> None:
    config = _token_pressure_config(valid_config_dict)
    cases = _eviction_cases(config)
    assert {case.cache_mode for case in cases} == set(CacheMode)


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_fillers"),
    [
        (256, 256),
        (512, 128),
        (1024, 64),
        (2048, 32),
        (4096, 16),
    ],
)
def test_token_pressure_derives_expected_filler_count(
    valid_config_dict: dict,
    prompt_tokens: int,
    expected_fillers: int,
) -> None:
    config = _token_pressure_config(valid_config_dict, prompt_tokens=[prompt_tokens])
    case = next(
        case
        for case in _eviction_cases(config)
        if case.cache_mode is CacheMode.NO_CACHE
    )

    artifacts = generate_workload(case, config, FakeTokenizer())
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert artifacts.populate_path is not None
    assert artifacts.num_population_prompts == (
        config.workload.requests_per_case + expected_fillers
    )
    assert artifacts.num_measurement_prompts == config.workload.requests_per_case
    assert metadata["pressure_fill_tokens"] == 65536
    assert metadata["derived_pressure_fill_requests"] == expected_fillers

    population = artifacts.populate_path.read_text(encoding="utf-8").splitlines()
    measurement = artifacts.measure_path.read_text(encoding="utf-8").splitlines()
    assert population[: config.workload.requests_per_case] == measurement


def test_token_pressure_workload_is_identical_across_cache_modes(
    valid_config_dict: dict,
) -> None:
    config = _token_pressure_config(valid_config_dict, prompt_tokens=[1024])
    artifacts = [
        generate_workload(case, config, FakeTokenizer())
        for case in _eviction_cases(config)
    ]
    metadata = [
        json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
        for artifact in artifacts
    ]

    assert {case.cache_mode for case in _eviction_cases(config)} == set(CacheMode)
    assert len({artifact.measure_path.read_bytes() for artifact in artifacts}) == 1
    assert len(
        {
            artifact.populate_path.read_bytes()
            for artifact in artifacts
            if artifact.populate_path is not None
        }
    ) == 1
    assert len({item["generator_seed"] for item in metadata}) == 1
    assert len({item["derived_pressure_fill_requests"] for item in metadata}) == 1
