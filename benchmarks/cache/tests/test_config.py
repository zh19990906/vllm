# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.cache.config import (
    SuiteConfig,
    assert_owned_child,
    create_owned_directory,
    load_suite_config,
    sanitize_environment,
)


def test_load_suite_config_normalizes_inf_and_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(
        """
schema_version: 1
model:
  id: /models/example
  served_name: example
  dtype: auto
  max_model_len: 32768
  trust_remote_code: false
parallelism:
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
server:
  host: 127.0.0.1
  port: 8100
  startup_timeout_seconds: 900
  shutdown_timeout_seconds: 60
  extra_args: []
  env: {}
cache:
  gpu_memory_utilization: 0.9
  cpu_bytes_to_use: 68719476736
  offload_block_size: 64
  eviction_policy: lru
  filesystem:
    enabled: true
    root_dir: ./kv
    read_threads: 32
    write_threads: 16
workload:
  seed: 1
  tokenizer: /models/example
  prompt_tokens: [1024]
  output_tokens: 128
  concurrency: [1, 8]
  request_rate: [inf, 4.0]
  requests_per_case: 8
  shared_prefix_ratios: [0.0, 0.5, 0.9]
  warmup_requests: 2
  token_length_tolerance: 2
results:
  root_dir: ./results
  keep_server_logs: true
  fail_fast: false
""",
        encoding="utf-8",
    )

    config = load_suite_config(config_path)

    assert config.parallelism.tensor_parallel_size == 2
    assert config.workload.request_rate == ["inf", 4.0]
    assert config.cache.filesystem.root_dir == (tmp_path / "kv").resolve()
    assert config.results.root_dir == (tmp_path / "results").resolve()


def test_unknown_key_is_rejected(valid_config_dict: dict) -> None:
    valid_config_dict["server"]["startp_timeout_seconds"] = 5
    with pytest.raises(ValidationError, match="startp_timeout_seconds"):
        SuiteConfig.model_validate(valid_config_dict)


def test_invalid_shared_prefix_ratio_is_rejected(valid_config_dict: dict) -> None:
    valid_config_dict["workload"]["shared_prefix_ratios"] = [1.1]
    with pytest.raises(ValidationError):
        SuiteConfig.model_validate(valid_config_dict)


def test_environment_is_sanitized() -> None:
    assert sanitize_environment(
        {"CUDA_VISIBLE_DEVICES": "0,1", "HF_TOKEN": "secret", "api_key": "x"}
    ) == {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "HF_TOKEN": "<redacted>",
        "api_key": "<redacted>",
    }


def test_owned_child_rejects_root_and_unmarked_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be below"):
        assert_owned_child(tmp_path, tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    with pytest.raises(ValueError, match="ownership marker"):
        assert_owned_child(child, tmp_path)


def test_create_owned_directory_marks_and_validates(tmp_path: Path) -> None:
    child = create_owned_directory(tmp_path / "child", tmp_path)
    assert (child / ".vllm-cache-benchmark-owned").is_file()
    assert_owned_child(child, tmp_path)


def test_model_and_tokenizer_must_not_be_blank(valid_config_dict: dict) -> None:
    valid_config_dict["model"]["id"] = "   "
    valid_config_dict["workload"]["tokenizer"] = ""
    with pytest.raises(ValidationError):
        SuiteConfig.model_validate(valid_config_dict)


def test_required_lists_must_not_be_empty(valid_config_dict: dict) -> None:
    valid_config_dict["workload"]["prompt_tokens"] = []
    with pytest.raises(ValidationError, match="prompt_tokens"):
        SuiteConfig.model_validate(valid_config_dict)


def test_create_owned_directory_rejects_existing_unmarked_child(tmp_path: Path) -> None:
    child = tmp_path / "existing"
    child.mkdir()
    (child / "user-data.txt").write_text("do not claim", encoding="utf-8")

    with pytest.raises(ValueError, match="ownership marker"):
        create_owned_directory(child, tmp_path)

    assert not (child / ".vllm-cache-benchmark-owned").exists()
