from __future__ import annotations

import json
from pathlib import Path

from benchmarks.cache.workload import build_benchmark_command, generate_workload


class FakeTokenizer:
    all_special_ids: list[int] = []
    vocab_size = 10000

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(part) for part in text.split()] if text else []

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class ExpandingTokenizer(FakeTokenizer):
    expansion_interval = 16

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        source = [int(part) for part in text.split()] if text else []
        encoded: list[int] = []
        for index, token_id in enumerate(source, start=1):
            encoded.append(token_id)
            if index % self.expansion_interval == 0:
                encoded.append(self.vocab_size - 1)
        return encoded


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_shared_prefix_workload_is_deterministic(
    suite_config, shared_prefix_case
) -> None:
    first = generate_workload(shared_prefix_case, suite_config, FakeTokenizer())
    first_text = first.measure_path.read_text(encoding="utf-8")
    second = generate_workload(shared_prefix_case, suite_config, FakeTokenizer())
    assert second.measure_path.read_text(encoding="utf-8") == first_text


def test_shared_prefixes_match_and_suffixes_differ(
    suite_config, shared_prefix_case
) -> None:
    artifacts = generate_workload(
        shared_prefix_case, suite_config, FakeTokenizer()
    )
    rows = _rows(artifacts.measure_path)
    prefix_len = round(shared_prefix_case.prompt_tokens * shared_prefix_case.prefix_ratio)
    encoded = [FakeTokenizer().encode(row["prompt"]) for row in rows]
    assert len({tuple(tokens[:prefix_len]) for tokens in encoded}) == 1
    assert len({tuple(tokens[prefix_len:]) for tokens in encoded}) == len(rows)


def test_warm_exact_has_population_and_identical_measurement(
    suite_config, warm_exact_case
) -> None:
    artifacts = generate_workload(warm_exact_case, suite_config, FakeTokenizer())
    assert artifacts.populate_path is not None
    assert artifacts.populate_path.read_bytes() == artifacts.measure_path.read_bytes()


def test_cold_unique_has_no_population_and_unique_prompts(
    suite_config, cold_case
) -> None:
    artifacts = generate_workload(cold_case, suite_config, FakeTokenizer())
    assert artifacts.populate_path is None
    prompts = [row["prompt"] for row in _rows(artifacts.measure_path)]
    assert len(prompts) == suite_config.workload.requests_per_case
    assert len(set(prompts)) == len(prompts)


def test_mixed_prefix_population_excludes_cold_subset(
    suite_config, mixed_prefix_case
) -> None:
    artifacts = generate_workload(mixed_prefix_case, suite_config, FakeTokenizer())
    assert artifacts.populate_path is not None
    measure_rows = _rows(artifacts.measure_path)
    populate_rows = _rows(artifacts.populate_path)
    exact_warm_count = sum(1 for i in range(len(measure_rows)) if i % 4 == 1)
    assert len(populate_rows) == exact_warm_count + 2


def test_metadata_records_hashes_and_observed_lengths(
    suite_config, cold_case
) -> None:
    artifacts = generate_workload(cold_case, suite_config, FakeTokenizer())
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["case_id"] == cold_case.case_id
    assert metadata["files"]["measure"]["sha256"]
    assert set(metadata["files"]["measure"]["observed_token_lengths"]) == {
        cold_case.prompt_tokens
    }


def test_expanding_tokenizer_converges_to_requested_length(
    suite_config, warm_exact_case
) -> None:
    artifacts = generate_workload(
        warm_exact_case, suite_config, ExpandingTokenizer()
    )
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    lengths = metadata["files"]["measure"]["observed_token_lengths"]
    tolerance = suite_config.workload.token_length_tolerance
    assert all(
        abs(length - warm_exact_case.prompt_tokens) <= tolerance
        for length in lengths
    )


def test_expanding_tokenizer_preserves_shared_encoded_prefix(
    suite_config, shared_prefix_case
) -> None:
    tokenizer = ExpandingTokenizer()
    artifacts = generate_workload(shared_prefix_case, suite_config, tokenizer)
    rows = _rows(artifacts.measure_path)
    prefix_len = round(
        shared_prefix_case.prompt_tokens * shared_prefix_case.prefix_ratio
    )
    encoded = [tokenizer.encode(row["prompt"]) for row in rows]
    assert len({tuple(tokens[:prefix_len]) for tokens in encoded}) == 1


def test_benchmark_command_uses_native_custom_dataset(
    suite_config, cold_case, tmp_path: Path
) -> None:
    command = build_benchmark_command(
        cold_case,
        suite_config,
        tmp_path / "measure.jsonl",
        tmp_path / "native-result.json",
        num_prompts=8,
    )
    assert command[:3] == ["vllm", "bench", "serve"]
    assert command[command.index("--dataset-name") + 1] == "custom"
    assert command[command.index("--custom-output-len") + 1] == "-1"
    assert "--disable-shuffle" in command
    assert "--skip-chat-template" in command
    assert command[command.index("--metric-percentiles") + 1] == "50,95,99"
    assert command[command.index("--result-filename") + 1] == "native-result.json"
