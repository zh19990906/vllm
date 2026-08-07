from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from benchmarks.cache.config import SuiteConfig, create_owned_directory
from benchmarks.cache.scenarios import ExecutionCase


class TokenizerProtocol(Protocol):
    vocab_size: int
    all_special_ids: list[int]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = True
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class WorkloadArtifacts:
    populate_path: Path | None
    measure_path: Path
    metadata_path: Path
    num_population_prompts: int
    num_measurement_prompts: int


class WorkloadGenerationError(RuntimeError):
    """Raised when decoded prompts cannot meet the configured token length."""


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write_text(path, content)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workload_identity(case: ExecutionCase) -> dict[str, object]:
    return {
        "workload_kind": case.workload_kind,
        "prompt_tokens": case.prompt_tokens,
        "prefix_ratio": case.prefix_ratio,
        "repetition": case.repetition,
    }


def _generator_seed(config: SuiteConfig, case: ExecutionCase) -> int:
    identity = json.dumps(
        _workload_identity(case), sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(
        f"{config.workload.seed}:{identity}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, byteorder="big")


def _allowed_tokens(tokenizer: TokenizerProtocol) -> tuple[int, ...]:
    prohibited = set(tokenizer.all_special_ids)
    allowed = tuple(
        token_id
        for token_id in range(tokenizer.vocab_size)
        if token_id not in prohibited
    )
    if not allowed:
        raise WorkloadGenerationError("tokenizer has no non-special tokens")
    return allowed


def _next_suffix_length(
    *,
    current_suffix_length: int,
    requested_length: int,
    observed_length: int,
    fixed_prefix_length: int,
) -> int:
    target_suffix_length = requested_length - fixed_prefix_length
    observed_suffix_length = observed_length - fixed_prefix_length

    if observed_suffix_length > 0 and current_suffix_length > 0:
        candidate = round(
            current_suffix_length
            * target_suffix_length
            / observed_suffix_length
        )
    elif observed_length < requested_length:
        candidate = current_suffix_length + 1
    else:
        candidate = current_suffix_length - 1

    candidate = max(0, candidate)
    if candidate == current_suffix_length and observed_length != requested_length:
        candidate += 1 if observed_length < requested_length else -1
        candidate = max(0, candidate)
    return candidate


def _sample_prompt(
    *,
    rng: random.Random,
    tokenizer: TokenizerProtocol,
    allowed_tokens: tuple[int, ...],
    requested_length: int,
    tolerance: int,
    fixed_prefix: tuple[int, ...] = (),
    required_encoded_prefix: tuple[int, ...] | None = None,
    seen: set[tuple[int, ...]] | None = None,
) -> tuple[str, list[int]]:
    target_suffix_length = requested_length - len(fixed_prefix)
    if target_suffix_length < 0:
        raise WorkloadGenerationError(
            f"prefix length {len(fixed_prefix)} exceeds requested length "
            f"{requested_length}"
        )

    candidate_suffix_length = target_suffix_length
    suffix_tokens: list[int] = []
    tried_suffix_lengths: set[int] = set()
    last_observed = -1
    for _ in range(32):
        while len(suffix_tokens) < candidate_suffix_length:
            suffix_tokens.append(rng.choice(allowed_tokens))

        token_ids = list(fixed_prefix)
        token_ids.extend(suffix_tokens[:candidate_suffix_length])
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        last_observed = len(encoded)
        tried_suffix_lengths.add(candidate_suffix_length)

        if abs(last_observed - requested_length) > tolerance:
            next_candidate = _next_suffix_length(
                current_suffix_length=candidate_suffix_length,
                requested_length=requested_length,
                observed_length=last_observed,
                fixed_prefix_length=len(fixed_prefix),
            )
            direction = 1 if last_observed < requested_length else -1
            while (
                next_candidate in tried_suffix_lengths
                and next_candidate + direction >= 0
            ):
                next_candidate += direction
            candidate_suffix_length = next_candidate
            continue

        encoded_tuple = tuple(encoded)
        prefix_matches = required_encoded_prefix is None or tuple(
            encoded[: len(required_encoded_prefix)]
        ) == required_encoded_prefix
        is_unique = seen is None or encoded_tuple not in seen
        if not prefix_matches or not is_unique:
            candidate_suffix_length = target_suffix_length
            suffix_tokens.clear()
            tried_suffix_lengths.clear()
            continue

        if seen is not None:
            seen.add(encoded_tuple)
        return prompt, encoded

    raise WorkloadGenerationError(
        f"unable to generate prompt with requested length {requested_length}; "
        f"last observed length was {last_observed} after 32 attempts"
    )


def _row(prompt: str, output_tokens: int) -> dict[str, object]:
    return {"prompt": prompt, "output_tokens": output_tokens}


def _generate_unique_rows(
    count: int,
    case: ExecutionCase,
    config: SuiteConfig,
    tokenizer: TokenizerProtocol,
    rng: random.Random,
    allowed_tokens: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[int]]:
    rows: list[dict[str, object]] = []
    lengths: list[int] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(count):
        prompt, encoded = _sample_prompt(
            rng=rng,
            tokenizer=tokenizer,
            allowed_tokens=allowed_tokens,
            requested_length=case.prompt_tokens,
            tolerance=config.workload.token_length_tolerance,
            seen=seen,
        )
        rows.append(_row(prompt, config.workload.output_tokens))
        lengths.append(len(encoded))
    return rows, lengths


def _generate_shared_rows(
    count: int,
    ratio: float,
    case: ExecutionCase,
    config: SuiteConfig,
    tokenizer: TokenizerProtocol,
    rng: random.Random,
    allowed_tokens: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[int]]:
    prefix_length = round(case.prompt_tokens * ratio)
    fixed_prefix = tuple(rng.choice(allowed_tokens) for _ in range(prefix_length))
    rows: list[dict[str, object]] = []
    lengths: list[int] = []
    seen: set[tuple[int, ...]] = set()
    encoded_prefix: tuple[int, ...] | None = None

    for _ in range(count):
        prompt, encoded = _sample_prompt(
            rng=rng,
            tokenizer=tokenizer,
            allowed_tokens=allowed_tokens,
            requested_length=case.prompt_tokens,
            tolerance=config.workload.token_length_tolerance,
            fixed_prefix=fixed_prefix,
            required_encoded_prefix=encoded_prefix,
            seen=seen,
        )
        if encoded_prefix is None:
            encoded_prefix = tuple(encoded[:prefix_length])
        rows.append(_row(prompt, config.workload.output_tokens))
        lengths.append(len(encoded))
    return rows, lengths


def _generate_mixed_rows(
    case: ExecutionCase,
    config: SuiteConfig,
    tokenizer: TokenizerProtocol,
    rng: random.Random,
    allowed_tokens: tuple[int, ...],
) -> tuple[
    list[dict[str, object]],
    list[int],
    list[dict[str, object]],
    list[int],
]:
    measurement: list[dict[str, object]] = []
    measurement_lengths: list[int] = []
    population: list[dict[str, object]] = []
    population_lengths: list[int] = []

    cold_seen: set[tuple[int, ...]] = set()
    exact_seen: set[tuple[int, ...]] = set()
    shared_state: dict[
        float, tuple[tuple[int, ...], tuple[int, ...] | None, set[tuple[int, ...]]]
    ] = {}

    for index in range(config.workload.requests_per_case):
        group = index % 4
        if group in (0, 1):
            seen = cold_seen if group == 0 else exact_seen
            prompt, encoded = _sample_prompt(
                rng=rng,
                tokenizer=tokenizer,
                allowed_tokens=allowed_tokens,
                requested_length=case.prompt_tokens,
                tolerance=config.workload.token_length_tolerance,
                seen=seen,
            )
            row = _row(prompt, config.workload.output_tokens)
            measurement.append(row)
            measurement_lengths.append(len(encoded))
            if group == 1:
                population.append(row)
                population_lengths.append(len(encoded))
            continue

        ratio = 0.5 if group == 2 else 0.9
        prefix_length = round(case.prompt_tokens * ratio)
        if ratio not in shared_state:
            shared_state[ratio] = (
                tuple(rng.choice(allowed_tokens) for _ in range(prefix_length)),
                None,
                set(),
            )
        fixed_prefix, encoded_prefix, seen = shared_state[ratio]
        prompt, encoded = _sample_prompt(
            rng=rng,
            tokenizer=tokenizer,
            allowed_tokens=allowed_tokens,
            requested_length=case.prompt_tokens,
            tolerance=config.workload.token_length_tolerance,
            fixed_prefix=fixed_prefix,
            required_encoded_prefix=encoded_prefix,
            seen=seen,
        )
        if encoded_prefix is None:
            encoded_prefix = tuple(encoded[:prefix_length])
            shared_state[ratio] = (fixed_prefix, encoded_prefix, seen)
            representative = _row(prompt, config.workload.output_tokens)
            population.append(representative)
            population_lengths.append(len(encoded))
        measurement.append(_row(prompt, config.workload.output_tokens))
        measurement_lengths.append(len(encoded))

    return measurement, measurement_lengths, population, population_lengths


def generate_workload(
    case: ExecutionCase,
    config: SuiteConfig,
    tokenizer: TokenizerProtocol,
) -> WorkloadArtifacts:
    """Generate deterministic JSONL artifacts for an execution case."""
    result_dir = create_owned_directory(case.result_dir, config.results.root_dir)
    seed = _generator_seed(config, case)
    rng = random.Random(seed)
    allowed_tokens = _allowed_tokens(tokenizer)
    count = config.workload.requests_per_case

    populate_rows: list[dict[str, object]] | None
    populate_lengths: list[int] | None
    if case.workload_kind == "cold-unique":
        measure_rows, measure_lengths = _generate_unique_rows(
            count, case, config, tokenizer, rng, allowed_tokens
        )
        populate_rows = None
        populate_lengths = None
    elif case.workload_kind in ("warm-exact-prefix", "restart-persistence"):
        measure_rows, measure_lengths = _generate_unique_rows(
            count, case, config, tokenizer, rng, allowed_tokens
        )
        populate_rows = list(measure_rows)
        populate_lengths = list(measure_lengths)
    elif case.workload_kind == "shared-prefix":
        measure_rows, measure_lengths = _generate_shared_rows(
            count,
            case.prefix_ratio,
            case,
            config,
            tokenizer,
            rng,
            allowed_tokens,
        )
        populate_rows = [measure_rows[0]]
        populate_lengths = [measure_lengths[0]]
    elif case.workload_kind == "mixed-prefix":
        (
            measure_rows,
            measure_lengths,
            populate_rows,
            populate_lengths,
        ) = _generate_mixed_rows(
            case, config, tokenizer, rng, allowed_tokens
        )
    else:
        raise WorkloadGenerationError(
            f"unsupported workload kind: {case.workload_kind}"
        )

    measure_path = result_dir / "measure.jsonl"
    _write_jsonl(measure_path, measure_rows)

    populate_path: Path | None = None
    if populate_rows is not None:
        populate_path = result_dir / "populate.jsonl"
        _write_jsonl(populate_path, populate_rows)

    files: dict[str, dict[str, object]] = {
        "measure": {
            "path": str(measure_path),
            "sha256": _file_sha256(measure_path),
            "requested_token_length": case.prompt_tokens,
            "observed_token_lengths": measure_lengths,
            "rows": len(measure_rows),
        }
    }
    if populate_path is not None and populate_lengths is not None:
        files["populate"] = {
            "path": str(populate_path),
            "sha256": _file_sha256(populate_path),
            "requested_token_length": case.prompt_tokens,
            "observed_token_lengths": populate_lengths,
            "rows": len(populate_rows or []),
        }

    metadata_path = result_dir / "metadata.json"
    metadata = {
        "case_id": case.case_id,
        "cache_mode": case.cache_mode.value,
        "workload_kind": case.workload_kind,
        "prompt_tokens": case.prompt_tokens,
        "prefix_ratio": case.prefix_ratio,
        "concurrency": case.concurrency,
        "request_rate": case.request_rate,
        "repetition": case.repetition,
        "generator_seed": seed,
        "files": files,
    }
    _atomic_write_text(
        metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )

    return WorkloadArtifacts(
        populate_path=populate_path,
        measure_path=measure_path,
        metadata_path=metadata_path,
        num_population_prompts=len(populate_rows or []),
        num_measurement_prompts=len(measure_rows),
    )


def build_benchmark_command(
    case: ExecutionCase,
    config: SuiteConfig,
    dataset_path: Path,
    native_result_path: Path,
    *,
    num_prompts: int,
) -> list[str]:
    """Build a native ``vllm bench serve`` custom-dataset command."""
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        config.server.host,
        "--port",
        str(config.server.port),
        "--endpoint",
        "/v1/completions",
        "--model",
        config.model.served_name,
        "--tokenizer",
        config.workload.tokenizer,
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset_path),
        "--custom-output-len",
        "-1",
        "--disable-shuffle",
        "--skip-chat-template",
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(case.request_rate),
        "--max-concurrency",
        str(case.concurrency),
        "--num-warmups",
        "0",
        "--seed",
        str(config.workload.seed),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(native_result_path.parent),
        "--result-filename",
        native_result_path.name,
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
        "--disable-tqdm",
    ]
