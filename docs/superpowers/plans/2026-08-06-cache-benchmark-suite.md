# Cache Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible external benchmark suite that compares native vLLM cache modes without modifying the inference hot path.

**Architecture:** The suite is a Python package under `benchmarks/cache/`. It validates one strict YAML configuration, expands it into isolated execution cases, generates deterministic JSONL workloads, launches `vllm serve`, invokes native `vllm bench serve`, snapshots Prometheus and host/GPU evidence, persists append-only normalized records, and regenerates CSV/Markdown reports. Every measured case starts from a controlled server/cache state; the filesystem persistence case uses two server lifecycles with the same cache directory.

**Tech Stack:** Python 3.10+, Pydantic v2, PyYAML, pytest, `requests`, `prometheus_client`, `psutil`, stdlib `subprocess`/`pathlib`/`csv`/`json`, native `vllm serve`, native `vllm bench serve`.

## Global Constraints

- Base commit is vLLM v0.26.0: `568afb3a13806beb53bb2e6bd518269357b237c0`.
- Do not modify Scheduler, Attention, PagedAttention, KVCacheManager, BlockPool, OffloadingConnector, or any inference-core source file.
- Add no third-party dependency: Pydantic, PyYAML, requests, prometheus_client, psutil, transformers/tokenizer support, and pytest already exist in the repository environment.
- Reject unknown YAML keys.
- Execute commands as argument arrays with `shell=False`.
- Default server bind address is `127.0.0.1`.
- Dry-run mode must not start processes, issue HTTP requests, or delete cache data.
- Never serialize environment values whose key contains `TOKEN`, `PASSWORD`, `SECRET`, `CREDENTIAL`, `API_KEY`, `ACCESS_KEY`, or `PRIVATE_KEY`, case-insensitively.
- Only delete directories created beneath the configured results root or filesystem cache root and containing the suite marker file `.vllm-cache-benchmark-owned`.
- Missing metrics are represented by `null` plus a reason; never convert missing data to zero.
- Unit and fake-integration tests must run without a GPU or model download.

---

## File Map

Create these production files:

- `benchmarks/cache/__init__.py`: package metadata only.
- `benchmarks/cache/config.py`: strict YAML schema, normalization, secret sanitization, safe-path checks.
- `benchmarks/cache/scenarios.py`: cache-mode matrix and exact `vllm serve` command/environment construction.
- `benchmarks/cache/workload.py`: deterministic prompt construction, JSONL persistence, native benchmark command construction.
- `benchmarks/cache/process.py`: managed subprocess lifecycle, readiness polling, graceful/forced shutdown.
- `benchmarks/cache/metrics.py`: Prometheus parsing/deltas, native-result normalization, resource sampling, environment evidence.
- `benchmarks/cache/report.py`: append-only JSONL writer, summary rows, CSV and Markdown generation.
- `benchmarks/cache/run_suite.py`: CLI and orchestration state machine.
- `benchmarks/cache/README.md`: operating guide and result interpretation.
- `benchmarks/cache/configs/example-7b.yaml`
- `benchmarks/cache/configs/example-70b.yaml`
- `benchmarks/cache/configs/example-397b.yaml`

Create these tests:

- `benchmarks/cache/tests/conftest.py`
- `benchmarks/cache/tests/test_config.py`
- `benchmarks/cache/tests/test_scenarios.py`
- `benchmarks/cache/tests/test_workload.py`
- `benchmarks/cache/tests/test_process.py`
- `benchmarks/cache/tests/test_metrics.py`
- `benchmarks/cache/tests/test_report.py`
- `benchmarks/cache/tests/test_run_suite.py`

No existing source file should be modified.

---

### Task 1: Strict Configuration and Safety Primitives

**Files:**
- Create: `benchmarks/cache/__init__.py`
- Create: `benchmarks/cache/config.py`
- Create: `benchmarks/cache/tests/conftest.py`
- Create: `benchmarks/cache/tests/test_config.py`

**Interfaces:**
- Produces: `SuiteConfig`, `load_suite_config(path: Path) -> SuiteConfig`, `sanitize_environment(env: Mapping[str, str]) -> dict[str, str]`, `assert_owned_child(path: Path, root: Path) -> None`, `create_owned_directory(path: Path, root: Path) -> Path`.
- Consumers: Tasks 2–8.

- [ ] **Step 1: Add failing tests for valid parsing and normalization**

```python
# benchmarks/cache/tests/test_config.py
from pathlib import Path

from benchmarks.cache.config import load_suite_config


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
```

- [ ] **Step 2: Add failing tests for unknown keys, invalid ratios, and unsafe paths**

```python
import pytest
from pydantic import ValidationError

from benchmarks.cache.config import assert_owned_child, sanitize_environment


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
    ) == {"CUDA_VISIBLE_DEVICES": "0,1", "HF_TOKEN": "<redacted>", "api_key": "<redacted>"}


def test_owned_child_rejects_root_and_unmarked_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be below"):
        assert_owned_child(tmp_path, tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    with pytest.raises(ValueError, match="ownership marker"):
        assert_owned_child(child, tmp_path)
```

- [ ] **Step 3: Run tests and confirm they fail**

Run:

```bash
pytest -q benchmarks/cache/tests/test_config.py
```

Expected: collection/import failure because `benchmarks.cache.config` does not exist.

- [ ] **Step 4: Implement strict Pydantic models and validators**

Implement `config.py` with `ConfigDict(extra="forbid", frozen=True)` on every model and these exact public models:

```python
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
RequestRate = Literal["inf"] | PositiveFloat


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    id: str
    served_name: str
    dtype: str = "auto"
    max_model_len: PositiveInt
    trust_remote_code: bool = False


class ParallelismConfig(StrictModel):
    tensor_parallel_size: PositiveInt = 1
    pipeline_parallel_size: PositiveInt = 1


class ServerConfig(StrictModel):
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8100
    startup_timeout_seconds: PositiveInt = 900
    shutdown_timeout_seconds: PositiveInt = 60
    extra_args: list[str] = []
    env: dict[str, str] = {}


class FilesystemCacheConfig(StrictModel):
    enabled: bool = True
    root_dir: Path
    read_threads: PositiveInt = 32
    write_threads: PositiveInt = 16


class CacheConfig(StrictModel):
    gpu_memory_utilization: Annotated[float, Field(gt=0.0, le=1.0)] = 0.9
    cpu_bytes_to_use: PositiveInt
    offload_block_size: PositiveInt = 64
    eviction_policy: Literal["lru", "arc"] = "lru"
    filesystem: FilesystemCacheConfig


class WorkloadConfig(StrictModel):
    seed: int = 1
    tokenizer: str
    prompt_tokens: list[PositiveInt]
    output_tokens: PositiveInt
    concurrency: list[PositiveInt]
    request_rate: list[RequestRate]
    requests_per_case: PositiveInt
    shared_prefix_ratios: list[Ratio]
    warmup_requests: Annotated[int, Field(ge=0)] = 2
    token_length_tolerance: Annotated[int, Field(ge=0, le=8)] = 2

    @field_validator("request_rate", mode="before")
    @classmethod
    def normalize_request_rates(cls, values: list[object]) -> list[object]:
        normalized: list[object] = []
        for value in values:
            if isinstance(value, str) and value.lower() == "inf":
                normalized.append("inf")
            elif isinstance(value, (int, float)) and value == float("inf"):
                normalized.append("inf")
            else:
                normalized.append(value)
        return normalized


class ResultsConfig(StrictModel):
    root_dir: Path
    keep_server_logs: bool = True
    fail_fast: bool = False


class SuiteConfig(StrictModel):
    schema_version: Literal[1]
    model: ModelConfig
    parallelism: ParallelismConfig
    server: ServerConfig
    cache: CacheConfig
    workload: WorkloadConfig
    results: ResultsConfig

    @model_validator(mode="after")
    def validate_required_ratios(self) -> "SuiteConfig":
        if not self.workload.prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        if not self.workload.concurrency:
            raise ValueError("concurrency must not be empty")
        if not self.workload.request_rate:
            raise ValueError("request_rate must not be empty")
        return self
```

`load_suite_config()` must use `yaml.safe_load`, resolve relative paths against `path.parent`, and return a revalidated copied model. Implement the safety helpers with `Path.resolve()`, `Path.is_relative_to()`, and marker name `.vllm-cache-benchmark-owned`.

- [ ] **Step 5: Run config tests**

```bash
pytest -q benchmarks/cache/tests/test_config.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add benchmarks/cache/__init__.py benchmarks/cache/config.py benchmarks/cache/tests/conftest.py benchmarks/cache/tests/test_config.py
git commit -m "feat: add cache benchmark configuration"
```

---

### Task 2: Cache Modes and Server Command Construction

**Files:**
- Create: `benchmarks/cache/scenarios.py`
- Create: `benchmarks/cache/tests/test_scenarios.py`

**Interfaces:**
- Consumes: `SuiteConfig`, `create_owned_directory()`.
- Produces: `CacheMode`, `ExecutionCase`, `build_execution_cases(config: SuiteConfig, run_dir: Path) -> list[ExecutionCase]`, `build_server_command(case: ExecutionCase, config: SuiteConfig) -> list[str]`, `build_server_environment(case: ExecutionCase, config: SuiteConfig) -> dict[str, str]`.
- Consumers: Tasks 3, 7, 8.

- [ ] **Step 1: Write failing tests for the four native cache modes**

```python
import json
from pathlib import Path

from benchmarks.cache.scenarios import (
    CacheMode,
    build_execution_cases,
    build_server_command,
)


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_server_commands_cover_four_cache_modes(
    suite_config, tmp_path: Path
) -> None:
    cases = build_execution_cases(suite_config, tmp_path)
    representative = {
        case.cache_mode: case
        for case in cases
        if case.workload_kind == "cold-unique"
        and case.prompt_tokens == suite_config.workload.prompt_tokens[0]
        and case.concurrency == suite_config.workload.concurrency[0]
        and case.request_rate == suite_config.workload.request_rate[0]
    }

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
```

- [ ] **Step 2: Write failing tests for deterministic identity and filesystem isolation**

```python
def test_case_ids_are_deterministic_and_unique(suite_config, tmp_path: Path) -> None:
    first = build_execution_cases(suite_config, tmp_path)
    second = build_execution_cases(suite_config, tmp_path)
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert len({case.case_id for case in first}) == len(first)


def test_tiered_cases_use_owned_per_case_directories(suite_config, tmp_path: Path) -> None:
    cases = build_execution_cases(suite_config, tmp_path)
    tiered = next(case for case in cases if case.cache_mode is CacheMode.TIERED_FS)
    assert tiered.filesystem_cache_dir is not None
    assert tiered.filesystem_cache_dir.is_relative_to(
        suite_config.cache.filesystem.root_dir
    )
```

- [ ] **Step 3: Run tests and confirm failure**

```bash
pytest -q benchmarks/cache/tests/test_scenarios.py
```

Expected: FAIL because `scenarios.py` does not exist.

- [ ] **Step 4: Implement scenario data types and matrix expansion**

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

RequestRate = str | float
WorkloadKind = Literal[
    "cold-unique",
    "warm-exact-prefix",
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
```

Matrix rules:

1. Generate `cold-unique`, `warm-exact-prefix`, `shared-prefix`, and `mixed-prefix` for all four cache modes.
2. Generate `restart-persistence` only for `tiered-fs`.
3. `shared-prefix` expands every configured ratio greater than zero; other workload kinds use ratio `0.0`, except `mixed-prefix`, whose internal composition is fixed in Task 3.
4. Expand every configured prompt length, concurrency, and request rate.
5. Set `repetition=0` in phase A; keep it in the identity for later repetitions.
6. Construct `case_id` from a canonical JSON identity hashed with SHA-256, prefixed by readable dimensions, for example `tiered-fs__shared-prefix__p4096__r0.900__c8__qinf__7c8a1d2e`.
7. Create result/cache directories only during real execution, not while building cases; store proposed paths in the dataclass.

- [ ] **Step 5: Implement exact server arguments**

All commands begin with:

```python
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
```

Append `--trust-remote-code` only when enabled, then append `server.extra_args` verbatim. Use `--no-enable-prefix-caching` for `no-cache`; use `--enable-prefix-caching` for the other modes.

CPU offload JSON must be:

```python
{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "cpu_bytes_to_use": config.cache.cpu_bytes_to_use,
        "block_size": config.cache.offload_block_size,
        "eviction_policy": config.cache.eviction_policy,
    },
}
```

Tiered FS adds:

```python
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
```

Serialize with `json.dumps(config, sort_keys=True, separators=(",", ":"))`. `build_server_environment()` merges `os.environ`, configured server env, and forces `PYTHONHASHSEED="0"` for tiered FS.

- [ ] **Step 6: Run scenario tests**

```bash
pytest -q benchmarks/cache/tests/test_scenarios.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add benchmarks/cache/scenarios.py benchmarks/cache/tests/test_scenarios.py
git commit -m "feat: generate cache benchmark scenarios"
```

---

### Task 3: Deterministic Workloads and Native Benchmark Commands

**Files:**
- Create: `benchmarks/cache/workload.py`
- Create: `benchmarks/cache/tests/test_workload.py`

**Interfaces:**
- Consumes: `SuiteConfig`, `ExecutionCase`.
- Produces: `TokenizerProtocol`, `WorkloadArtifacts`, `generate_workload(case, config, tokenizer) -> WorkloadArtifacts`, `build_benchmark_command(case, config, dataset_path, native_result_path, *, num_prompts) -> list[str]`.
- Consumers: Task 7.

- [ ] **Step 1: Write failing tests with a reversible fake tokenizer**

```python
from pathlib import Path

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


def test_shared_prefix_workload_is_deterministic(
    suite_config, shared_prefix_case, tmp_path: Path
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
    rows = [json.loads(line) for line in artifacts.measure_path.read_text().splitlines()]
    prefix_len = round(shared_prefix_case.prompt_tokens * shared_prefix_case.prefix_ratio)
    encoded = [FakeTokenizer().encode(row["prompt"]) for row in rows]
    assert len({tuple(tokens[:prefix_len]) for tokens in encoded}) == 1
    assert len({tuple(tokens[prefix_len:]) for tokens in encoded}) == len(rows)
```

- [ ] **Step 2: Add failing tests for warm/populate files and benchmark CLI**

```python
def test_warm_exact_has_population_and_identical_measurement(
    suite_config, warm_exact_case
) -> None:
    artifacts = generate_workload(warm_exact_case, suite_config, FakeTokenizer())
    assert artifacts.populate_path is not None
    assert artifacts.populate_path.read_bytes() == artifacts.measure_path.read_bytes()


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
```

- [ ] **Step 3: Run tests and confirm failure**

```bash
pytest -q benchmarks/cache/tests/test_workload.py
```

Expected: FAIL because `workload.py` does not exist.

- [ ] **Step 4: Implement workload data types and token generation**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


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
```

Generation algorithm:

1. Seed `random.Random` with `sha256(f"{config.workload.seed}:{case.case_id}")` converted to an integer.
2. Build an allowed token pool from `[0, tokenizer.vocab_size)` excluding `all_special_ids`.
3. Sample token IDs, decode, re-encode, and retry up to 32 times until the encoded length differs from the requested length by no more than `token_length_tolerance`; otherwise raise `WorkloadGenerationError` containing requested and observed lengths.
4. Write one JSON object per line with exactly `{"prompt": <string>, "output_tokens": <int>}`. This matches vLLM v0.26.0 `CustomDataset`.
5. Use UTF-8, final newline, stable key ordering, and atomic `Path.replace()`.

Workload rules:

- `cold-unique`: every full prompt is unique; no population file.
- `warm-exact-prefix`: generate `requests_per_case` unique prompts; population and measurement files contain the same rows in the same order.
- `shared-prefix`: one shared prefix of `round(prompt_tokens * prefix_ratio)` tokens and unique suffixes; population file contains one representative row to establish the prefix, measurement contains `requests_per_case` suffix variants.
- `mixed-prefix`: measurement rows are allocated by index modulo four: unique cold; fully warmed exact; shared ratio 0.5; shared ratio 0.9. Population contains the exact rows and one representative for each shared-prefix group required to warm those subsets.
- `restart-persistence`: same artifact layout as warm exact, but Task 7 stops the server after population and restarts before measurement.

Persist `metadata.json` with case identity, requested/observed token lengths, SHA-256 of each JSONL file, and generator seed.

- [ ] **Step 5: Implement native benchmark command construction**

Use this fixed command shape:

```python
[
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
```

Population commands use the same builder but write `population-result.json`; their latency metrics are not included in the final comparison.

- [ ] **Step 6: Run workload tests**

```bash
pytest -q benchmarks/cache/tests/test_workload.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add benchmarks/cache/workload.py benchmarks/cache/tests/test_workload.py
git commit -m "feat: generate reproducible cache workloads"
```

---

### Task 4: Managed Processes and Readiness

**Files:**
- Create: `benchmarks/cache/process.py`
- Create: `benchmarks/cache/tests/test_process.py`

**Interfaces:**
- Produces: `CommandResult`, `ManagedProcess`, `run_command()`, `start_server()`, `wait_for_server()`, `stop_server()`.
- Consumers: Tasks 5 and 7.

- [ ] **Step 1: Write failing tests for command capture and timeout**

```python
import sys
from pathlib import Path

from benchmarks.cache.process import run_command


def test_run_command_captures_output(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-c", "print('ok')"],
        env={},
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        timeout_seconds=5,
    )
    assert result.returncode == 0
    assert (tmp_path / "stdout.log").read_text().strip() == "ok"


def test_run_command_returns_timeout_record(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        env={},
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        timeout_seconds=0.1,
    )
    assert result.timed_out is True
    assert result.returncode is None
```

- [ ] **Step 2: Write failing tests for readiness and process-group shutdown**

Use a test HTTP server started with `python -m http.server` for a successful status probe, and a Python child process that spawns another sleeping process to verify `stop_server()` terminates the process group.

- [ ] **Step 3: Run tests and confirm failure**

```bash
pytest -q benchmarks/cache/tests/test_process.py
```

Expected: FAIL because `process.py` does not exist.

- [ ] **Step 4: Implement managed process types**

```python
@dataclass(frozen=True, slots=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int | None
    timed_out: bool
    started_at: str
    ended_at: str
    elapsed_seconds: float
    stdout_path: Path
    stderr_path: Path


@dataclass(slots=True)
class ManagedProcess:
    command: tuple[str, ...]
    process: subprocess.Popen[bytes]
    stdout_handle: BinaryIO
    stderr_handle: BinaryIO
    stdout_path: Path
    stderr_path: Path
    started_at: str
```

Implementation requirements:

- Use `subprocess.Popen(..., shell=False, start_new_session=True)`.
- Write stdout/stderr directly to binary log files.
- `wait_for_server(base_url, timeout_seconds)` polls `GET /v1/models` every 0.5 seconds with connect/read timeout 2 seconds and succeeds only on HTTP 200 with non-empty `data`.
- Abort readiness immediately with `ServerExitedError` if the process exits.
- `stop_server()` sends SIGTERM to the process group, waits the configured timeout, then sends SIGKILL; it always closes file handles.
- `run_command()` uses the same process-group behavior on timeout.
- Never call `shell=True`.

- [ ] **Step 5: Run process tests**

```bash
pytest -q benchmarks/cache/tests/test_process.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add benchmarks/cache/process.py benchmarks/cache/tests/test_process.py
git commit -m "feat: manage benchmark subprocesses"
```

---

### Task 5: Metrics, Resource Sampling, and Environment Evidence

**Files:**
- Create: `benchmarks/cache/metrics.py`
- Create: `benchmarks/cache/tests/test_metrics.py`

**Interfaces:**
- Consumes: `CommandResult`, `ExecutionCase`, `SuiteConfig`.
- Produces: `PrometheusSnapshot`, `ResourceSample`, `ResourceSampler`, `fetch_prometheus_snapshot()`, `compute_prometheus_delta()`, `normalize_native_result()`, `collect_environment_evidence()`.
- Consumers: Tasks 6 and 7.

- [ ] **Step 1: Write failing Prometheus parsing/delta tests**

```python
from benchmarks.cache.metrics import parse_prometheus_text, compute_prometheus_delta


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
    assert delta["vllm:kv_offload_tiering_lookup_sync_delay_seconds_count"]["value"] == 4
    assert delta["vllm:kv_offload_tiering_lookup_sync_delay_seconds_sum"]["value"] == 1.2
```

- [ ] **Step 2: Write failing native result and missing-metric tests**

```python
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
        )
    )
    normalized = normalize_native_result(native)
    assert normalized["benchmark"]["ttft_ms"]["p95"] == 30.0
    assert normalized["cache"]["prefix_cache_hits_tokens"]["value"] is None
    assert normalized["cache"]["prefix_cache_hits_tokens"]["reason"] == "metric_not_exposed"
```

Also test that a top-level one-element JSON list is accepted, because `--append-result` and version differences may produce list-shaped output.

- [ ] **Step 3: Write failing resource sampler tests**

Inject callables for RSS and GPU-memory sampling. Verify the sampler records peak and final values and records `gpu_sampler_unavailable` rather than raising when `nvidia-smi` is absent.

- [ ] **Step 4: Run tests and confirm failure**

```bash
pytest -q benchmarks/cache/tests/test_metrics.py
```

Expected: FAIL because `metrics.py` does not exist.

- [ ] **Step 5: Implement Prometheus snapshots and selected metric schema**

Use `prometheus_client.parser.text_string_to_metric_families`. Flatten samples by full sample name plus sorted labels. Capture all `vllm:` metrics in raw snapshots, then expose these selected metrics when present:

```python
SELECTED_METRICS = {
    "cpu_cache_usage_perc": "vllm:kv_offload_cpu_cache_usage_perc",
    "cpu_cache_write_usage_perc": "vllm:kv_offload_cpu_cache_write_usage_perc",
    "cpu_cache_read_usage_perc": "vllm:kv_offload_cpu_cache_read_usage_perc",
    "cpu_allocation_size": "vllm:kv_offload_cpu_allocation_size",
    "stores_skipped": "vllm:kv_offload_stores_skipped",
    "tiering_lookup_sync_delay_seconds": "vllm:kv_offload_tiering_lookup_sync_delay_seconds",
    "tiering_lookup_async_delay_seconds": "vllm:kv_offload_tiering_lookup_async_delay_seconds",
}
```

For prefix-cache evidence, search exact sample names first and then suffixes in this order:

```python
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
```

If none exist, write a null value and `metric_not_exposed`. Preserve the entire raw before/after snapshot so later schema revisions can recover metrics without rerunning benchmarks.

- [ ] **Step 6: Implement native normalization and resource sampling**

Normalized benchmark fields are:

```python
{
    "completed": int | None,
    "failed": int | None,
    "duration_seconds": float | None,
    "request_throughput": float | None,
    "output_token_throughput": float | None,
    "total_token_throughput": float | None,
    "ttft_ms": {"mean": ..., "median": ..., "p50": ..., "p95": ..., "p99": ...},
    "tpot_ms": {"mean": ..., "median": ..., "p50": ..., "p95": ..., "p99": ...},
    "itl_ms": {"mean": ..., "median": ..., "p50": ..., "p95": ..., "p99": ...},
    "e2el_ms": {"mean": ..., "median": ..., "p50": ..., "p95": ..., "p99": ...},
}
```

`ResourceSampler` polls once per second while the measured benchmark command runs:

- process-tree RSS through `psutil.Process(pid).children(recursive=True)`;
- system available/used memory through `psutil.virtual_memory()`;
- GPU used memory using `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`, summed for the server process tree.

Store peak, mean, final, sample count, and unavailable reason.

`collect_environment_evidence()` executes optional commands with 10-second limits: `vllm --version`, `python --version`, `git rev-parse HEAD`, `git status --porcelain`, `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`, `nvidia-smi topo -m`, `lscpu --json`, and `numactl --hardware`. Failure is captured as `{status: "unavailable", reason: ...}`.

- [ ] **Step 7: Run metrics tests**

```bash
pytest -q benchmarks/cache/tests/test_metrics.py
```

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

```bash
git add benchmarks/cache/metrics.py benchmarks/cache/tests/test_metrics.py
git commit -m "feat: collect cache benchmark metrics"
```

---

### Task 6: Append-Only Results and Comparison Reports

**Files:**
- Create: `benchmarks/cache/report.py`
- Create: `benchmarks/cache/tests/test_report.py`

**Interfaces:**
- Produces: `append_result(path: Path, record: Mapping[str, Any]) -> None`, `load_results(path: Path) -> list[dict[str, Any]]`, `build_summary_rows(records) -> list[dict[str, Any]]`, `write_summary_csv()`, `write_markdown_report()`, `rebuild_reports(run_dir: Path) -> None`.
- Consumers: Task 7.

- [ ] **Step 1: Write failing append/recovery tests**

```python
def test_jsonl_is_immediately_recoverable(tmp_path: Path) -> None:
    path = tmp_path / "scenario-results.jsonl"
    append_result(path, {"case_id": "a", "status": "completed"})
    append_result(path, {"case_id": "b", "status": "benchmark_error"})
    assert [row["case_id"] for row in load_results(path)] == ["a", "b"]
```

The implementation must flush and `os.fsync()` after every appended line.

- [ ] **Step 2: Write failing comparison identity tests**

```python
def test_report_compares_only_matching_workloads() -> None:
    rows = build_summary_rows(
        [
            synthetic_record("no-cache", prompt_tokens=1024, p95_ttft=100),
            synthetic_record("gpu-apc", prompt_tokens=1024, p95_ttft=50),
            synthetic_record("cpu-offload", prompt_tokens=4096, p95_ttft=40),
        ]
    )
    cpu_row = next(row for row in rows if row["cache_mode"] == "cpu-offload")
    assert cpu_row["p95_ttft_delta_vs_no_cache_pct"] is None
    assert cpu_row["comparison_reason"] == "matching_baseline_not_found"
```

- [ ] **Step 3: Write failing Markdown/CSV snapshot tests**

Verify report sections include environment summary, completed/failed count, absolute metrics, deltas vs `no-cache` and `gpu-apc`, missing-metric reasons, and commands/log paths.

- [ ] **Step 4: Run tests and confirm failure**

```bash
pytest -q benchmarks/cache/tests/test_report.py
```

Expected: FAIL because `report.py` does not exist.

- [ ] **Step 5: Implement atomic report generation**

Comparison key excludes cache mode and case ID and includes:

```python
COMPARISON_FIELDS = (
    "workload_kind",
    "prompt_tokens",
    "prefix_ratio",
    "concurrency",
    "request_rate",
    "repetition",
    "model_id",
    "model_revision",
    "tensor_parallel_size",
    "pipeline_parallel_size",
)
```

Compute percent delta as `(candidate - baseline) / baseline * 100`; report TTFT decreases as improvement using a separate positive field `p95_ttft_improvement_pct = (baseline - candidate) / baseline * 100`. Do not compare failed records or records with missing metrics.

Write `summary.csv` and `report.md` to temporary siblings and replace atomically. The Markdown report must explicitly label population runs as unmeasured setup and restart-persistence as a post-restart measurement.

- [ ] **Step 6: Run report tests**

```bash
pytest -q benchmarks/cache/tests/test_report.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add benchmarks/cache/report.py benchmarks/cache/tests/test_report.py
git commit -m "feat: report cache benchmark results"
```

---

### Task 7: End-to-End Runner, Dry Run, and Restart Persistence

**Files:**
- Create: `benchmarks/cache/run_suite.py`
- Create: `benchmarks/cache/tests/test_run_suite.py`

**Interfaces:**
- Consumes all interfaces from Tasks 1–6.
- Produces CLI: `python benchmarks/cache/run_suite.py --config PATH [--dry-run] [--rebuild-report RUN_DIR] [--case-id ID ...]`.

- [ ] **Step 1: Write failing dry-run test**

```python
def test_dry_run_writes_manifest_without_starting_processes(
    monkeypatch, config_path: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "benchmarks.cache.run_suite.start_server",
        lambda *args, **kwargs: pytest.fail("start_server called in dry run"),
    )
    exit_code = main(["--config", str(config_path), "--dry-run"])
    assert exit_code == 0
    run_dirs = list((tmp_path / "results").iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert {item["cache_mode"] for item in manifest["cases"]} == {
        "no-cache", "gpu-apc", "cpu-offload", "tiered-fs"
    }
```

- [ ] **Step 2: Write failing partial-result and cleanup test**

Inject fake start/run/stop functions. Make the second measured command fail. Assert the first completed JSONL record remains, the second error record contains `benchmark_error`, the server is stopped in a `finally` path, and later cases continue when `fail_fast=false`.

- [ ] **Step 3: Write failing restart-persistence state-machine test**

Record calls and assert exact order:

```text
start server -> population benchmark -> stop server -> start server using same FS path -> measured benchmark -> stop server
```

Assert Prometheus before/after snapshots and resource sampling wrap only the measured benchmark, not population.

- [ ] **Step 4: Run tests and confirm failure**

```bash
pytest -q benchmarks/cache/tests/test_run_suite.py
```

Expected: FAIL because `run_suite.py` does not exist.

- [ ] **Step 5: Implement CLI and import behavior**

Support both direct-script and module execution:

```python
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

Parser arguments:

```python
parser.add_argument("--config", type=Path)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--case-id", action="append", default=[])
parser.add_argument("--rebuild-report", type=Path)
```

`--config` is required unless `--rebuild-report` is present. Return process exit codes rather than calling `sys.exit()` inside testable functions.

- [ ] **Step 6: Implement run directory and manifest creation**

Run ID format is `YYYYMMDDTHHMMSSZ-<8-char-config-sha>`. Create:

```text
manifest.json
environment.json
scenarios.json
scenario-results.jsonl
summary.csv
report.md
workloads/
raw/<case-id>/
```

Manifest includes schema version, base commit, sanitized config, config fingerprint, selected case IDs, dry-run flag, and timestamps. Persist exact sanitized server/benchmark command arrays before execution.

- [ ] **Step 7: Implement one isolated execution case**

For every case:

1. Create/mark owned case result directory.
2. For tiered FS, create/mark the per-case cache directory; ordinary cases do not delete user paths.
3. Generate workload artifacts.
4. Start server and wait for `/v1/models`.
5. Run population benchmark if present.
6. For restart-persistence only: stop server, start it again with the same filesystem path, and wait again.
7. Snapshot Prometheus.
8. Start `ResourceSampler` bound to server PID.
9. Run measured native benchmark.
10. Stop sampler and snapshot Prometheus again.
11. Normalize result and append completed record.
12. In `finally`, stop any running server.
13. On error, append one categorized error record with stage, command result, timestamps, retryability, and raw log paths.
14. Rebuild CSV/Markdown after every record.

Error mapping is exact:

```python
ConfigurationError -> "configuration_error"
ServerExitedError -> "server_start_error"
ServerReadyTimeout -> "server_timeout"
CommandTimeout during benchmark -> "benchmark_error"
NativeResultParseError -> "parse_error"
MetricsCollectionError -> "metrics_error"
ShutdownError -> "shutdown_error"
KeyboardInterrupt -> "interrupted"
```

- [ ] **Step 8: Implement dry-run and report rebuild**

Dry run performs config loading, case expansion, command construction, workload metadata planning, path safety validation, and manifest/scenario output. It does not instantiate tokenizers, create filesystem cache directories, start processes, fetch metrics, or run environment commands that may be unavailable.

`--rebuild-report RUN_DIR` reads only `scenario-results.jsonl` and regenerates `summary.csv` and `report.md`.

- [ ] **Step 9: Run runner tests**

```bash
pytest -q benchmarks/cache/tests/test_run_suite.py
```

Expected: PASS.

- [ ] **Step 10: Commit Task 7**

```bash
git add benchmarks/cache/run_suite.py benchmarks/cache/tests/test_run_suite.py
git commit -m "feat: orchestrate cache benchmark suite"
```

---

### Task 8: Examples, Documentation, and Fake End-to-End Validation

**Files:**
- Create: `benchmarks/cache/README.md`
- Create: `benchmarks/cache/configs/example-7b.yaml`
- Create: `benchmarks/cache/configs/example-70b.yaml`
- Create: `benchmarks/cache/configs/example-397b.yaml`
- Modify: `benchmarks/cache/tests/test_run_suite.py`

**Interfaces:**
- Produces user-facing operating instructions and validated examples.

- [ ] **Step 1: Add failing tests that load all examples**

```python
@pytest.mark.parametrize(
    "name", ["example-7b.yaml", "example-70b.yaml", "example-397b.yaml"]
)
def test_example_config_is_valid(name: str) -> None:
    config = load_suite_config(Path("benchmarks/cache/configs") / name)
    assert config.schema_version == 1
    assert config.parallelism.tensor_parallel_size in {1, 2, 4, 8}
```

Examples use local placeholder model paths that are syntactically valid and clearly documented as values the operator must replace; they are not code placeholders in the implementation plan.

- [ ] **Step 2: Add a fake executable integration test**

Create temporary fake `vllm` and `nvidia-smi` executables on PATH. The fake `vllm serve` process exposes `/v1/models` and `/metrics`; the fake `vllm bench serve` parses `--result-dir/--result-filename` and writes representative JSON. Run one selected case through `main()` and assert manifest, environment, JSONL, CSV, Markdown, commands, and raw logs exist.

- [ ] **Step 3: Run tests and confirm they fail**

```bash
pytest -q benchmarks/cache/tests/test_run_suite.py -k "example or fake"
```

Expected: FAIL because docs/examples/fake fixture are absent.

- [ ] **Step 4: Write examples**

- `example-7b.yaml`: TP=1, 32 GiB CPU tier, prompt lengths 1024/4096, concurrency 1/8/32.
- `example-70b.yaml`: TP=4, 128 GiB CPU tier, prompt lengths 4096/8192/16384, concurrency 1/8/16.
- `example-397b.yaml`: TP=8, 512 GiB CPU tier, prompt lengths 4096/16384/32768, concurrency 1/4/8.
- All examples bind loopback and use `/mnt/nvme/vllm-kv-cache` only as an illustrative filesystem root.
- Include `token_length_tolerance: 2` and `request_rate: [inf]`.

- [ ] **Step 5: Write README**

Document exact commands:

```bash
python benchmarks/cache/run_suite.py \
  --config benchmarks/cache/configs/example-7b.yaml \
  --dry-run

python benchmarks/cache/run_suite.py \
  --config /path/to/machine-model.yaml

python benchmarks/cache/run_suite.py \
  --rebuild-report results/cache/<run-id>
```

Also document prerequisites, safe-directory marker behavior, four cache modes, cold/warm/shared/mixed/restart semantics, output files, missing metrics, how to select case IDs from dry-run output, TP=1/2/4/8 guidance, `PYTHONHASHSEED=0`, and manual smoke-test checklist.

- [ ] **Step 6: Run fake integration and all cache-suite tests**

```bash
pytest -q benchmarks/cache/tests
```

Expected: PASS without GPU/model downloads.

- [ ] **Step 7: Commit Task 8**

```bash
git add benchmarks/cache/README.md benchmarks/cache/configs benchmarks/cache/tests/test_run_suite.py
git commit -m "docs: add cache benchmark examples"
```

---

### Task 9: Static Checks, Full Verification, and Review Preparation

**Files:**
- Modify only files created by Tasks 1–8 if verification finds defects.

**Interfaces:**
- Produces a verified implementation branch ready for code review.

- [ ] **Step 1: Format and lint**

```bash
ruff format benchmarks/cache
ruff check benchmarks/cache
```

Expected: PASS with no changes remaining after the final format run.

- [ ] **Step 2: Run focused test suite**

```bash
pytest -q benchmarks/cache/tests
```

Expected: PASS.

- [ ] **Step 3: Run compile check**

```bash
python -m compileall -q benchmarks/cache
```

Expected: exit code 0.

- [ ] **Step 4: Verify dry run against each example**

```bash
for config in benchmarks/cache/configs/example-*.yaml; do
  python benchmarks/cache/run_suite.py --config "$config" --dry-run
done
```

Expected: each command exits 0, starts no `vllm serve` process, and prints the generated run directory and case count.

- [ ] **Step 5: Confirm no inference-core file changed**

```bash
git diff --name-only main...HEAD | grep -Ev '^(benchmarks/cache/|docs/superpowers/)' && exit 1 || true
```

Expected: no output.

- [ ] **Step 6: Review commit history and diff**

```bash
git status --short
git log --oneline main..HEAD
git diff --stat main...HEAD
```

Expected: clean working tree; focused commits; only benchmark suite, design, and plan files.

- [ ] **Step 7: Commit any verification-only fixes**

```bash
git add benchmarks/cache
git commit -m "test: finalize cache benchmark suite"
```

Skip this commit when verification required no changes.

---

## Manual Hardware Acceptance After Merge Candidate

Run these outside CI on the target NVIDIA host:

1. Small model, TP=1: complete `no-cache`, `gpu-apc`, and `cpu-offload` selected cases.
2. Representative multi-GPU model: execute the same workload identity at TP=2 or TP=4.
3. Local NVMe: complete `tiered-fs` and `restart-persistence` with one cache directory reused across the restart pair.
4. Confirm report contains native TTFT/TPOT/throughput, raw Prometheus snapshots, CPU offload gauges/histograms when exposed, resource samples, commands, and logs.
5. Confirm a repeated exact-prefix case shows cache evidence or explicitly reports that the metric is not exposed; never infer a hit solely from lower TTFT.
6. Record GPU model, driver, NUMA topology, model revision, tokenizer revision, TP/PP, and filesystem device in `environment.json`.

Phase A is accepted only after the no-GPU tests pass and the two manual smoke-test groups above succeed.

---

## Plan Self-Review Result

- Spec coverage: every design section maps to Tasks 1–9; restart persistence is explicitly covered in Tasks 3 and 7.
- Scope: all production changes remain under `benchmarks/cache/`; no inference-core hook is introduced.
- Type consistency: `SuiteConfig`, `ExecutionCase`, `WorkloadArtifacts`, process results, and normalized result records have one defining task and named consumers.
- Placeholder scan: no implementation step contains TBD/TODO or an unspecified error-handling instruction.
- Dependency check: every imported third-party package already exists in vLLM v0.26.0 requirements.
