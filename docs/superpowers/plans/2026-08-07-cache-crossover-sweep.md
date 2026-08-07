# Cache Restore-vs-Recompute Crossover Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add token-budgeted eviction pressure so the benchmark can compare filesystem KV restore against recompute across 256/512/1024/2048/4096 prompt lengths with approximately constant cache pressure.

**Architecture:** Extend the strict workload config with an optional `pressure_fill_tokens` budget while preserving `pressure_fill_requests`. Centralize derived filler-count calculation in the workload module, use the same helper for scenario enablement/audit metadata, and add a dedicated local crossover config that keeps GPU KV and CPU primary at 2 GiB. Existing generator-seed fairness rules and victim-first population ordering remain unchanged.

**Tech Stack:** Python 3.11, Pydantic v2, PyYAML, pytest, existing `benchmarks/cache` benchmark framework.

## Global Constraints

- Sweep prompt lengths: exactly 256, 512, 1024, 2048, 4096 tokens.
- Filler token budget: exactly 65,536 tokens.
- Victim count: `requests_per_case = 8`.
- GPU KV: 2 GiB via `--kv-cache-memory-bytes 2147483648`.
- CPU primary: 2 GiB.
- Offload chunk: 64 tokens.
- Eviction policy: LRU.
- Compare only `tiered-fs` and `no-cache` `eviction-restore` cases during first hardware sweep.
- `pressure_fill_requests` remains backward compatible.
- `pressure_fill_tokens` defaults to 0.
- At most one of `pressure_fill_requests` and `pressure_fill_tokens` may be non-zero; both zero is valid.
- Workloads for matching cache modes at a given prompt length must remain byte-identical.
- Do not silently reseed or resample after workload-generation failure.

---

### Task 1: Strict pressure-token configuration

**Files:**
- Modify: `benchmarks/cache/config.py`
- Test: `benchmarks/cache/tests/test_config.py`

**Interfaces:**
- Consumes: existing `NonNegativeInt` alias and `WorkloadConfig`.
- Produces: `WorkloadConfig.pressure_fill_tokens: int` with default `0`; model-level validation rejecting simultaneous non-zero request and token pressure.

- [ ] **Step 1: Write failing config tests**

Add tests that validate all three legal states and the conflict:

```python
def test_pressure_fill_tokens_defaults_to_zero(suite_config):
    assert suite_config.workload.pressure_fill_tokens == 0


def test_pressure_fill_tokens_is_accepted(config_payload):
    config_payload["workload"]["pressure_fill_requests"] = 0
    config_payload["workload"]["pressure_fill_tokens"] = 65536
    config = SuiteConfig.model_validate(config_payload)
    assert config.workload.pressure_fill_tokens == 65536


def test_pressure_modes_are_mutually_exclusive(config_payload):
    config_payload["workload"]["pressure_fill_requests"] = 64
    config_payload["workload"]["pressure_fill_tokens"] = 65536
    with pytest.raises(ValueError, match="at most one"):
        SuiteConfig.model_validate(config_payload)
```

- [ ] **Step 2: Run the focused config tests and verify RED**

Run:

```bash
python -m pytest -q benchmarks/cache/tests/test_config.py -k 'pressure_fill'
```

Expected: tests referencing `pressure_fill_tokens` fail because strict config rejects the unknown field and no exclusivity validator exists.

- [ ] **Step 3: Implement strict config semantics**

Add to `WorkloadConfig`:

```python
pressure_fill_requests: NonNegativeInt = 0
pressure_fill_tokens: NonNegativeInt = 0
```

Add a model validator:

```python
@model_validator(mode="after")
def validate_pressure_mode(self) -> "WorkloadConfig":
    if self.pressure_fill_requests > 0 and self.pressure_fill_tokens > 0:
        raise ValueError(
            "at most one of pressure_fill_requests and pressure_fill_tokens "
            "may be non-zero"
        )
    return self
```

- [ ] **Step 4: Run focused config tests and verify GREEN**

Run the same pytest command. Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cache/config.py benchmarks/cache/tests/test_config.py
git commit -m "Add token-budgeted cache pressure config"
```

---

### Task 2: Derive filler requests from token budget

**Files:**
- Modify: `benchmarks/cache/workload.py`
- Modify: `benchmarks/cache/tests/test_eviction_restore_workload.py`

**Interfaces:**
- Consumes: `WorkloadConfig.pressure_fill_requests`, `WorkloadConfig.pressure_fill_tokens`, `ExecutionCase.prompt_tokens`.
- Produces: `_pressure_fill_request_count(case: ExecutionCase, config: SuiteConfig) -> int` and metadata fields `pressure_fill_tokens` and `derived_pressure_fill_requests`.

- [ ] **Step 1: Write failing derivation tests**

Add tests covering the exact sweep values:

```python
@pytest.mark.parametrize(
    ("prompt_tokens", "expected"),
    [(256, 256), (512, 128), (1024, 64), (2048, 32), (4096, 16)],
)
def test_token_pressure_derives_expected_filler_count(
    suite_config, prompt_tokens, expected
):
    config = suite_config.model_copy(
        update={
            "workload": suite_config.workload.model_copy(
                update={
                    "pressure_fill_requests": 0,
                    "pressure_fill_tokens": 65536,
                    "prompt_tokens": [prompt_tokens],
                }
            )
        }
    )
    case = next(
        case for case in build_execution_cases(config, config.results.root_dir / "token-pressure")
        if case.workload_kind == "eviction-restore"
        and case.prompt_tokens == prompt_tokens
    )
    artifacts = generate_workload(case, config, FakeTokenizer())
    metadata = json.loads(artifacts.metadata_path.read_text())
    assert artifacts.num_population_prompts == config.workload.requests_per_case + expected
    assert metadata["pressure_fill_tokens"] == 65536
    assert metadata["derived_pressure_fill_requests"] == expected
```

Keep/add a backward-compatibility assertion that request-count pressure still uses the configured count exactly.

- [ ] **Step 2: Run the focused workload tests and verify RED**

Run:

```bash
python -m pytest -q benchmarks/cache/tests/test_eviction_restore_workload.py
```

Expected: token-budget cases fail because filler count is still taken only from `pressure_fill_requests` and metadata lacks the new fields.

- [ ] **Step 3: Implement one filler-count helper**

In `benchmarks/cache/workload.py`, import `math` and add:

```python
def _pressure_fill_request_count(
    case: ExecutionCase,
    config: SuiteConfig,
) -> int:
    token_budget = config.workload.pressure_fill_tokens
    if token_budget > 0:
        return math.ceil(token_budget / case.prompt_tokens)
    return config.workload.pressure_fill_requests
```

Use this helper in `_generate_eviction_restore_rows`:

```python
pressure_fill_requests = _pressure_fill_request_count(case, config)
population_count = victim_count + pressure_fill_requests
```

Record in metadata:

```python
"pressure_fill_requests": config.workload.pressure_fill_requests,
"pressure_fill_tokens": config.workload.pressure_fill_tokens,
"derived_pressure_fill_requests": _pressure_fill_request_count(case, config),
```

Do not include either pressure field in `_workload_identity`; matching cache modes must keep the same seed, and increasing pressure must extend the deterministic stream rather than changing its victims.

- [ ] **Step 4: Verify victim order and compatibility**

Run:

```bash
python -m pytest -q benchmarks/cache/tests/test_eviction_restore_workload.py
```

Expected: all tests pass, including first `requests_per_case` population rows equaling measurement rows and legacy fixed-request pressure behavior.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cache/workload.py benchmarks/cache/tests/test_eviction_restore_workload.py
git commit -m "Derive cache pressure from token budget"
```

---

### Task 3: Enable scenarios under either pressure mechanism

**Files:**
- Modify: `benchmarks/cache/scenarios.py`
- Modify: `benchmarks/cache/tests/test_scenarios.py`

**Interfaces:**
- Consumes: `WorkloadConfig.pressure_fill_requests`, `WorkloadConfig.pressure_fill_tokens`.
- Produces: `eviction-restore` cases whenever either configured pressure value is non-zero.

- [ ] **Step 1: Write failing scenario test**

Add a test that sets request pressure to zero and token pressure to 65,536, then checks `eviction-restore` exists for every cache mode and configured prompt length:

```python
def test_token_pressure_enables_eviction_restore_cases(suite_config):
    workload = suite_config.workload.model_copy(
        update={"pressure_fill_requests": 0, "pressure_fill_tokens": 65536}
    )
    config = suite_config.model_copy(update={"workload": workload})
    cases = build_execution_cases(config, config.results.root_dir / "token-pressure")
    eviction_cases = [c for c in cases if c.workload_kind == "eviction-restore"]
    assert eviction_cases
    assert {c.cache_mode for c in eviction_cases} == set(CacheMode)
```

Also retain the existing assertion that both pressure fields at zero produce no eviction-restore cases.

- [ ] **Step 2: Run focused scenario tests and verify RED**

Run:

```bash
python -m pytest -q benchmarks/cache/tests/test_scenarios.py -k 'eviction_restore or pressure'
```

Expected: token-pressure enablement test fails.

- [ ] **Step 3: Implement scenario enablement**

Replace the request-only gate with a local boolean:

```python
pressure_enabled = (
    config.workload.pressure_fill_requests > 0
    or config.workload.pressure_fill_tokens > 0
)
```

Append `eviction-restore` only when `pressure_enabled` is true.

- [ ] **Step 4: Run focused scenario tests and verify GREEN**

Run the same pytest command. Expected: selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cache/scenarios.py benchmarks/cache/tests/test_scenarios.py
git commit -m "Enable eviction restore for token pressure"
```

---

### Task 4: Prove fairness across cache modes with token pressure

**Files:**
- Modify: `benchmarks/cache/tests/test_eviction_restore_workload.py`
- Modify only if needed: `benchmarks/cache/workload.py`

**Interfaces:**
- Consumes: `_generator_seed` behavior and token-pressure filler derivation from Tasks 1-2.
- Produces: regression proof that matching `no-cache`, `gpu-apc`, `cpu-offload`, and `tiered-fs` cases have byte-identical workload artifacts for a given prompt length.

- [ ] **Step 1: Add cross-mode token-pressure test**

Configure `[256, 512, 1024, 2048, 4096]` prompt lengths with `pressure_fill_tokens=65536` and generate all matching eviction-restore cases using `FakeTokenizer`. For each prompt length assert:

```python
assert len({artifact.measure_path.read_bytes() for artifact in artifacts}) == 1
assert len({artifact.populate_path.read_bytes() for artifact in artifacts}) == 1
assert len({metadata["generator_seed"] for metadata in metadata_rows}) == 1
assert len({metadata["derived_pressure_fill_requests"] for metadata in metadata_rows}) == 1
```

- [ ] **Step 2: Run test and inspect result**

Run:

```bash
python -m pytest -q benchmarks/cache/tests/test_eviction_restore_workload.py -k 'cache_modes or identical'
```

Expected: PASS if the earlier implementation preserved seed identity. If it fails, do not change the seed to include pressure/cache mode; fix only the source of nondeterminism.

- [ ] **Step 3: Commit test evidence**

```bash
git add benchmarks/cache/tests/test_eviction_restore_workload.py benchmarks/cache/workload.py
git commit -m "Test token pressure workload fairness"
```

---

### Task 5: Add the hardware crossover configuration

**Files:**
- Create: `benchmarks/cache/configs/local-crossover.yaml`

**Interfaces:**
- Consumes: token-pressure semantics from Tasks 1-3.
- Produces: one reproducible five-length local hardware configuration.

- [ ] **Step 1: Add the configuration**

Create `benchmarks/cache/configs/local-crossover.yaml` with:

```yaml
schema_version: 1
model:
  id: /mnt/model/Qwen2.5-7B-Instruct
  served_name: qwen2.5-7b
  dtype: auto
  max_model_len: 32768
  trust_remote_code: false
parallelism:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
server:
  host: 127.0.0.1
  port: 8100
  startup_timeout_seconds: 900
  shutdown_timeout_seconds: 60
  extra_args:
    - --kv-cache-memory-bytes
    - "2147483648"
  env:
    PYTHONSAFEPATH: "1"
    VLLM_USE_FLASHINFER_SAMPLER: "0"
    TORCH_CUDA_ARCH_LIST: "12.0+PTX"
cache:
  gpu_memory_utilization: 0.9
  cpu_bytes_to_use: 2147483648
  offload_block_size: 64
  eviction_policy: lru
  filesystem:
    enabled: true
    root_dir: /tmp/vllm-kv-cache
    read_threads: 32
    write_threads: 16
workload:
  seed: 1
  tokenizer: /mnt/model/Qwen2.5-7B-Instruct
  prompt_tokens: [256, 512, 1024, 2048, 4096]
  output_tokens: 1
  concurrency: [1]
  request_rate: [inf]
  requests_per_case: 8
  pressure_fill_requests: 0
  pressure_fill_tokens: 65536
  shared_prefix_ratios: [0.0, 0.5, 0.9]
  warmup_requests: 2
  token_length_tolerance: 2
results:
  root_dir: /code/results/cache
  keep_server_logs: true
  fail_fast: false
```

- [ ] **Step 2: Dry-run and validate the generated matrix**

Run:

```bash
python benchmarks/cache/run_suite.py \
  --config benchmarks/cache/configs/local-crossover.yaml \
  --dry-run
```

Expected: each cache mode has five eviction-restore cases, one per prompt length; hardware execution will later select only the five no-cache and five tiered-fs cases.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/cache/configs/local-crossover.yaml
git commit -m "Add cache crossover sweep config"
```

---

### Task 6: Full benchmark-suite verification

**Files:**
- Verify all modified files from Tasks 1-5.

**Interfaces:**
- Produces: evidence that the new feature does not regress existing cache benchmark behavior.

- [ ] **Step 1: Run the full cache tests**

Run:

```bash
python -m pytest -q benchmarks/cache/tests
```

Expected: all tests pass. The pre-existing fake E2E fixture escaping issue may require the same runner-only patch used in prior verification; do not commit that unrelated fixture change as part of this feature.

- [ ] **Step 2: Compile the benchmark package**

Run:

```bash
python -m compileall -q benchmarks/cache
```

Expected: exit code 0.

- [ ] **Step 3: Inspect final diff against the PR base**

Run:

```bash
git diff --stat fix/cache-workload-fairness...HEAD
git diff fix/cache-workload-fairness...HEAD -- \
  benchmarks/cache/config.py \
  benchmarks/cache/scenarios.py \
  benchmarks/cache/workload.py \
  benchmarks/cache/tests \
  benchmarks/cache/configs/local-crossover.yaml
```

Confirm no temporary workflow or unrelated runtime code is included.

- [ ] **Step 4: Hardware preflight after pulling the final head**

On `/code/vllm`, run the suite dry-run and select exactly the ten crossover cases. Before hardware execution, generate workloads using the real Qwen2.5 tokenizer and verify derived filler counts are 256/128/64/32/16 and each population replays its first eight rows as measurement victims.

- [ ] **Step 5: Execute the hardware sweep**

Run the five `tiered-fs__eviction-restore` and five `no-cache__eviction-restore` cases. Preserve every run directory and collect P95 TTFT plus path-evidence metrics. Reject any tiered point without eight secondary async lookups or any no-cache point showing external KV transfer.

- [ ] **Step 6: Build crossover table**

For each prompt length calculate:

```text
delta_ms = fs_restore_p95_ms - recompute_p95_ms
```

Report:

```text
prompt_tokens | recompute_p95_ms | fs_restore_p95_ms | delta_ms | preferred_path
```

A sign change in `delta_ms` brackets the crossover region. Do not fit a higher-order model from only five points.
