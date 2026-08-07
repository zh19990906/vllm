# Cache Workload Fairness and Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make matching cache benchmark cases use byte-identical workloads across cache modes and execution controls, while making token-length adjustment converge on non-stable tokenizers by searching a stable token stream.

**Architecture:** Keep case IDs and execution metadata unchanged, but derive RNG state from a separate workload-content identity that excludes cache mode, concurrency, request rate, paths, and case IDs. Inside `_sample_prompt`, keep one lazily extended suffix token pool for each prompt and vary only the prefix length of that pool during the existing bounded search.

**Tech Stack:** Python 3.10+, pytest, Pydantic-backed benchmark configuration, Hugging Face tokenizer-compatible protocol.

## Global Constraints

- Keep `token_length_tolerance` unchanged; do not widen tolerance.
- Keep the existing 32-attempt retry bound.
- Keep case IDs, configuration schemas, result schemas, and benchmark commands unchanged.
- Preserve deterministic output, uniqueness checks, exact-warm population/measurement identity, and encoded shared-prefix guarantees.
- Work only in `benchmarks/cache` plus design/plan/test documentation; do not modify vLLM inference-core code.

---

### Task 1: Lock cross-mode workload fairness with failing tests

**Files:**
- Modify: `benchmarks/cache/tests/test_workload.py`
- Read: `benchmarks/cache/scenarios.py`
- Test: `benchmarks/cache/tests/test_workload.py`

**Interfaces:**
- Consumes: `build_execution_cases(config, run_dir) -> list[ExecutionCase]`, `generate_workload(case, config, tokenizer) -> WorkloadArtifacts`.
- Produces: regression tests requiring identical JSONL content and generator seed for matching workload shapes across cache modes and execution controls.

- [ ] **Step 1: Add a helper that selects matching execution cases**

Add imports for `CacheMode` and `build_execution_cases`, then add:

```python
def _matching_cases(suite_config, tmp_path: Path, workload_kind: str, prefix_ratio=0.0):
    cases = build_execution_cases(suite_config, tmp_path / "matching-run")
    return [
        case
        for case in cases
        if case.workload_kind == workload_kind
        and case.prefix_ratio == prefix_ratio
        and case.prompt_tokens == suite_config.workload.prompt_tokens[0]
        and case.concurrency == suite_config.workload.concurrency[0]
        and case.request_rate == suite_config.workload.request_rate[0]
    ]
```

- [ ] **Step 2: Add a cross-cache-mode warm-exact fairness test**

```python
def test_matching_cache_modes_use_identical_warm_exact_workloads(
    suite_config, tmp_path: Path
) -> None:
    cases = _matching_cases(suite_config, tmp_path, "warm-exact-prefix")
    artifacts = [
        generate_workload(case, suite_config, FakeTokenizer()) for case in cases
    ]
    measure = [item.measure_path.read_bytes() for item in artifacts]
    populate = [item.populate_path.read_bytes() for item in artifacts if item.populate_path]
    seeds = [
        json.loads(item.metadata_path.read_text(encoding="utf-8"))["generator_seed"]
        for item in artifacts
    ]
    assert len({case.case_id for case in cases}) == len(cases)
    assert len(set(measure)) == 1
    assert len(set(populate)) == 1
    assert len(set(seeds)) == 1
```

- [ ] **Step 3: Add shared-prefix cross-mode fairness coverage**

```python
def test_matching_cache_modes_use_identical_shared_prefix_workloads(
    suite_config, tmp_path: Path
) -> None:
    cases = _matching_cases(suite_config, tmp_path, "shared-prefix", 0.5)
    measure = [
        generate_workload(case, suite_config, FakeTokenizer()).measure_path.read_bytes()
        for case in cases
    ]
    assert len(set(measure)) == 1
```

- [ ] **Step 4: Add execution-control independence coverage**

Select two no-cache warm-exact cases with the same workload shape but different `concurrency`/`request_rate`, generate them with `FakeTokenizer`, and assert byte-identical `measure.jsonl` and `populate.jsonl`.

- [ ] **Step 5: Verify RED**

Run:

```bash
pytest -q benchmarks/cache/tests/test_workload.py \
  -k 'matching_cache_modes or execution_controls'
```

Expected: fairness tests fail because `_generator_seed` currently hashes `case.case_id`, which differs across cache mode/concurrency/request-rate.

- [ ] **Step 6: Commit failing fairness tests**

```bash
git add benchmarks/cache/tests/test_workload.py
git commit -m "test: require comparable cache workloads"
```

---

### Task 2: Make workload RNG identity independent of execution mode

**Files:**
- Modify: `benchmarks/cache/workload.py`
- Test: `benchmarks/cache/tests/test_workload.py`

**Interfaces:**
- Produces: `_workload_identity(case: ExecutionCase) -> dict[str, object]` and a mode-independent `_generator_seed(config, case) -> int`.

- [ ] **Step 1: Add explicit workload-content identity**

```python
def _workload_identity(case: ExecutionCase) -> dict[str, object]:
    return {
        "workload_kind": case.workload_kind,
        "prompt_tokens": case.prompt_tokens,
        "prefix_ratio": case.prefix_ratio,
        "repetition": case.repetition,
    }
```

- [ ] **Step 2: Derive the generator seed from the root seed plus canonical workload identity**

```python
def _generator_seed(config: SuiteConfig, case: ExecutionCase) -> int:
    identity = json.dumps(
        _workload_identity(case), sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(
        f"{config.workload.seed}:{identity}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, byteorder="big")
```

- [ ] **Step 3: Verify GREEN for fairness tests**

Run the Task 1 test selection. Expected: PASS.

- [ ] **Step 4: Run all workload tests**

```bash
pytest -q benchmarks/cache/tests/test_workload.py
```

Expected: all existing deterministic, prefix, uniqueness, metadata, and command tests remain green.

- [ ] **Step 5: Commit the seed fix**

```bash
git add benchmarks/cache/workload.py benchmarks/cache/tests/test_workload.py
git commit -m "fix: share benchmark workloads across cache modes"
```

---

### Task 3: Lock stable-token-stream convergence with a failing test

**Files:**
- Modify: `benchmarks/cache/tests/test_workload.py`
- Test: `benchmarks/cache/tests/test_workload.py`

**Interfaces:**
- Consumes: `generate_workload` public behavior.
- Produces: regression coverage that deterministically fails when each adjustment attempt resamples the entire suffix.

- [ ] **Step 1: Add a value-sensitive tokenizer double**

```python
class ValueExpandingTokenizer(FakeTokenizer):
    vocab_size = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        source = [int(part) for part in text.split()] if text else []
        encoded: list[int] = []
        for token_id in source:
            encoded.append(token_id)
            if token_id == 1:
                encoded.extend((0, 0))
        return encoded
```

This makes encoded length depend strongly on the sampled token values, not only source length.

- [ ] **Step 2: Add the convergence regression**

```python
def test_value_sensitive_tokenizer_converges_with_bounded_search(
    suite_config, warm_exact_case
) -> None:
    artifacts = generate_workload(
        warm_exact_case, suite_config, ValueExpandingTokenizer()
    )
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    lengths = metadata["files"]["measure"]["observed_token_lengths"]
    tolerance = suite_config.workload.token_length_tolerance
    assert all(
        abs(length - warm_exact_case.prompt_tokens) <= tolerance
        for length in lengths
    )
```

- [ ] **Step 3: Verify RED**

Run:

```bash
pytest -q benchmarks/cache/tests/test_workload.py \
  -k 'value_sensitive_tokenizer'
```

Expected: `WorkloadGenerationError` after 32 attempts because the current implementation applies each proportional correction to a freshly resampled suffix.

- [ ] **Step 4: Commit the failing convergence test**

```bash
git add benchmarks/cache/tests/test_workload.py
git commit -m "test: cover value-sensitive tokenizer convergence"
```

---

### Task 4: Search a stable suffix token stream during length adjustment

**Files:**
- Modify: `benchmarks/cache/workload.py`
- Test: `benchmarks/cache/tests/test_workload.py`

**Interfaces:**
- Keeps `_sample_prompt(...) -> tuple[str, list[int]]` unchanged externally.
- Adds only private helpers if needed for untried candidate selection.

- [ ] **Step 1: Keep one lazily extended suffix token pool per prompt**

At the start of `_sample_prompt`, create `suffix_tokens: list[int] = []` and `tried_suffix_lengths: set[int] = set()`.

Before each candidate build, extend the pool only as needed:

```python
while len(suffix_tokens) < candidate_suffix_length:
    suffix_tokens.append(rng.choice(allowed_tokens))

token_ids = list(fixed_prefix)
token_ids.extend(suffix_tokens[:candidate_suffix_length])
```

- [ ] **Step 2: Avoid repeated candidate lengths**

After `_next_suffix_length(...)`, use the observed error direction to select the nearest non-negative untried candidate:

```python
direction = 1 if last_observed < requested_length else -1
while candidate_suffix_length in tried_suffix_lengths:
    candidate_suffix_length += direction
    if candidate_suffix_length < 0:
        break
```

Record the current candidate in `tried_suffix_lengths` after its encoded length is observed. If a downward search would go below zero, leave the loop to fail through the existing bounded error path rather than changing tolerance.

- [ ] **Step 3: Verify GREEN for the value-sensitive tokenizer test**

Run the Task 3 test selection. Expected: PASS within the existing 32 attempts and configured tolerance.

- [ ] **Step 4: Re-run all workload tests**

```bash
pytest -q benchmarks/cache/tests/test_workload.py
```

Expected: all workload tests pass.

- [ ] **Step 5: Commit the convergence fix**

```bash
git add benchmarks/cache/workload.py benchmarks/cache/tests/test_workload.py
git commit -m "fix: stabilize cache prompt length search"
```

---

### Task 5: Full verification and pull request

**Files:**
- Verify: `benchmarks/cache/workload.py`
- Verify: `benchmarks/cache/tests/`
- Verify: `docs/superpowers/specs/2026-08-07-cache-workload-fairness-convergence-design.md`
- Verify: `docs/superpowers/plans/2026-08-07-cache-workload-fairness-convergence.md`

**Interfaces:**
- Produces: a verified branch and PR; no merge is performed without explicit user instruction.

- [ ] **Step 1: Run the complete cache benchmark test suite**

```bash
pytest -q benchmarks/cache/tests
```

Expected: all tests pass.

- [ ] **Step 2: Compile the benchmark package**

```bash
python -m compileall -q benchmarks/cache
```

Expected: exit code 0.

- [ ] **Step 3: Verify no inference-core files changed**

Compare the branch against `main`; expected changed production code is limited to `benchmarks/cache/workload.py`, with tests and docs alongside it.

- [ ] **Step 4: Remove any temporary GitHub-hosted verification workflow used for RED/GREEN testing**

The PR must not contain temporary runner-only workflow files.

- [ ] **Step 5: Open a pull request**

Title: `Fix cache workload fairness and convergence`

The PR body must state the hardware-observed failures (cross-mode prompt mismatch and 32-attempt Qwen2.5 convergence failures), the invariants preserved, and exact verification results.
