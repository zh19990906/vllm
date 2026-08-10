# Cache Workload Token-Length Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cache benchmark prompt generation converge to the requested encoded token length for tokenizers whose arbitrary token sequences expand or contract across `decode -> encode`.

**Architecture:** Keep the existing deterministic token-ID sampling and strict round-trip validation. Add a focused helper that adjusts the sampled suffix length from the observed encoded length, then use it only when a candidate misses the configured tolerance. Preserve fixed prefixes, uniqueness checks, retry bounds, case IDs, schemas, and benchmark commands.

**Tech Stack:** Python 3.10+, pytest, Pydantic-backed benchmark configuration, Hugging Face tokenizer-compatible protocol.

## Global Constraints

- Keep `token_length_tolerance` strict; do not raise the configured tolerance.
- Keep the existing 32-attempt retry bound.
- Keep case IDs and configuration schemas unchanged.
- Preserve deterministic output for identical seed and case ID.
- Preserve identical population and measurement rows for exact-warm and restart-persistence workloads.
- Preserve the encoded shared prefix across all rows in a shared-prefix workload.
- Do not modify cache-mode commands, Scheduler, attention code, KV-cache internals, result schemas, or reporting.

---

### Task 1: Add regression coverage for non-stable tokenizer round trips

**Files:**

- Modify: `benchmarks/cache/tests/test_workload.py`

**Interfaces:**

- Consumes: `generate_workload(case, config, tokenizer) -> WorkloadArtifacts`
- Produces: regression tests that require adaptive token-count convergence while exercising existing public workload behavior.

- [ ] **Step 1: Add a tokenizer test double that expands round-trip token counts**

Add this test utility next to `FakeTokenizer`:

```python
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
```

This preserves deterministic text generation while making `len(encode(decode(ids))) > len(ids)` for sufficiently long inputs.

- [ ] **Step 2: Add a failing exact-warm convergence test**

```python
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
```

- [ ] **Step 3: Add a failing shared-prefix regression test**

```python
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
```

- [ ] **Step 4: Verify RED**

Run:

```bash
pytest -q benchmarks/cache/tests/test_workload.py \
  -k 'expanding_tokenizer'
```

Expected: the convergence test fails with `WorkloadGenerationError` reporting an observed length above the requested length after 32 attempts. The shared-prefix test may fail for the same root cause.

- [ ] **Step 5: Commit the failing tests**

```bash
git add benchmarks/cache/tests/test_workload.py
git commit -m "test: cover expanding tokenizer workloads"
```

---

### Task 2: Adapt sampled suffix length from observed encoded length

**Files:**

- Modify: `benchmarks/cache/workload.py`
- Test: `benchmarks/cache/tests/test_workload.py`

**Interfaces:**

- Consumes: current sampled suffix length, requested total length, observed encoded total length, and fixed-prefix source length.
- Produces: `_next_suffix_length(...) -> int`, a non-negative next candidate length that moves toward the requested encoded total.

- [ ] **Step 1: Add the adjustment helper**

Add this helper above `_sample_prompt`:

```python
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
```

- [ ] **Step 2: Use an adaptive candidate length in `_sample_prompt`**

Replace the fixed `suffix_length` sampling loop with:

```python
    target_suffix_length = requested_length - len(fixed_prefix)
    if target_suffix_length < 0:
        raise WorkloadGenerationError(...)

    candidate_suffix_length = target_suffix_length
    last_observed = -1
    for _ in range(32):
        token_ids = list(fixed_prefix)
        token_ids.extend(
            rng.choice(allowed_tokens)
            for _ in range(candidate_suffix_length)
        )
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        last_observed = len(encoded)
        if abs(last_observed - requested_length) > tolerance:
            candidate_suffix_length = _next_suffix_length(
                current_suffix_length=candidate_suffix_length,
                requested_length=requested_length,
                observed_length=last_observed,
                fixed_prefix_length=len(fixed_prefix),
            )
            continue
```

Keep the existing encoded-prefix, uniqueness, and acceptance checks unchanged after the tolerance check.

- [ ] **Step 3: Verify GREEN for the regression tests**

Run:

```bash
pytest -q benchmarks/cache/tests/test_workload.py \
  -k 'expanding_tokenizer'
```

Expected: both tests pass.

- [ ] **Step 4: Run all workload tests**

Run:

```bash
pytest -q benchmarks/cache/tests/test_workload.py
```

Expected: all workload tests pass, including deterministic generation, uniqueness, exact-warm identity, metadata, and command construction.

- [ ] **Step 5: Commit the minimal implementation**

```bash
git add benchmarks/cache/workload.py benchmarks/cache/tests/test_workload.py
git commit -m "fix: adapt cache prompt token generation"
```

---

### Task 3: Full verification and pull request

**Files:**

- Verify: `benchmarks/cache/workload.py`
- Verify: `benchmarks/cache/tests/test_workload.py`
- Verify: `benchmarks/cache/tests/`

**Interfaces:**

- Consumes: completed adaptive generator implementation.
- Produces: verified branch and pull request ready to merge.

- [ ] **Step 1: Run the complete cache benchmark test suite**

```bash
pytest -q benchmarks/cache/tests
```

Expected: all tests pass.

- [ ] **Step 2: Compile the cache benchmark package**

```bash
python -m compileall -q benchmarks/cache
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 3: Review the diff scope**

```bash
git diff main...HEAD -- \
  benchmarks/cache/workload.py \
  benchmarks/cache/tests/test_workload.py \
  docs/superpowers/specs/2026-08-06-cache-workload-token-length-fix-design.md \
  docs/superpowers/plans/2026-08-06-cache-workload-token-length-fix.md
```

Expected: no cache execution-path, scheduler, attention, command-building, schema, or report changes.

- [ ] **Step 4: Open a pull request**

Use title:

```text
Fix cache workload token-length convergence
```

Include the Qwen2.5 failure evidence, the adaptive suffix-length approach, unchanged strict tolerance, and test results in the PR body.
