# Cache Workload Fairness and Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make matching cache benchmark cases use byte-identical workloads across cache modes and execution controls, while making token-length adjustment converge on non-stable tokenizers by searching a stable token stream.

**Architecture:** Keep case IDs and execution metadata unchanged. Derive RNG state from a workload-content identity that excludes cache mode, concurrency, request rate, paths, and case IDs. Inside `_sample_prompt`, keep one lazily extended suffix token pool for each prompt and vary only the prefix length of that pool during the bounded search.

**Constraints:**

- Keep `token_length_tolerance` unchanged.
- Keep the 32-attempt retry bound.
- Keep case IDs, configuration schemas, result schemas, and benchmark commands unchanged.
- Preserve determinism, uniqueness, exact-warm identity, and encoded shared-prefix guarantees.
- Limit production changes to `benchmarks/cache`; do not modify inference-core code.

## Task 1: Lock workload fairness with failing tests

**Files:**
- `benchmarks/cache/tests/test_workload.py`
- `benchmarks/cache/scenarios.py`

Add regression tests requiring:

- matching warm-exact cases across all cache modes to produce byte-identical `measure.jsonl` and `populate.jsonl`;
- matching shared-prefix cases across all cache modes to produce byte-identical measurement datasets;
- matching workload shapes with different concurrency/request-rate controls to produce identical workload bytes;
- case IDs to remain distinct even when workload content is identical.

Verify RED with:

```bash
pytest -q benchmarks/cache/tests/test_workload.py \
  -k 'matching_cache_modes or execution_controls'
```

Expected root cause: `_generator_seed` hashes `case.case_id`, so execution metadata changes RNG state.

## Task 2: Separate workload identity from case identity

**File:** `benchmarks/cache/workload.py`

Add an explicit private workload identity containing only:

- `workload_kind`
- `prompt_tokens`
- `prefix_ratio`
- `repetition`

Canonicalize that identity with JSON and combine it with `config.workload.seed` before hashing. Exclude cache mode, concurrency, request rate, case ID, and paths.

Verify the Task 1 tests turn GREEN, then run all workload tests.

## Task 3: Lock content-sensitive convergence with a failing test

**File:** `benchmarks/cache/tests/test_workload.py`

Add a tokenizer double whose encoded length depends strongly on token values, not only source length. The current implementation should fail because every proportional correction is applied to a freshly resampled suffix.

Verify RED with:

```bash
pytest -q benchmarks/cache/tests/test_workload.py \
  -k 'value_sensitive_tokenizer'
```

Expected: `WorkloadGenerationError` after the existing 32-attempt bound.

## Task 4: Search a stable suffix token stream

**File:** `benchmarks/cache/workload.py`

Within `_sample_prompt`:

1. Create one local suffix token pool.
2. Lazily extend it only when a candidate needs more source tokens.
3. Build candidates from `fixed_prefix + suffix_tokens[:candidate_suffix_length]`.
4. Reuse the same pool while encoded length remains outside tolerance.
5. Track tried suffix lengths and move to a nearby untried length if the proportional correction repeats a candidate.
6. Only resample a new content stream after length is valid but encoded-prefix or uniqueness validation rejects the candidate.
7. Keep the existing 32-attempt failure behavior and strict tolerance.

Verify the value-sensitive regression turns GREEN and all workload tests remain green.

## Task 5: Full verification and pull request

Run:

```bash
pytest -q benchmarks/cache/tests
python -m compileall -q benchmarks/cache
```

Then:

- compare the branch with `main` and confirm production changes are limited to `benchmarks/cache/workload.py`;
- remove temporary GitHub-hosted verification workflow files;
- run repository pre-commit checks;
- open/update PR `Fix cache workload fairness and convergence` with hardware RED evidence and exact verification results;
- do not merge without explicit user instruction.
