# PR #3: Cache Workload Fairness and Convergence

Date completed: **2026-08-07**.

## Summary

PR #3 fixed two correctness problems in the cache benchmark workload generator that made
hardware comparisons unreliable:

1. logically matching cache modes could receive different prompt bytes because workload
   RNG identity depended on case-specific execution identity;
2. token-length correction discarded and regenerated the random suffix after each
   adjustment, so a content-sensitive tokenizer could oscillate and exhaust the bounded
   search even when a valid length was nearby.

This PR was foundational for the later eviction/restore benchmark and shadow cost-model
hardware validation because a cache comparison is meaningless if each cache mode measures
a different workload.

## Remote identity

- PR: `#3`
- Title: `Fix cache workload fairness and convergence`
- Branch: `fix/cache-workload-fairness`
- Final head: `4fa33f8267de7cbc4d95208886de8e63f028cb3a`
- Base at PR metadata observation: `main@da4895d13c081d85ac4d83a43bbfcccb6a4388fa`
- Merge commit: `abc426ae063e90c837c48dfbe75fabe919c82575`
- Changed files: 4
- GitHub-reported diff size: `+347 / -9`

## Problem 1: cache-mode fairness

### Symptom

The benchmark matrix included cases such as no-cache, GPU APC, CPU offload, and tiered
filesystem offload that were intended to compare cache behavior for the same prompt data.
However, workload RNG state was derived from a case identity that also encoded execution
choices such as cache mode.

As a result, matching benchmark cases could produce different measurement datasets.
Performance differences would then combine cache behavior and prompt-content differences.

### Design correction

PR #3 introduced a workload-content identity that contains only properties that are
supposed to change prompt content:

- workload kind;
- prompt token length;
- prefix ratio;
- repetition.

It intentionally excludes execution controls and output identity, including:

- cache mode;
- concurrency;
- request rate;
- case ID;
- result paths.

Case IDs and result schemas remained unchanged. Only prompt-content seeding was separated
from execution identity.

### TDD evidence

The PR recorded a RED state in which three fairness tests failed. Matching cache modes
produced four distinct measurement datasets, and changing execution controls changed
prompt bytes.

After separating workload identity from case identity, those tests passed and matching
cache modes produced byte-identical workload data.

## Problem 2: tokenizer convergence

### Symptom

Hardware validation with Qwen2.5-7B exposed failures such as:

```text
unable to generate prompt with requested length 1024
```

The observed encoded length could move around the target and still exhaust the 32-attempt
bound. During one smoke sequence, examples included an observed length near 998 for GPU APC
and 1039 for tiered filesystem for a requested 1024-token workload.

### Root cause

The length-adjustment loop resampled an entirely new random suffix after every correction.
For a tokenizer whose round-trip behavior depends on token content, changing the content
while also changing the requested size made the search unstable. The algorithm was not
performing a monotonic correction over one candidate stream.

### Design correction

The generator now reuses one lazily extended suffix token stream while searching for the
requested encoded length. New random content is not generated merely because the previous
encoded length was too short or too long.

Content is resampled only after length is already valid and a separate encoded-prefix or
uniqueness requirement rejects the candidate.

### TDD evidence

A value-sensitive tokenizer regression test failed before the fix with
`WorkloadGenerationError`, with the final observed length still away from 1024 after all
attempts. The same targeted test passed after stabilizing the suffix stream.

## Hardware-oriented outcome

Later benchmark work used a small real-tokenizer tolerance/preflight window around target
sizes. Observed acceptable encoded ranges were:

| Requested | Observed tolerance range |
|---:|---:|
| 256 | 254-258 |
| 512 | 510-514 |
| 1024 | 1022-1026 |
| 2048 | 2046-2050 |
| 4096 | 4094-4098 |

No-cache and tiered-filesystem workloads were subsequently verified byte-identical for
matching cases, giving the later crossover measurements a meaningful fairness baseline.

## Verification recorded by the PR

GitHub-hosted Python 3.11 verification recorded:

```text
python -m pytest -q benchmarks/cache/tests
54 passed in 3.25s
```

`python -m compileall -q benchmarks/cache` also passed.

The full-suite runner required a temporary correction to an unrelated fake-executable
fixture because that fixture generated an invalid Python script from a nested newline
escape. PR #3 intentionally did not change that unrelated test.

See:
[`../incidents/fake-vllm-newline-fixture.md`](../incidents/fake-vllm-newline-fixture.md).

## Scope discipline

The final PR changed only:

- `benchmarks/cache/workload.py`;
- `benchmarks/cache/tests/test_workload.py`;
- its design specification;
- its implementation plan.

It did not modify scheduler, attention, KV cache manager, block pool, offloading runtime,
CUDA kernels, or other inference-core code.

## Why this matters later

PR #5 and PR #7 hardware work depends on comparing restore versus recompute for equivalent
prompt content. PR #3 established that invariant. If a future benchmark again shows
unexpected cross-mode differences, verify workload byte identity before interpreting the
result as a cache-performance effect.

## Related records

- [`../incidents/cache-workload-tokenizer-fairness-and-convergence.md`](../incidents/cache-workload-tokenizer-fairness-and-convergence.md)
- [`2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md`](2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md)
- [`../validation/cache-crossover-baseline.md`](../validation/cache-crossover-baseline.md)
