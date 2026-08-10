# Incident: Cache Workload Fairness and Tokenizer Convergence

Status: **resolved by PR #3**.

First confirmed during hardware benchmark development on 2026-08-07.

## Symptom

Two apparently different benchmark problems were observed:

1. matching cache-mode cases produced different prompt bytes even when they were supposed
   to compare the same workload;
2. prompt generation could fail with errors such as:

```text
unable to generate prompt with requested length 1024
```

The length failure occurred across multiple cache modes, including GPU APC and tiered
filesystem cases.

## Impact

The fairness issue invalidated direct cache-mode performance comparisons because measured
latency could be influenced by different prompt content.

The convergence issue made real-tokenizer benchmarks flaky and could prevent long prompt
cases from reaching cache validation at all.

Together, they could be misdiagnosed as cache/offload behavior even though the failure was
in workload construction.

## Root cause: workload identity

Prompt RNG state was derived from a case identity that included execution-specific fields.
Cache mode, concurrency, request rate, or output identity could therefore indirectly change
prompt content.

For a fair cache comparison, prompt-content identity must be independent from execution
controls.

## Root cause: suffix resampling

The token-length correction loop regenerated the random suffix after each adjustment.
With a content-sensitive tokenizer, changing both the length target and token content on
every iteration made the search unstable.

A nearby valid token length could exist while repeated full resampling still exhausted the
bounded attempt count.

## Resolution

PR #3 introduced a content identity based on only:

- workload kind;
- prompt length;
- prefix ratio;
- repetition.

It excludes cache mode, concurrency, request rate, case ID, and result path.

The generator also reuses one lazily extended suffix stream while correcting encoded
length. Content is resampled only after the encoded length is valid but another semantic
constraint, such as uniqueness or encoded-prefix validation, rejects the candidate.

## Evidence

Before the fairness fix, targeted tests observed four distinct measurement datasets across
matching cache modes, and execution-control changes altered prompt bytes.

After the fix, matching cache modes produced byte-identical workload data.

Before the convergence fix, a value-sensitive tokenizer regression test exhausted the
attempt bound. After stabilizing the suffix stream, the same test passed.

Real-tokenizer preflight later accepted these observed windows:

| Requested | Observed range |
| ---: | ---: |
| 256 | 254-258 |
| 512 | 510-514 |
| 1024 | 1022-1026 |
| 2048 | 2046-2050 |
| 4096 | 4094-4098 |

## False conclusions to avoid

- Do not attribute a prompt-generation exception to KV restore logic before inspecting the
  workload generator.
- Do not interpret cross-mode latency differences until workload byte identity is verified.
- Do not fix convergence by simply increasing the attempt count while continuing to
  regenerate unrelated random content every iteration.
- Do not reintroduce case ID or cache mode into workload-content seeding for convenience.

## Regression protection

PR #3 added tests for:

- cross-cache-mode workload identity;
- independence from execution controls;
- value-sensitive tokenizer convergence.

The benchmark history should preserve byte-identical matching workloads as an explicit
fairness invariant.

## Related work

- PR #3: `Fix cache workload fairness and convergence`
- PR #3 merge commit: `abc426ae063e90c837c48dfbe75fabe919c82575`
- [`../history/2026-08-07-pr3-cache-workload-fairness-convergence.md`](../history/2026-08-07-pr3-cache-workload-fairness-convergence.md)
- [`../validation/cache-crossover-baseline.md`](../validation/cache-crossover-baseline.md)
