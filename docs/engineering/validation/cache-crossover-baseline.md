# Cache Restore vs Recompute Crossover Baseline

Validation period: 2026-08-07 through 2026-08-10.

## Question

When a prefix already has KV available outside GPU memory, is restoring that KV always
cheaper than recomputing the prefix?

The answer on the validation system was **no**. Cost depends strongly on the source tier.

## Environment

Historical hardware evidence records:

- model: Qwen2.5-7B-Instruct;
- vLLM installed wheel: 0.26.0 during the recorded hardware run;
- Python: 3.11.11;
- GPU: NVIDIA RTX PRO 5000 72GB Blackwell;
- observed GPU memory: 73415 MiB;
- driver: 580.126.09;
- two GPUs were visible, with the benchmark cases using the configured single-device
  execution path unless otherwise noted by a scenario.

These values describe the validation environment. They are not general performance
claims for all hardware, drivers, models, block sizes, or storage devices.

## Fairness prerequisites

The comparison relies on the workload correctness fixes from PR #3:

- matching cache modes use the same workload-content identity;
- execution controls do not perturb prompt bytes;
- tokenizer length correction uses a stable suffix stream;
- matching no-cache and tiered-filesystem workloads were checked byte-identical.

Without these invariants, latency differences cannot be attributed cleanly to cache
behavior.

## Measured recompute baseline

P95 TTFT for no-cache/recompute cases:

| Prompt tokens | Recompute P95 TTFT |
|---:|---:|
| 256 | 26.414 ms |
| 512 | 44.961 ms |
| 1024 | 81.705 ms |
| 2048 | 152.461 ms |
| 4096 | 308.424 ms |

## Measured filesystem restore baseline

P95 TTFT for pressure cases restoring from the filesystem secondary tier:

| Prompt tokens | Filesystem restore P95 TTFT |
|---:|---:|
| 256 | 31.119 ms |
| 512 | 56.979 ms |
| 1024 | 108.132 ms |
| 2048 | 244.266 ms |
| 4096 | 651.127 ms |

Across the measured 256-4096-token range, filesystem restore was slower than recompute at
every point.

The ratio worsened at longer prompts; at 4096 tokens, 651.127 ms is about 2.11 times the
308.424 ms recompute P95.

## Filesystem promotion measurements

The validation profile also recorded secondary promotion timing seeds:

| Prompt tokens | Filesystem promotion time |
|---:|---:|
| 256 | 13.916 ms |
| 512 | 35.230 ms |
| 1024 | 81.458 ms |
| 2048 | 171.505 ms |
| 4096 | 498.874 ms |

These measurements were later used only as validation profile data for PR #7's shadow
model. They are not production defaults.

## CPU-primary contrast

A 1024-token CPU-primary restore measured approximately:

```text
24.490 ms
```

That is much lower than the 1024-token recompute baseline of 81.705 ms.

The important design conclusion is therefore not "restore is slow". It is:

> Restore and recompute cost must be compared using the actual source tier and current
> runtime conditions.

## Evidence that filesystem cases were real external restores

The 4096-token filesystem case was investigated to exclude a false warm-GPU or CPU-primary
hit. Historical evidence recorded:

- 8 secondary asynchronous lookups;
- average secondary async lookup time of about 498.87 ms;
- 32704 external cache-hit tokens;
- 1,875,378,176 total CPU-to-GPU transfer bytes across 8 measured requests.

This supported the conclusion that the measured filesystem result represented real lower-
tier restore behavior.

A later 1024-token shadow validation run also completed successfully with cache metric
evidence available and a measured P95 TTFT of 99.529 ms for that specific calibrated run.
That later number should not replace the original 108.132 ms seed; it was a separate run
with shadow runtime instrumentation/calibration active.

## Design consequence

These results motivated PR #7's per-tier shadow cost model.

The seed profile used for that validation was:

```text
recompute_ms:
256   26.414
512   44.961
1024  81.705
2048  152.461
4096  308.424

cpu_primary restore_ms:
1024  24.490

filesystem restore_ms:
256   31.119
512   56.979
1024  108.132
2048  244.266
4096  651.127

filesystem promotion_ms:
256   13.916
512   35.230
1024  81.458
2048  171.505
4096  498.874
```

The profile must remain validation input. Do not hardcode these machine/model-specific
numbers as production behavior.

## Limitations

- The sweep covers only the recorded model, hardware, driver, cache configuration, and
  prompt range.
- P95 TTFT includes more than raw storage I/O; it is an end-to-end request-level measure.
- Filesystem cost can change with storage device, filesystem state, thread counts, memory
  pressure, and promotion path.
- CPU-primary evidence is a 1024-token anchor rather than a full CPU-primary curve.
- The result does not establish that recompute should be enforced. PR #7 intentionally
  validated shadow prediction while preserving actual restore behavior.

## Reuse guidance

Before repeating the full sweep, identify what new hypothesis the run will answer. If the
model, storage hardware, cache layout, vLLM implementation, or decision algorithm has not
changed, the historical sweep is usually sufficient as a baseline.

If rerunning, preserve:

- exact code head;
- model and tokenizer identity;
- cache capacities and block size;
- generated workload identity/hash;
- native/runtime provenance;
- external hit tokens;
- transfer bytes;
- per-tier lookup/promotion evidence;
- raw result directory.

## Related records

- [`../history/2026-08-07-pr3-cache-workload-fairness-convergence.md`](../history/2026-08-07-pr3-cache-workload-fairness-convergence.md)
- [`../history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md`](../history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md)
- [`pr7-shadow-cost-model-hardware-validation.md`](pr7-shadow-cost-model-hardware-validation.md)
