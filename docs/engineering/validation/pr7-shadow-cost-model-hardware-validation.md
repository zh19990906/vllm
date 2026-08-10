# PR #7 Shadow Cost Model Hardware Validation

Validation completed: **2026-08-10**.

Final PR head:

```text
96de0c823721c374527dbb0b3a49fdc7eccba341
```

Merged as:

```text
37f65141108e112a317fe4a5d8215a4c21c3c00e
```

## Validation question

PR #7 was not intended to improve TTFT directly. The hardware acceptance question was:

> Can the runtime correctly predict restore versus recompute cost using real source
> provenance and runtime calibration while leaving the actual KV restore execution path
> unchanged?

The safety requirement was stronger than a performance comparison. Even when shadow
prediction preferred recompute, the request still had to execute the pre-existing restore
path.

## Runtime provenance requirement

Hardware results were accepted only after proving that the benchmark's server subprocess
used the PR #7 Python modules while retaining native extensions from the installed wheel.

Validation used exact source-over-wheel for six runtime modules:

```text
vllm/v1/kv_offload/cost_model.py
vllm/v1/kv_offload/base.py
vllm/v1/kv_offload/tiering/spec.py
vllm/v1/kv_offload/tiering/manager.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
```

Native components such as FlashAttention and `_C_stable_libtorch` remained from the
installed wheel.

See:
[`../incidents/native-wheel-exact-overlay.md`](../incidents/native-wheel-exact-overlay.md).

## Seed profile

The validation-only profile came from earlier cache crossover measurements:

| Tokens | Recompute | FS restore | FS promotion |
|---:|---:|---:|---:|
| 256 | 26.414 ms | 31.119 ms | 13.916 ms |
| 512 | 44.961 ms | 56.979 ms | 35.230 ms |
| 1024 | 81.705 ms | 108.132 ms | 81.458 ms |
| 2048 | 152.461 ms | 244.266 ms | 171.505 ms |
| 4096 | 308.424 ms | 651.127 ms | 498.874 ms |

CPU-primary restore seed:

```text
1024 tokens -> 24.490 ms
```

The profile is hardware-validation data, not a production default.

## CPU-primary 1024 anchor

### Expected

```text
source=cpu_primary
preferred=restore
confidence=high
actual_path=restore
```

### Observed

Across 8 high-confidence measured decisions:

- source: `cpu_primary`;
- preferred: `restore`;
- predicted restore average: about 24.49 ms;
- predicted recompute average: about 81.705 ms;
- external KV hit tokens: 8192;
- CPU-to-GPU transfer bytes: 469,762,048;
- actual path: restore.

### Result

**PASS.**

This proved that the model could identify the expected cheap CPU-primary restore without
changing the execution path.

## Filesystem 1024 anchor

### Expected

```text
source=secondary:filesystem
preferred=recompute
confidence=high
actual_path=restore
```

### Observed

Across 8 measured high-confidence decisions:

- source: `secondary:filesystem`;
- preferred: `recompute`;
- promotion observations: 8;
- runtime scale after online observations: approximately 0.8995;
- external KV hit tokens: 8192;
- CPU-to-GPU transfer bytes: 469,762,048;
- actual path: restore.

One recorded filesystem 1024 benchmark report had:

```text
case: tiered-fs__eviction-restore__p1024__r0.000__c1__qinf__b9b23a4b
status: completed
P95 TTFT: 99.529 ms
request throughput: 10.518 req/s
result directory: /code/results/cache/20260810T011713Z-cae3a925
```

### Result

**PASS.**

This was the core shadow-invariance proof. The cost model predicted recompute while the
actual request still restored cached KV.

## Filesystem five-point sweep

Tokens covered:

```text
256 / 512 / 1024 / 2048 / 4096
```

### 512, 1024, 2048, 4096

Observed behavior was stable:

```text
source=secondary:filesystem
preferred=recompute
confidence=high
actual_path=restore
```

These points matched the seed profile and preserved execution invariance.

### 256 adaptive crossover

The p256 point initially appeared inconsistent because a single sweep could contain both
restore and recompute shadow preferences.

This was investigated independently and reproduced twice. The decisions followed the
online runtime scale rather than changing arbitrarily.

The approximate crossover threshold was:

```text
runtime_scale ~= 0.8488
```

When EWMA calibration moved the scale across that threshold, the shadow preference moved
with it. Every observed request still executed restore.

The static seed profile itself prefers recompute at p256. The mixed runtime result is a
consequence of online calibration, not evidence that the seed table or safety boundary is
wrong.

### Result

**PASS with expected adaptive p256 behavior.**

Do not rewrite this validation as "all five points always predict recompute." That was the
pre-run expectation, but the observed p256 result is more precise and should be preserved.

## Execution-invariance evidence

The hardware validation intentionally checked data-path signals, not only log text.
Evidence included:

- external cache-hit token counts;
- CPU-to-GPU transfer bytes;
- source provenance;
- promotion observation counters;
- runtime-scale metric updates;
- actual restore path remaining active when shadow preferred recompute.

The matching 1024 CPU-primary and filesystem anchors both recorded 8192 external hit tokens
and 469,762,048 CPU-to-GPU bytes across 8 measured requests, while the shadow preference
differed by source tier. This is strong evidence that the cost instrumentation observed the
existing path rather than replacing it.

## Software validation accompanying hardware work

Final PR-scoped validation recorded:

- 59 focused tests passing;
- runtime-module `compileall` passing;
- targeted Ruff check passing;
- targeted Ruff format check passing;
- `git diff --check` passing;
- clean validation worktree;
- GitHub mypy passing on Python 3.10, 3.11, 3.12, and 3.13.

A separate exact source-over-wheel scheduler integration run recorded:

```text
93 passed in 6.27s
```

Repository-wide pre-commit was not green because of unrelated existing
`benchmarks/cache/**` Ruff/format, markdownlint, and SPDX issues. PR #7's own 14-file
runtime diff was kept free of those benchmark cleanup changes.

## Archived evidence

Historical local archive retained after completion:

```text
/code/cleanup-backup/pr7-shadow-hardware-evidence.tar.gz
```

The larger raw benchmark evidence remained under `/code/results` rather than being checked
into the repository.

These paths are machine-local historical observations and must be verified before future
use.

## Acceptance summary

| Gate | Result |
|---|---|
| CPU-primary 1024 predicts restore | PASS |
| CPU-primary actual path remains restore | PASS |
| Filesystem 1024 predicts recompute | PASS |
| Filesystem actual path remains restore | PASS |
| Secondary promotion observations update EWMA | PASS |
| FS 512/1024/2048/4096 prefer recompute | PASS |
| p256 adaptive crossover explained and reproduced | PASS |
| Actual path preserved throughout p256 investigation | PASS |
| Exact feature runtime proven in server subprocess | PASS |
| Focused PR checks | PASS |
| Python 3.10-3.13 mypy | PASS |
| Repository-wide cache pre-commit baseline | Known unrelated red baseline |

## What this validation does not authorize

- It does not authorize `mode=enforce`.
- It does not prove that recompute is always cheaper than filesystem restore on other
  systems.
- It does not establish production default cost curves.
- It does not justify online learning of recompute or CPU-primary cost in this phase.
- It does not justify re-running the expensive sweep without a changed hypothesis.

## Related records

- [`cache-crossover-baseline.md`](cache-crossover-baseline.md)
- [`../history/2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md`](../history/2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md)
- [`../incidents/native-wheel-exact-overlay.md`](../incidents/native-wheel-exact-overlay.md)
- [`../incidents/cache-benchmark-ci-baseline.md`](../incidents/cache-benchmark-ci-baseline.md)
