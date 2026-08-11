# Issue #13/#14 Validation and Roadmap Consolidation

Observed: **2026-08-11**.

This record captures the transition from the single-environment restore/recompute
measurement work through calibrated shadow prediction, then documents the repository
cleanup that made Issue #15 the unambiguous next P0 research stage.

## Scope

This is a historical progress record. It does not replace `CURRENT_STATE.md` and should not
be read as a permanently current branch snapshot.

The documentation refresh that produced this record started from
`main@ceb895f9abd916524ca7178aac3f42dc230a48af`.

## Issue #13 / PR #25: establish the real crossover evidence

Issue #13 converted the earlier point measurements into structured, reproducible hardware
evidence suitable for cost-model calibration.

PR #25 merged as:

```text
3295d83ff76ec8792942e6ec7faf9adbb4afe39e
```

The important conclusions were:

- CPU-primary restore has a measured P50 crossover bracket between 192 and 216 requested
  prompt tokens in the baseline environment.
- Tiered-filesystem restore showed no P50 crossover in the measured 256-4096 requested
  token range; recompute remained faster at every measured point.
- Source tier is therefore a first-class decision input; a cache hit alone is not enough
  to conclude that restore is preferable.
- The initial 2 GiB CPU sweep was rejected as a CPU-restore curve because the victims were
  evicted from CPU and recomputed instead of restored.
- The deterministic requested-token 208 workload was unavailable and was not silently
  reseeded.
- Requested prompt tokens and actual restored/external KV tokens were explicitly separated
  so the next-stage cost model could calibrate on the runtime quantity it actually uses.
- The filesystem result was described as lower-tier/tiered-fs on the container's local
  overlay-backed filesystem. Physical NVMe provenance was not established and must not be
  inferred from that result.

The structured source artifact is:

- `docs/engineering/validation/2026-08-10-issue13-restore-recompute-crossover.json`

The human-readable validation report is:

- `docs/engineering/validation/2026-08-10-issue13-restore-recompute-crossover.md`

## Issue #14 / PR #26: calibrate the shadow cost model

Issue #14 used the Issue #13 dataset to calibrate the shadow model without changing the
runtime restore/recompute execution path.

PR #26 merged as:

```text
ddbe6650778c78ef01dec9ecbb424fa1a4bcf553
```

The final P95 acceptance evidence was:

| Metric | Before calibration | After calibration |
| --- | ---: | ---: |
| Decision correctness | 13/14 | 14/14 |
| Decision accuracy | 0.929 | 1.000 |
| Principal macro-MAPE | 34.021% | 0.090% |

The only before-calibration wrong decision was the CPU-primary 128-requested /
104-external-token anchor. The dominant error source was the old sparse CPU restore curve,
which had only a distant 1024-token measured point and therefore extrapolated poorly in the
short-token region.

The calibrated profile uses actual external KV tokens as the cost-curve coordinate and
keeps CPU-primary and filesystem restore curves separate.

The Phase 1 result stayed intentionally shadow-only:

- no active scheduler restore/recompute choice was enabled;
- no production runtime cost formula was changed;
- no runtime enforcement path was added;
- the existing secondary-promotion EWMA behavior was regression-tested rather than
  redesigned.

The checked-in artifacts are:

- `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json`
- `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.md`

## CI and delivery lessons from PR #26

The Pod validation environment did not have `pytest`, so local completion evidence used
pure-Python functional assertions, evaluator gates, Ruff, formatting, compile checks,
structured JSON validation, and changed-file invariants. GitHub Actions remained the
final source of truth for repository-wide pre-commit.

The first PR #26 CI run exposed static issues rather than a calibration failure:

- `MAPE` needed to be allowlisted for the repository typos checker;
- three new Python files needed SPDX headers;
- one test fixture needed an explicit `dict[str, Any]` annotation for mypy;
- markdownlint formatting was required in the new docs and one older Issue #13 validation
  file because pre-commit runs against all files.

Those failures were fixed without changing the calibration result or the runtime execution
path. The final authoritative pre-commit run passed before merge.

This reinforced two repository practices:

1. distinguish feature failures from repository-wide static baseline failures;
2. keep hardware/model evidence separate from CI formatting and delivery mechanics.

## PR #22 retirement and durable documentation cleanup

PR #22 had been left open on an old roadmap baseline. By the time Issue #14 was complete,
its mutable roadmap edits were stale and the branch had significantly diverged from
`main`.

Rather than merging obsolete current-state text, the useful point-in-time records were
preserved through PR #27:

- `docs/engineering/history/2026-08-10-pr5-finalization-and-roadmap-transition.md`
- `docs/engineering/handoffs/2026-08-10-post-pr5-roadmap.md`

PR #22 was then closed without merge.

PR #28 refreshed `docs/engineering/CURRENT_STATE.md` to the actual Issue #15-era roadmap.
PR #29 immediately corrected one self-referential documentation mistake: a mutable status
file should record the snapshot it was prepared from rather than claim that snapshot is a
permanently current `main` head.

After that cleanup, the durable interpretation is:

```text
#13 measurement complete
  -> #14 calibration complete
  -> #15 generalization is current P0
  -> #16 active decision follows only after #15 evidence
```

## Current research boundary handed to Issue #15

Issue #14 proves that a calibrated shadow profile can fit the one-model, one-machine,
concurrency=1 baseline very accurately. It does **not** prove that the same profile
transfers across models, load, hardware, topology, or storage paths.

Issue #15 therefore owns the next scientific question: which parts of the cost model are
portable, which must be environment-specific, and which require online calibration.

The next stage should not immediately launch a broad 1-8 GPU sweep. It should use the
smallest additional matrix that can distinguish model, load, and environment effects while
retaining interpretable provenance.

## Related records

- [`../CURRENT_STATE.md`](../CURRENT_STATE.md)
- [`../validation/2026-08-10-issue13-restore-recompute-crossover.md`](../validation/2026-08-10-issue13-restore-recompute-crossover.md)
- [`../validation/2026-08-10-issue14-shadow-cost-model-calibration.md`](../validation/2026-08-10-issue14-shadow-cost-model-calibration.md)
- [`2026-08-10-pr5-finalization-and-roadmap-transition.md`](2026-08-10-pr5-finalization-and-roadmap-transition.md)
- [`../handoffs/2026-08-10-post-pr5-roadmap.md`](../handoffs/2026-08-10-post-pr5-roadmap.md)
- [`../handoffs/2026-08-11-issue15-generalization-handoff.md`](../handoffs/2026-08-11-issue15-generalization-handoff.md)
