# Current Engineering State

Observed: **2026-08-11**.

This file is intentionally mutable. Verify GitHub issue, PR, branch, and commit metadata
before acting if this file is older than the current session.

## Repository state

- Repository: `zh19990906/vllm`
- Default branch: `main`
- This refresh was prepared from
  `main@ceb895f9abd916524ca7178aac3f42dc230a48af`, after the PR #27-#29 documentation
  consolidation completed.
- Do not treat that snapshot SHA as a permanently current branch head. Updating this file
  creates newer commits; live GitHub metadata remains authoritative for the actual head.
- PR #27 preserved the useful point-in-time history from superseded PR #22 without merging
  its stale mutable roadmap state.
- PR #28 refreshed this file to the Issue #15-era roadmap, and PR #29 corrected the
  snapshot wording so this mutable document does not recursively claim its own merge as a
  permanently current head.
- GitHub is the authoritative remote. The observed development workflow synchronizes
  GitHub branches/tags to a Gitee mirror used by Pod workspaces.
- Pod workspaces are for build/test/hardware validation and are not an authoritative
  GitHub write path.

## Project objective

The long-term objective is an adaptive hierarchical KV cache system that can make runtime
decisions from measured cost instead of treating every reusable KV as an unconditional
restore win.

The core decision chain is:

```text
measure restore vs recompute
  -> calibrate shadow prediction
  -> validate generalization
  -> enable active restore/recompute choice
  -> build multi-tier placement
  -> build adaptive admission/eviction
```

## Roadmap status

Parent roadmap: **Issue #9**.

### Completed work

- **Issue #10**: workload fairness and token-length convergence, completed by PR #3.
- **Issue #11**: real eviction/lower-tier restore benchmark, completed by PR #5.
- **Issue #12**: shadow restore/recompute cost model, completed by PR #7.
- **Issue #13**: systematic restore-vs-recompute crossover measurement, completed by
  PR #25.
- **Issue #14**: calibration of the shadow cost model from real measurements, completed by
  PR #26.
- **Issue #20**: repository `benchmarks/cache` pre-commit/static baseline cleanup,
  completed.

### Current P0 core work

**Issue #15 is the current primary research issue.**

Issue #15 validates whether the calibrated shadow model generalizes beyond the single
model, single-machine, concurrency=1 baseline used for Issues #13 and #14. The result must
show prediction error and decision accuracy under at least two meaningfully different
model, hardware, or load conditions, identify which parameters transfer across
environments, and identify which parameters require online calibration.

### Next P0 stage

**Issue #16 remains open and follows Issue #15.**

Issue #16 promotes shadow-only restore/recompute advice into a real runtime choice with
explicit enablement, observability, fallback, and safe behavior outside validated regions.
Do not enable active behavior merely because the Issue #14 baseline gate passed.

### Parallel work

- **Issue #17**: NUMA topology discovery and CPU KV placement foundations, open.
- **Issue #21**: permanent fix for the fake `vllm` fixture nested-newline escaping defect,
  open.

Issue #17 can proceed in parallel with Issue #15. Issue #21 is maintenance and should not
block the main restore/recompute research path unless it directly blocks a required test.

### Later work

- **Issue #18**: GPU/CPU/NVMe multi-tier KV placement, open; depends on active
  restore/recompute behavior and NUMA foundations.
- **Issue #19**: adaptive online KV admission/eviction, open; follows multi-tier placement
  and active restore/recompute decisions.

## Recently completed core evidence

### Issue #13 / PR #25: crossover measurement

- PR #25 merge commit:
  `3295d83ff76ec8792942e6ec7faf9adbb4afe39e`
- Baseline environment: one model, one GPU Pod, concurrency=1.
- CPU-primary restore showed a P50 crossover bracket at 192-216 requested prompt tokens.
- Tiered-filesystem restore showed no P50 crossover in the measured 256-4096 requested
  token range; recompute remained faster throughout that measured range.
- Restore provenance and structured calibration input were recorded for the next stage.
- The filesystem evidence is lower-tier/tiered-fs on the container's local
  overlay-backed filesystem; it is not evidence of physical NVMe performance.

See:
[`validation/2026-08-10-issue13-restore-recompute-crossover.md`](validation/2026-08-10-issue13-restore-recompute-crossover.md).

### Issue #14 / PR #26: shadow cost-model calibration

- PR #26 merge commit:
  `ddbe6650778c78ef01dec9ecbb424fa1a4bcf553`
- Primary acceptance metric: P95.
- Before calibration: 13/14 correct decisions and principal P95 macro-MAPE 34.021%.
- After calibration: 14/14 correct decisions and principal P95 macro-MAPE 0.090%.
- Calibration uses runtime actual `external_tokens`, not requested prompt tokens.
- The Phase 1 result remains shadow-only and does not modify the active restore/recompute
  execution path.
- Repository-wide GitHub pre-commit passed on the final PR head before merge.

See:
[`validation/2026-08-10-issue14-shadow-cost-model-calibration.md`](validation/2026-08-10-issue14-shadow-cost-model-calibration.md).

### PR #27-#29: retire stale roadmap state and restore durable documentation

PR #27 preserved only these point-in-time records from superseded PR #22:

- [`history/2026-08-10-pr5-finalization-and-roadmap-transition.md`](history/2026-08-10-pr5-finalization-and-roadmap-transition.md)
- [`handoffs/2026-08-10-post-pr5-roadmap.md`](handoffs/2026-08-10-post-pr5-roadmap.md)

Those files are historical snapshots. They intentionally retain their 2026-08-10 framing
and must not be read as the current roadmap. PR #22 itself was closed without merge.

PR #28 refreshed this mutable status file to the Issue #15-era roadmap. PR #29 corrected
the refresh metadata so it records a preparation snapshot rather than a self-invalidating
claim about the permanently current `main` head.

The consolidated progress record is:

[`history/2026-08-10-to-2026-08-11-issue13-14-and-roadmap-consolidation.md`](history/2026-08-10-to-2026-08-11-issue13-14-and-roadmap-consolidation.md).

## Current objective: design Issue #15 generalization validation

The next design/implementation cycle should stay narrow enough to produce interpretable
new evidence rather than immediately launching a broad 1-8 GPU sweep.

Recommended sequence:

1. Reuse the checked-in Issue #13 measurement schema and Issue #14 calibrated evaluator.
2. Choose the smallest additional conditions that satisfy Issue #15's generalization goal,
   with at least two meaningfully different model, hardware, or load conditions overall.
3. Define acceptance metrics before running hardware experiments, including prediction
   error and decision accuracy.
4. Separate transferable profile parameters from environment-specific or online-calibrated
   parameters.
5. Record failure boundaries and missing model inputs instead of hiding them with
   high-cardinality fitting.
6. Save structured results and a concise validation report under
   `docs/engineering/validation/`.
7. Keep execution shadow-only until Issue #15 evidence justifies moving to Issue #16.

## Current continuation record

Use this handoff for the next Issue #15 session:

[`handoffs/2026-08-11-issue15-generalization-handoff.md`](handoffs/2026-08-11-issue15-generalization-handoff.md).

It records the completed #13/#14 inputs, the minimum discriminative experiment shape,
metrics that must be fixed before expensive runs, and stop conditions that prevent an
unnecessary broad sweep.

## Important interpretation rules

- Requested prompt tokens and runtime actual external KV tokens are not interchangeable.
  Cost-model calibration and evaluation should use the runtime quantity used by the model.
- Do not call a local filesystem tier "NVMe" unless the storage provenance actually proves
  physical NVMe behavior.
- Do not treat a shadow decision change as proof that the active runtime path changed.
- Do not repeat the Issue #13 baseline sweep merely to reproduce archived numbers; rerun
  expensive hardware experiments only when Issue #15 introduces a new condition or a new
  hypothesis.
- A current all-files CI failure should be classified against current `main` before being
  attributed to a feature branch. Issue #20's old cache static baseline is completed.
- The fake `vllm` fixture defect is still open as Issue #21 and should be treated as a known
  maintenance item until it is permanently resolved.

## Source-of-truth order

When records disagree, prefer:

1. current GitHub issue, PR, branch, and commit metadata;
2. current repository source and checked-in structured artifacts;
3. raw benchmark/hardware artifacts;
4. validation documents;
5. history and handoff snapshots.

The point-in-time handoffs under `docs/engineering/handoffs/` are useful provenance, but
this file is the mutable repository summary that should be updated as the roadmap advances.
