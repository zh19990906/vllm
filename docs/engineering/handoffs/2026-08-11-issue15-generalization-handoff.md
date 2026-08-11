# Handoff: Issue #15 Generalization Validation

Observed: **2026-08-11**.

This is a point-in-time continuation record for the next P0 research stage. Re-verify the
live GitHub issue, current `main`, available models, and hardware inventory before launching
expensive validation work.

## Current objective

Issue #15 asks whether the Issue #14 calibrated shadow cost model generalizes beyond the
single-model, single-machine, concurrency=1 baseline.

The issue completion criteria require:

- at least two meaningfully different model, hardware, or load conditions;
- prediction error and decision-accuracy comparisons;
- an explicit distinction between parameters that transfer across environments and
  parameters that require online calibration;
- identified model failure boundaries and any missing feature/input needed to explain
  them;
- checked-in validation results under `docs/engineering/validation/`.

Issue #15 remains shadow-only research. Issue #16 owns active restore/recompute execution.

## Inputs already complete

### Issue #13 measurement dataset

Use these as the baseline evidence and schema reference:

- `docs/engineering/validation/2026-08-10-issue13-restore-recompute-crossover.json`
- `docs/engineering/validation/2026-08-10-issue13-restore-recompute-crossover.md`

Important baseline facts:

- one model / one known GPU Pod / concurrency=1;
- CPU-primary P50 crossover bracketed at 192-216 requested prompt tokens;
- no tiered-filesystem P50 crossover was observed across 256-4096 requested tokens;
- source tier and actual transfer/restore evidence are required for interpretation;
- runtime external KV tokens are not interchangeable with requested prompt tokens;
- the local filesystem evidence does not prove physical NVMe behavior.

### Issue #14 calibrated evaluator and profile

Use the checked-in calibration implementation and validation artifacts rather than
re-deriving the baseline manually:

- `benchmarks/cache/cost_model_calibration.py`
- `benchmarks/cache/evaluate_cost_model.py`
- `benchmarks/cache/profiles/issue12-shadow-cost-baseline.json`
- `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json`
- `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.md`

The baseline P95 gate improved from 13/14 to 14/14 decision correctness and from 34.021% to
0.090% principal macro-MAPE.

Do not interpret that fit quality as proof of generalization.

## Recommended experiment design

Keep the first Issue #15 matrix intentionally small and discriminative.

### Control

Treat the Issue #13/#14 environment as the control. Do not rerun the full baseline sweep
unless a new implementation change or provenance question requires it.

### New condition A: load/concurrency shift

Prefer the same model and hardware with a materially higher concurrency or request-load
condition. The purpose is to isolate contention effects while holding model and machine
constant.

Choose the lowest load that produces a meaningful change in cache, CPU, PCIe, or
secondary-tier contention rather than arbitrarily selecting the largest available
concurrency.

### New condition B: model or hardware shift

Add one condition that changes a different axis from condition A:

- a second model on the same machine, **or**
- the same model on a meaningfully different machine/hardware path.

Pick the option that can be proven from the available environment inventory. Do not claim a
hardware/storage distinction that the environment cannot actually establish.

### Tier coverage

For every condition where the tier exists, preserve separate analysis for:

- CPU-primary restore;
- secondary filesystem/NVMe restore.

Do not collapse tier observations into a single generic "restore" curve.

## Metrics and acceptance design

Before running hardware experiments, write down the exact evaluation gates for the new
conditions.

At minimum record:

- decision correctness and decision accuracy;
- recompute prediction error;
- CPU-primary restore prediction error;
- secondary-tier restore prediction error;
- macro or otherwise explicitly defined aggregate error;
- actual restore-minus-recompute margin for boundary-sensitive anchors;
- source tier, external KV tokens, transfer evidence, model identity, concurrency, and
  hardware/environment provenance.

Reuse the Issue #14 evaluator where possible, but do not choose new thresholds after seeing
the results. If the Issue #14 fixed profile fails, preserve the failure instead of fitting
it away before recording the evidence.

## Generalization questions to answer

The final Issue #15 report should make each of these explicit:

1. Does the Issue #14 external-token recompute curve transfer across the new conditions?
2. Does the CPU-primary restore curve transfer, or does memory/PCIe/NUMA behavior require a
   machine-specific scale or new feature?
3. Does secondary-tier restore require a storage/path-specific model?
4. Does concurrency primarily change a global scale, or does it change the curve shape and
   restore/recompute decision boundary?
5. Which runtime observations are sufficient for online calibration without high-cardinality
   state?
6. Where does the current model become low-confidence or systematically wrong?

## Repository and execution rules

- GitHub is the authoritative write path.
- Pod workspaces are build/test/hardware environments; do not depend on Pod-to-GitHub push.
- Preserve deterministic workload identity when comparing cache modes or model variants.
- Save raw run directories outside Git, but record run directories, commands, environment,
  and structured results in validation documents.
- Keep the active runtime behavior unchanged throughout Issue #15.
- Do not describe container local filesystem measurements as physical NVMe unless storage
  provenance proves that claim.
- Do not silently substitute requested prompt tokens for runtime external KV tokens.

## Deliverables for closing Issue #15

A complete Issue #15 delivery should include:

1. a written design/plan that fixes the experiment matrix and acceptance criteria before
   expensive runs;
2. structured machine-readable results suitable for later calibration work;
3. a concise validation report under `docs/engineering/validation/`;
4. before/after or control/new-condition prediction-error and decision-accuracy tables;
5. an explicit classification of transferable versus environment-specific parameters;
6. failure-boundary evidence and any required feature/input additions;
7. a handoff to Issue #16 that states the validated regions where active decisions may be
   considered safe to design.

## Stop conditions

Do not expand the matrix just because more GPUs or models are available. Stop and analyze
when the minimum Issue #15 criteria are met or when a clear model failure exposes a missing
input that should be designed before more measurements.

Do not start Issue #16 enforcement while the Issue #15 evidence is incomplete.

## Related records

- [`../CURRENT_STATE.md`](../CURRENT_STATE.md)
- [`../history/2026-08-10-to-2026-08-11-issue13-14-and-roadmap-consolidation.md`](../history/2026-08-10-to-2026-08-11-issue13-14-and-roadmap-consolidation.md)
- [`../validation/2026-08-10-issue13-restore-recompute-crossover.md`](../validation/2026-08-10-issue13-restore-recompute-crossover.md)
- [`../validation/2026-08-10-issue14-shadow-cost-model-calibration.md`](../validation/2026-08-10-issue14-shadow-cost-model-calibration.md)
