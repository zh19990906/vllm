# Issue #14 Shadow Cost Model Calibration Design

## Summary

Calibrate the existing shadow-only KV offload cost model with the real hardware data produced by #13, while keeping the execution path behaviorally unchanged.

The primary calibration target is **P95 TTFT**. P50 and P99 remain diagnostic views. Completion requires **shadow decision accuracy >= 95%** on the validated #13 baseline and **P95 cost-prediction MAPE <= 15%** for the principal recompute/restore curves. Samples close to the crossover are reported with an explicit decision margin rather than treated as equivalent to large-cost mistakes.

The first implementation is deliberately conservative: retain the current `OffloadCostModel` decision equation and piecewise-linear `CostCurve`, recalibrate the profile on the runtime's actual `external_tokens` axis, and add an offline calibration/evaluation tool. Runtime formula changes are permitted only if this evidence-driven approach cannot meet the agreed acceptance criteria.

## Context

#12 introduced an opt-in shadow cost model that distinguishes CPU-primary from secondary-tier provenance, predicts restore versus recompute cost, and calibrates secondary-tier promotion online with a bounded EWMA. It intentionally does not change the actual restore path.

#13 produced a systematic real-hardware baseline on Qwen2.5-7B-Instruct, one RTX PRO 5000 72GB Blackwell GPU, concurrency 1, and deterministic eviction/restore workloads. The findings relevant to #14 are:

- valid CPU-primary P50 crossover lies between 192 and 216 requested prompt tokens;
- CPU-primary restore is substantially faster than recompute from 256 through 4096 requested tokens;
- tiered-fs lower-tier restore is slower than recompute at every measured point from 256 through 4096;
- configured cache mode is not sufficient provenance: the initial 2 GiB CPU sweep actually recomputed after CPU eviction;
- requested prompt tokens are not identical to the actual external KV tokens consumed by the runtime decision;
- filesystem evidence proves lower-tier/external restore, but not physical NVMe provenance.

The existing #12 recompute profile is already close to the #13 P95 recompute measurements. The major baseline weakness is the CPU-primary restore profile: it contains only one 1024-token sample and therefore uses proportional low-confidence extrapolation away from that point. #13 shows CPU restore has substantial fixed overhead and grows much more slowly than a line through the origin, so the single-point profile produces systematic short-prefix error.

## Goals

- Define reproducible cost-prediction and decision-accuracy metrics.
- Rebuild the calibration profile from #13 real measurements on the same token axis used by runtime decisions.
- Explain the dominant baseline prediction errors with evidence.
- Reach the agreed P95 acceptance thresholds on the #13 baseline.
- Keep shadow mode behaviorally inert.
- Produce a structured calibration result and engineering validation report consumable by #15 and #16.

## Non-Goals

- Do not enable active restore/recompute enforcement.
- Do not change scheduler matched-token behavior, lookup semantics, allocation, transfer jobs, cache contents, or actual execution path.
- Do not add model-runner timing instrumentation in #14.
- Do not solve cross-model, cross-concurrency, or cross-hardware generalization; that belongs to #15.
- Do not claim physical NVMe provenance from the tiered-fs measurements.
- Do not introduce high-cardinality runtime state or labels.
- Do not fit a polynomial, neural model, or opaque regression merely to improve baseline accuracy.

## Calibration Target and Acceptance Criteria

### Primary percentile

P95 TTFT is the primary target for calibration and decision scoring.

Reasons:

1. The existing #12 recompute profile numerically aligns closely with the #13 P95 recompute curve, so P95 preserves the original profile's operational meaning.
2. Future active decisions should avoid trading average improvement for tail-latency regressions.
3. #13 demonstrated that P50 and P95 crossover boundaries can differ, so the percentile must be explicit rather than implicit.

P50 and P99 are still reported for diagnosis and later generalization work, but they do not determine #14 completion.

### Cost error metrics

For each scored sample and each applicable path:

```text
absolute_error_ms = abs(predicted_ms - actual_ms)
relative_error = absolute_error_ms / actual_ms
```

For one curve:

```text
MAPE = mean(relative_error) * 100
```

Report three primary curve MAPEs separately:

- recompute;
- CPU-primary restore;
- tiered-fs restore.

Define the principal combined P95 MAPE as the macro-average of those three curve MAPEs:

```text
principal_mape = mean(recompute_mape, cpu_restore_mape, tiered_fs_restore_mape)
```

This avoids overweighting recompute merely because it is paired with more than one restore source.

Acceptance threshold:

```text
principal P95 cost-prediction MAPE <= 15%
```

### Decision accuracy

For each paired restore/recompute anchor sample:

```text
actual_preferred = restore   if actual_restore_p95 < actual_recompute_p95
                   recompute otherwise

predicted_preferred = restore if predicted_restore_p95 < predicted_recompute_p95
                      recompute otherwise
```

Equal costs resolve to recompute, matching the runtime model.

Primary metric:

```text
decision_accuracy = correct_decisions / scored_decisions
```

Acceptance threshold:

```text
decision_accuracy >= 95%
```

The #13 compact artifact contains 9 CPU anchor decisions (128, 192, 216, 224, 256, 512, 1024, 2048, 4096 requested tokens) and 5 tiered-fs anchor decisions (256, 512, 1024, 2048, 4096), for 14 scored decisions. On this fixed baseline, `>=95%` therefore requires **14/14 anchor decisions correct**; 13/14 is only 92.9%.

### Decision margin

Also record:

```text
actual_margin_ms = actual_restore_p95 - actual_recompute_p95
predicted_margin_ms = predicted_restore_p95 - predicted_recompute_p95
```

Near-crossover points with small absolute actual margins remain in the accuracy denominator, but the report marks them explicitly as boundary-sensitive samples.

For #14 reporting, a boundary-sensitive sample is any sample where:

```text
abs(actual_margin_ms) <= 1.0 ms
```

This label is diagnostic only and does not relax the acceptance threshold.

## Source of Truth and Scored Dataset

The canonical calibration input is:

```text
docs/engineering/validation/2026-08-10-issue13-restore-recompute-crossover.json
```

The evaluator reads structured measurements from this artifact rather than copying #13 result values into source code.

### Full-cost anchor samples

Full cost scoring uses rows with absolute recompute/restore P95 values:

- `wide_curve`: recompute + CPU-primary restore + tiered-fs restore for 256, 512, 1024, 2048, 4096 requested tokens;
- `cpu_crossover_points`: recompute + CPU-primary restore for 128, 192, 216, 224 requested tokens.

For the recompute curve MAPE, score the unique requested-token anchors from the union of those sets: 128, 192, 216, 224, 256, 512, 1024, 2048, 4096. Wide rows do not count recompute twice merely because both CPU and tiered-fs decisions use the same paired no-cache measurement.

### Boundary repeat samples

`boundary_repeats` stores repeat deltas but not complete per-repeat absolute TTFT values. Therefore those repeats:

- do **not** enter cost MAPE;
- do **not** enter the 14-anchor decision-accuracy denominator;
- are retained as supplemental P95 direction-stability checks at 192 and 216 requested tokens.

The evaluator records these as `repeat_direction_checks` and verifies that the repeat P95 delta signs are consistent with the corresponding anchor-side interpretation available in the artifact.

### Exclusions

Invalid or ambiguous samples must not silently enter calibration:

- the 2 GiB `cpu-offload` sweep is excluded from CPU restore calibration because runtime evidence proves it recomputed;
- the 208 requested-token workload-generation failure is not a latency sample;
- only restore samples backed by the #13 validation evidence are eligible.

## Runtime Axis: External KV Tokens

The existing runtime model calls `CostCurve.estimate()` using `LoadProvenance.external_tokens`. Therefore #14 calibrates profile curves on **actual external KV tokens per measurement request**, not requested prompt length.

For a benchmark row with 8 measured requests:

```text
external_tokens_per_request = external_kv_tokens_total / 8
```

The evaluator retains both axes in output:

- `requested_tokens` for benchmark provenance and crossover interpretation;
- `external_tokens` for cost-model calibration and runtime prediction.

## Duplicate External-Token Samples

Different requested prompt lengths may collapse to the same runtime `external_tokens` value because of block/chunk boundaries. For example, #13 requested 216 and 224 both produced 192 external tokens per request.

Profile construction handles duplicates as follows:

1. keep every full-cost raw anchor sample in the evaluator dataset;
2. group calibration candidates by `(curve, external_tokens)`;
3. use the **median P95 latency** as the profile sample for that runtime token count;
4. score predictions against every original eligible anchor sample, not merely against the aggregated median.

Median aggregation avoids inventing a new runtime feature while remaining robust to small run-to-run variation.

## Profile Construction

### Recompute

Construct a P95 recompute curve from paired no-cache anchor rows.

A no-cache row has no external KV transfer of its own, so its runtime x-coordinate is the `external_tokens` value of the paired valid restore workload with the same deterministic benchmark workload. The #13 artifact's fairness checks and workload hashes are the evidence for this pairing.

When multiple anchors map to the same external-token count, use the median rule above.

### CPU-primary restore

Build a multi-point CPU-primary P95 restore curve from valid 8 GiB CPU restore anchors, including the short-prefix crossover points and the wide 256-4096 range.

The 2 GiB configured CPU-offload sweep is explicitly excluded.

Expected effect: replacing the old single 1024-token sample removes proportional-through-origin extrapolation across the measured CPU range. Inside the measured range, ordinary piecewise-linear interpolation becomes high confidence.

### tiered-fs restore

Build a P95 tiered-fs restore curve from valid tiered-fs lower-tier restore anchors at the wide measured points.

Retain filesystem/tiered-fs terminology. Do not rename the tier to NVMe.

### Promotion curve / EWMA

#14 does not initially change the existing secondary promotion EWMA equation or its low-cardinality `(tier, token_bucket)` state.

The static #13 acceptance criteria are evaluated against the calibrated seed profile with runtime scale 1.0. EWMA convergence/stability is separately tested with deterministic unit tests and, if needed, a focused shadow-only runtime validation.

## Before/After Evaluation

The validation report must distinguish calibration improvement from merely restating the fitted profile.

Evaluate two profiles against the same anchor dataset:

1. **before:** the #12 shadow profile values already recorded in the repository's #12 cost-model test/design artifacts;
2. **after:** the deterministic external-token P95 profile derived from #13 by this design.

The evaluator should expose a reusable `evaluate_profile(dataset, profile)` path so the same scoring logic is used for both. The CLI may accept an optional baseline profile file or the validation driver may construct one from the recorded #12 profile; the evaluator must not scrape Python test source at runtime.

The final validation artifact records the provenance of the before-profile values explicitly.

## Offline Calibration Evaluator

Add a small benchmark/engineering tool with no dependency on a running vLLM server. Recommended location:

```text
benchmarks/cache/evaluate_cost_model.py
```

Responsibilities:

1. load and validate the #13 structured artifact;
2. build eligible anchor samples and supplemental repeat-direction checks;
3. derive runtime external-token axes from paired workloads;
4. aggregate duplicate external-token profile points with medians;
5. instantiate the existing `OffloadCostModel` from a supplied or derived profile;
6. run predictions for every eligible anchor sample;
7. calculate absolute error, relative error, per-curve MAPE, principal macro-MAPE, decision accuracy, and margins;
8. validate supplemental repeat-direction checks;
9. emit a compact machine-readable result plus human-readable summary;
10. fail nonzero in explicit `--check` mode if acceptance thresholds are missed.

Suggested CLI:

```bash
python benchmarks/cache/evaluate_cost_model.py \
  --input docs/engineering/validation/2026-08-10-issue13-restore-recompute-crossover.json \
  --percentile p95 \
  --output /tmp/issue14-calibration.json \
  --check
```

The evaluator is not a general model-fitting framework. It implements only the deterministic calibration rules defined in this spec.

## Output Schema

The structured result should include at least:

```text
schema_version
source_artifact
percentile
acceptance_thresholds
profile
samples[]
  source
  requested_tokens
  external_tokens
  actual_restore_ms
  actual_recompute_ms
  predicted_restore_ms
  predicted_recompute_ms
  restore_abs_error_ms
  recompute_abs_error_ms
  restore_relative_error
  recompute_relative_error
  actual_preferred
  predicted_preferred
  actual_margin_ms
  predicted_margin_ms
  boundary_sensitive
repeat_direction_checks[]
aggregate
  recompute_mape
  cpu_restore_mape
  tiered_fs_restore_mape
  principal_macro_mape
  decision_accuracy
  decision_correct
  decision_total
  acceptance_passed
excluded_samples[]
```

## Runtime Code Change Policy

### Phase 1: profile + evaluator only

Do not modify `OffloadCostModel`, `CostCurve`, scheduler, provenance, metrics, or manager behavior unless the calibrated external-token profile cannot satisfy the agreed acceptance criteria.

This is the preferred outcome because it demonstrates that the current runtime abstraction is sufficient for the measured baseline.

### Phase 2: evidence-triggered model change

Only if Phase 1 fails `decision_accuracy >= 95%` or principal P95 MAPE `<= 15%`, inspect residuals and make the smallest explainable model change.

Escalation order:

1. verify token-axis pairing and duplicate aggregation;
2. verify source/provenance classification;
3. distinguish fixed overhead from token-proportional cost if residuals show systematic curvature outside piecewise interpolation coverage;
4. only then consider extending curve semantics.

Any formula change requires new pure-unit tests and must preserve configuration backward compatibility unless there is a documented reason not to.

Do not introduce requested-token or transfer-byte runtime features in #14 solely to fit this baseline. Those are candidate generalization features for #15 if the external-token/source-tier model fails across environments.

## EWMA Validation

The existing secondary-tier correction remains:

```text
sample_scale = clamp(observed_promotion_ms / promotion_seed_ms,
                     sample_scale_min,
                     sample_scale_max)
runtime_scale_new = alpha * sample_scale + (1 - alpha) * runtime_scale_old
restore_estimate_ms = restore_seed_ms * runtime_scale
```

Required deterministic tests:

- converges monotonically toward a stationary sample scale;
- bucket isolation remains intact;
- clamp boundaries remain effective;
- no update for CPU-primary or tiers without a promotion curve;
- one bucket's observations do not change another bucket's prediction;
- repeated stable observations produce bounded diminishing updates.

If a focused real shadow run is required, it must remain behaviorally inert. No broad GPU sweep is required unless offline/unit evidence exposes a gap.

## Error-Source Analysis

The validation report must explicitly evaluate:

1. **Token-axis mismatch:** requested prompt tokens differ from runtime external tokens.
2. **Sparse CPU profile / fixed overhead:** the original single CPU sample forces proportional extrapolation.
3. **Source-tier dependence:** CPU-primary and tiered-fs have opposite decisions over much of the same token range.
4. **Secondary-tier nonlinear behavior:** tiered-fs P95 has a large tail increase at 2048/4096.
5. **Runtime EWMA:** EWMA corrects runtime drift but cannot repair a wrong static tier profile or provenance classification.
6. **Concurrency / hardware / model:** these are #15 generalization dimensions, not #14 baseline calibration defects.

## Testing Strategy

### Pure evaluator tests

Use a small synthetic artifact fixture to test:

- schema validation;
- invalid-sample exclusion;
- requested-to-external token pairing;
- unique recompute scoring;
- duplicate external-token median aggregation;
- P95 selection;
- cost-error and macro-MAPE calculations;
- equal-cost decision rule;
- 14-anchor accuracy accounting;
- boundary-sensitive labeling;
- repeat-direction handling without absolute repeat TTFT;
- acceptance pass/fail behavior;
- deterministic JSON output.

### Existing cost-model tests

Keep current pure `CostCurve` and `OffloadCostModel` tests unless evidence requires Phase 2 changes.

Add calibrated-profile regression assertions covering at least:

- short CPU recompute-faster side;
- short CPU restore-faster side;
- wide CPU restore side;
- all measured tiered-fs points remain recompute-preferred.

### Repository checks

Run focused tests, compile checks, targeted Ruff formatting/linting, JSON validation, and `git diff --check`.

Do not require Pod full `pre-commit --all-files`; repository-wide pre-commit remains GitHub Actions responsibility because Pod network access cannot reliably install remote hooks.

## Validation Artifact

Record final results under:

```text
docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.md
docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json
```

The report must include:

- source #13 artifact and baseline commit;
- before-profile provenance and before/after metrics;
- calibrated external-token P95 profile;
- per-anchor decision table;
- supplemental repeat-direction checks;
- acceptance thresholds and pass/fail;
- dominant error-source explanation;
- EWMA validation result;
- exclusions and limitations;
- exact commands used;
- explicit statement that execution remained shadow-only.

## Scope Boundaries / Handoff

If #14 passes on the fixed baseline, #15 owns validation across different model, concurrency/load, and hardware conditions and decides whether additional low-cardinality features such as transfer bytes or hardware/profile identity are required.

#16 may only enable active restore/recompute selection after #14 and the necessary #15 generalization evidence are accepted. The active path must consume the same explicit provenance and calibrated cost semantics rather than reintroducing `cache hit => restore`.

## Completion Criteria

#14 is complete when all of the following are true:

- [ ] P95 prediction-error and decision-accuracy metrics are implemented and documented.
- [ ] #13 data are transformed into an external-token calibrated profile with invalid samples excluded by evidence.
- [ ] Before/after evaluation identifies and verifies the dominant baseline errors.
- [ ] All 14 anchor shadow decisions are correct, satisfying the agreed >=95% threshold.
- [ ] Principal P95 macro-MAPE on the #13 anchor baseline is <=15%.
- [ ] Boundary repeat direction checks are retained as supplemental evidence.
- [ ] EWMA convergence/isolation/clamp behavior is verified.
- [ ] Runtime behavior remains shadow-only and unchanged.
- [ ] Calibration method and results are recorded in `docs/engineering/validation/`.
