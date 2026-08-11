# Issue #14: shadow cost model calibration validation

## Executive conclusion

Phase 1 passes the approved P95 calibration gate. Decision correctness improves from 13/14 (0.929) to 14/14 (1.000). The calibrated principal macro-MAPE is 0.090%, below the 15.0% limit.

This Phase 1 result changes calibration data and validation tooling only. It does not change the runtime cost formula or activate restore/recompute enforcement.

## Source data and environment

- Primary calibration input: `2026-08-10-issue13-restore-recompute-crossover.json`.
- Primary metric: P95 TTFT.
- #13 baseline model: Qwen2.5-7B-Instruct.
- Measurement shape: one GPU Pod, concurrency = 1.
- Calibration coordinate: actual external KV tokens per request, not requested prompt tokens.
- Filesystem measurements are lower-tier / tiered-fs measurements on the container overlay-backed local filesystem; they do not establish physical NVMe provenance.

## Acceptance criteria

- Decision accuracy minimum: 95%.
- On 14 formal anchors, the >=95% requirement implies 14/14; 13/14 is only 92.9%.
- Principal P95 macro-MAPE maximum: 15.0%.
- Boundary-sensitive diagnostic threshold: |actual restore - recompute margin| <= 1.0 ms.
- Boundary-sensitive anchors remain in the acceptance denominator; there is no threshold relaxation.

## Before-calibration baseline

- Decision correctness: 13/14 (0.929).
- Recompute MAPE: 13.585%.
- CPU restore MAPE: 77.085%.
- Tiered-filesystem restore MAPE: 11.393%.
- Principal macro-MAPE: 34.021%.
- The #12 CPU restore profile had only one measured point at 1024 tokens. Below that point, `CostCurve` used proportional low-confidence extrapolation, which severely underestimated short-token CPU restore cost.
- The observed wrong decision is: `cpu_primary` requested=128 external=104 actual=recompute predicted=restore.

## Calibrated external-token P95 profile

The profile uses low-cardinality, tier-separated empirical P95 curves on the runtime external-token axis.

### Recompute

| External tokens | P95 ms |
| ---: | ---: |
| 104 | 19.660 |
| 168 | 22.186 |
| 192 | 25.082 |
| 232 | 26.663 |
| 512 | 44.813 |
| 1024 | 81.258 |
| 2016 | 152.433 |
| 4088 | 309.140 |

### CPU-primary restore

| External tokens | P95 ms |
| ---: | ---: |
| 104 | 21.220 |
| 168 | 21.830 |
| 192 | 22.212 |
| 232 | 21.872 |
| 512 | 23.057 |
| 1024 | 24.687 |
| 2016 | 29.173 |
| 4088 | 35.213 |

### Tiered-filesystem restore

| External tokens | P95 ms |
| ---: | ---: |
| 232 | 36.007 |
| 512 | 59.159 |
| 1024 | 101.799 |
| 2016 | 320.793 |
| 4088 | 648.235 |

Requested-token anchors 216 and 224 both map to 192 external tokens. Their P95 costs are aggregated by median to form the single 192-token profile point, while both original decision anchors remain separately scored.

## After-calibration metrics

- Decision correctness: 14/14 (1.000).
- Recompute MAPE: 0.186%.
- CPU restore MAPE: 0.083%.
- Tiered-filesystem restore MAPE: 0.000%.
- Principal macro-MAPE: 0.090%.
- Acceptance result: PASS.

## Per-anchor decision evidence

| Source | Requested | External | Actual recompute ms | Actual restore ms | Actual | Calibrated prediction | Boundary-sensitive |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| cpu_primary | 128 | 104 | 19.660 | 21.220 | recompute | recompute | false |
| cpu_primary | 192 | 168 | 22.186 | 21.830 | restore | restore | true |
| cpu_primary | 216 | 192 | 24.872 | 22.130 | restore | restore | false |
| cpu_primary | 224 | 192 | 25.291 | 22.295 | restore | restore | false |
| cpu_primary | 256 | 232 | 26.663 | 21.872 | restore | restore | false |
| secondary:filesystem | 256 | 232 | 26.663 | 36.007 | recompute | recompute | false |
| cpu_primary | 512 | 512 | 44.813 | 23.057 | restore | restore | false |
| secondary:filesystem | 512 | 512 | 44.813 | 59.159 | recompute | recompute | false |
| cpu_primary | 1024 | 1024 | 81.258 | 24.687 | restore | restore | false |
| secondary:filesystem | 1024 | 1024 | 81.258 | 101.799 | recompute | recompute | false |
| cpu_primary | 2048 | 2016 | 152.433 | 29.173 | restore | restore | false |
| secondary:filesystem | 2048 | 2016 | 152.433 | 320.793 | recompute | recompute | false |
| cpu_primary | 4096 | 4088 | 309.140 | 35.213 | restore | restore | false |
| secondary:filesystem | 4096 | 4088 | 309.140 | 648.235 | recompute | recompute | false |

## Boundary repeat direction checks

- Requested 192: P95 restore-minus-recompute deltas = [-0.342283, -0.30389, -0.158826]; all restore-faster = True.
- Requested 216: P95 restore-minus-recompute deltas = [-3.242693, -3.360189, -2.266975]; all restore-faster = True.

Repeat observations are supplemental direction-stability evidence only. They do not inflate the 14-anchor decision denominator.

## Error-source analysis

- The dominant before-calibration error is the CPU restore curve: CPU restore MAPE is 77.085% before calibration versus 0.083% after calibration.
- The old single 1024-token CPU sample forces proportional extrapolation below range and causes the 128-requested / 104-external-token anchor to be predicted as restore when the measured P95 result prefers recompute.
- Recompute already tracks the #13 baseline substantially better than the old CPU restore curve; calibration aligns it to the same external-token measurement coordinate.
- Filesystem restore remains slower than recompute at all five measured wide anchors, and the calibrated decision remains recompute for each.

## EWMA validation

The existing secondary-promotion EWMA behavior remains unchanged. With alpha=0.2 and a stationary observed scale of 2.0, the validated runtime-scale sequence is `1.2, 1.36, 1.488, 1.5904, 1.67232`; updates are monotonic toward the stationary scale and diminish each observation.
Existing tests continue to cover bucket isolation, sample-scale clamping, and no update for CPU-primary or tiers without a promotion curve.
The filesystem promotion seed curve is inherited from #12 because #13 does not provide a clean per-anchor promotion-latency profile.

## Exclusions and provenance caveats

- The 2 GiB CPU sweep is excluded from CPU restore calibration: external hits/tokens and CPU-to-GPU transfer evidence were zero, so those victims were recomputed rather than restored.
- Requested token count 208 is excluded because deterministic workload generation failed; the workload was not reseeded or resampled.
- Requested prompt tokens are not interchangeable with actual external KV restored tokens.
- `tiered-fs` means the measured lower-tier / overlay-backed local filesystem path and must not be described as proven physical NVMe.

## Commands and checks

Primary gate command:

```bash
python benchmarks/cache/evaluate_cost_model.py \
  --input docs/engineering/validation/2026-08-10-issue13-restore-recompute-crossover.json \
  --before-profile benchmarks/cache/profiles/issue12-shadow-cost-baseline.json \
  --percentile p95 \
  --output /tmp/issue14-calibration.json \
  --check
```

Observed compact result:

```text
after: decision=14/14 accuracy=1.000 macro_mape=0.090% passed=True
```

Targeted validation also covered deterministic JSON output, `--check` success/failure exit codes, configurable boundary diagnostics, calibrated CPU/filesystem decisions, EWMA convergence, Ruff, compileall, and `git diff --check`.

## Completion against #14

- Prediction error and decision-accuracy metrics are defined.
- The #13 baseline is calibrated on the runtime external-token axis.
- Dominant before-calibration error is identified and explained.
- The approved 14/14 decision gate passes.
- The approved <=15% principal P95 macro-MAPE gate passes.
- Boundary repeats and invalid/excluded measurements are preserved.
- Phase 1 remains shadow-only.

## Limitations and handoff

- Phase 1 is calibrated to the #13 one-GPU, concurrency=1 baseline; #15 owns model, concurrency, workload, tier/device, and hardware generalization.
- #16 owns active restore-versus-recompute enforcement. This work does not change the active execution path.
- No new GPU measurements were required because the offline calibration gate passed with the existing #13 evidence.