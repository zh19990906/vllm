# Issue #16 active restore/recompute decision handoff

## Purpose

This handoff converts Issue #15 evidence into a bounded eligibility map for
Issue #16 design. It does **not** enable active restore/recompute behavior.

Issue #16 must retain an explicit fallback outside eligible regions and must not
treat diagnostic Issue #15 scalars as production calibration.

## Required model changes before broad active coverage

- Add a bounded online/environment scale mechanism before using C-load or 14B CPU
  evidence for active decisions.
- Add an explanatory feature/input for model/secondary-filesystem behavior before
  any 14B filesystem region can become eligible.
- Preserve confidence/fallback behavior for short-token filesystem extrapolation.
- Treat restore allocation/capacity failure as invalid restore evidence, never as a
  successful restore sample.

## Eligibility map

| Region | Tier | External-token region | Fixed-profile verdict | Confidence | Eligibility | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| C0 7B/C1 CPU-primary measured baseline | cpu_primary | 104-4088 | pass in Issue #14 control | measured baseline; retain boundary fallback | **eligible** | Issue #14 achieved 14/14 decisions and 0.090% principal macro-MAPE. Eligibility means active-design consideration only, not enabled runtime behavior. |
| C0 7B/C1 filesystem measured baseline | secondary:filesystem | 232-4088 | pass in Issue #14 control | measured baseline | **eligible** | The checked-in calibrated control is accurate in its measured filesystem range. This is overlay-backed filesystem evidence, not physical NVMe. |
| C-load 7B/C2 CPU-primary | cpu_primary | 104-4088 | fail | high-confidence formal samples exist | **needs_online_calibration** | Raw CPU restore MAPE is 37.919%; one scalar reduces residual MAPE to 3.020%. Recompute also requires an environment/load scale. |
| C-load 7B/C2 filesystem high-confidence region | secondary:filesystem | 232-4088 | fail | high after short-token extrapolation region | **needs_online_calibration** | Raw filesystem MAPE is 49.074%; one scalar reduces residual to 5.204%. No active use until a bounded online scale mechanism is validated. |
| C-load 7B/C2 filesystem short-token extrapolation | secondary:filesystem | 104-168 | fail / low-confidence | low | **ineligible** | The 104-external-token decision is wrong with a clear actual margin; low-confidence evidence never expands Issue #16 eligibility. |
| C-model 14B/C1 CPU-primary | cpu_primary | 104-4088 | fail | high-confidence formal samples exist | **needs_online_calibration** | Raw CPU restore MAPE is 38.997%; one scalar reduces residual to 2.814%. Recompute scale is also model-sensitive (1.991 diagnostic scalar). |
| C-model 14B/C1 filesystem valid measured region | secondary:filesystem | 104-1024 | fail | mixed; 104-168 low-confidence | **ineligible** | A single filesystem scalar leaves 35.600% residual MAPE, so the curve requires a missing input/feature rather than scale-only calibration. |
| C-model 14B/C1 filesystem long-token p4096 | secondary:filesystem | invalid/unavailable | invalid restore evidence | invalid | **ineligible** | The p4096 formal restore was partial and hit KV offload allocation failure; it is explicitly excluded. |

## Safety boundary for Issue #16

An `eligible` row means only that Issue #16 may design and test an active decision
for that measured region. It is not production enablement.

`needs_online_calibration` rows remain outside active coverage until the calibration
mechanism itself has validation evidence and bounded fallback.

`ineligible` rows must fall back to the existing safe behavior until a later issue
provides new evidence.

The current evidence does not justify cross-machine, multi-GPU, NUMA-placement, or
physical-NVMe active coverage.

## Source evidence

- C-load raw evaluation SHA256: `97f7e307dbc5435d179cf0de12a2aad21f05b0cbead2e4e7b6f9f207112e42b1`.
- C-model raw evaluation SHA256: `e45b9d49e5e2f738c55b213561eb5fdf6d6251a9bf7f54100414432ff9c3c52a`.
- C-load diagnostic SHA256: `52949a0f2bd3cecb3dc442211cff8f81faa938496dcfeeb2a32a38904a4f8283`.
- C-model diagnostic SHA256: `2510f3fc5f56f9243d7c6f37b3c2bf517be1b0a8406fb1894ce18423067655f9`.
- Final structured report:
  `docs/engineering/validation/2026-08-11-issue15-generalization-validation.json`.
