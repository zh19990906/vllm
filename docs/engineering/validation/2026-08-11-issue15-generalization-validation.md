# Issue #15: cost model generalization validation

## Executive conclusion

The Issue #14 frozen shadow cost profile does **not** transfer unchanged to either
pre-registered new condition.

- C-load selected concurrency 2 as the first materially contended load.
  Frozen-profile high-confidence accuracy is
  7/8
  (0.875) with principal macro-MAPE
  44.454%.
- C-model uses Qwen2.5-14B-Instruct at concurrency 1.
  Frozen-profile high-confidence accuracy is
  6/7
  (0.857) with principal macro-MAPE
  53.322%.
- C-load is an `environment_specific_scale_candidate`: all three principal curves
  fall below 15% residual MAPE after one observational scalar.
- C-model is `curve_shape_or_missing_feature` overall: recompute and CPU restore are
  scale-like, but tiered-filesystem restore remains at
  35.600% residual MAPE
  after scalar correction.
- No active scheduler, inference, restore/recompute enforcement, or KV-offload runtime
  behavior is changed by this validation.

Negative evidence satisfies the Issue #15 research goal: the result identifies where
the current profile transfers structurally, where online scale is required, and where
an additional model/path feature is required before a wider sweep.

## Scope and provenance

- Approved design:
  `docs/superpowers/specs/2026-08-11-issue15-generalization-validation-design.md`.
- Approved plan:
  `docs/superpowers/plans/2026-08-11-issue15-generalization-validation.md`.
- Experiment base: `main@b43b3d83048f55f00f85e3e7d230d1b98c25b4f9`.
- Frozen profile:
  `benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json`.
- Primary percentile: P95.
- Cost coordinate: runtime actual external KV tokens per request.
- Hardware held constant for the two new conditions:
  GPU `GPU-5516e45d-3e50-69ef-f0f2-8ecff465beea`, TP1.
- This matrix validates load and model-scale generalization on one machine. It does
  **not** establish cross-machine, PCIe-topology, NUMA-placement, or physical-NVMe
  generalization.
- `secondary:filesystem` remains container-local overlay-backed filesystem evidence.

## C0 archived control

Issue #14 on Qwen2.5-7B-Instruct, one GPU, concurrency 1 produced:

- 14/14 correct P95 decisions;
- principal macro-MAPE 0.090%;
- CPU-primary and filesystem curves calibrated separately;
- no active enforcement.

The control is reused rather than rerun.

## C-load selection

The same-session requested-1024 sentinel selected concurrency 2 and stopped the
pre-registered `2 -> 4 -> 8` search.

| Principal path | C1 -> C2 absolute P95 change |
| --- | ---: |
| recompute | 91.982% |
| CPU-primary restore | 62.153% |
| secondary filesystem restore | 70.693% |

Restore provenance remained valid. C4 and C8 were not run.

## Primary frozen-profile results

| Condition | High-confidence decisions | Accuracy | Recompute MAPE | CPU restore MAPE | Filesystem MAPE | Principal macro-MAPE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C-load, 7B/C2 | 7/8 | 0.875 | 46.370% | 37.919% | 49.074% | 44.454% | fixed-profile fail |
| C-model, 14B/C1 | 6/7 | 0.857 | 47.834% | 38.997% | 73.134% | 53.322% | fixed-profile fail |

Primary artifacts were saved before diagnostics:

- C-load SHA256:
  `97f7e307dbc5435d179cf0de12a2aad21f05b0cbead2e4e7b6f9f207112e42b1`.
- C-model SHA256:
  `e45b9d49e5e2f738c55b213561eb5fdf6d6251a9bf7f54100414432ff9c3c52a`.

The C-model raw SHA was rechecked after diagnostics and remained unchanged.

## Scale-versus-shape diagnostics

### C-load

| Curve | Raw MAPE | Diagnostic scale | Residual MAPE | Classification |
| --- | ---: | ---: | ---: | --- |
| cpu_restore | 37.919% | 1.6387 | 3.020% | environment_specific_scale_candidate |
| recompute | 46.370% | 1.8764 | 3.381% | environment_specific_scale_candidate |
| tiered_fs_restore | 49.074% | 1.9186 | 5.204% | environment_specific_scale_candidate |

C-load therefore supports a low-cardinality **load/environment scale** interpretation.
The scalar values are diagnostic only and are not production calibration.

### C-model

| Curve | Raw MAPE | Diagnostic scale | Residual MAPE | Classification |
| --- | ---: | ---: | ---: | --- |
| cpu_restore | 38.997% | 1.6051 | 2.814% | environment_specific_scale_candidate |
| recompute | 47.834% | 1.9909 | 4.864% | environment_specific_scale_candidate |
| tiered_fs_restore | 73.134% | 4.1458 | 35.600% | curve_shape_or_missing_feature |

The 14B filesystem path cannot be repaired by a single scalar. That is the clearest
Issue #15 evidence that the current external-token-only curve is missing an explanatory
input for model/secondary-path behavior.

## Restore provenance and exclusions

C-load accepted 10 formal restore samples and excluded none.

C-model accepted 9 formal restore samples. Two p4096 observations remain explicitly
preserved as exclusions:

- old 16 GiB CPU-primary p4096:
  `no_external_kv_tokens`;
- 2 GiB tiered-filesystem p4096:
  `kv_offload_allocation_failure`.

A controlled 24 GiB CPU correction provided a valid p4096 CPU restore:

- runtime external KV tokens per request: 4088;
- P95 recompute: 621.064 ms;
- P95 CPU restore: 61.819 ms;
- recompute and restore provenance both point to the same corrected run.

The filesystem p4096 observation is not upgraded by increasing the tier capacity because
that would change the intended secondary-filesystem condition. It remains invalid evidence.

## Failure boundaries

- **C-load CPU, requested 128 / external 104:** frozen decision is wrong, but the
  actual restore-minus-recompute margin is only -0.128 ms. This is boundary-sensitive,
  not a clear-margin high-confidence failure.
- **C-model CPU, requested 128 / external 104:** frozen decision is wrong with
  -0.501 ms actual margin, again boundary-sensitive.
- **Filesystem short-token extrapolation:** external 104 is low-confidence and wrong
  in both new conditions, with clear actual recompute preference.
- **C-model filesystem shape:** one scalar still leaves 35.600% residual MAPE.
- **C-model filesystem p4096:** partial restore plus allocation failure makes the
  observation invalid rather than a model-scoring sample.

The formal anchors already provide enough local evidence for the generalization question,
so no broader sweep is justified.

## Answers to the ten Issue #15 research questions

### 1. Does recompute transfer?

No as a fixed absolute curve. C-load raw recompute MAPE is 46.370% and C-model is
47.834%. A single scalar reduces residuals to 3.381% and 4.864%, respectively.
Recompute therefore appears structurally reusable but environment/model-scale sensitive.

### 2. Does CPU-primary restore transfer?

Not unchanged. Raw CPU restore MAPE is 37.919% under C-load and 38.997% under
C-model. Single-scalar residuals are 3.020% and 2.814%, so CPU restore is a strong
`environment_specific_scale_candidate`.

### 3. What do machine, memory, PCIe, and NUMA results imply?

This matrix does not vary machines, GPU topology, PCIe, or NUMA placement, so it supports
no cross-machine claim. Memory/capacity is nevertheless operationally important:
insufficient CPU capacity caused invalid 14B restore provenance until the controlled
CPU-primary correction, and filesystem staging failed partially at p4096. NUMA-specific
placement remains separate work.

### 4. Is the secondary path generic?

No. C-load filesystem behavior is scale-like, but 14B/C1 filesystem behavior is
shape/missing-feature limited. The lower tier must remain a distinct curve and cannot be
collapsed into CPU restore or described as physical NVMe.

### 5. Is concurrency drift scale or shape?

For the selected C2 load, it is predominantly scale. Recompute, CPU restore, and
filesystem restore changed materially relative to the C1 sentinel, yet one scalar per
curve reduces residual MAPE below 15% for all three.

### 6. Which parameters appear fixed?

The useful fixed structure is the runtime external-token coordinate, tier separation,
workload identity, decision comparison, and restore-provenance rules. The Issue #14
absolute latency values are not globally fixed outside C0.

### 7. Which parameters require online calibration?

At minimum, bounded per-environment/load scale is required for recompute and CPU restore
outside C0, and for C-load filesystem. These Issue #15 scalars are diagnostics, not a
production online-calibration implementation.

### 8. Which low-cardinality runtime observations are justified?

Evidence supports considering model/model-class identity, concurrency/load identity,
runtime external KV tokens, source tier, transfer count/bytes, restore allocation
validity, and a bounded environment/path scale observation. These are candidates for a
small runtime model, not justification for high-cardinality fitting.

### 9. Where are low-confidence or systematic error regions?

Filesystem external 104-168 is low-confidence in both new conditions; external 104 is
wrong in both. The 14B filesystem curve has a systematic residual-shape failure even after
scaling, and its p4096 restore is invalid.

### 10. Should a new feature precede a wider sweep?

Yes. The minimal matrix already distinguishes global scale drift from a real
model/secondary-path shape failure. A bounded online-scale mechanism and an additional
model/path explanatory input should be designed before spending GPU time on a broader
factorial sweep.

## Issue #16 eligibility

The machine-readable artifact and
`docs/engineering/handoffs/2026-08-11-issue16-active-decision-handoff.md`
contain the bounded eligibility map.

The essential rule is:

- C0 measured regions may be considered for active-decision design with fallback;
- C-load and 14B CPU regions require validated online calibration first;
- 14B filesystem and low-confidence short-token filesystem regions are ineligible;
- no entry in this report enables active runtime behavior by itself.

## Stop decision

Stop the Issue #15 experimental matrix.

The two new conditions already answer the transfer/scale/shape question and expose a
specific missing-feature problem. C4/C8, additional GPUs, and more models are not run
merely because resources are available.
