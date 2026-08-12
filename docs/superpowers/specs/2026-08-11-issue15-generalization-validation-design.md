# Issue #15 Generalization Validation Design

Date: 2026-08-11

Issue: #15 `[P0] 验证 cost model 在不同模型、并发和硬件下的泛化能力`

Base: `main@b43b3d83048f55f00f85e3e7d230d1b98c25b4f9`

## Objective

Validate whether the Issue #14 calibrated shadow restore/recompute cost model transfers beyond the single-model, single-machine, concurrency=1 control environment, while keeping the runtime decision path shadow-only.

The design is intentionally small and discriminative. It is meant to distinguish:

1. parameter transfer across load and model-scale changes;
2. environment-specific multiplicative scaling;
3. curve-shape failure that indicates missing runtime inputs or features.

Issue #15 succeeds by producing trustworthy generalization evidence, including negative evidence. It does not require the current cost model to pass in every new condition.

## Source-of-truth and scope

The experiment follows this source priority:

1. live GitHub issue, PR, branch, and commit metadata;
2. current repository source and checked-in structured artifacts;
3. raw benchmark and hardware artifacts;
4. validation documents;
5. history and handoff snapshots.

Existing Issue #13/#14 data are the control. The complete baseline sweep must not be rerun unless a new implementation change or provenance problem invalidates reuse.

### In scope

- frozen-profile holdout evaluation;
- one load/concurrency shift on the same model and hardware;
- one model-scale shift on the same hardware;
- CPU-primary and secondary-filesystem restore analyzed separately;
- explicit transfer/scale/shape classification;
- structured machine-readable results and a validation report;
- a bounded eligibility handoff to Issue #16.

### Out of scope

- active restore/recompute enforcement;
- scheduler behavior changes;
- broad 1-8 GPU sweeps;
- claiming physical NVMe behavior without storage provenance;
- high-cardinality post-hoc fitting that erases fixed-profile failures;
- NUMA placement work owned by Issue #17.

## Control evidence

The existing Issue #13/#14 control uses:

- model: `Qwen2.5-7B-Instruct`;
- one NVIDIA RTX PRO 5000 72GB Blackwell GPU;
- concurrency = 1;
- CPU-primary and secondary filesystem restore measured separately;
- runtime actual `external_tokens` as the cost-model quantity.

Important retained control facts:

- CPU-primary P50 restore/recompute crossover: 192-216 requested prompt tokens;
- no secondary-filesystem P50 crossover from 256-4096 requested prompt tokens;
- Issue #14 P95 calibrated result: 14/14 decisions correct and principal macro-MAPE 0.090%;
- filesystem evidence is container local overlay-backed lower-tier restore and is not physical NVMe evidence.

Requested prompt tokens are workload anchors only. Cost-model evaluation must use runtime actual external KV tokens.

## Current Pod inventory used for design

The design-session inventory observed:

- GPU0: NVIDIA RTX PRO 5000 72GB Blackwell, UUID `GPU-5516e45d-3e50-69ef-f0f2-8ecff465beea`;
- GPU1: NVIDIA RTX PRO 5000 72GB Blackwell, UUID `GPU-74f68875-4f31-d1fb-f276-b2bb9cc7c80d`;
- GPU topology matches the Issue #13 recorded topology;
- available model candidates include `Qwen2.5-7B-Instruct` and `Qwen2.5-14B-Instruct`;
- three local untracked cache benchmark YAML files are intentionally preserved in `/code/vllm` and must not be removed by cleanup operations.

The localized `lscpu` grep did not capture CPU/NUMA fields in this inventory pass. Formal experiment provenance must therefore recapture CPU/NUMA with locale-independent commands before measurements are accepted.

## Alternatives considered

### Approach A: load shift plus model-scale shift

Use the existing 7B/C1 control, add one materially contended 7B load condition, and add Qwen2.5-14B at C1 on the same GPU.

Advantages:

- changes only one primary axis per new condition;
- keeps hardware constant;
- directly tests whether external-token-only curves miss model compute or KV geometry;
- satisfies the Issue #15 minimum evidence target without a broad sweep.

This is the selected approach.

### Approach B: load shift first, choose the second axis after results

This minimizes immediate GPU work but makes the second condition result-dependent and weakens pre-registration.

Rejected for the initial design.

### Approach C: full 7B/14B x low/high concurrency factorial

This measures model-by-contention interaction directly but roughly doubles the initial matrix before establishing whether the current model already has a clear failure boundary.

Deferred unless the selected minimal matrix leaves a specific hypothesis unresolved.

## Selected experiment matrix

### C0: control

Reuse checked-in Issue #13/#14 evidence for Qwen2.5-7B-Instruct, GPU0, concurrency=1, tensor parallel size 1.

Do not repeat the complete baseline sweep.

A small same-session drift/provenance sentinel is allowed before new-condition measurements. Sentinel data do not enter the generalization accuracy aggregate.

### C-load: minimum materially contended load

Hold constant:

- model: Qwen2.5-7B-Instruct;
- GPU: GPU0;
- tensor parallel size: 1;
- machine and tier configuration.

Probe concurrency in this fixed order:

`2 -> 4 -> 8`

The load-selection sentinel is fixed at **1024 requested prompt tokens**. Compare each candidate against a same-session concurrency=1 sentinel at the same requested anchor and with the same tier/workload lifecycle.

The first concurrency that satisfies the contention criterion becomes the only formal C-load concurrency.

Contention criterion relative to the same-session concurrency=1 sentinel:

- at least one of recompute, CPU-primary restore, or secondary-filesystem restore changes in P95 latency by at least 20%; and
- at least one other principal path changes by at least 10%; and
- required restore provenance remains valid.

Stop probing immediately when the criterion is met.

If concurrency=8 still does not meet the criterion, record that the pre-registered range did not produce material contention. Do not silently expand to 16/32.

### C-model: model-scale shift

Use:

- model: Qwen2.5-14B-Instruct;
- GPU: GPU0;
- tensor parallel size: 1;
- concurrency=1;
- the same benchmark lifecycle and tier definitions as the control where feasible.

The initial Issue #15 claim will therefore cover load generalization and model-scale generalization, not cross-machine generalization.

## Formal workload anchors

For both C-load and C-model, the first formal measurement set uses requested prompt-token anchors:

`128, 192, 256, 1024, 4096`

These anchors are chosen to cover:

- the short-token decision-boundary region;
- an early point beyond the prior crossover region;
- medium and long points that can reveal global-scale versus curve-shape drift.

Runtime `external_tokens`, not requested prompt length, are the cost-model input used for evaluation.

## Tier coverage

For every formal condition where the tier exists, preserve separate measurements and evaluation for:

1. recompute;
2. CPU-primary restore;
3. `secondary:filesystem` restore.

Do not collapse the two restore tiers into one generic restore curve.

Use `filesystem` or `tiered-fs` terminology unless physical storage provenance establishes a stronger device claim.

## Frozen-profile evaluation semantics

The primary Issue #15 evidence is a holdout evaluation of the frozen Issue #14 calibrated profile on new conditions.

The current `benchmarks/cache/evaluate_cost_model.py` derives a calibrated profile from its input before evaluating it. That behavior is valid for Issue #14 calibration but cannot serve as the primary Issue #15 holdout path.

Implementation for Issue #15 must therefore provide an explicit way to evaluate a supplied frozen calibrated profile without deriving or mutating it from the new-condition dataset.

Any recalibration or scale fitting happens only after the raw fixed-profile result has been saved and classified.

## Primary percentile and metrics

The primary percentile remains P95 to preserve comparability with Issue #14.

For each formal new condition record:

- decision correct count;
- decision total;
- decision accuracy;
- recompute MAPE;
- CPU-primary restore MAPE;
- secondary-filesystem restore MAPE;
- principal macro-MAPE;
- per-anchor actual and predicted recompute latency;
- per-anchor actual and predicted restore latency;
- actual restore-minus-recompute margin;
- predicted restore-minus-recompute margin;
- actual and predicted preferred path;
- cost-model confidence;
- runtime actual external KV tokens;
- model identity;
- concurrency/load identity;
- GPU UUID and environment provenance;
- source tier and transfer evidence.

Principal macro-MAPE is the equal-weight mean of:

1. recompute MAPE;
2. CPU-primary restore MAPE;
3. secondary-filesystem restore MAPE.

This prevents a tier with more samples from dominating the aggregate.

## Fixed-profile transfer gate

A new condition is classified as `fixed_profile_transfer_pass` only if all of the following hold for valid high-confidence samples:

1. each of the three principal curves has at least one valid high-confidence formal sample;
2. decision accuracy >= 95%;
3. principal macro-MAPE <= 15%;
4. no principal curve has MAPE > 20%;
5. there is no incorrect high-confidence decision whose actual absolute restore-minus-recompute margin is > 1 ms.

If any principal curve has zero valid high-confidence formal samples, classify the condition as `insufficient_evidence` rather than transfer pass, even if the available decisions are correct.

The 15% macro-MAPE and 95% decision-accuracy thresholds preserve the Issue #14 gate. The additional 20% per-curve cap prevents one badly transferred curve from being hidden by the macro average.

The 1 ms margin retains the Issue #14 boundary-sensitive convention and distinguishes near-boundary ambiguity from clearly wrong decisions.

Because the first formal matrix is intentionally small, the 95% accuracy gate may effectively require all high-confidence decisions to be correct. This conservatism is intentional before Issue #16 active-decision design.

## Low-confidence treatment

Low-confidence samples remain part of the evidence but do not expand the validated region.

Rules:

- report their prediction error and decision correctness separately;
- do not count a low-confidence correct prediction as evidence that Issue #16 may safely expand active coverage;
- treat systematic low-confidence errors as a useful generalization failure boundary;
- do not hide low-confidence extrapolation by refitting before recording the raw result.

## Failure diagnosis: transfer versus scale versus shape

If a frozen-profile curve fails the primary error gate, first preserve the failure artifact.

Then compute one diagnostic multiplicative scale per principal curve:

`scale = median(actual_latency / frozen_predicted_latency)`

Re-evaluate residual error using only that single scalar.

Classify each curve as:

- `transferable`: raw MAPE <= 15%;
- `environment_specific_scale_candidate`: raw MAPE > 15% and one-scalar residual MAPE <= 15%;
- `curve_shape_or_missing_feature`: one-scalar residual MAPE remains > 15%.

Also prefer `curve_shape_or_missing_feature` when latency ratios vary directionally with token size or a single scalar cannot repair decision-boundary behavior.

The diagnostic scalar is not a production calibration result. Its purpose is to distinguish global scale drift from missing explanatory inputs.

## Candidate missing inputs to evaluate from failure evidence

Do not add these features speculatively. Use the measured failure pattern to decide whether they are required:

- model identity or model compute class;
- per-token KV bytes / total transferred KV bytes;
- KV geometry such as layer/head dimensions;
- memory or PCIe bandwidth/topology indicators;
- NUMA locality;
- concurrency or queue/contention observations;
- secondary storage/path characteristics.

The 7B-to-14B shift is particularly useful because external-token equality does not imply equal model recompute cost or equal KV-transfer bytes.

## Boundary refinement

Do not expand the full sweep when a boundary question appears.

Additional local anchors are allowed only when at least one trigger occurs:

- P95 `abs(actual restore-minus-recompute margin) <= 1 ms`;
- adjacent measured anchors change actual preferred-path sign;
- frozen prediction chooses the wrong path at a clearly non-boundary anchor.

Additional measurements must be local to the implicated boundary and stop once the relevant bracket or failure region is established.

Do not pursue a single-token exact threshold when a reliable bracket answers the generalization question.

## Experiment lifecycle

### Phase 0: provenance and drift sentinel

Before formal measurements:

1. use an Issue #15 worktree based on the then-current live `main`;
2. preserve the three intentional local YAML files in `/code/vllm`;
3. recapture GPU UUID, GPU topology, CPU/NUMA, memory, filesystem, model path, and relevant software/version provenance;
4. verify GPU0 identity against control provenance;
5. run the 7B/C1 1024-requested-token sentinel to detect obvious session drift;
6. preflight Qwen2.5-14B-Instruct loading and deterministic workload generation for the formal anchors.

Sentinel data are operational validity checks and do not enter the fixed-profile generalization aggregate.

### Phase 1: C-load selection

Probe the pre-registered concurrency candidates using the 1024-requested-token sentinel.

Choose the first candidate meeting the material-contention criterion. Stop the search at that point.

### Phase 2: frozen-profile formal measurements

Run the five formal anchors for C-load and C-model, separately measuring recompute, CPU-primary restore, and secondary-filesystem restore where valid.

Create the structured measurement dataset before any post-hoc calibration or diagnostic fitting.

Evaluate the frozen Issue #14 calibrated profile against this dataset.

### Phase 3: failure diagnosis

Only if the frozen profile fails:

- compute per-curve diagnostic scalar classification;
- inspect token-dependent residual patterns;
- perform boundary-local refinement only when a trigger applies;
- identify the smallest plausible missing runtime feature/input.

### Phase 4: stop or expand

Stop when C-load and C-model provide enough evidence to classify transfer, scale drift, shape failure, and validated regions.

Expand to a third new condition only if the existing evidence leaves a specific competing hypothesis unresolved. Additional available GPUs or models are not sufficient justification by themselves.

## Raw artifact schema requirements

The machine-readable measurement artifact must contain enough information to reproduce interpretation without reading Markdown.

Each formal sample should include at least:

- schema version;
- condition ID;
- run/artifact identity;
- model name and model path;
- GPU UUID and visible GPU count;
- concurrency and request-load controls;
- requested prompt tokens;
- requests per case;
- runtime external KV tokens total;
- runtime external KV tokens per request used by the cost model;
- source tier;
- percentile;
- actual recompute latency;
- actual restore latency;
- CPU-to-GPU transfer count/bytes where applicable;
- secondary-tier lookup/transfer evidence where applicable;
- workload identity/hash evidence;
- environment provenance reference.

The evaluation artifact should additionally include:

- frozen profile identity and source commit/artifact;
- predicted recompute and restore latency;
- actual and predicted margins;
- decision correctness;
- confidence;
- per-curve errors;
- aggregate errors;
- transfer/scale/shape classification;
- diagnostic scalar only when the diagnostic phase runs.

Raw run directories remain outside Git. Checked-in artifacts record their paths/identities, commands, provenance, and structured summaries.

## Error handling and invalid samples

A configured cache mode is not enough to claim restore provenance.

Reject or explicitly exclude samples when runtime evidence shows that the intended restore path did not occur, including zero external KV use or missing expected transfer evidence.

Record invalid samples with a reason instead of silently substituting recompute measurements.

Workload-generation failures must be recorded deterministically. Do not reseed repeatedly to force a pre-registered anchor to succeed. If an anchor is unavailable for a model, record the failure and use a documented local substitute only if necessary to preserve the scientific question.

## Issue #15 decision-log policy

Issue #15 comments act as an execution decision journal and index. They do not replace checked-in specs or validation artifacts.

Add an Issue #15 comment at these milestones:

1. design approval and frozen experiment/acceptance rules;
2. implementation/experiment plan approval, with spec/plan paths;
3. any material deviation from the pre-registered design;
4. first meaningful failure boundary or an explicit stop/expand decision;
5. final validation and Issue #16 handoff.

A material change is any change that can alter experiment interpretation, including:

- acceptance thresholds;
- anchor set;
- primary percentile;
- contention criterion;
- selected model or hardware axis;
- tier/device provenance interpretation;
- fixed-profile versus recalibrated evaluation semantics.

Material changes must be recorded in Issue #15 before new measurements use the changed condition. Minor implementation details that do not alter interpretation do not need issue-level comments.

Acceptance thresholds are now pre-registered. A later discovery may explain why a gate is unsuitable, but measured results must not be used to silently move the gate.

## Stop conditions

Stop the first Issue #15 matrix when either:

- the two new conditions are sufficient to answer the transfer/scale/shape questions and identify validated regions; or
- a clear failure exposes a missing input that should be designed before more measurements.

Do not expand solely because more GPU or model capacity is available.

Do not begin Issue #16 enforcement while Issue #15 evidence is incomplete.

## Issue #16 eligibility handoff

The final report should provide a bounded eligibility map instead of a global `cost model works` statement.

Each region should specify:

- model/model-class evidence;
- concurrency/load region;
- source tier;
- external-token range;
- confidence status;
- fixed-profile transfer status;
- any required online scale or runtime feature;
- whether the region is eligible for active-decision design consideration.

Low-confidence extrapolation and fixed-profile failure regions are explicitly ineligible unless a later validated mechanism addresses them.

## Implementation and verification design

After this design is reviewed, implementation planning should use TDD and remain narrow.

Expected implementation work is limited to capabilities required for Issue #15 evidence, such as:

- evaluating an explicitly supplied frozen profile without recalibration;
- representing the new structured generalization dataset and classifications;
- preserving existing Issue #14 calibration behavior for its original use case.

Verification should include focused Python assertions/tests for evaluator semantics and artifact calculations, compile checks, targeted Ruff checks, benchmark-specific validation, and GitHub Actions repository-wide pre-commit as the authoritative all-files environment.

Do not assume pytest is installed on the Pod and do not install it merely for convenience.

## Completion criteria

Issue #15 is ready to close only when the repository contains:

- this approved design and an approved implementation/experiment plan;
- at least two meaningfully different non-control model/load/hardware conditions overall;
- structured machine-readable results;
- a validation report under `docs/engineering/validation/`;
- fixed-profile prediction-error and decision-accuracy comparisons;
- transferable versus environment-specific parameter classification;
- failure-boundary evidence and required feature/input notes;
- a bounded Issue #16 eligibility handoff.

Active restore/recompute enforcement remains Issue #16 work.
