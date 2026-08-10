# PR #7: KV Offload Shadow Cost Model

Period covered: **2026-08-07 through 2026-08-10**.

Final status: **merged**.

## Why this work existed

The cache benchmark work established that a lower-tier KV hit is not automatically cheaper
than recomputing the same prefix.

On the validation machine, filesystem restore P95 TTFT was slower than recompute at every
measured point from 256 through 4096 prompt tokens, while a CPU-primary 1024-token restore
was much faster than recompute. The useful decision therefore depends on the source tier
and current cost, not simply on whether cached KV exists.

This led to a per-tier cost model.

## Remote identity

- PR: `#7`
- Title: `Add KV offload shadow cost model`
- Branch: `feature/kv-offload-shadow-cost-model`
- Final head: `96de0c823721c374527dbb0b3a49fdc7eccba341`
- Base branch: `main`
- Merge commit: `37f65141108e112a317fe4a5d8215a4c21c3c00e`
- Merged: 2026-08-10
- Changed files: 14
- GitHub-reported final diff size: `+3755 / -11`

## Requirement change: enforce idea to shadow-only phase

An early direction considered letting the scheduler choose restore or recompute directly.
That was intentionally narrowed before completion.

The accepted first phase is **shadow-only**:

- predict whether restore or recompute would be cheaper;
- record source provenance, predicted costs, confidence, and runtime calibration;
- expose low-cardinality Prometheus metrics;
- preserve the existing matched-token and restore behavior.

The scheduler's real return behavior remains unchanged. `mode: enforce` is rejected by
configuration validation.

This safety boundary was the central acceptance criterion. A filesystem prediction of
`recompute` is expected to coexist with an actual restore path during this phase.

## Pure cost model

PR #7 added `vllm/v1/kv_offload/cost_model.py` with pure prediction and calibration logic.
The main concepts are:

- `CostCurve`;
- `CurveEstimate`;
- `LoadProvenance`;
- `ShadowDecision`;
- `RuntimeObservation`;
- `OffloadCostModel`.

### Curve behavior

- exact sample: high confidence;
- interpolation inside measured points: high confidence;
- extrapolation outside the measured range: low confidence;
- a single-point CPU-primary curve may scale proportionally outside its sample, but the
  extrapolated result is low confidence;
- equal estimated costs choose recompute.

Validation curves are configuration data. The measured benchmark profile is not hardcoded
as a production default.

## Runtime calibration

Secondary-tier promotion observations update a runtime scale using an EWMA. The historical
validation configuration used:

- `ewma_alpha = 0.2`;
- sample scale clamp `[0.25, 4.0]`.

The final estimator scales the whole seeded restore estimate:

```text
restore_estimate_ms = restore_seed_ms * runtime_scale
```

This is important because an earlier design discussion considered scaling only a promotion
component. The merged implementation uses the equation above; future work must not silently
switch formulas based on old discussion notes.

Runtime scale is bucketed by secondary tier and token bucket. Recompute and CPU-primary
costs are not learned online in this phase.

## Provenance design

The model needs to know where the matched KV logically came from.

PR #7 added optional no-op interfaces in the generic offloading base so non-tiering
implementations are not forced to participate. Tiering then records request-scoped logical
source information.

Source rules include:

- direct primary hit: `cpu_primary`;
- secondary hit: `secondary:<tier_key>`;
- multiple source types: `mixed`, with low confidence.

A crucial semantic rule is that a block promoted from secondary storage into CPU primary
retains its logical secondary provenance for the request that caused the promotion. If the
source were rewritten to `cpu_primary` after promotion, the shadow model would lose the
costly lower-tier origin it is trying to evaluate.

## Promotion timing

The promotion observation timer starts at the first secondary-tier `RETRY`, not at the
later `HIT`.

This aligns the runtime observation with the historical async lookup delay measurement.
Starting only at `HIT` would omit waiting time and make the online calibration artificially
cheap.

## Tier-key compatibility fix

The cost model needs stable keys for secondary tiers. A manager-only
`cost_model_tier_key` field was introduced for disambiguation when needed.

Real filesystem tier constructors do not accept that manager-only field. The runtime tier
configuration therefore strips it before constructing the actual secondary tier. This was
an important compatibility fix found during integration.

## Scheduler hook and fail-open behavior

After the final external matched-token boundary is known, the scheduler:

1. determines the matched offload keys;
2. obtains request load provenance;
3. asks the cost model for a shadow decision;
4. records statistics and debug information;
5. continues the original request path.

Shadow/provenance/telemetry exceptions are fail-open: they are logged and the existing
request behavior continues.

The implementation does not change:

- matched-token count;
- async lookup result;
- allocation;
- transfer jobs;
- cache contents;
- actual restore path.

## Metrics

The merged work exposes low-cardinality metrics for:

- shadow decisions;
- predicted restore seconds;
- predicted recompute seconds;
- runtime scale;
- secondary promotion observations.

The recorded names include:

```text
vllm:kv_offload_cost_shadow_decisions
vllm:kv_offload_cost_predicted_restore_seconds
vllm:kv_offload_cost_predicted_recompute_seconds
vllm:kv_offload_cost_runtime_scale
vllm:kv_offload_cost_observations
```

Request IDs, block hashes, paths, and device identifiers are not used as Prometheus labels.

## Native integration problem discovered during validation

A source checkout alone could not provide the installed wheel's native extensions. At the
same time, merely merging PR #7 and PR #5 into a disposable worktree did not prove that
`vllm serve` used PR #7 Python modules, because the benchmark launches the installed
`vllm` executable.

The reliable validation method became **exact source-over-wheel**:

- import and retain the installed wheel/runtime base;
- overlay only the six PR #7 Python runtime modules;
- keep native modules such as FlashAttention and `_C_stable_libtorch` from the wheel;
- make the benchmark subprocess use a PATH shim that installs the exact module overlay
  before entering the real vLLM CLI.

The six runtime modules were:

```text
vllm/v1/kv_offload/cost_model.py
vllm/v1/kv_offload/base.py
vllm/v1/kv_offload/tiering/spec.py
vllm/v1/kv_offload/tiering/manager.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
```

Validation explicitly checked that those modules came from the feature worktree while the
native extensions came from the installed wheel.

See:
[`../incidents/native-wheel-exact-overlay.md`](../incidents/native-wheel-exact-overlay.md).

## Real scheduler integration

Using installed wheel/native components with the exact six-module Python overlay, the
existing scheduler suite completed:

```text
93 passed in 6.27s
```

This provided a stronger compatibility signal than pure source-first imports.

## Hardware validation

### CPU-primary 1024 anchor

The CPU-primary pressure configuration kept the victim in primary CPU storage.

Observed across 8 measured requests:

- source: `cpu_primary`;
- preferred path: `restore`;
- confidence: high;
- predicted restore average: about 24.49 ms;
- predicted recompute average: about 81.705 ms;
- external KV hit tokens: 8192;
- CPU-to-GPU bytes: 469,762,048;
- actual path: restore.

This validated the positive case: the model can identify a lower-cost CPU-primary restore
without changing execution.

### Filesystem 1024 anchor

With a smaller CPU primary tier, the victim reached the filesystem secondary tier.

Observed across 8 measured requests:

- source: `secondary:filesystem`;
- preferred path: `recompute`;
- confidence: high;
- promotion observations: 8;
- runtime scale after observations: approximately 0.8995;
- external KV hit tokens: 8192;
- CPU-to-GPU bytes: 469,762,048;
- actual path: restore.

This was the central shadow-invariance proof: the model predicted recompute, but the real
request still restored cached KV.

### Filesystem five-point sweep

The sweep covered 256, 512, 1024, 2048, and 4096 prompt tokens.

At 512, 1024, 2048, and 4096, observed high-confidence shadow decisions preferred
recompute while actual execution remained restore.

The 256-token point did not stay uniformly recompute. Repeated investigation reproduced a
runtime-calibration crossover: the online EWMA moved the restore scale across the decision
boundary during the run. The approximate runtime-scale threshold was 0.8488. Decisions
tracked the current scale as designed, and every observed request still used the restore
execution path.

The seeded static profile itself predicts recompute for all five filesystem points. The
p256 mixed result is therefore expected adaptive behavior from runtime calibration, not a
violation of the shadow-only safety boundary.

See:
[`../validation/pr7-shadow-cost-model-hardware-validation.md`](../validation/pr7-shadow-cost-model-hardware-validation.md).

## Final scoped verification

Final validation on head `96de0c823721c374527dbb0b3a49fdc7eccba341` recorded:

- 59 focused shadow-cost tests;
- runtime-module `compileall` pass;
- targeted Ruff check pass;
- targeted Ruff format check pass;
- `git diff --check` pass;
- clean validation worktree;
- GitHub mypy pass on Python 3.10, 3.11, 3.12, and 3.13.

Repository-wide standard pre-commit remained red because existing
`benchmarks/cache/**` files had Ruff/format, markdownlint, and SPDX problems outside the
14-file PR #7 diff. Those benchmark baseline changes were deliberately not folded into the
runtime PR.

## Cleanup before merge

The temporary shadow-cost-model TDD workflow was removed before completion. PR #5
benchmark validation edits were kept out of PR #7.

Hardware evidence was archived locally, including:

```text
/code/cleanup-backup/pr7-shadow-hardware-evidence.tar.gz
```

Additional historical backups included local staged/unstaged patches and a scenario backup.
Large benchmark results remained under `/code/results` rather than being committed to Git.

Temporary worktrees, test virtual environments, and transient scripts were cleaned after
validation, while the main working repositories and result/archive directories were
retained.

## Final outcome

PR #7 merged as commit:

```text
37f65141108e112a317fe4a5d8215a4c21c3c00e
```

The delivered feature is deliberately observational. It established cost and provenance
telemetry that can support a future enforcement design, but it did not authorize or
implement a restore-versus-recompute execution switch.

Any future enforcement phase must be designed and reviewed separately rather than treating
this shadow implementation as implicit approval.
