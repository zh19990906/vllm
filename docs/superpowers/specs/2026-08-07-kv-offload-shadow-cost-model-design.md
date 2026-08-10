# KV Offload Shadow Cost Model Design

## Summary

Add an opt-in shadow cost model for hierarchical KV offload reads. The model predicts whether restoring an offloaded prefix or recomputing it would be cheaper, but in the first phase it never changes the actual request path. It records provenance, estimated costs, confidence, and runtime calibration data so the prediction can be validated on real hardware before any enforcement mode is introduced.

The design is intentionally isolated from Attention, PagedAttention, CUDA graphs, and kernels. It reuses the existing offloading scheduler/manager boundary and preserves existing `LookupResult` semantics.

## Goals

- Distinguish CPU-primary hits from secondary-tier promotions.
- Estimate restore cost and recompute cost per request prefix.
- Seed the model from measured benchmark profiles rather than hard-coded model-specific constants.
- Calibrate observable secondary-tier promotion cost online with EWMA.
- Expose low-cardinality aggregate metrics and request-level debug logs.
- Keep shadow mode behaviorally inert: no change to matched-token counts, transfer jobs, scheduling decisions, or cache contents.
- Provide an interface that can later support an explicit enforcement mode without redesigning provenance or cost calculation.

## Non-Goals

- No change to the actual restore/recompute choice in this phase.
- No online learning of recompute cost in v1.
- No model-runner or attention timing instrumentation.
- No changes to PagedAttention, attention kernels, CUDA graphs, block hashing, or KV layout.
- No new remote-tier policy.
- No high-order regression or fitted crossover formula.
- No request IDs in Prometheus labels.

## Safety Invariant

When shadow mode is enabled, all existing control-flow outputs that affect request execution must remain unchanged.

In particular, the scheduler path remains equivalent to:

```python
return num_hit_tokens, bool(num_hit_tokens)
```

Shadow prediction may only add state observation, metrics, and debug logging. It must not alter `num_hit_tokens`, the async flag, load/store jobs, cache lookup outcomes, allocation behavior, or actual restore execution.

When the cost model is absent or configured `mode: off`, no shadow model object is created and no additional provenance accounting is required beyond negligible default no-op calls.

## Architecture

### 1. Pure Cost Model Module

Add `vllm/v1/kv_offload/cost_model.py` containing small, independently testable data types and pure logic:

- `CostCurve`
- `CacheCostProfile`
- `RuntimeCorrection`
- `LoadProvenance`
- `ShadowDecision`
- `OffloadCostModel`

The module must not import scheduler, model-runner, GPU, filesystem, or tier-manager implementations.

### 2. Provenance Interface

Keep `LookupResult` unchanged. Do not add values such as `HIT_CPU` or `HIT_FS` because that would widen the offloading protocol across all connector implementations.

Instead, add a narrow optional manager-side query interface at the offloading boundary, conceptually:

```python
def get_load_provenance(
    self,
    req_context: ReqContext,
    num_tokens: int,
) -> LoadProvenance | None:
    ...
```

The default implementation is a no-op returning `None`. `TieringOffloadingManager` overrides it because it is the component that actually knows whether a key was found in CPU primary or in a specific secondary tier.

The scheduler consumes this summary only after `_lookup()` has converged on the final number of external matched tokens. The query itself must be idempotent for a given request state: reading provenance must not destructively remove source markers, because scheduler retries, preemption, or repeated inspection must not silently reclassify a previously promoted prefix. Provenance is cleared only by the defined request/reset lifecycle or when a new request state replaces it.

### 3. Tiering Provenance State

`TieringOffloadingManager` must preserve the logical source of restored data across asynchronous promotion.

Existing behavior is:

1. primary lookup misses;
2. a secondary tier reports a hit;
3. promotion is initiated into CPU primary;
4. a later scheduler step sees an ordinary primary hit.

Without persistent provenance, step 4 would incorrectly classify a filesystem restore as a CPU-primary restore.

Per-request provenance state therefore records which keys were promoted and from which secondary tier. Promotion changes physical location but not logical source for the pending request.

Lifecycle:

```text
on_new_request
    -> initialize empty provenance

primary HIT with no prior promotion marker
    -> source is cpu_primary

secondary HIT + successful promotion initiation
    -> mark key source as secondary:<tier_key>

promotion completes
    -> keep the original secondary source marker

scheduler obtains final matched prefix
    -> summarize provenance for that prefix without destructive consumption

request finish/reset
    -> clear provenance
```

A failed promotion must not create a secondary-hit provenance record for tokens that cannot actually be restored.

Each secondary manager must expose or be mapped to a stable, bounded `tier_key` used by both provenance and `profile.tiers`. The cost model must not depend on human-readable log text or object identity. For the current storage tier the configured key may be `filesystem`; future tier implementations use their own stable keys.

### 4. Scheduler Shadow Hook

The hook belongs in `OffloadingConnectorScheduler.get_num_new_matched_tokens()` after the final `num_hit_tokens` value is known and before returning it.

Behavior:

```text
num_hit_tokens == 0
    -> no decision

num_hit_tokens > 0
    -> obtain provenance summary
    -> call OffloadCostModel.shadow_decide(...)
    -> record metrics/debug event
    -> return the original num_hit_tokens and async flag unchanged
```

This is also the intended future enforcement decision point, but enforcement is out of scope for this spec.

## Provenance Semantics

`LoadProvenance` represents the actual external prefix selected by the scheduler, not merely the most recently looked-up block.

Minimum fields:

- `source`
- `external_tokens`
- `secondary_promoted_tokens`
- `lookup_sync_seconds` when available
- `lookup_async_seconds` when available
- confidence metadata if source composition is ambiguous

Source classification for v1:

- all selected tokens are direct CPU-primary hits -> `cpu_primary`
- all selected tokens originated from one secondary tier -> `secondary:<tier_key>`
- CPU and secondary are mixed -> `mixed`
- multiple secondary tiers are mixed -> `mixed`

`mixed` decisions are allowed in shadow mode but always carry low confidence. V1 does not attempt a complex weighted multi-tier execution model.

## Configuration

Use the existing `kv_connector_extra_config` namespace. Add a `cache_cost_model` sub-configuration consumed only by the offloading connector implementation.

The default is off.

Example:

```yaml
kv_connector_extra_config:
  cache_cost_model:
    mode: shadow
    ewma_alpha: 0.2
    sample_scale_min: 0.25
    sample_scale_max: 4.0

    profile:
      recompute_ms:
        256: 26.414
        512: 44.961
        1024: 81.705
        2048: 152.461
        4096: 308.424

      tiers:
        cpu_primary:
          restore_ms:
            1024: 24.490

        filesystem:
          restore_ms:
            256: 31.119
            512: 56.979
            1024: 108.132
            2048: 244.266
            4096: 651.127

          promotion_ms:
            256: 13.916
            512: 35.230
            1024: 81.458
            2048: 171.505
            4096: 498.874
```

The example numbers are benchmark profile data, not source-code defaults. The implementation must not hard-code them.

Profile identity remains an operator responsibility in v1. A profile is valid only for the model/hardware/cache-format environment in which it was measured. A later iteration may add explicit profile fingerprint validation.

## Cost Model

### Curves

Use piecewise-linear interpolation between measured points. Do not fit a polynomial or a single global linear formula.

For a request with `external_tokens = n`:

```text
recompute_cost_ms = recompute_curve(n)
restore_seed_ms = tier_restore_curve(source, n)
restore_cost_ms = restore_seed_ms * runtime_scale(source, bucket(n))
```

Decision:

```text
preferred = restore   if restore_cost_ms < recompute_cost_ms
preferred = recompute otherwise
```

Equal estimated cost resolves to `recompute` as the conservative choice because it avoids unnecessary I/O and transfer work.

### Extrapolation and Confidence

- Exact measured point -> high confidence, subject to source confidence.
- Interpolation between two measured points -> high confidence.
- Outside a measured range -> low confidence.
- A tier with only one measured point, such as the initial CPU-primary profile, may produce a simple proportional estimate for shadow observability, but the result is low confidence away from the measured point.
- `mixed` provenance always lowers confidence regardless of curve coverage.

Shadow mode must still record low-confidence predictions; it simply must not treat them as future enforcement-ready evidence.

## Online Calibration

V1 updates only secondary-tier promotion correction because it is already directly observable in the tiering manager.

It does not online-learn recompute cost because there is no clean scheduler/manager observation for exact prefill compute time without adding model-runner instrumentation.

It does not online-learn CPU-primary restore cost in v1 because CPU-to-GPU transfer timing is not yet exposed to the cost-model boundary as a clean per-request observation.

### EWMA

Maintain a runtime scale per `(source tier, token bucket)`.

Initialization:

```text
runtime_scale = 1.0
```

For a completed secondary promotion observation:

```text
sample_scale = observed_promotion_ms / seeded_promotion_ms
sample_scale = clamp(sample_scale, sample_scale_min, sample_scale_max)
new_scale = alpha * sample_scale + (1 - alpha) * previous_scale
```

Defaults:

```text
ewma_alpha = 0.2
sample_scale_min = 0.25
sample_scale_max = 4.0
```

The clamp prevents one transient storage or scheduling stall from destabilizing the model.

Token buckets should be derived deterministically from the configured promotion curve sample points so the profile and runtime correction use the same scale boundaries. Secondary promotion observations are assigned using the actual promoted-token count represented by the completed promotion job; they are not inferred from total prompt length.

## Shadow Decision Record

A request-level debug record should contain enough information to audit a prediction without adding high-cardinality metrics:

```text
request_id=<id>
external_tokens=2048
source=secondary:filesystem
recompute_estimate_ms=152.46
restore_seed_ms=244.27
runtime_scale=1.08
restore_estimate_ms=263.81
preferred=recompute
confidence=high
mode=shadow
actual_path=restore
```

`actual_path=restore` is expected in shadow mode because the model does not change behavior.

## Metrics

Prometheus metrics must use bounded labels only. Never use request ID as a label.

Candidate aggregate metrics:

- `vllm:kv_offload_cost_shadow_decisions_total{source,preferred,confidence}`
- `vllm:kv_offload_cost_predicted_restore_ms`
- `vllm:kv_offload_cost_predicted_recompute_ms`
- `vllm:kv_offload_cost_runtime_scale{source,token_bucket}`
- `vllm:kv_offload_cost_observations_total{source}`

Histogram/gauge/counter details should follow existing offloading metrics patterns. Metric names may be adjusted during implementation to match repository naming conventions, but label cardinality and semantics must remain as specified.

## Error Handling

Shadow mode must fail open with respect to request execution.

- Invalid `cache_cost_model` configuration is a startup configuration error.
- Missing curve coverage at runtime produces a low-confidence or unavailable prediction, not a request failure.
- Missing provenance produces no shadow decision for that lookup.
- Missing runtime observation leaves the current runtime scale unchanged.
- Cost-model calculation exceptions must not change the matched-token result. Implementation should keep pure validation strict enough that runtime calculation failures are exceptional and observable in logs/tests.

## Testing Strategy

Implementation follows TDD.

### Pure Cost Model Tests

Cover:

- exact curve sample points;
- interpolation;
- below-range and above-range extrapolation confidence;
- single-point CPU-primary estimation;
- equal-cost resolves to recompute;
- 1024 CPU seed predicts restore;
- filesystem seeds at 256, 512, 1024, 2048, and 4096 predict recompute;
- EWMA alpha behavior;
- sample-scale clamp;
- independent tier/bucket corrections.

### Provenance Tests

Cover:

- direct CPU-primary hit;
- secondary hit marks source;
- promoted key later appearing as primary HIT still retains secondary source;
- repeated provenance queries are idempotent;
- failed promotion is not misclassified as a successful secondary restore;
- request finish cleans provenance;
- cache reset cleans provenance;
- mixed CPU/secondary prefix is classified `mixed` with low confidence;
- multiple secondary sources are classified `mixed` with low confidence;
- secondary `tier_key` maps deterministically to the configured profile tier.

### Scheduler Invariance Tests

For identical mocked manager results, compare shadow disabled and enabled:

- identical `num_hit_tokens`;
- identical async flag;
- identical load jobs;
- identical allocation-facing behavior;
- identical manager lookup/touch behavior except for the explicit provenance query/stat recording;
- additional shadow decision stats only when enabled.

### Configuration Tests

Cover:

- missing `cache_cost_model` -> off;
- `mode: off` -> off;
- valid shadow profile parses;
- invalid alpha/ranges/curve points fail startup validation;
- unknown provenance tier key cannot silently select another tier profile;
- benchmark data are never defaulted implicitly.

## Hardware Validation

Use the already-proven Qwen2.5-7B environment and pressure workload. Do not expand the matrix until the two anchors pass.

### Anchor 1: CPU Primary Hit at 1024

Expected:

```text
shadow preferred = restore
actual path = restore
```

### Anchor 2: Filesystem Restore at 1024

Expected:

```text
shadow preferred = recompute
actual path = restore
```

The second line proves shadow invariance.

### Filesystem Sweep

For 256, 512, 1024, 2048, and 4096 external-token cases, expected shadow preference is `recompute` using the measured seed profile.

Also verify:

- external KV token counts are unchanged versus the baseline workload;
- CPU-to-GPU restored bytes are unchanged;
- actual cache path remains restore;
- runtime scale for the filesystem tier receives observations and updates;
- shadow instrumentation does not cause a material TTFT regression relative to measurement noise.

The success criterion is prediction correctness and execution invariance, not improved TTFT.

## Future Enforcement Phase

A later separately reviewed design may introduce:

```yaml
cache_cost_model:
  mode: enforce
```

Enforcement would reuse the same provenance and cost-model API and the same scheduler decision point. It must not be implemented as part of this shadow-mode work.

Before enforcement is allowed, hardware evidence must show that:

- known CPU-primary cases reliably prefer restore;
- known filesystem cases reliably prefer recompute;
- provenance classification is stable;
- online correction does not cause unstable decision oscillation;
- low-confidence and mixed-source cases have an explicit safe policy.

## Acceptance Criteria

The shadow-mode implementation is complete when all of the following are true:

1. Shadow mode is opt-in and disabled by default.
2. Existing `LookupResult` semantics are unchanged.
3. Tiering provenance survives secondary-to-primary promotion correctly and provenance reads are idempotent.
4. Secondary provenance uses a stable tier key that maps explicitly to the configured profile.
5. Cost curves use configured benchmark seed data with piecewise-linear interpolation.
6. Secondary-tier runtime correction uses bounded EWMA.
7. Recompute and CPU-primary curves are not falsely presented as online-learned in v1.
8. Scheduler execution outputs are invariant between off and shadow modes.
9. Prometheus labels remain low cardinality.
10. Unit tests cover cost logic, provenance lifecycle, configuration, and scheduler invariance.
11. Hardware validation produces the expected CPU-primary and filesystem shadow decisions while the actual request path remains unchanged.
