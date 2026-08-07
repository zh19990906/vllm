# KV Offload Shadow Cost Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in shadow cost model that distinguishes CPU-primary restores from secondary-tier promotions, predicts restore versus recompute cost, calibrates secondary promotion cost online, and records the decision without changing the actual request path.

**Architecture:** Keep the model as pure logic in `vllm/v1/kv_offload/cost_model.py`; let `TieringOffloadingManager` remain the source of truth for logical load provenance and secondary-promotion observations; let `OffloadingConnectorScheduler` make and record the shadow decision only after the final external hit length is known. Reuse the existing offloading stats/Prometheus plumbing and preserve `LookupResult`, scheduler matched-token counts, async flags, transfer jobs, cache allocation, and restore execution exactly.

**Tech Stack:** Python 3.11, dataclasses, vLLM V1 KV offloading/tiering interfaces, `OffloadingConnectorStats`, pytest, unittest.mock, existing Qwen2.5-7B cache benchmark suite for hardware validation.

## Global Constraints

- Shadow mode is opt-in and disabled by default.
- `LookupResult` semantics are unchanged.
- No changes to Attention, PagedAttention, CUDA graphs, kernels, block hashing, or KV layout.
- No online learning of recompute cost in v1.
- No model-runner or attention timing instrumentation.
- No online learning of CPU-primary restore cost in v1.
- Benchmark seed numbers are configuration data and must never become source-code defaults.
- Equal predicted restore/recompute cost resolves to `recompute`.
- Request IDs must never be Prometheus labels.
- Shadow prediction must not alter `num_hit_tokens`, the async flag, load/store jobs, cache lookup outcomes, allocation behavior, or actual restore execution.
- Missing provenance or missing runtime curve coverage skips the decision or lowers confidence; it never fails the request.
- `mode: enforce` is out of scope and must not be accepted as a working mode in this implementation.
- Runtime feature branch is based on `main` after merged PR #6 (`bcbb26fa8ed90d2bd1de57f70168ec3d188c8c9c`); do not merge benchmark PR #5 into the runtime branch.

---

## File Structure

- Create `vllm/v1/kv_offload/cost_model.py` — pure configuration parsing, cost curves, provenance/result types, shadow decision logic, token buckets, and bounded EWMA correction.
- Create `tests/v1/kv_offload/test_cost_model.py` — pure unit coverage for curves, config validation, decisions, mixed-source behavior, and EWMA.
- Modify `vllm/v1/kv_offload/base.py` — add default no-op cost-model/provenance interfaces without widening `LookupResult`.
- Modify `vllm/v1/kv_offload/tiering/spec.py` — parse the opt-in model once, resolve stable secondary-tier profile keys, pass shared model/provenance metadata to the manager, and expose the same model to the scheduler.
- Modify `vllm/v1/kv_offload/tiering/manager.py` — record logical source per looked-up key, preserve secondary provenance after promotion, time promotion waves, feed bounded EWMA, and expose an idempotent provenance summary.
- Create `tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py` — focused manager tests using mocks/fakes rather than filesystem or GPU dependencies.
- Modify `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py` — register low-cardinality shadow decision, prediction, observation, and runtime-scale metrics using existing stats plumbing.
- Modify `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` — obtain matched-prefix provenance, calculate a shadow decision, emit aggregate stats/debug logging, and return the original matched-token result unchanged.
- Modify `tests/v1/kv_connector/unit/offloading_connector/utils.py` — let `MockOffloadingSpec` expose an opt-in `OffloadCostModel` for scheduler tests.
- Modify `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` — prove shadow execution invariance and metric emission.

---

### Task 1: Pure Cost Model, Configuration, and EWMA

**Files:**
- Create: `vllm/v1/kv_offload/cost_model.py`
- Create: `tests/v1/kv_offload/test_cost_model.py`

**Interfaces:**
- Consumes: only Python stdlib (`dataclasses`, `math`, `bisect`, `collections.abc`, `typing`).
- Produces:
  - `Confidence = Literal["high", "low"]`
  - `PreferredPath = Literal["restore", "recompute"]`
  - `CurveEstimate(value_ms: float, confidence: Confidence)`
  - `LoadProvenance(source: str, external_tokens: int, secondary_promoted_tokens: int | None, sources: tuple[str, ...], confidence: Confidence, lookup_sync_seconds: float | None = None, lookup_async_seconds: float | None = None)`
  - `ShadowDecision(preferred: PreferredPath, restore_estimate_ms: float, recompute_estimate_ms: float, runtime_scale: float, confidence: Confidence)`
  - `RuntimeObservation(source: str, token_bucket: int, observed_ms: float, seeded_ms: float, sample_scale: float, runtime_scale: float)`
  - `CostCurve.from_mapping(raw: Mapping[object, object]) -> CostCurve`
  - `CostCurve.estimate(tokens: int) -> CurveEstimate`
  - `CostCurve.bucket_for(tokens: int) -> int`
  - `OffloadCostModel.from_extra_config(extra_config: Mapping[str, Any]) -> OffloadCostModel | None`
  - `OffloadCostModel.shadow_decide(provenance: LoadProvenance) -> ShadowDecision | None`
  - `OffloadCostModel.observe_secondary_promotion(tier_key: str, tokens: int, observed_ms: float) -> RuntimeObservation | None`

- [ ] **Step 1: Write failing curve/config tests**

Add tests that define a profile locally; never import hardware seed constants from production code:

```python
import pytest

from vllm.v1.kv_offload.cost_model import CostCurve, OffloadCostModel


def test_cost_curve_exact_interpolation_and_extrapolation():
    curve = CostCurve.from_mapping({256: 20.0, 512: 40.0, 1024: 80.0})

    assert curve.estimate(512).value_ms == pytest.approx(40.0)
    assert curve.estimate(512).confidence == "high"
    assert curve.estimate(768).value_ms == pytest.approx(60.0)
    assert curve.estimate(768).confidence == "high"

    # Outside measured coverage: proportional endpoint extrapolation, low confidence.
    assert curve.estimate(128).value_ms == pytest.approx(10.0)
    assert curve.estimate(128).confidence == "low"
    assert curve.estimate(2048).value_ms == pytest.approx(160.0)
    assert curve.estimate(2048).confidence == "low"


def test_single_point_curve_is_exact_only_at_sample():
    curve = CostCurve.from_mapping({1024: 24.49})
    assert curve.estimate(1024).confidence == "high"
    assert curve.estimate(2048).value_ms == pytest.approx(48.98)
    assert curve.estimate(2048).confidence == "low"


def test_shadow_model_is_off_by_default():
    assert OffloadCostModel.from_extra_config({}) is None
    assert OffloadCostModel.from_extra_config({"cache_cost_model": {"mode": "off"}}) is None
```

Also parameterize invalid inputs: empty shadow profile, non-positive token sample, non-positive/non-finite latency, `ewma_alpha <= 0`, `ewma_alpha > 1`, non-positive sample clamp, min > max, and `mode: enforce`.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
python -m pytest -q tests/v1/kv_offload/test_cost_model.py
```

Expected: collection/import failure because `vllm.v1.kv_offload.cost_model` does not exist.

- [ ] **Step 3: Implement parsing and `CostCurve` minimally**

Create the new module with strict finite-positive validation and deterministic sample ordering. Inside measured coverage, interpolate linearly. Outside coverage, scale from the nearest endpoint proportionally to token count and mark confidence low; this avoids negative extrapolated costs. A single-point curve uses the same proportional rule away from its exact sample.

Core implementation shape:

```python
@dataclass(frozen=True, slots=True)
class CurveEstimate:
    value_ms: float
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class CostCurve:
    samples: tuple[tuple[int, float], ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[object, object]) -> "CostCurve": ...

    def estimate(self, tokens: int) -> CurveEstimate:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        # exact -> high; inside neighbors -> linear/high;
        # outside -> nearest endpoint proportional/low.
        ...

    def bucket_for(self, tokens: int) -> int:
        # Ceiling bucket over configured sample token counts; last sample above range.
        for sample_tokens, _ in self.samples:
            if tokens <= sample_tokens:
                return sample_tokens
        return self.samples[-1][0]
```

Parse `cache_cost_model` only when `mode == "shadow"`. Require `profile.recompute_ms` and at least one tier restore curve. A tier may omit `promotion_ms`; that tier remains statically predictable but receives no EWMA update.

- [ ] **Step 4: Add failing decision/EWMA tests**

Use the measured shape as test fixture data, not defaults:

```python
PROFILE = {
    "cache_cost_model": {
        "mode": "shadow",
        "ewma_alpha": 0.2,
        "sample_scale_min": 0.25,
        "sample_scale_max": 4.0,
        "profile": {
            "recompute_ms": {256: 26.414, 512: 44.961, 1024: 81.705,
                             2048: 152.461, 4096: 308.424},
            "tiers": {
                "cpu_primary": {"restore_ms": {1024: 24.490}},
                "filesystem": {
                    "restore_ms": {256: 31.119, 512: 56.979, 1024: 108.132,
                                   2048: 244.266, 4096: 651.127},
                    "promotion_ms": {256: 13.916, 512: 35.230, 1024: 81.458,
                                     2048: 171.505, 4096: 498.874},
                },
            },
        },
    }
}
```

Cover:

```python
def test_cpu_1024_prefers_restore(): ...

@pytest.mark.parametrize("tokens", [256, 512, 1024, 2048, 4096])
def test_filesystem_samples_prefer_recompute(tokens): ...

def test_equal_cost_prefers_recompute(): ...

def test_ewma_updates_only_matching_tier_bucket(): ...

def test_ewma_clamps_sample_scale_before_update(): ...

def test_mixed_sources_use_conservative_max_and_low_confidence(): ...
```

For mixed provenance, if every component source has a restore curve, estimate every component at the full `external_tokens`, take the maximum restore estimate, and force confidence to low. If any component source has no curve, return `None`. This is shadow-only conservative observability, not an enforcement policy.

- [ ] **Step 5: Run tests to verify RED for missing decision methods**

Run the same test file. Expected: curve/config tests pass while decision/EWMA tests fail because the methods/types are not implemented yet.

- [ ] **Step 6: Implement decision and bounded EWMA**

Implement source mapping exactly:

```python
def _profile_key(source: str) -> str | None:
    if source == "cpu_primary":
        return "cpu_primary"
    if source.startswith("secondary:"):
        return source.removeprefix("secondary:")
    return None
```

For a secondary source, use the runtime scale at `(tier_key, promotion_curve.bucket_for(tokens))`; default scale is `1.0`. The approved v1 equation is:

```python
restore_estimate_ms = restore_seed_ms * runtime_scale
```

Do not silently change this to component-only correction in v1.

EWMA:

```python
sample_scale = observed_ms / promotion_seed_ms
sample_scale = min(max(sample_scale, sample_scale_min), sample_scale_max)
new_scale = ewma_alpha * sample_scale + (1.0 - ewma_alpha) * old_scale
```

Return a `RuntimeObservation` so the manager can emit bounded metrics without reading private dictionaries.

- [ ] **Step 7: Run pure tests and commit**

Run:

```bash
python -m pytest -q tests/v1/kv_offload/test_cost_model.py
python -m compileall -q vllm/v1/kv_offload/cost_model.py
```

Expected: all new tests pass and compileall exits 0.

Commit:

```bash
git add vllm/v1/kv_offload/cost_model.py tests/v1/kv_offload/test_cost_model.py
git commit -m "feat: add KV offload shadow cost model"
```

---

### Task 2: Base Interfaces and Tiering Spec Wiring

**Files:**
- Modify: `vllm/v1/kv_offload/base.py`
- Modify: `vllm/v1/kv_offload/tiering/spec.py`
- Create: `tests/v1/kv_offload/tiering/test_shadow_cost_spec.py`

**Interfaces:**
- Consumes: `OffloadCostModel`, `LoadProvenance` from Task 1.
- Produces:
  - `OffloadingManager.get_load_provenance(keys: Collection[OffloadKey], req_context: ReqContext, external_tokens: int) -> LoadProvenance | None` defaulting to `None`.
  - `OffloadingSpec.get_cost_model() -> OffloadCostModel | None` defaulting to `None`.
  - `TieringOffloadingSpec.get_cost_model() -> OffloadCostModel | None` returning the shared instance created from config.
  - Stable secondary tier keys passed to `TieringOffloadingManager` in the same order as `secondary_tiers`.

- [ ] **Step 1: Write failing no-op interface tests**

Add a tiny concrete fake manager/spec in the test module or extend an existing test fake and assert default behavior:

```python
assert manager.get_load_provenance([], ReqContext("r"), 64) is None
assert spec.get_cost_model() is None
```

These tests protect compatibility for non-tiering offload implementations.

- [ ] **Step 2: Run targeted test and verify RED**

Run:

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_spec.py
```

Expected: attribute/method failures because the default interfaces do not exist.

- [ ] **Step 3: Add default no-op interfaces in `base.py`**

Use `TYPE_CHECKING` imports to avoid runtime cycles:

```python
if TYPE_CHECKING:
    from vllm.v1.kv_offload.cost_model import LoadProvenance, OffloadCostModel


class OffloadingManager(ABC):
    ...
    def get_load_provenance(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        external_tokens: int,
    ) -> "LoadProvenance | None":
        return None


class OffloadingSpec(ABC):
    ...
    def get_cost_model(self) -> "OffloadCostModel | None":
        return None
```

Do not make either method abstract; existing implementations must remain source-compatible.

- [ ] **Step 4: Write failing tier-key/config wiring tests**

Test `TieringOffloadingSpec` helper behavior without constructing mmap resources. Add a small class/static helper `_resolve_cost_model_tier_keys()` and test it directly:

```python
def test_unique_secondary_types_default_to_type_key():
    assert TieringOffloadingSpec._resolve_cost_model_tier_keys(
        [{"type": "filesystem"}, {"type": "network"}], enabled=True
    ) == ("filesystem", "network")


def test_duplicate_secondary_types_require_explicit_cost_keys():
    with pytest.raises(ValueError, match="cost_model_tier_key"):
        TieringOffloadingSpec._resolve_cost_model_tier_keys(
            [{"type": "filesystem"}, {"type": "filesystem"}], enabled=True
        )


def test_explicit_keys_disambiguate_duplicate_types():
    assert TieringOffloadingSpec._resolve_cost_model_tier_keys(
        [
            {"type": "filesystem", "cost_model_tier_key": "local_ssd"},
            {"type": "filesystem", "cost_model_tier_key": "slow_disk"},
        ],
        enabled=True,
    ) == ("local_ssd", "slow_disk")
```

Reject empty/non-string keys and duplicate explicit keys while shadow is enabled. When shadow is off, return type-derived keys without adding validation overhead that could break existing configurations.

- [ ] **Step 5: Wire one shared model through `TieringOffloadingSpec`**

In `__init__`:

```python
self._cost_model = OffloadCostModel.from_extra_config(self.extra_config)
self._cost_model_tier_keys = self._resolve_cost_model_tier_keys(
    self.secondary_tier_configs,
    enabled=self._cost_model is not None,
)
```

Override:

```python
@override
def get_cost_model(self) -> OffloadCostModel | None:
    return self._cost_model
```

When constructing the manager, pass the shared object and deterministic metadata:

```python
tiering_manager = TieringOffloadingManager(
    primary_tier=primary_tier,
    secondary_tiers=secondary_tiers,
    cost_model=self._cost_model,
    secondary_tier_keys=self._cost_model_tier_keys,
    tokens_per_chunk_by_group=tuple(
        tokens_per_block * self.blocks_per_chunk
        for tokens_per_block in self.tokens_per_block
    ),
)
```

The model object passed to manager and returned to scheduler must be the same instance so manager EWMA updates are immediately visible to scheduler decisions.

- [ ] **Step 6: Run compatibility/spec tests and commit**

Run:

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_spec.py
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
python -m compileall -q vllm/v1/kv_offload/base.py vllm/v1/kv_offload/tiering/spec.py
```

Expected: new tests pass; existing scheduler tests remain green.

Commit:

```bash
git add vllm/v1/kv_offload/base.py vllm/v1/kv_offload/tiering/spec.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_spec.py
git commit -m "feat: wire shadow cost model into tiering"
```

---

### Task 3: Tiering Provenance and Secondary-Promotion Observations

**Files:**
- Modify: `vllm/v1/kv_offload/tiering/manager.py`
- Create: `tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py`

**Interfaces:**
- Consumes: shared `OffloadCostModel`, stable `secondary_tier_keys`, `tokens_per_chunk_by_group`, and base `get_load_provenance()` interface.
- Produces: idempotent request-prefix provenance plus manager-side EWMA observations; no lookup result changes.

- [ ] **Step 1: Write failing direct-CPU and secondary-promotion provenance tests**

Build a lightweight fake secondary tier and `MagicMock` primary tier. Create request state with `manager.on_new_request(ReqContext("r"))`.

Test direct CPU:

```python
primary.lookup.return_value = LookupResult.HIT
assert manager.lookup(key, ctx) is LookupResult.HIT
prov = manager.get_load_provenance([key], ctx, 64)
assert prov is not None
assert prov.source == "cpu_primary"
assert prov.external_tokens == 64
assert prov.secondary_promoted_tokens == 0
```

Test secondary persistence across promotion:

```python
primary.lookup.side_effect = [LookupResult.MISS, LookupResult.HIT]
secondary.lookup.return_value = LookupResult.HIT
monkeypatch.setattr(manager, "_initiate_promotion", lambda *args: True)

assert manager.lookup(key, ctx) is LookupResult.RETRY
assert manager.lookup(key, ctx) is LookupResult.HIT
prov = manager.get_load_provenance([key], ctx, 64)
assert prov.source == "secondary:filesystem"
assert prov.secondary_promoted_tokens == 64
```

Call `get_load_provenance()` twice and assert equal results to prove the read is idempotent and does not consume state.

- [ ] **Step 2: Run provenance tests and verify RED**

Run:

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
```

Expected: constructor/signature/state/provenance failures.

- [ ] **Step 3: Add cost-only request state and tier mapping**

Extend `RequestState` with fields that are allocated/used only when a model exists:

```python
key_sources: dict[OffloadKey, str] | None = None
promotion_started_at: dict[str, float] | None = None
promotion_keys: dict[str, set[OffloadKey]] | None = None
promotion_pending_jobs: dict[str, int] | None = None
promotion_elapsed_seconds: dict[str, float] | None = None
```

In `on_new_request()`, initialize these dicts only if `self._cost_model is not None`; otherwise leave them `None`.

In manager `__init__`, preserve backward-compatible defaults:

```python
def __init__(
    self,
    primary_tier: CPUPrimaryTierOffloadingManager,
    secondary_tiers: list[SecondaryTierManager] | None = None,
    *,
    cost_model: OffloadCostModel | None = None,
    secondary_tier_keys: tuple[str, ...] | None = None,
    tokens_per_chunk_by_group: tuple[int, ...] = (),
): ...
```

Create object-to-key mapping only when enabled:

```python
self._secondary_tier_keys = {
    tier: key for tier, key in zip(self.secondary_tiers, secondary_tier_keys or ())
}
```

Assert matching lengths when a model is enabled.

- [ ] **Step 4: Record logical source without changing lookup behavior**

In `lookup()`:

```python
if primary_hit is LookupResult.HIT:
    if req_state is not None and req_state.key_sources is not None:
        req_state.key_sources.setdefault(key, "cpu_primary")
    return LookupResult.HIT
```

After a secondary `HIT`, only when `_initiate_promotion()` returns true:

```python
tier_key = self._secondary_tier_keys[tier]
source = f"secondary:{tier_key}"
req_state.key_sources[key] = source
req_state.promotion_started_at.setdefault(tier_key, lookup_start)
req_state.promotion_keys.setdefault(tier_key, set()).add(key)
```

Do not mark secondary provenance when promotion allocation fails. Preserve the existing returned `MISS`/`RETRY` values exactly.

- [ ] **Step 5: Implement idempotent provenance summarization and lifecycle tests**

Add tests for:

- failed promotion does not leave a secondary source marker;
- CPU + secondary selected keys -> `source == "mixed"`, `confidence == "low"`, `secondary_promoted_tokens is None`;
- two different secondary source keys -> mixed/low;
- unknown selected key -> `None` rather than guessing;
- `on_request_finished()` removes provenance with the request state;
- `reset_cache()` clears cost provenance on active requests while preserving the existing active-request `RequestState` lifecycle.

Implement:

```python
@override
def get_load_provenance(
    self,
    keys: Collection[OffloadKey],
    req_context: ReqContext,
    external_tokens: int,
) -> LoadProvenance | None:
    if self._cost_model is None or external_tokens <= 0:
        return None
    state = self._req_state.get(req_context.req_id)
    if state is None or state.key_sources is None:
        return None
    selected = tuple(keys)
    if not selected:
        return None
    sources = tuple(sorted({state.key_sources[k] for k in selected if k in state.key_sources}))
    if any(k not in state.key_sources for k in selected) or not sources:
        return None
    ...
```

For pure secondary provenance, set `secondary_promoted_tokens=external_tokens`; for pure CPU, `0`; for mixed, `None` and low confidence. Populate lookup timing fields only when an applicable manager observation exists; otherwise leave them `None`.

- [ ] **Step 6: Write failing promotion-observation/EWMA test**

Use a fake monotonic clock via monkeypatch. Exercise a successful promotion wave, `_flush_pending_promotions()`, fake completion from `tier.get_finished_jobs()`, and `_process_finished_jobs()`.

Assert:

```python
observation = model.observe_secondary_promotion  # spy/wrap real method
# one call only after all jobs for (request, tier) finish
# source key is filesystem; token span lands in expected deterministic bucket
```

Also test a failed completed promotion removes affected secondary provenance and does not update EWMA.

- [ ] **Step 7: Implement promotion wave timing and token-span calculation**

Track `job_id -> tier_key` for promotion jobs. Increment a per-request/per-tier pending count when `_flush_pending_promotions()` submits a promotion. On completion decrement it; when it reaches zero, calculate elapsed time from first successful lookup to final promotion completion.

Calculate promoted token span without double-counting KV groups:

```python
def _token_span_for_keys(self, keys: Collection[OffloadKey]) -> int:
    per_group: dict[int, int] = {}
    for key in set(keys):
        group_idx = get_offload_group_idx(key)
        per_group[group_idx] = (
            per_group.get(group_idx, 0) + self._tokens_per_chunk_by_group[group_idx]
        )
    return max(per_group.values(), default=0)
```

This treats multiple KV groups as alternative representations of the same token span rather than summing duplicate model-layer coverage.

On a completed successful pure-tier wave:

```python
runtime_observation = self._cost_model.observe_secondary_promotion(
    tier_key,
    promoted_tokens,
    elapsed_seconds * 1000.0,
)
```

Store the elapsed seconds in request state for later provenance/debugging. The manager metrics for the observation are added in Task 4; do not add ad-hoc logging/Prometheus code in this task.

- [ ] **Step 8: Run provenance tests plus existing tiering/scheduler coverage and commit**

Run:

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
python -m compileall -q vllm/v1/kv_offload/tiering/manager.py
```

Expected: all targeted tests pass.

Commit:

```bash
git add vllm/v1/kv_offload/tiering/manager.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
git commit -m "feat: track KV restore provenance"
```

---

### Task 4: Shadow Scheduler Hook and Low-Cardinality Metrics

**Files:**
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py`
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
- Modify: `vllm/v1/kv_offload/tiering/manager.py`
- Modify: `tests/v1/kv_connector/unit/offloading_connector/utils.py`
- Modify: `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py`

**Interfaces:**
- Consumes: `OffloadingSpec.get_cost_model()`, manager `get_load_provenance()`, `ShadowDecision`, manager `RuntimeObservation`.
- Produces: aggregate Prometheus-compatible stats and request-level debug records; scheduler return values stay byte-for-byte semantically identical.

- [ ] **Step 1: Add failing metric-definition tests**

Extend existing offloading metric tests or add assertions in scheduler tests for these exact internal metric names:

```python
_ConnectorMetricName.COST_SHADOW_DECISIONS
_ConnectorMetricName.COST_PREDICTED_RESTORE
_ConnectorMetricName.COST_PREDICTED_RECOMPUTE
_ConnectorMetricName.COST_RUNTIME_SCALE
_ConnectorMetricName.COST_OBSERVATIONS
```

Define externally rendered metrics using seconds for time values to match existing vLLM conventions:

```text
vllm:kv_offload_cost_shadow_decisions
vllm:kv_offload_cost_predicted_restore_seconds
vllm:kv_offload_cost_predicted_recompute_seconds
vllm:kv_offload_cost_runtime_scale
vllm:kv_offload_cost_observations
```

Metadata:

```python
COST_SHADOW_DECISIONS: counter labels=("source", "preferred", "confidence")
COST_PREDICTED_RESTORE: histogram labels=("source",)
COST_PREDICTED_RECOMPUTE: histogram labels=("source",)
COST_RUNTIME_SCALE: gauge labels=("source", "token_bucket")
COST_OBSERVATIONS: counter labels=("source",)
```

Use bounded histogram buckets `(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)` seconds.

- [ ] **Step 2: Run targeted metrics/scheduler tests and verify RED**

Run:

```bash
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py -k 'cost or shadow'
```

Expected: missing metric constants/definitions and missing scheduler behavior.

- [ ] **Step 3: Register metrics using existing connector plumbing**

Add the constants to `_ConnectorMetricName` and definitions to `get_connector_metric_definitions()`. Do not create a new Prometheus class or direct global metric singleton. `OffloadingConnectorStats` already supports labeled counters, gauges, and histograms.

- [ ] **Step 4: Make the test `MockOffloadingSpec` expose the cost model**

In `MockOffloadingSpec.__init__`:

```python
self.cost_model = OffloadCostModel.from_extra_config(self.extra_config)
```

Override:

```python
def get_cost_model(self) -> OffloadCostModel | None:
    return self.cost_model
```

Because `MagicMock(spec=OffloadingManager)` now includes `get_load_provenance`, scheduler tests can configure the exact provenance without replacing the manager type.

- [ ] **Step 5: Write the scheduler invariance test before implementation**

Create one disabled runner and one shadow runner with identical block sizing, prompt, mocked hit behavior, and load result. For shadow, configure a small profile such as recompute 100 ms versus CPU restore 20 ms and return CPU provenance.

Verify both actual paths load the same GPU blocks and that shadow adds only decision stats:

```python
assert disabled_loaded == shadow_loaded
assert disabled_num_hit_tokens == shadow_num_hit_tokens
assert disabled_async_flag == shadow_async_flag

reduced = _reduce_kv_connector_stats(shadow_runner)
assert reduced[
    f'{_ConnectorMetricName.COST_SHADOW_DECISIONS}:("cpu_primary", "restore", "high")'
] == 1
```

Also assert shadow disabled never calls `manager.get_load_provenance()`.

Add a filesystem provenance test whose configured shadow preference is `recompute` while `expected_loaded=...` proves the actual restore still executes.

- [ ] **Step 6: Add a pure matched-key helper and shadow hook**

In scheduler `__init__`:

```python
self._cost_model = spec.get_cost_model()
```

Add a side-effect-free helper that mirrors the final `_lookup()` chunk bounds using `req_status.num_locally_computed_tokens`, `num_hit_tokens`, each group's `tokens_per_chunk`, and `group_state.offload_keys`:

```python
def _get_matched_external_keys(
    self,
    req_status: RequestOffloadState,
    num_hit_tokens: int,
) -> tuple[OffloadKey, ...]:
    num_cached_tokens = req_status.num_locally_computed_tokens + num_hit_tokens
    keys: list[OffloadKey] = []
    for group_config, group_state in zip(
        self.config.kv_group_configs, req_status.group_states
    ):
        start = req_status.num_locally_computed_tokens // group_config.tokens_per_chunk
        end = cdiv(num_cached_tokens, group_config.tokens_per_chunk)
        keys.extend(group_state.offload_keys[start:end])
    return tuple(keys)
```

After `_lookup()` has resolved and `update_num_hit_chunks()` has run:

```python
if self._cost_model is not None and num_hit_tokens is not None and num_hit_tokens > 0:
    matched_keys = self._get_matched_external_keys(req_status, num_hit_tokens)
    provenance = self.manager.get_load_provenance(
        matched_keys, req_status.req_context, num_hit_tokens
    )
    if provenance is not None:
        decision = self._cost_model.shadow_decide(provenance)
        if decision is not None:
            self._record_shadow_decision(request.request_id, provenance, decision)
```

Then preserve the existing tail exactly:

```python
self._touch(req_status)
return num_hit_tokens, bool(num_hit_tokens)
```

Do not use the decision to change `num_hit_tokens` or `skip_reading_prefix_cache`.

- [ ] **Step 7: Record decision stats and debug log**

`_record_shadow_decision()` emits:

```python
self._connector_stats.increase_counter(
    _ConnectorMetricName.COST_SHADOW_DECISIONS,
    labelvalues=(provenance.source, decision.preferred, decision.confidence),
)
self._connector_stats.observe_histogram(
    _ConnectorMetricName.COST_PREDICTED_RESTORE,
    decision.restore_estimate_ms / 1000.0,
    labelvalues=(provenance.source,),
)
self._connector_stats.observe_histogram(
    _ConnectorMetricName.COST_PREDICTED_RECOMPUTE,
    decision.recompute_estimate_ms / 1000.0,
    labelvalues=(provenance.source,),
)
```

Debug log fields: request ID, external tokens, source, restore seed/estimate, recompute estimate, runtime scale, preferred, confidence, `mode=shadow`, and `actual_path=restore`. The request ID appears only in the log message, never metric labels.

- [ ] **Step 8: Emit manager EWMA observation metrics**

When Task 3 gets a non-`None` `RuntimeObservation`, emit through `self._stats`:

```python
source = f"secondary:{tier_key}"
self._stats.increase_counter(
    _ConnectorMetricName.COST_OBSERVATIONS,
    labelvalues=(source,),
)
self._stats.set_gauge(
    _ConnectorMetricName.COST_RUNTIME_SCALE,
    runtime_observation.runtime_scale,
    labelvalues=(source, str(runtime_observation.token_bucket)),
)
```

No request ID labels and no unbounded path/device strings.

- [ ] **Step 9: Run scheduler/metric tests and commit**

Run:

```bash
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
python -m pytest -q tests/v1/kv_offload/test_cost_model.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_spec.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
python -m compileall -q \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
  vllm/v1/kv_offload/tiering/manager.py
```

Expected: all targeted tests pass.

Commit:

```bash
git add \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
  vllm/v1/kv_offload/tiering/manager.py \
  tests/v1/kv_connector/unit/offloading_connector/utils.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
git commit -m "feat: record KV offload shadow decisions"
```

---

### Task 5: Full Source Verification and Shadow-Safety Regression Gate

**Files:**
- Modify only if verification exposes a defect in files already listed above.
- Test all touched offloading/tiering tests plus repository formatting/type checks that are available in the environment.

**Interfaces:**
- Consumes: Tasks 1-4 complete.
- Produces: evidence that shadow mode is default-off and behaviorally inert before hardware testing.

- [ ] **Step 1: Run the complete focused unit suite**

Run:

```bash
python -m pytest -q \
  tests/v1/kv_offload/test_cost_model.py \
  tests/v1/kv_offload/tiering \
  tests/v1/kv_connector/unit/offloading_connector
```

Expected: all tests pass. Do not weaken existing tests to make the new feature pass.

- [ ] **Step 2: Run compile checks**

```bash
python -m compileall -q \
  vllm/v1/kv_offload \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading
```

Expected: exit 0.

- [ ] **Step 3: Run repository lint/type hooks that are present**

First inspect available commands rather than assuming dependencies:

```bash
command -v pre-commit || true
command -v ruff || true
```

If `pre-commit` exists:

```bash
pre-commit run --files \
  vllm/v1/kv_offload/cost_model.py \
  vllm/v1/kv_offload/base.py \
  vllm/v1/kv_offload/tiering/spec.py \
  vllm/v1/kv_offload/tiering/manager.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
  tests/v1/kv_offload/test_cost_model.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_spec.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py \
  tests/v1/kv_connector/unit/offloading_connector/utils.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
```

If only `ruff` exists, run `ruff check` on the same Python files. Record unavailable tooling honestly; do not claim it ran.

- [ ] **Step 4: Inspect final diff for forbidden behavior changes**

Run:

```bash
git diff main...HEAD -- \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
  vllm/v1/kv_offload/tiering/manager.py
```

Reviewer checklist:

- `LookupResult` enum untouched.
- Existing `_lookup()` return values unchanged.
- `get_num_new_matched_tokens()` final return still `num_hit_tokens, bool(num_hit_tokens)`.
- No branch changes `skip_reading_prefix_cache` from the shadow decision.
- No shadow decision controls `_initiate_promotion`, `prepare_load`, allocation, or transfer job creation.
- `mode: enforce` is not implemented.
- No hardware benchmark numbers exist as defaults in production source.

- [ ] **Step 5: Commit verification-only fixes if needed**

If no fixes were needed, do not create an empty commit. If a verification defect required a scoped change, rerun the failing check plus the complete focused unit suite, then commit only that fix:

```bash
git add <only-files-changed-for-verification-fix>
git commit -m "fix: harden KV offload shadow model"
```

---

### Task 6: Hardware Shadow Validation on the Existing Crossover Workload

**Files:**
- Do not modify runtime branch source for this task.
- Use a disposable local integration branch/worktree that combines `feature/kv-offload-shadow-cost-model` with `feature/cache-eviction-restore-benchmark` only for hardware validation.

**Interfaces:**
- Consumes: verified runtime branch plus PR #5 benchmark workload.
- Produces: CPU-primary and filesystem shadow-decision evidence while actual restore behavior remains unchanged.

- [ ] **Step 1: Create a disposable integration checkout without polluting either PR**

On the hardware pod after fetching both branches:

```bash
git fetch origin main feature/kv-offload-shadow-cost-model feature/cache-eviction-restore-benchmark

git switch -C validation/kv-offload-shadow-cost-model \
  origin/feature/kv-offload-shadow-cost-model

git merge --no-edit origin/feature/cache-eviction-restore-benchmark
```

This branch is validation-only. Do not push it as the runtime PR and do not merge its benchmark commits back into `feature/kv-offload-shadow-cost-model`.

- [ ] **Step 2: Add the measured profile only to the disposable benchmark config**

In the validation checkout, edit `benchmarks/cache/configs/local-crossover.yaml` server connector extra config to include:

```yaml
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

Set the filesystem secondary tier's `cost_model_tier_key: filesystem` next to its existing `type`. These values remain validation config only.

- [ ] **Step 3: CPU-primary 1024 anchor**

Use the previously proven CPU-control shape: GPU KV fixed 2 GiB, CPU primary 4 GiB, no secondary promotion needed for the victims. Run only the 1024 eviction/restore case.

Expected:

```text
shadow preferred = restore
actual path = restore
secondary cost observation count = 0
```

Also verify the same 8 CPU->GPU loads and approximately the same external token count as the prior control; shadow does not change transfer bytes.

- [ ] **Step 4: Filesystem 1024 anchor**

Use GPU KV 2 GiB + CPU primary 2 GiB + filesystem secondary and run only p1024.

Expected:

```text
shadow preferred = recompute
actual path = restore
```

Evidence requirements:

- external KV transfer remains approximately the full victim prefix;
- 8 CPU->GPU restore operations remain present;
- secondary async/promotion activity remains present;
- `vllm:kv_offload_cost_shadow_decisions{source="secondary:filesystem",preferred="recompute",confidence="high"}` increments;
- `vllm:kv_offload_cost_observations{source="secondary:filesystem"}` increments;
- runtime scale gauge for the 1024 bucket appears.

- [ ] **Step 5: Filesystem 256-4096 sweep**

Run only the five tiered-fs eviction/restore cases. Expected shadow preference for every measured point:

```text
256  -> recompute
512  -> recompute
1024 -> recompute
2048 -> recompute
4096 -> recompute
```

The actual path must still restore in every completed case. Compare external token totals and CPU->GPU bytes to the previously measured baseline; material differences indicate a shadow-invariance bug.

- [ ] **Step 6: Check instrumentation overhead and summarize evidence**

Compare P95 TTFT to the prior baseline values:

```text
256:  31.119 ms
512:  56.979 ms
1024: 108.132 ms
2048: 244.266 ms
4096: 651.127 ms
```

Do not require an arbitrary fixed percentage pass/fail threshold from one run. Treat a consistent regression outside normal run-to-run noise as a reason to repeat the affected single case and investigate instrumentation overhead before enabling enforcement work.

Record for each case:

```text
prompt_tokens
actual_p95_ttft_ms
external_tokens
cpu_to_gpu_bytes
shadow_source
shadow_preferred
shadow_confidence
runtime_scale_bucket
runtime_scale
actual_path
```

- [ ] **Step 7: Return to the runtime branch and keep validation-only edits out of the PR**

```bash
git switch feature/kv-offload-shadow-cost-model
git status --short
```

Expected: clean runtime branch; no benchmark config or PR #5 files added to its diff.

---

## Final Verification Before Completion

Before claiming the shadow implementation is complete, use `superpowers:verification-before-completion` and collect fresh evidence for:

```bash
python -m pytest -q \
  tests/v1/kv_offload/test_cost_model.py \
  tests/v1/kv_offload/tiering \
  tests/v1/kv_connector/unit/offloading_connector

python -m compileall -q \
  vllm/v1/kv_offload \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading

git diff --check main...HEAD
git status --short
```

Then verify hardware anchors and the five filesystem shadow decisions from Task 6. If native pre-commit/ruff is unavailable or fails for unrelated pre-existing repository debt, report the exact command and failure rather than calling the branch fully green.
