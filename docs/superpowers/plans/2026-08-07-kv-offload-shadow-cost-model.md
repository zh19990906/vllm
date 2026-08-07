# KV Offload Shadow Cost Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in shadow cost model that distinguishes CPU-primary restores from secondary-tier promotions, predicts restore versus recompute cost, calibrates secondary promotion cost online, and records the decision without changing the actual request path.

**Architecture:** Keep cost calculation as pure logic in `vllm/v1/kv_offload/cost_model.py`. `TieringOffloadingManager` remains the source of truth for logical load provenance and promotion observations, while `OffloadingConnectorScheduler` asks for the final matched-prefix provenance and records a shadow decision after `_lookup()` resolves. Reuse the existing `OffloadingConnectorStats`/Prometheus plumbing; do not add a second metrics transport.

**Tech Stack:** Python 3.11, dataclasses, vLLM V1 KV offloading/tiering interfaces, `OffloadingConnectorStats`, pytest, `unittest.mock`, and the existing Qwen2.5-7B cache benchmark suite for hardware validation.

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
- Missing provenance or missing curve coverage skips the decision or lowers confidence; it never fails the request.
- `mode: enforce` is out of scope and must fail configuration validation in this implementation.
- Runtime feature branch is based on `main` after merged PR #6 (`bcbb26fa8ed90d2bd1de57f70168ec3d188c8c9c`); do not merge benchmark PR #5 into the runtime branch.

---

## File Structure

- Create `vllm/v1/kv_offload/cost_model.py` — pure config parsing, curves, provenance/result types, shadow decisions, buckets, and EWMA.
- Create `tests/v1/kv_offload/test_cost_model.py` — pure unit coverage.
- Modify `vllm/v1/kv_offload/base.py` — default no-op provenance/cost-model interfaces.
- Modify `vllm/v1/kv_offload/tiering/spec.py` — parse one shared model and resolve stable secondary-tier keys.
- Modify `vllm/v1/kv_offload/tiering/manager.py` — source provenance, promotion-wave timing, EWMA observations, lifecycle cleanup.
- Create `tests/v1/kv_offload/tiering/test_shadow_cost_spec.py` — base/spec compatibility and stable tier-key tests.
- Create `tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py` — manager provenance/observation tests with mocks only.
- Modify `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py` — bounded-label metric definitions.
- Modify `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` — inert shadow hook and debug record.
- Modify `tests/v1/kv_connector/unit/offloading_connector/utils.py` — expose the opt-in model from `MockOffloadingSpec`.
- Modify `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` — execution-invariance and shadow-metric tests.

---

### Task 1: Pure Cost Model, Configuration, and EWMA

**Files:**
- Create: `vllm/v1/kv_offload/cost_model.py`
- Create: `tests/v1/kv_offload/test_cost_model.py`

**Interfaces:**
- Produces `Confidence = Literal["high", "low"]` and `PreferredPath = Literal["restore", "recompute"]`.
- Produces immutable dataclasses:
  - `CurveEstimate(value_ms: float, confidence: Confidence)`
  - `LoadProvenance(source: str, external_tokens: int, secondary_promoted_tokens: int | None, sources: tuple[str, ...], confidence: Confidence, lookup_sync_seconds: float | None = None, lookup_async_seconds: float | None = None)`
  - `ShadowDecision(preferred: PreferredPath, restore_seed_ms: float, restore_estimate_ms: float, recompute_estimate_ms: float, runtime_scale: float, confidence: Confidence)`
  - `RuntimeObservation(tier_key: str, token_bucket: int, observed_ms: float, seeded_ms: float, sample_scale: float, runtime_scale: float)`
- Produces `CostCurve.from_mapping()`, `CostCurve.estimate()`, `CostCurve.bucket_for()`.
- Produces `OffloadCostModel.from_extra_config()`, `shadow_decide()`, and `observe_secondary_promotion()`.

- [ ] **Step 1: Write RED tests for curve semantics and default-off config**

Create `tests/v1/kv_offload/test_cost_model.py` with these concrete tests first:

```python
import pytest

from vllm.v1.kv_offload.cost_model import CostCurve, OffloadCostModel


def test_cost_curve_exact_interpolation_and_extrapolation():
    curve = CostCurve.from_mapping({256: 20.0, 512: 40.0, 1024: 80.0})
    assert curve.estimate(512).value_ms == pytest.approx(40.0)
    assert curve.estimate(512).confidence == "high"
    assert curve.estimate(768).value_ms == pytest.approx(60.0)
    assert curve.estimate(768).confidence == "high"
    assert curve.estimate(128).value_ms == pytest.approx(10.0)
    assert curve.estimate(128).confidence == "low"
    assert curve.estimate(2048).value_ms == pytest.approx(160.0)
    assert curve.estimate(2048).confidence == "low"


def test_single_point_curve_is_exact_only_at_sample():
    curve = CostCurve.from_mapping({1024: 24.49})
    assert curve.estimate(1024).value_ms == pytest.approx(24.49)
    assert curve.estimate(1024).confidence == "high"
    assert curve.estimate(2048).value_ms == pytest.approx(48.98)
    assert curve.estimate(2048).confidence == "low"


def test_shadow_model_is_off_by_default():
    assert OffloadCostModel.from_extra_config({}) is None
    assert OffloadCostModel.from_extra_config(
        {"cache_cost_model": {"mode": "off"}}
    ) is None
```

Add a parameterized validation test containing these invalid configs: shadow with missing profile, empty `recompute_ms`, token sample `0`, latency `0.0`, latency `float("inf")`, `ewma_alpha=0.0`, `ewma_alpha=1.1`, clamp min `0.0`, clamp min greater than max, and `mode="enforce"`. Each must raise `ValueError`.

- [ ] **Step 2: Run RED test**

```bash
python -m pytest -q tests/v1/kv_offload/test_cost_model.py
```

Expected: import/collection failure because `vllm.v1.kv_offload.cost_model` does not exist.

- [ ] **Step 3: Implement strict parsing and curve math**

Create the module using only stdlib imports. `CostCurve.from_mapping()` must coerce integer-like string keys such as `"1024"`, reject booleans, require positive integer tokens, require finite positive numeric latency, sort by token count, and reject duplicate token counts after coercion.

Use these exact estimate rules:

```python
# Exact sample: measured value, high confidence.
# Between two samples: y0 + (tokens - x0) * (y1 - y0) / (x1 - x0), high confidence.
# Below range: first_ms * tokens / first_tokens, low confidence.
# Above range: last_ms * tokens / last_tokens, low confidence.
```

`bucket_for(tokens)` uses the first configured sample token count greater than or equal to `tokens`, otherwise the last sample token count.

`OffloadCostModel.from_extra_config()` reads `extra_config["cache_cost_model"]`. Only `off` and `shadow` are accepted; `shadow` requires `profile.recompute_ms` plus at least one `profile.tiers.<tier>.restore_ms`. `promotion_ms` is optional per tier. Defaults are exactly `ewma_alpha=0.2`, `sample_scale_min=0.25`, `sample_scale_max=4.0`; benchmark curves have no defaults.

- [ ] **Step 4: Add RED decision and EWMA tests**

Use this test-only profile:

```python
PROFILE = {
    "cache_cost_model": {
        "mode": "shadow",
        "ewma_alpha": 0.2,
        "sample_scale_min": 0.25,
        "sample_scale_max": 4.0,
        "profile": {
            "recompute_ms": {
                256: 26.414,
                512: 44.961,
                1024: 81.705,
                2048: 152.461,
                4096: 308.424,
            },
            "tiers": {
                "cpu_primary": {"restore_ms": {1024: 24.490}},
                "filesystem": {
                    "restore_ms": {
                        256: 31.119,
                        512: 56.979,
                        1024: 108.132,
                        2048: 244.266,
                        4096: 651.127,
                    },
                    "promotion_ms": {
                        256: 13.916,
                        512: 35.230,
                        1024: 81.458,
                        2048: 171.505,
                        4096: 498.874,
                    },
                },
            },
        },
    }
}
```

Add these full assertions:

```python
from vllm.v1.kv_offload.cost_model import LoadProvenance


def test_cpu_1024_prefers_restore():
    model = OffloadCostModel.from_extra_config(PROFILE)
    assert model is not None
    decision = model.shadow_decide(
        LoadProvenance("cpu_primary", 1024, 0, ("cpu_primary",), "high")
    )
    assert decision is not None
    assert decision.preferred == "restore"
    assert decision.restore_estimate_ms == pytest.approx(24.490)
    assert decision.recompute_estimate_ms == pytest.approx(81.705)
    assert decision.runtime_scale == pytest.approx(1.0)


@pytest.mark.parametrize("tokens", [256, 512, 1024, 2048, 4096])
def test_filesystem_samples_prefer_recompute(tokens):
    model = OffloadCostModel.from_extra_config(PROFILE)
    assert model is not None
    decision = model.shadow_decide(
        LoadProvenance(
            "secondary:filesystem",
            tokens,
            tokens,
            ("secondary:filesystem",),
            "high",
        )
    )
    assert decision is not None
    assert decision.preferred == "recompute"
    assert decision.confidence == "high"


def test_equal_cost_prefers_recompute():
    config = {
        "cache_cost_model": {
            "mode": "shadow",
            "profile": {
                "recompute_ms": {1024: 50.0},
                "tiers": {"cpu_primary": {"restore_ms": {1024: 50.0}}},
            },
        }
    }
    model = OffloadCostModel.from_extra_config(config)
    assert model is not None
    decision = model.shadow_decide(
        LoadProvenance("cpu_primary", 1024, 0, ("cpu_primary",), "high")
    )
    assert decision is not None
    assert decision.preferred == "recompute"


def test_ewma_updates_only_matching_tier_bucket():
    model = OffloadCostModel.from_extra_config(PROFILE)
    assert model is not None
    observation = model.observe_secondary_promotion("filesystem", 1024, 162.916)
    assert observation is not None
    assert observation.token_bucket == 1024
    assert observation.sample_scale == pytest.approx(2.0)
    assert observation.runtime_scale == pytest.approx(1.2)
    untouched = model.shadow_decide(
        LoadProvenance(
            "secondary:filesystem", 512, 512, ("secondary:filesystem",), "high"
        )
    )
    updated = model.shadow_decide(
        LoadProvenance(
            "secondary:filesystem", 1024, 1024, ("secondary:filesystem",), "high"
        )
    )
    assert untouched is not None and updated is not None
    assert untouched.runtime_scale == pytest.approx(1.0)
    assert updated.runtime_scale == pytest.approx(1.2)


def test_ewma_clamps_sample_scale_before_update():
    model = OffloadCostModel.from_extra_config(PROFILE)
    assert model is not None
    observation = model.observe_secondary_promotion("filesystem", 1024, 8145.8)
    assert observation is not None
    assert observation.sample_scale == pytest.approx(4.0)
    assert observation.runtime_scale == pytest.approx(1.6)


def test_mixed_sources_use_conservative_max_and_low_confidence():
    model = OffloadCostModel.from_extra_config(PROFILE)
    assert model is not None
    decision = model.shadow_decide(
        LoadProvenance(
            "mixed",
            1024,
            None,
            ("cpu_primary", "secondary:filesystem"),
            "low",
        )
    )
    assert decision is not None
    assert decision.restore_seed_ms == pytest.approx(108.132)
    assert decision.preferred == "recompute"
    assert decision.confidence == "low"
```

For `mixed`, estimate every component at the full external token count, choose the maximum restore estimate, and force low confidence. If any component lacks a restore curve, return `None`.

- [ ] **Step 5: Run RED for missing decision/EWMA implementation**

```bash
python -m pytest -q tests/v1/kv_offload/test_cost_model.py
```

Expected: curve/config tests pass; decision/EWMA tests fail on missing implementation.

- [ ] **Step 6: Implement decision and bounded EWMA**

Source-to-profile mapping is exactly:

```python
def _profile_key(source: str) -> str | None:
    if source == "cpu_primary":
        return "cpu_primary"
    if source.startswith("secondary:"):
        return source.removeprefix("secondary:")
    return None
```

For one secondary source, choose its runtime scale by `promotion_curve.bucket_for(external_tokens)`, defaulting to `1.0`. Preserve the approved v1 equation:

```python
restore_estimate_ms = restore_seed_ms * runtime_scale
```

EWMA is exactly:

```python
sample_scale = observed_ms / promotion_seed_ms
sample_scale = min(max(sample_scale, sample_scale_min), sample_scale_max)
new_scale = ewma_alpha * sample_scale + (1.0 - ewma_alpha) * old_scale
```

Return `RuntimeObservation(tier_key, bucket, observed_ms, promotion_seed_ms, sample_scale, new_scale)` from a successful observation. Return `None` when the tier has no `promotion_ms` curve.

- [ ] **Step 7: Verify and commit Task 1**

```bash
python -m pytest -q tests/v1/kv_offload/test_cost_model.py
python -m compileall -q vllm/v1/kv_offload/cost_model.py
git add vllm/v1/kv_offload/cost_model.py tests/v1/kv_offload/test_cost_model.py
git commit -m "feat: add KV offload shadow cost model"
```

Expected before commit: all Task 1 tests pass and compileall exits 0.

---

### Task 2: Base Interfaces and Tiering Spec Wiring

**Files:**
- Modify: `vllm/v1/kv_offload/base.py`
- Modify: `vllm/v1/kv_offload/tiering/spec.py`
- Create: `tests/v1/kv_offload/tiering/test_shadow_cost_spec.py`

**Interfaces:**
- `OffloadingManager.get_load_provenance(keys, req_context, external_tokens) -> LoadProvenance | None` defaults to `None`.
- `OffloadingSpec.get_cost_model() -> OffloadCostModel | None` defaults to `None`.
- `TieringOffloadingSpec.get_cost_model()` returns the exact shared model instance passed into the manager.

- [ ] **Step 1: Write RED default-interface and tier-key tests**

Use unbound default methods so no abstract fake class is needed:

```python
import pytest

from vllm.v1.kv_offload.base import OffloadingManager, OffloadingSpec, ReqContext
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


def test_base_shadow_interfaces_are_noop():
    assert OffloadingManager.get_load_provenance(
        object(), (), ReqContext("r"), 64
    ) is None
    assert OffloadingSpec.get_cost_model(object()) is None


def test_unique_secondary_types_default_to_type_key():
    keys = TieringOffloadingSpec._resolve_cost_model_tier_keys(
        [{"type": "fs"}, {"type": "network"}], enabled=True
    )
    assert keys == ("fs", "network")


def test_duplicate_types_require_explicit_cost_model_keys():
    with pytest.raises(ValueError, match="cost_model_tier_key"):
        TieringOffloadingSpec._resolve_cost_model_tier_keys(
            [{"type": "fs"}, {"type": "fs"}], enabled=True
        )


def test_explicit_keys_disambiguate_duplicate_types():
    keys = TieringOffloadingSpec._resolve_cost_model_tier_keys(
        [
            {"type": "fs", "cost_model_tier_key": "local_ssd"},
            {"type": "fs", "cost_model_tier_key": "slow_disk"},
        ],
        enabled=True,
    )
    assert keys == ("local_ssd", "slow_disk")
```

Also test empty/non-string explicit keys and duplicate explicit keys raise `ValueError` only when enabled. When disabled, the helper returns type-derived strings and does not add new validation failures to existing configs.

- [ ] **Step 2: Run RED**

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_spec.py
```

Expected: missing default methods/helper.

- [ ] **Step 3: Add non-abstract no-op interfaces in `base.py`**

Under `TYPE_CHECKING`, import `LoadProvenance` and `OffloadCostModel`. Add these concrete methods without changing any abstract requirements:

```python
def get_load_provenance(
    self,
    keys: Collection[OffloadKey],
    req_context: ReqContext,
    external_tokens: int,
) -> "LoadProvenance | None":
    return None
```

and on `OffloadingSpec`:

```python
def get_cost_model(self) -> "OffloadCostModel | None":
    return None
```

- [ ] **Step 4: Parse and share one model in `TieringOffloadingSpec`**

After `secondary_tier_configs` validation in `__init__`:

```python
self._cost_model = OffloadCostModel.from_extra_config(self.extra_config)
self._cost_model_tier_keys = self._resolve_cost_model_tier_keys(
    self.secondary_tier_configs,
    enabled=self._cost_model is not None,
)
```

Implement `_resolve_cost_model_tier_keys()` as a `@staticmethod`. When enabled, each key is `cost_model_tier_key` if provided, otherwise `type`; all keys must be non-empty strings and unique. If duplicate `type` values appear without explicit distinct keys, raise `ValueError` mentioning `cost_model_tier_key`.

Override:

```python
@override
def get_cost_model(self) -> OffloadCostModel | None:
    return self._cost_model
```

Pass the shared model and metadata into manager construction:

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

- [ ] **Step 5: Verify and commit Task 2**

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_spec.py
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
python -m compileall -q vllm/v1/kv_offload/base.py vllm/v1/kv_offload/tiering/spec.py
git add vllm/v1/kv_offload/base.py vllm/v1/kv_offload/tiering/spec.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_spec.py
git commit -m "feat: wire shadow cost model into tiering"
```

Expected before commit: new spec tests and existing scheduler tests pass.

---

### Task 3: Tiering Provenance and Promotion Observations

**Files:**
- Modify: `vllm/v1/kv_offload/tiering/manager.py`
- Create: `tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py`

**Interfaces:**
- Manager constructor gains keyword-only optional `cost_model`, `secondary_tier_keys`, and `tokens_per_chunk_by_group`, preserving existing callers.
- `get_load_provenance()` is read-only/idempotent.
- Successful promotion waves update the shared model once per completed request/tier wave; failed waves do not update it.

- [ ] **Step 1: Write RED provenance tests with mocks**

Use this deterministic helper in the test file:

```python
from unittest.mock import MagicMock

from vllm.v1.kv_offload.base import LookupResult, ReqContext, RequestOffloadingContext
from vllm.v1.kv_offload.cost_model import OffloadCostModel
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.tiering.base import SecondaryTierManager
from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager


def make_manager(model: OffloadCostModel):
    primary = MagicMock(spec=CPUOffloadingManager)
    secondary = MagicMock(spec=SecondaryTierManager)
    secondary.tier_type = "fs"
    secondary.get_finished_jobs.return_value = []
    secondary.on_new_request.return_value = RequestOffloadingContext()
    manager = TieringOffloadingManager(
        primary,
        [secondary],
        cost_model=model,
        secondary_tier_keys=("filesystem",),
        tokens_per_chunk_by_group=(64,),
    )
    return manager, primary, secondary
```

Use a one-point test model (`recompute_ms=100`, filesystem `restore_ms=200`, `promotion_ms=50`) and `make_offload_key(b"hash", 0)`.

Direct CPU test:

```python
primary.lookup.return_value = LookupResult.HIT
manager.on_new_request(ctx)
assert manager.lookup(key, ctx) is LookupResult.HIT
first = manager.get_load_provenance([key], ctx, 64)
second = manager.get_load_provenance([key], ctx, 64)
assert first == second
assert first is not None
assert first.source == "cpu_primary"
assert first.secondary_promoted_tokens == 0
```

Secondary persistence test:

```python
primary.lookup.side_effect = [LookupResult.MISS, LookupResult.HIT]
secondary.lookup.return_value = LookupResult.HIT
monkeypatch.setattr(manager, "_initiate_promotion", lambda tier, key, ctx: True)
manager.on_new_request(ctx)
assert manager.lookup(key, ctx) is LookupResult.RETRY
assert manager.lookup(key, ctx) is LookupResult.HIT
provenance = manager.get_load_provenance([key], ctx, 64)
assert provenance is not None
assert provenance.source == "secondary:filesystem"
assert provenance.secondary_promoted_tokens == 64
```

Add tests for failed promotion (returns `MISS`, no provenance), CPU+secondary mixed source, two secondary keys with different source strings, unknown selected key returning `None`, request finish cleanup, and reset clearing cost-specific state on active requests.

- [ ] **Step 2: Run RED**

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
```

Expected: constructor/provenance failures.

- [ ] **Step 3: Add cost-only state without default-off accounting**

Extend `RequestState` with nullable fields:

```python
key_sources: dict[OffloadKey, str] | None = None
promotion_started_at: dict[str, float] | None = None
promotion_keys: dict[str, set[OffloadKey]] | None = None
promotion_pending_jobs: dict[str, int] | None = None
promotion_elapsed_seconds: dict[str, float] | None = None
```

In `on_new_request()`, initialize these five dictionaries only when `self._cost_model is not None`.

Manager constructor signature becomes:

```python
def __init__(
    self,
    primary_tier: CPUPrimaryTierOffloadingManager,
    secondary_tiers: list[SecondaryTierManager] | None = None,
    *,
    cost_model: OffloadCostModel | None = None,
    secondary_tier_keys: tuple[str, ...] | None = None,
    tokens_per_chunk_by_group: tuple[int, ...] = (),
):
```

When enabled, assert `len(secondary_tier_keys) == len(self.secondary_tiers)` and save `{tier: key}`. Add `self._promotion_job_tier_keys: dict[JobId, str] = {}` only for observation bookkeeping.

- [ ] **Step 4: Preserve logical source across promotion**

For primary HIT:

```python
if primary_hit is LookupResult.HIT:
    if req_state is not None and req_state.key_sources is not None:
        req_state.key_sources.setdefault(key, "cpu_primary")
    return LookupResult.HIT
```

After a secondary reports HIT, call `_initiate_promotion()` exactly as today. Only when it succeeds, set:

```python
tier_key = self._secondary_tier_keys[tier]
req_state.key_sources[key] = f"secondary:{tier_key}"
req_state.promotion_started_at.setdefault(tier_key, lookup_start)
req_state.promotion_keys.setdefault(tier_key, set()).add(key)
```

Then return the same existing `LookupResult.RETRY`; if allocation fails, return the existing `MISS` and do not mark provenance.

- [ ] **Step 5: Implement idempotent `get_load_provenance()`**

The method must not mutate request state. Convert selected keys to a tuple, reject empty selections, and return `None` if any selected key lacks a recorded source.

Classification:

```text
all sources == cpu_primary                 -> source=cpu_primary, secondary_promoted_tokens=0, confidence=high
one unique source starting secondary:      -> that source, secondary_promoted_tokens=external_tokens, confidence=high
more than one unique source                -> source=mixed, secondary_promoted_tokens=None, confidence=low
```

Set `sources` to the sorted tuple of unique source strings. For pure secondary, if `promotion_elapsed_seconds[tier_key]` exists, expose it as `lookup_async_seconds`; otherwise leave timing fields `None`.

On `reset_cache()`, clear the five cost-only dictionaries for every retained active request after pending promotions have been drained/invalidated. Keep the existing non-cost `RequestState` lifecycle unchanged. Finished request deletion already removes provenance.

- [ ] **Step 6: Add RED promotion-wave EWMA tests**

Use the actual promotion path with `CPULoadStoreSpec`:

```python
from unittest.mock import MagicMock

from vllm.v1.kv_offload.base import PrepareStoreOutput
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.tiering.base import JobResult

primary.lookup.return_value = LookupResult.MISS
primary.prepare_write.return_value = PrepareStoreOutput(
    keys_to_store=[key],
    store_spec=CPULoadStoreSpec([3]),
    evicted_keys=[],
)
secondary.lookup.return_value = LookupResult.HIT
manager.on_new_request(ctx)

real_observe = model.observe_secondary_promotion
spy = MagicMock(side_effect=real_observe)
model.observe_secondary_promotion = spy
```

Monkeypatch `time.monotonic` so the first secondary lookup starts at `100.000`, sync accounting observes `100.001`, and promotion completion observes `100.050`. Call `manager.lookup(key, ctx)`, then `_flush_pending_promotions()`. Configure `secondary.get_finished_jobs.return_value = [JobResult(job_id=0, success=True)]` and call `_process_finished_jobs()`.

Assert:

```python
spy.assert_called_once()
tier_key, tokens, observed_ms = spy.call_args.args
assert tier_key == "filesystem"
assert tokens == 64
assert observed_ms == pytest.approx(50.0)
```

Repeat with `success=False`; assert `spy.assert_not_called()` and `get_load_provenance([key], ctx, 64) is None`.

- [ ] **Step 7: Implement promotion-wave timing/token span**

When `_flush_pending_promotions()` creates a promotion `JobMetadata`, store its tier key in `_promotion_job_tier_keys`, increment `state.promotion_pending_jobs[tier_key]`, and keep all promoted keys in `state.promotion_keys[tier_key]`.

On promotion completion, decrement the count. Failed completion removes each failed job key from `state.key_sources`. When the count reaches zero on success, calculate token span by grouping keys by `get_offload_group_idx()` and taking the maximum per-group span:

```python
def _token_span_for_keys(self, keys: Collection[OffloadKey]) -> int:
    per_group: dict[int, int] = {}
    for key in set(keys):
        group_idx = get_offload_group_idx(key)
        per_group[group_idx] = (
            per_group.get(group_idx, 0)
            + self._tokens_per_chunk_by_group[group_idx]
        )
    return max(per_group.values(), default=0)
```

Elapsed time is `time.monotonic() - state.promotion_started_at[tier_key]`. Save it in `promotion_elapsed_seconds`, call `observe_secondary_promotion(tier_key, token_span, elapsed * 1000.0)`, then clear the completed wave's started-at, key set, and pending counter so a later wave starts fresh.

- [ ] **Step 8: Verify and commit Task 3**

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
python -m compileall -q vllm/v1/kv_offload/tiering/manager.py
git add vllm/v1/kv_offload/tiering/manager.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
git commit -m "feat: track KV restore provenance"
```

Expected before commit: all targeted tests pass.

---

### Task 4: Shadow Scheduler Hook and Metrics

**Files:**
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py`
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
- Modify: `vllm/v1/kv_offload/tiering/manager.py`
- Modify: `tests/v1/kv_connector/unit/offloading_connector/utils.py`
- Modify: `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py`

**Interfaces:**
- Scheduler reads the same shared model via `spec.get_cost_model()`.
- Manager runtime observations and scheduler shadow decisions both feed the existing `OffloadingConnectorStats` transport.

- [ ] **Step 1: Write RED metric-definition tests**

Add exact constants under `_ConnectorMetricName` and assert `get_connector_metric_definitions()` eventually contains them:

```text
COST_SHADOW_DECISIONS = vllm:kv_offload_cost_shadow_decisions
COST_PREDICTED_RESTORE = vllm:kv_offload_cost_predicted_restore_seconds
COST_PREDICTED_RECOMPUTE = vllm:kv_offload_cost_predicted_recompute_seconds
COST_RUNTIME_SCALE = vllm:kv_offload_cost_runtime_scale
COST_OBSERVATIONS = vllm:kv_offload_cost_observations
```

Metadata is fixed:

```text
shadow decisions: counter labels source, preferred, confidence
predicted restore: histogram label source
predicted recompute: histogram label source
runtime scale: gauge labels source, token_bucket
observations: counter label source
```

Prediction histogram buckets are `(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)` seconds.

- [ ] **Step 2: Run RED metric test selection**

```bash
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py -k 'cost or shadow'
```

Expected: no matching tests yet or failures once the new RED tests are added; metric constants/definitions are absent.

- [ ] **Step 3: Register metrics through existing plumbing**

Add the five constants to `_ConnectorMetricName` and return `OffloadingCounterMetadata`, `OffloadingGaugeMetadata`, or `OffloadingHistogramMetadata` from `get_connector_metric_definitions()` with the label tuples above. Do not create direct Prometheus globals; `OffloadPromMetrics` already creates/binds definitions from metadata.

- [ ] **Step 4: Make `MockOffloadingSpec` expose the model**

Import `OffloadCostModel` in `tests/.../utils.py`. At the end of `MockOffloadingSpec.__init__`:

```python
self.cost_model = OffloadCostModel.from_extra_config(self.extra_config)
```

Add:

```python
def get_cost_model(self) -> OffloadCostModel | None:
    return self.cost_model
```

- [ ] **Step 5: Write RED scheduler-invariance tests**

Add a helper profile in `test_scheduler.py`:

```python
def _shadow_extra_config(source: str) -> dict:
    tier_key = "cpu_primary" if source == "cpu_primary" else "filesystem"
    restore = 20.0 if tier_key == "cpu_primary" else 200.0
    return {
        "cache_cost_model": {
            "mode": "shadow",
            "profile": {
                "recompute_ms": {12: 100.0},
                "tiers": {tier_key: {"restore_ms": {12: restore}}},
            },
        }
    }
```

For a 12-token chunk (`block_size=4`, `blocks_per_chunk=3`), create the same warmed-hit runner shape used by existing `expected_loaded=(0, 1, 2)` tests. Wrap `get_num_new_matched_tokens` to record its return values without changing them:

```python
matched_results: list[tuple[int | None, bool]] = []
original = runner.connector_scheduler.get_num_new_matched_tokens


def recording_get_num_new_matched_tokens(request, num_computed_tokens):
    result = original(request, num_computed_tokens)
    matched_results.append(result)
    return result

runner.connector_scheduler.get_num_new_matched_tokens = (
    recording_get_num_new_matched_tokens
)
```

For shadow CPU provenance, configure:

```python
runner.manager.get_load_provenance.return_value = LoadProvenance(
    "cpu_primary", 12, 0, ("cpu_primary",), "high"
)
runner.connector_scheduler._maximal_prefix_lookup = lambda keys, ctx: 1
runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=(0, 1, 2))
assert (12, True) in matched_results
```

Assert reduced stats contain:

```python
key = (
    f"{_ConnectorMetricName.COST_SHADOW_DECISIONS}:"
    "('cpu_primary', 'restore', 'high')"
)
assert _reduce_kv_connector_stats(runner)[key] == 1
```

Add a second test with `source="secondary:filesystem"`, restore seed `200.0`, recompute seed `100.0`; expected shadow preference is recompute while `expected_loaded=(0, 1, 2)` still proves actual restore.

Add a default-off test that runs the same warmed hit without `cache_cost_model` and asserts `runner.manager.get_load_provenance.assert_not_called()`.

- [ ] **Step 6: Implement side-effect-free matched-key helper and shadow hook**

In scheduler `__init__`:

```python
self._cost_model = spec.get_cost_model()
```

Add:

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
        start = (
            req_status.num_locally_computed_tokens
            // group_config.tokens_per_chunk
        )
        end = cdiv(num_cached_tokens, group_config.tokens_per_chunk)
        keys.extend(group_state.offload_keys[start:end])
    return tuple(keys)
```

After `req_status.update_num_hit_chunks(...)` and before `_touch()`:

```python
if self._cost_model is not None and num_hit_tokens is not None and num_hit_tokens > 0:
    matched_keys = self._get_matched_external_keys(req_status, num_hit_tokens)
    provenance = self.manager.get_load_provenance(
        matched_keys,
        req_status.req_context,
        num_hit_tokens,
    )
    if provenance is not None:
        decision = self._cost_model.shadow_decide(provenance)
        if decision is not None:
            self._record_shadow_decision(request.request_id, provenance, decision)
```

Keep the existing final two lines exactly:

```python
self._touch(req_status)
return num_hit_tokens, bool(num_hit_tokens)
```

- [ ] **Step 7: Emit shadow decision stats/debug log**

`_record_shadow_decision()` increases the decision counter and observes restore/recompute estimates after converting milliseconds to seconds. Labels are only `(source, preferred, confidence)` or `(source,)`.

Debug log must include these fields: request ID, external tokens, source, `restore_seed_ms`, `restore_estimate_ms`, `recompute_estimate_ms`, runtime scale, preferred path, confidence, `mode=shadow`, and `actual_path=restore`. The request ID is log text only.

- [ ] **Step 8: Emit manager observation metrics**

When Task 3 gets a non-`None` `RuntimeObservation`, emit:

```python
source = f"secondary:{runtime_observation.tier_key}"
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

No request IDs, filesystem paths, device names, or hashes become labels.

- [ ] **Step 9: Verify and commit Task 4**

```bash
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
python -m pytest -q tests/v1/kv_offload/test_cost_model.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_spec.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
python -m compileall -q \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
  vllm/v1/kv_offload/tiering/manager.py
git add \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
  vllm/v1/kv_offload/tiering/manager.py \
  tests/v1/kv_connector/unit/offloading_connector/utils.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
git commit -m "feat: record KV offload shadow decisions"
```

Expected before commit: all targeted tests pass.

---

### Task 5: Full Source Verification and Shadow-Safety Gate

**Files:**
- No new files.
- If a verification failure exposes a defect, return to Tasks 1-4, add a RED regression test in the owning task's test file, implement the smallest fix, and rerun this gate from the beginning.

- [ ] **Step 1: Run focused full unit coverage**

```bash
python -m pytest -q \
  tests/v1/kv_offload/test_cost_model.py \
  tests/v1/kv_offload/tiering \
  tests/v1/kv_connector/unit/offloading_connector
```

Expected: all tests pass.

- [ ] **Step 2: Compile touched subsystems**

```bash
python -m compileall -q \
  vllm/v1/kv_offload \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading
```

Expected: exit 0.

- [ ] **Step 3: Run available lint hooks without overstating unavailable tooling**

```bash
command -v pre-commit || true
command -v ruff || true
```

If `pre-commit` is present, run:

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

If `pre-commit` is absent and `ruff` is present, run `ruff check` over the same Python paths. If neither exists, record that exact fact; do not claim lint passed.

- [ ] **Step 4: Inspect forbidden behavior changes**

```bash
git diff main...HEAD -- \
  vllm/v1/kv_offload/base.py \
  vllm/v1/kv_offload/tiering/manager.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
```

Reviewer gate:

- `LookupResult` enum has no new members.
- Existing tiering `lookup()` result branches still return the same HIT/HIT_PENDING/RETRY/MISS values.
- `get_num_new_matched_tokens()` still ends with `return num_hit_tokens, bool(num_hit_tokens)`.
- No decision writes `skip_reading_prefix_cache`.
- No decision controls `_initiate_promotion`, `prepare_load`, allocation, or transfer job construction.
- `mode: enforce` is rejected by config parsing.
- Measured Qwen benchmark numbers appear only in tests/docs/validation config, not production defaults.

- [ ] **Step 5: Check patch hygiene**

```bash
git diff --check main...HEAD
git status --short
```

Expected: `git diff --check` exits 0. Working tree should contain only intentional changes; before moving to hardware validation, commit any intentional source/test change through its owning Task 1-4 commit cycle rather than making an unreviewed verification-only patch.

---

### Task 6: Hardware Shadow Validation with PR #5 Workload

**Files:**
- Do not add benchmark files to the runtime branch.
- Use a disposable local validation branch that combines the runtime branch with `feature/cache-eviction-restore-benchmark`.

- [ ] **Step 1: Create the disposable integration branch on the hardware pod**

```bash
cd /code/vllm
git fetch origin main feature/kv-offload-shadow-cost-model feature/cache-eviction-restore-benchmark
git switch -C validation/kv-offload-shadow-cost-model \
  origin/feature/kv-offload-shadow-cost-model
git merge --no-edit origin/feature/cache-eviction-restore-benchmark
```

Do not push this validation branch as the runtime PR.

- [ ] **Step 2: Add validation-only model injection to the benchmark checkout**

PR #5's suite builds `kv_connector_extra_config` in `benchmarks/cache/scenarios.py`, so `local-crossover.yaml` alone cannot inject a new connector-specific dictionary. In the disposable branch only, add this immediately after `_offloading_config()` creates `extra`:

```python
raw_cost_model = os.environ.get("VLLM_CACHE_COST_MODEL_JSON")
if raw_cost_model:
    extra["cache_cost_model"] = json.loads(raw_cost_model)
```

In the tiered-fs secondary dict, add:

```python
"cost_model_tier_key": "filesystem",
```

Do not commit these two validation-only benchmark edits to `feature/kv-offload-shadow-cost-model`.

- [ ] **Step 3: Export the measured seed profile**

```bash
export VLLM_CACHE_COST_MODEL_JSON='{"mode":"shadow","ewma_alpha":0.2,"sample_scale_min":0.25,"sample_scale_max":4.0,"profile":{"recompute_ms":{"256":26.414,"512":44.961,"1024":81.705,"2048":152.461,"4096":308.424},"tiers":{"cpu_primary":{"restore_ms":{"1024":24.49}},"filesystem":{"restore_ms":{"256":31.119,"512":56.979,"1024":108.132,"2048":244.266,"4096":651.127},"promotion_ms":{"256":13.916,"512":35.23,"1024":81.458,"2048":171.505,"4096":498.874}}}}}'
```

- [ ] **Step 4: CPU-primary 1024 anchor using TieringOffloadingManager**

Keep tiered-fs enabled but temporarily set `cache.cpu_bytes_to_use` in `benchmarks/cache/configs/local-crossover.yaml` to `4294967296` (4 GiB). Run only:

```bash
python benchmarks/cache/run_suite.py \
  --config benchmarks/cache/configs/local-crossover.yaml \
  --case-id tiered-fs__eviction-restore__p1024__r0.000__c1__qinf__18e9853b
```

If the regenerated case ID differs because the branch's case hash changed, obtain the exact ID first with:

```bash
python benchmarks/cache/run_suite.py \
  --config benchmarks/cache/configs/local-crossover.yaml \
  --dry-run
```

and select the single `tiered-fs`, `eviction-restore`, `prompt_tokens=1024` case from the generated `scenarios.json`.

Expected evidence:

```text
shadow preferred = restore
actual path = restore
source = cpu_primary
secondary cost observation count = 0
```

Also verify 8 CPU-to-GPU loads and external token volume remain consistent with the prior CPU-primary control.

- [ ] **Step 5: Filesystem 1024 anchor**

Restore `cache.cpu_bytes_to_use` to `2147483648` (2 GiB), keep GPU KV fixed at 2 GiB, and run the single tiered-fs p1024 eviction/restore case.

Expected:

```text
shadow preferred = recompute
actual path = restore
source = secondary:filesystem
```

Prometheus delta must show one or more `vllm:kv_offload_cost_observations{source="secondary:filesystem"}` observations and a `vllm:kv_offload_cost_runtime_scale{source="secondary:filesystem",token_bucket="1024"}` gauge. Existing external-prefix and CPU-to-GPU byte evidence must remain consistent with the pre-shadow baseline.

- [ ] **Step 6: Run the five filesystem cases**

Run the tiered-fs eviction/restore cases at `256, 512, 1024, 2048, 4096` tokens. Expected shadow preference is `recompute` for all five; actual path remains restore.

Record:

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

Compare P95 TTFT against prior filesystem baselines `31.119, 56.979, 108.132, 244.266, 651.127 ms`. Do not invent a one-run percentage threshold; repeat only an affected single case if instrumentation shows a consistent regression outside ordinary run-to-run noise.

- [ ] **Step 7: Leave validation-only benchmark edits behind**

```bash
git switch feature/kv-offload-shadow-cost-model
git status --short
git diff --name-only main...HEAD
```

Expected: the runtime branch contains only the runtime/spec/plan/test files from Tasks 1-5; no `benchmarks/cache/scenarios.py` or `local-crossover.yaml` change appears in its diff.

---

## Final Verification Before Completion

Before claiming completion, invoke `superpowers:verification-before-completion` and collect fresh output from:

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

Then include Task 6's CPU-primary anchor, filesystem anchor, and five filesystem shadow decisions in the verification evidence. If pre-commit/ruff is unavailable or fails for unrelated pre-existing repository debt, report the exact command and output; do not describe the branch as fully green on that check.
