# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "cost_model.py"
_SPEC = importlib.util.spec_from_file_location(
    "vllm_cost_model_under_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_COST_MODEL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _COST_MODEL
_SPEC.loader.exec_module(_COST_MODEL)

CostCurve = _COST_MODEL.CostCurve
CurveEstimate = _COST_MODEL.CurveEstimate
LoadProvenance = _COST_MODEL.LoadProvenance
OffloadCostModel = _COST_MODEL.OffloadCostModel


PROFILE: dict[str, Any] = {
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


CALIBRATED_P95_PROFILE = {
    "cache_cost_model": {
        "mode": "shadow",
        "ewma_alpha": 0.2,
        "sample_scale_min": 0.25,
        "sample_scale_max": 4.0,
        "profile": {
            "recompute_ms": {
                104: 19.660,
                168: 22.186,
                192: 25.082,
                232: 26.663,
                512: 44.813,
                1024: 81.258,
                2016: 152.433,
                4088: 309.140,
            },
            "tiers": {
                "cpu_primary": {
                    "restore_ms": {
                        104: 21.220,
                        168: 21.830,
                        192: 22.212,
                        232: 21.872,
                        512: 23.057,
                        1024: 24.687,
                        2016: 29.173,
                        4088: 35.213,
                    }
                },
                "filesystem": {
                    "restore_ms": {
                        232: 36.007,
                        512: 59.159,
                        1024: 101.799,
                        2016: 320.793,
                        4088: 648.235,
                    },
                    "promotion_ms": PROFILE["cache_cost_model"]["profile"]["tiers"][
                        "filesystem"
                    ]["promotion_ms"],
                },
            },
        },
    }
}


def test_curve_exact_interpolation_and_outside_confidence() -> None:
    curve = CostCurve.from_mapping({256: 20.0, 512: 40.0, 1024: 80.0})

    assert curve.estimate(512) == CurveEstimate(40.0, "high")
    assert curve.estimate(768).value_ms == pytest.approx(60.0)
    assert curve.estimate(768).confidence == "high"
    assert curve.estimate(128).value_ms == pytest.approx(10.0)
    assert curve.estimate(128).confidence == "low"
    assert curve.estimate(2048).value_ms == pytest.approx(160.0)
    assert curve.estimate(2048).confidence == "low"


def test_single_point_curve_scales_with_low_confidence_off_sample() -> None:
    curve = CostCurve.from_mapping({1024: 24.49})

    assert curve.estimate(1024).value_ms == pytest.approx(24.49)
    assert curve.estimate(1024).confidence == "high"
    assert curve.estimate(2048).value_ms == pytest.approx(48.98)
    assert curve.estimate(2048).confidence == "low"


def test_bucket_uses_ceiling_sample_and_last_sample_above_range() -> None:
    curve = CostCurve.from_mapping({256: 20.0, 512: 40.0, 1024: 80.0})

    assert curve.bucket_for(1) == 256
    assert curve.bucket_for(256) == 256
    assert curve.bucket_for(257) == 512
    assert curve.bucket_for(900) == 1024
    assert curve.bucket_for(4096) == 1024


def test_cost_model_is_off_by_default() -> None:
    assert OffloadCostModel.from_extra_config({}) is None
    assert (
        OffloadCostModel.from_extra_config({"cache_cost_model": {"mode": "off"}})
        is None
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"cache_cost_model": {"mode": "shadow"}},
        {
            "cache_cost_model": {
                "mode": "shadow",
                "profile": {"recompute_ms": {}},
            }
        },
        {
            "cache_cost_model": {
                "mode": "shadow",
                "profile": {
                    "recompute_ms": {0: 1.0},
                    "tiers": {"cpu_primary": {"restore_ms": {1: 1.0}}},
                },
            }
        },
        {
            "cache_cost_model": {
                "mode": "shadow",
                "profile": {
                    "recompute_ms": {1: 0.0},
                    "tiers": {"cpu_primary": {"restore_ms": {1: 1.0}}},
                },
            }
        },
        {
            "cache_cost_model": {
                "mode": "shadow",
                "profile": {
                    "recompute_ms": {1: math.inf},
                    "tiers": {"cpu_primary": {"restore_ms": {1: 1.0}}},
                },
            }
        },
        {
            "cache_cost_model": {
                "mode": "shadow",
                "ewma_alpha": 0.0,
                "profile": {
                    "recompute_ms": {1: 1.0},
                    "tiers": {"cpu_primary": {"restore_ms": {1: 1.0}}},
                },
            }
        },
        {
            "cache_cost_model": {
                "mode": "shadow",
                "ewma_alpha": 1.1,
                "profile": {
                    "recompute_ms": {1: 1.0},
                    "tiers": {"cpu_primary": {"restore_ms": {1: 1.0}}},
                },
            }
        },
        {
            "cache_cost_model": {
                "mode": "shadow",
                "sample_scale_min": 0.0,
                "profile": {
                    "recompute_ms": {1: 1.0},
                    "tiers": {"cpu_primary": {"restore_ms": {1: 1.0}}},
                },
            }
        },
        {
            "cache_cost_model": {
                "mode": "shadow",
                "sample_scale_min": 2.0,
                "sample_scale_max": 1.0,
                "profile": {
                    "recompute_ms": {1: 1.0},
                    "tiers": {"cpu_primary": {"restore_ms": {1: 1.0}}},
                },
            }
        },
        {"cache_cost_model": {"mode": "enforce"}},
    ],
)
def test_invalid_shadow_config_raises_value_error(raw: dict) -> None:
    with pytest.raises(ValueError):
        OffloadCostModel.from_extra_config(raw)


def test_curve_accepts_integer_like_string_keys() -> None:
    curve = CostCurve.from_mapping({"256": 20.0, "512": 40.0})
    assert curve.samples == ((256, 20.0), (512, 40.0))


@pytest.mark.parametrize(
    "raw",
    [
        {True: 1.0},
        {0: 1.0},
        {-1: 1.0},
        {1: 0.0},
        {1: math.nan},
        {1: math.inf},
        {"1": 1.0, 1: 2.0},
    ],
)
def test_invalid_curve_samples_raise_value_error(raw: dict) -> None:
    with pytest.raises(ValueError):
        CostCurve.from_mapping(raw)


def _profile_model() -> Any:
    model = OffloadCostModel.from_extra_config(PROFILE)
    assert model is not None
    return model


def test_cpu_1024_prefers_restore() -> None:
    model = _profile_model()
    decision = model.shadow_decide(
        LoadProvenance(
            source="cpu_primary",
            external_tokens=1024,
            secondary_promoted_tokens=0,
            sources=("cpu_primary",),
            confidence="high",
        )
    )

    assert decision is not None
    assert decision.preferred == "restore"
    assert decision.restore_seed_ms == pytest.approx(24.490)
    assert decision.restore_estimate_ms == pytest.approx(24.490)
    assert decision.recompute_estimate_ms == pytest.approx(81.705)
    assert decision.runtime_scale == pytest.approx(1.0)
    assert decision.confidence == "high"


@pytest.mark.parametrize("tokens", [256, 512, 1024, 2048, 4096])
def test_filesystem_measured_points_prefer_recompute(tokens: int) -> None:
    model = _profile_model()
    decision = model.shadow_decide(
        LoadProvenance(
            source="secondary:filesystem",
            external_tokens=tokens,
            secondary_promoted_tokens=tokens,
            sources=("secondary:filesystem",),
            confidence="high",
        )
    )

    assert decision is not None
    assert decision.preferred == "recompute"
    assert decision.confidence == "high"


def test_equal_cost_prefers_recompute() -> None:
    model = OffloadCostModel.from_extra_config(
        {
            "cache_cost_model": {
                "mode": "shadow",
                "profile": {
                    "recompute_ms": {1024: 10.0},
                    "tiers": {"cpu_primary": {"restore_ms": {1024: 10.0}}},
                },
            }
        }
    )
    assert model is not None

    decision = model.shadow_decide(
        LoadProvenance(
            source="cpu_primary",
            external_tokens=1024,
            secondary_promoted_tokens=0,
            sources=("cpu_primary",),
            confidence="high",
        )
    )
    assert decision is not None
    assert decision.preferred == "recompute"


def test_single_point_cpu_extrapolation_is_low_confidence() -> None:
    model = _profile_model()
    decision = model.shadow_decide(
        LoadProvenance(
            source="cpu_primary",
            external_tokens=2048,
            secondary_promoted_tokens=0,
            sources=("cpu_primary",),
            confidence="high",
        )
    )

    assert decision is not None
    assert decision.preferred == "restore"
    assert decision.restore_estimate_ms == pytest.approx(48.98)
    assert decision.confidence == "low"


def test_ewma_updates_matching_tier_bucket() -> None:
    model = _profile_model()
    observation = model.observe_secondary_promotion("filesystem", 1024, 162.916)

    assert observation is not None
    assert observation.tier_key == "filesystem"
    assert observation.token_bucket == 1024
    assert observation.seeded_ms == pytest.approx(81.458)
    assert observation.sample_scale == pytest.approx(2.0)
    assert observation.runtime_scale == pytest.approx(1.2)

    decision = model.shadow_decide(
        LoadProvenance(
            source="secondary:filesystem",
            external_tokens=1024,
            secondary_promoted_tokens=1024,
            sources=("secondary:filesystem",),
            confidence="high",
        )
    )
    assert decision is not None
    assert decision.runtime_scale == pytest.approx(1.2)
    assert decision.restore_estimate_ms == pytest.approx(108.132 * 1.2)


def test_ewma_isolated_by_bucket() -> None:
    model = _profile_model()
    observation = model.observe_secondary_promotion("filesystem", 512, 70.46)
    assert observation is not None
    assert observation.token_bucket == 512
    assert observation.runtime_scale == pytest.approx(1.2)

    decision = model.shadow_decide(
        LoadProvenance(
            source="secondary:filesystem",
            external_tokens=1024,
            secondary_promoted_tokens=1024,
            sources=("secondary:filesystem",),
            confidence="high",
        )
    )
    assert decision is not None
    assert decision.runtime_scale == pytest.approx(1.0)


def test_ewma_clamps_sample_scale_before_update() -> None:
    model = _profile_model()
    observation = model.observe_secondary_promotion("filesystem", 1024, 8145.8)

    assert observation is not None
    assert observation.sample_scale == pytest.approx(4.0)
    assert observation.runtime_scale == pytest.approx(1.6)


def test_tier_without_promotion_curve_does_not_update_ewma() -> None:
    model = _profile_model()
    assert model.observe_secondary_promotion("cpu_primary", 1024, 25.0) is None
    assert model.observe_secondary_promotion("unknown", 1024, 25.0) is None


def test_mixed_sources_use_conservative_max_and_low_confidence() -> None:
    model = _profile_model()
    decision = model.shadow_decide(
        LoadProvenance(
            source="mixed",
            external_tokens=1024,
            secondary_promoted_tokens=None,
            sources=("cpu_primary", "secondary:filesystem"),
            confidence="low",
        )
    )

    assert decision is not None
    assert decision.restore_seed_ms == pytest.approx(108.132)
    assert decision.restore_estimate_ms == pytest.approx(108.132)
    assert decision.preferred == "recompute"
    assert decision.confidence == "low"


def test_mixed_source_missing_curve_returns_no_decision() -> None:
    model = _profile_model()
    assert (
        model.shadow_decide(
            LoadProvenance(
                source="mixed",
                external_tokens=1024,
                secondary_promoted_tokens=None,
                sources=("cpu_primary", "secondary:network"),
                confidence="low",
            )
        )
        is None
    )


def _calibrated_model() -> Any:
    model = OffloadCostModel.from_extra_config(CALIBRATED_P95_PROFILE)
    assert model is not None
    return model


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (104, "recompute"),
        (168, "restore"),
        (192, "restore"),
        (232, "restore"),
        (512, "restore"),
        (1024, "restore"),
        (2016, "restore"),
        (4088, "restore"),
    ],
)
def test_calibrated_cpu_p95_decisions(
    tokens: int,
    expected: str,
) -> None:
    decision = _calibrated_model().shadow_decide(
        LoadProvenance(
            source="cpu_primary",
            external_tokens=tokens,
            secondary_promoted_tokens=0,
            sources=("cpu_primary",),
            confidence="high",
        )
    )
    assert decision is not None
    assert decision.preferred == expected


@pytest.mark.parametrize(
    "tokens",
    [232, 512, 1024, 2016, 4088],
)
def test_calibrated_filesystem_p95_still_prefers_recompute(
    tokens: int,
) -> None:
    decision = _calibrated_model().shadow_decide(
        LoadProvenance(
            source="secondary:filesystem",
            external_tokens=tokens,
            secondary_promoted_tokens=tokens,
            sources=("secondary:filesystem",),
            confidence="high",
        )
    )
    assert decision is not None
    assert decision.preferred == "recompute"


def test_ewma_converges_monotonically_toward_stationary_scale() -> None:
    model = _profile_model()
    observed_ms = 81.458 * 2.0
    scales = []

    for _ in range(5):
        observation = model.observe_secondary_promotion(
            "filesystem",
            1024,
            observed_ms,
        )
        assert observation is not None
        scales.append(observation.runtime_scale)

    assert scales == sorted(scales)
    assert all(scale < 2.0 for scale in scales)
    assert scales == pytest.approx([1.2, 1.36, 1.488, 1.5904, 1.67232])


def test_ewma_stable_observations_have_diminishing_updates() -> None:
    model = _profile_model()
    observed_ms = 81.458 * 2.0
    scales = []

    for _ in range(5):
        observation = model.observe_secondary_promotion(
            "filesystem",
            1024,
            observed_ms,
        )
        assert observation is not None
        scales.append(observation.runtime_scale)

    increments = [right - left for left, right in zip(scales, scales[1:])]
    assert increments[0] > increments[1] > increments[2] > increments[3] > 0
