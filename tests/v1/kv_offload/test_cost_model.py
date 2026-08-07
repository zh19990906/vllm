# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import math
from pathlib import Path
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "cost_model.py"
_SPEC = importlib.util.spec_from_file_location("vllm_cost_model_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_COST_MODEL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _COST_MODEL
_SPEC.loader.exec_module(_COST_MODEL)

CostCurve = _COST_MODEL.CostCurve
CurveEstimate = _COST_MODEL.CurveEstimate
OffloadCostModel = _COST_MODEL.OffloadCostModel


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
