# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure cost-model primitives for shadow KV offload decisions."""

from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Literal

Confidence = Literal["high", "low"]


@dataclass(frozen=True, slots=True)
class CurveEstimate:
    value_ms: float
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class CostCurve:
    samples: tuple[tuple[int, float], ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[object, object]) -> "CostCurve":
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("cost curve must be a non-empty mapping")

        parsed: dict[int, float] = {}
        for raw_tokens, raw_value in raw.items():
            tokens = cls._parse_tokens(raw_tokens)
            if tokens in parsed:
                raise ValueError(f"duplicate cost curve token sample: {tokens}")
            parsed[tokens] = cls._parse_latency_ms(raw_value)

        return cls(samples=tuple(sorted(parsed.items())))

    @staticmethod
    def _parse_tokens(raw: object) -> int:
        if isinstance(raw, bool):
            raise ValueError("cost curve token samples must be positive integers")
        if type(raw) is int:
            tokens = raw
        elif isinstance(raw, str):
            try:
                tokens = int(raw)
            except ValueError as exc:
                raise ValueError(
                    "cost curve token samples must be positive integers"
                ) from exc
        else:
            raise ValueError("cost curve token samples must be positive integers")

        if tokens <= 0:
            raise ValueError("cost curve token samples must be positive integers")
        return tokens

    @staticmethod
    def _parse_latency_ms(raw: object) -> float:
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError("cost curve latency samples must be finite and positive")
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("cost curve latency samples must be finite and positive")
        return value

    def estimate(self, tokens: int) -> CurveEstimate:
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("tokens must be a positive integer")

        sample_tokens = tuple(token_count for token_count, _ in self.samples)
        index = bisect_left(sample_tokens, tokens)

        if index < len(self.samples) and self.samples[index][0] == tokens:
            return CurveEstimate(self.samples[index][1], "high")

        if index == 0:
            first_tokens, first_ms = self.samples[0]
            return CurveEstimate(first_ms * tokens / first_tokens, "low")

        if index == len(self.samples):
            last_tokens, last_ms = self.samples[-1]
            return CurveEstimate(last_ms * tokens / last_tokens, "low")

        left_tokens, left_ms = self.samples[index - 1]
        right_tokens, right_ms = self.samples[index]
        fraction = (tokens - left_tokens) / (right_tokens - left_tokens)
        value_ms = left_ms + fraction * (right_ms - left_ms)
        return CurveEstimate(value_ms, "high")

    def bucket_for(self, tokens: int) -> int:
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("tokens must be a positive integer")

        for sample_tokens, _ in self.samples:
            if tokens <= sample_tokens:
                return sample_tokens
        return self.samples[-1][0]


class OffloadCostModel:
    """Validated shadow cost-model configuration.

    Decision and online-calibration behavior is added separately after its
    dedicated tests are introduced. This class already owns the parsed curves
    so configuration validation has a single source of truth.
    """

    def __init__(
        self,
        *,
        recompute_curve: CostCurve,
        tier_restore_curves: dict[str, CostCurve],
        tier_promotion_curves: dict[str, CostCurve],
        ewma_alpha: float,
        sample_scale_min: float,
        sample_scale_max: float,
    ) -> None:
        self.recompute_curve = recompute_curve
        self.tier_restore_curves = tier_restore_curves
        self.tier_promotion_curves = tier_promotion_curves
        self.ewma_alpha = ewma_alpha
        self.sample_scale_min = sample_scale_min
        self.sample_scale_max = sample_scale_max

    @classmethod
    def from_extra_config(
        cls, extra_config: Mapping[str, Any]
    ) -> "OffloadCostModel | None":
        raw_config = extra_config.get("cache_cost_model")
        if raw_config is None:
            return None
        if not isinstance(raw_config, Mapping):
            raise ValueError("cache_cost_model must be a mapping")

        mode = raw_config.get("mode", "off")
        if mode == "off":
            return None
        if mode != "shadow":
            raise ValueError("cache_cost_model.mode must be 'off' or 'shadow'")

        ewma_alpha = cls._parse_positive_number(
            raw_config.get("ewma_alpha", 0.2), "ewma_alpha"
        )
        if ewma_alpha > 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")

        sample_scale_min = cls._parse_positive_number(
            raw_config.get("sample_scale_min", 0.25), "sample_scale_min"
        )
        sample_scale_max = cls._parse_positive_number(
            raw_config.get("sample_scale_max", 4.0), "sample_scale_max"
        )
        if sample_scale_min > sample_scale_max:
            raise ValueError("sample_scale_min must be <= sample_scale_max")

        profile = raw_config.get("profile")
        if not isinstance(profile, Mapping):
            raise ValueError("cache_cost_model.profile must be a mapping")

        recompute_raw = profile.get("recompute_ms")
        if not isinstance(recompute_raw, Mapping) or not recompute_raw:
            raise ValueError("cache_cost_model.profile.recompute_ms is required")
        recompute_curve = CostCurve.from_mapping(recompute_raw)

        tiers_raw = profile.get("tiers")
        if not isinstance(tiers_raw, Mapping) or not tiers_raw:
            raise ValueError("cache_cost_model.profile.tiers must be non-empty")

        restore_curves: dict[str, CostCurve] = {}
        promotion_curves: dict[str, CostCurve] = {}
        for raw_tier_key, raw_tier_profile in tiers_raw.items():
            if not isinstance(raw_tier_key, str) or not raw_tier_key:
                raise ValueError("cost-model tier keys must be non-empty strings")
            if not isinstance(raw_tier_profile, Mapping):
                raise ValueError(f"tier profile '{raw_tier_key}' must be a mapping")

            restore_raw = raw_tier_profile.get("restore_ms")
            if not isinstance(restore_raw, Mapping) or not restore_raw:
                raise ValueError(
                    f"tier profile '{raw_tier_key}' requires restore_ms"
                )
            restore_curves[raw_tier_key] = CostCurve.from_mapping(restore_raw)

            promotion_raw = raw_tier_profile.get("promotion_ms")
            if promotion_raw is not None:
                if not isinstance(promotion_raw, Mapping) or not promotion_raw:
                    raise ValueError(
                        f"tier profile '{raw_tier_key}'.promotion_ms must be "
                        "a non-empty mapping"
                    )
                promotion_curves[raw_tier_key] = CostCurve.from_mapping(promotion_raw)

        return cls(
            recompute_curve=recompute_curve,
            tier_restore_curves=restore_curves,
            tier_promotion_curves=promotion_curves,
            ewma_alpha=ewma_alpha,
            sample_scale_min=sample_scale_min,
            sample_scale_max=sample_scale_max,
        )

    @staticmethod
    def _parse_positive_number(raw: object, name: str) -> float:
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"{name} must be finite and positive")
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return value
