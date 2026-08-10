# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure cost-model primitives for shadow KV offload decisions."""

import math
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Confidence = Literal["high", "low"]
PreferredPath = Literal["restore", "recompute"]


@dataclass(frozen=True, slots=True)
class CurveEstimate:
    value_ms: float
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class LoadProvenance:
    source: str
    external_tokens: int
    secondary_promoted_tokens: int | None
    sources: tuple[str, ...]
    confidence: Confidence
    lookup_sync_seconds: float | None = None
    lookup_async_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ShadowDecision:
    preferred: PreferredPath
    restore_seed_ms: float
    restore_estimate_ms: float
    recompute_estimate_ms: float
    runtime_scale: float
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    tier_key: str
    token_bucket: int
    observed_ms: float
    seeded_ms: float
    sample_scale: float
    runtime_scale: float


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
    """Validated shadow cost model with bounded secondary-tier calibration."""

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
        self._runtime_scales: dict[tuple[str, int], float] = {}

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
                raise ValueError(f"tier profile '{raw_tier_key}' requires restore_ms")
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

    def shadow_decide(self, provenance: LoadProvenance) -> ShadowDecision | None:
        if provenance.external_tokens <= 0:
            return None

        recompute = self.recompute_curve.estimate(provenance.external_tokens)
        if provenance.source == "mixed":
            restore_choice = self._estimate_mixed_restore(provenance)
            if restore_choice is None:
                return None
            restore_seed_ms, restore_estimate_ms, runtime_scale = restore_choice
            confidence: Confidence = "low"
        else:
            profile_key = self._profile_key(provenance.source)
            if profile_key is None:
                return None
            restore_curve = self.tier_restore_curves.get(profile_key)
            if restore_curve is None:
                return None
            restore = restore_curve.estimate(provenance.external_tokens)
            runtime_scale = self._runtime_scale(
                provenance.source, profile_key, provenance.external_tokens
            )
            restore_seed_ms = restore.value_ms
            restore_estimate_ms = restore.value_ms * runtime_scale
            confidence = (
                "high"
                if provenance.confidence == "high"
                and restore.confidence == "high"
                and recompute.confidence == "high"
                else "low"
            )

        preferred: PreferredPath = (
            "restore" if restore_estimate_ms < recompute.value_ms else "recompute"
        )
        return ShadowDecision(
            preferred=preferred,
            restore_seed_ms=restore_seed_ms,
            restore_estimate_ms=restore_estimate_ms,
            recompute_estimate_ms=recompute.value_ms,
            runtime_scale=runtime_scale,
            confidence=confidence,
        )

    def _estimate_mixed_restore(
        self, provenance: LoadProvenance
    ) -> tuple[float, float, float] | None:
        if not provenance.sources:
            return None

        candidates: list[tuple[float, float, float]] = []
        for source in provenance.sources:
            profile_key = self._profile_key(source)
            if profile_key is None:
                return None
            restore_curve = self.tier_restore_curves.get(profile_key)
            if restore_curve is None:
                return None
            restore = restore_curve.estimate(provenance.external_tokens)
            runtime_scale = self._runtime_scale(
                source, profile_key, provenance.external_tokens
            )
            candidates.append(
                (restore.value_ms, restore.value_ms * runtime_scale, runtime_scale)
            )

        return max(candidates, key=lambda item: item[1])

    def observe_secondary_promotion(
        self, tier_key: str, tokens: int, observed_ms: float
    ) -> RuntimeObservation | None:
        promotion_curve = self.tier_promotion_curves.get(tier_key)
        if promotion_curve is None:
            return None
        if isinstance(observed_ms, bool) or not isinstance(observed_ms, int | float):
            raise ValueError("observed_ms must be finite and positive")
        observed_value = float(observed_ms)
        if not math.isfinite(observed_value) or observed_value <= 0:
            raise ValueError("observed_ms must be finite and positive")

        seeded_ms = promotion_curve.estimate(tokens).value_ms
        token_bucket = promotion_curve.bucket_for(tokens)
        sample_scale = observed_value / seeded_ms
        sample_scale = min(
            max(sample_scale, self.sample_scale_min), self.sample_scale_max
        )

        key = (tier_key, token_bucket)
        old_scale = self._runtime_scales.get(key, 1.0)
        runtime_scale = (
            self.ewma_alpha * sample_scale + (1.0 - self.ewma_alpha) * old_scale
        )
        self._runtime_scales[key] = runtime_scale

        return RuntimeObservation(
            tier_key=tier_key,
            token_bucket=token_bucket,
            observed_ms=observed_value,
            seeded_ms=seeded_ms,
            sample_scale=sample_scale,
            runtime_scale=runtime_scale,
        )

    def _runtime_scale(self, source: str, tier_key: str, tokens: int) -> float:
        if not source.startswith("secondary:"):
            return 1.0
        promotion_curve = self.tier_promotion_curves.get(tier_key)
        if promotion_curve is None:
            return 1.0
        bucket = promotion_curve.bucket_for(tokens)
        return self._runtime_scales.get((tier_key, bucket), 1.0)

    @staticmethod
    def _profile_key(source: str) -> str | None:
        if source == "cpu_primary":
            return "cpu_primary"
        if source.startswith("secondary:"):
            tier_key = source.removeprefix("secondary:")
            return tier_key or None
        return None

    @staticmethod
    def _parse_positive_number(raw: object, name: str) -> float:
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"{name} must be finite and positive")
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return value
