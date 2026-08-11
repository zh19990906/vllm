# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import unittest
from pathlib import Path

from benchmarks.cache.cost_model_calibration import load_profile_artifact


_REPO_ROOT = Path(__file__).resolve().parents[3]


class FrozenProfileArtifactTests(unittest.TestCase):
    def test_issue14_frozen_profile_matches_calibration_result(self) -> None:
        calibration_path = (
            _REPO_ROOT
            / "docs/engineering/validation/"
            "2026-08-10-issue14-shadow-cost-model-calibration.json"
        )
        artifact_path = (
            _REPO_ROOT
            / "benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json"
        )

        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(
            artifact["cache_cost_model"],
            calibration["calibrated_profile"],
        )
        self.assertEqual(
            load_profile_artifact(artifact_path),
            calibration["calibrated_profile"],
        )
        self.assertEqual(artifact["provenance"]["issue"], 14)
        self.assertEqual(
            artifact["provenance"]["profile_role"],
            "frozen_holdout_seed",
        )


if __name__ == "__main__":
    unittest.main()


def _generalization_condition_fixture() -> dict:
    return {
        "schema_version": 1,
        "issue": 15,
        "condition": {
            "id": "c-model",
            "model": "/mnt/model/Qwen2.5-14B-Instruct",
            "served_model": "qwen2.5-14b",
            "concurrency": 1,
            "request_rate": "inf",
            "requests_per_case": 8,
            "tensor_parallel_size": 1,
            "gpu_uuid": "GPU-test",
            "environment_artifact": "/code/results/cache/run/environment.json",
            "run_directories": {
                "recompute": "/code/results/cache/recompute",
                "cpu_primary": "/code/results/cache/cpu",
                "secondary:filesystem": "/code/results/cache/fs",
            },
        },
        "samples": [
            {
                "source": "secondary:filesystem",
                "requested_tokens": 256,
                "external_kv_tokens_total": 1856,
                "external_kv_tokens_per_request": 232,
                "latency_ms": {
                    "recompute": {"p95": 30.0},
                    "restore": {"p95": 38.0},
                },
                "workload": {
                    "measure_sha256": "same-measure-sha",
                    "populate_sha256": "same-populate-sha",
                },
                "transfer_evidence": {
                    "tiered_fs_async_lookups": 8,
                },
            },
            {
                "source": "cpu_primary",
                "requested_tokens": 256,
                "external_kv_tokens_total": 1856,
                "external_kv_tokens_per_request": 232,
                "latency_ms": {
                    "recompute": {"p95": 30.0},
                    "restore": {"p95": 24.0},
                },
                "workload": {
                    "measure_sha256": "same-measure-sha",
                    "populate_sha256": "same-populate-sha",
                },
                "transfer_evidence": {
                    "cpu_to_gpu_transfers": 8,
                    "cpu_to_gpu_bytes": 106430464,
                },
            },
        ],
        "excluded_samples": [],
    }


class GeneralizationConditionLoaderTests(unittest.TestCase):
    def test_loader_uses_per_request_external_tokens_and_retains_metadata(
        self,
    ) -> None:
        from tempfile import TemporaryDirectory

        from benchmarks.cache.cost_model_generalization import (
            load_generalization_condition,
        )

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "condition.json"
            path.write_text(
                json.dumps(_generalization_condition_fixture()),
                encoding="utf-8",
            )

            condition = load_generalization_condition(path, percentile="p95")

        self.assertEqual(condition.condition_id, "c-model")
        self.assertEqual(
            condition.model,
            "/mnt/model/Qwen2.5-14B-Instruct",
        )
        self.assertEqual(condition.served_model, "qwen2.5-14b")
        self.assertEqual(condition.concurrency, 1)
        self.assertEqual(condition.tensor_parallel_size, 1)
        self.assertEqual(condition.gpu_uuid, "GPU-test")
        self.assertEqual(condition.dataset.percentile, "p95")
        self.assertEqual(condition.dataset.requests_per_case, 8)

        keys = [
            (sample.requested_tokens, sample.source)
            for sample in condition.dataset.decision_samples
        ]
        self.assertEqual(
            keys,
            [
                (256, "cpu_primary"),
                (256, "secondary:filesystem"),
            ],
        )
        self.assertTrue(
            all(
                sample.external_tokens == 232
                for sample in condition.dataset.decision_samples
            )
        )

        self.assertEqual(len(condition.sample_metadata), 2)
        self.assertEqual(
            condition.run_directories["secondary:filesystem"],
            "/code/results/cache/fs",
        )

    def test_loader_rejects_inconsistent_external_token_total(self) -> None:
        from tempfile import TemporaryDirectory

        from benchmarks.cache.cost_model_generalization import (
            load_generalization_condition,
        )

        payload = _generalization_condition_fixture()
        payload["samples"][0]["external_kv_tokens_total"] = 1857

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid-condition.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "external KV token total",
            ):
                load_generalization_condition(path, percentile="p95")


class GeneralizationConditionValidationTests(unittest.TestCase):
    def _write_payload(self, payload: dict) -> tuple[object, Path]:
        from tempfile import TemporaryDirectory

        tempdir = TemporaryDirectory()
        path = Path(tempdir.name) / "condition.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return tempdir, path

    def test_loader_rejects_wrong_schema_version(self) -> None:
        from benchmarks.cache.cost_model_generalization import (
            load_generalization_condition,
        )

        payload = _generalization_condition_fixture()
        payload["schema_version"] = 2
        tempdir, path = self._write_payload(payload)
        try:
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_generalization_condition(path)
        finally:
            tempdir.cleanup()

    def test_loader_rejects_wrong_issue(self) -> None:
        from benchmarks.cache.cost_model_generalization import (
            load_generalization_condition,
        )

        payload = _generalization_condition_fixture()
        payload["issue"] = 14
        tempdir, path = self._write_payload(payload)
        try:
            with self.assertRaisesRegex(ValueError, "issue"):
                load_generalization_condition(path)
        finally:
            tempdir.cleanup()

    def test_loader_rejects_non_positive_requests_per_case(self) -> None:
        from benchmarks.cache.cost_model_generalization import (
            load_generalization_condition,
        )

        payload = _generalization_condition_fixture()
        payload["condition"]["requests_per_case"] = 0
        payload["samples"] = []
        tempdir, path = self._write_payload(payload)
        try:
            with self.assertRaisesRegex(ValueError, "requests_per_case"):
                load_generalization_condition(path)
        finally:
            tempdir.cleanup()

    def test_loader_rejects_unsupported_source(self) -> None:
        from benchmarks.cache.cost_model_generalization import (
            load_generalization_condition,
        )

        payload = _generalization_condition_fixture()
        payload["samples"][0]["source"] = "nvme"
        tempdir, path = self._write_payload(payload)
        try:
            with self.assertRaisesRegex(ValueError, "source"):
                load_generalization_condition(path)
        finally:
            tempdir.cleanup()

    def test_loader_rejects_unsupported_percentile(self) -> None:
        from benchmarks.cache.cost_model_generalization import (
            load_generalization_condition,
        )

        payload = _generalization_condition_fixture()
        payload["samples"] = []
        tempdir, path = self._write_payload(payload)
        try:
            with self.assertRaisesRegex(ValueError, "percentile"):
                load_generalization_condition(path, percentile="p90")
        finally:
            tempdir.cleanup()


class FrozenConditionEvaluationTests(unittest.TestCase):
    def test_evaluation_preserves_wrong_supplied_profile_prediction(
        self,
    ) -> None:
        from copy import deepcopy
        from tempfile import TemporaryDirectory

        from benchmarks.cache.cost_model_calibration import (
            load_profile_artifact,
        )
        from benchmarks.cache.cost_model_generalization import (
            evaluate_frozen_condition,
            load_generalization_condition,
        )

        payload = _generalization_condition_fixture()

        with TemporaryDirectory() as tmp:
            condition_path = Path(tmp) / "condition.json"
            condition_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            condition = load_generalization_condition(condition_path)

        frozen_path = (
            _REPO_ROOT
            / "benchmarks/cache/profiles/"
            "issue14-shadow-cost-calibrated.json"
        )
        profile = deepcopy(load_profile_artifact(frozen_path))

        # Deliberately make the supplied CPU restore prediction wrong.
        # Holdout evaluation must preserve this wrong prediction rather than
        # deriving a new curve from the condition data.
        profile["profile"]["tiers"]["cpu_primary"]["restore_ms"]["232"] = 50.0

        result = evaluate_frozen_condition(
            condition,
            profile,
            profile_identity="deliberately-wrong-fixture-profile",
        )

        self.assertEqual(result["mode"], "frozen_profile_holdout")
        self.assertEqual(
            result["profile_identity"],
            "deliberately-wrong-fixture-profile",
        )
        self.assertNotIn("calibrated_profile", result)

        cpu_row = next(
            row
            for row in result["evaluation"]["samples"]
            if row["source"] == "cpu_primary"
        )

        self.assertEqual(cpu_row["actual_preferred"], "restore")
        self.assertEqual(cpu_row["predicted_preferred"], "recompute")
        self.assertFalse(cpu_row["decision_correct"])
        self.assertAlmostEqual(cpu_row["predicted_restore_ms"], 50.0)


def _synthetic_evaluation_row(
    *,
    source: str,
    requested_tokens: int,
    actual_recompute_ms: float = 100.0,
    predicted_recompute_ms: float = 100.0,
    actual_restore_ms: float,
    predicted_restore_ms: float,
    confidence: str = "high",
) -> dict:
    actual_margin_ms = actual_restore_ms - actual_recompute_ms
    predicted_margin_ms = predicted_restore_ms - predicted_recompute_ms
    actual_preferred = "restore" if actual_margin_ms < 0 else "recompute"
    predicted_preferred = (
        "restore" if predicted_margin_ms < 0 else "recompute"
    )

    return {
        "source": source,
        "requested_tokens": requested_tokens,
        "external_tokens": requested_tokens,
        "actual_recompute_ms": actual_recompute_ms,
        "actual_restore_ms": actual_restore_ms,
        "predicted_recompute_ms": predicted_recompute_ms,
        "predicted_restore_ms": predicted_restore_ms,
        "actual_margin_ms": actual_margin_ms,
        "predicted_margin_ms": predicted_margin_ms,
        "actual_preferred": actual_preferred,
        "predicted_preferred": predicted_preferred,
        "boundary_sensitive": abs(actual_margin_ms) <= 1.0,
        "recompute_abs_error_ms": abs(
            predicted_recompute_ms - actual_recompute_ms
        ),
        "recompute_relative_error": abs(
            predicted_recompute_ms - actual_recompute_ms
        )
        / actual_recompute_ms,
        "restore_abs_error_ms": abs(
            predicted_restore_ms - actual_restore_ms
        ),
        "restore_relative_error": abs(
            predicted_restore_ms - actual_restore_ms
        )
        / actual_restore_ms,
        "decision_correct": predicted_preferred == actual_preferred,
        "confidence": confidence,
        "runtime_scale": 1.0,
    }


class FrozenTransferGateTests(unittest.TestCase):
    def _evaluate_rows(self, rows: list[dict]) -> dict:
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from benchmarks.cache.cost_model_generalization import (
            evaluate_frozen_condition,
            load_generalization_condition,
        )

        payload = _generalization_condition_fixture()

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "condition.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            condition = load_generalization_condition(path)

        synthetic = {
            "samples": rows,
            "aggregate": {
                "sentinel": "Issue 15 must recompute the high-confidence gate",
            },
        }

        with patch(
            "benchmarks.cache.cost_model_generalization.evaluate_profile",
            return_value=synthetic,
        ):
            return evaluate_frozen_condition(
                condition,
                {"mode": "shadow", "profile": {}},
                profile_identity="synthetic-frozen-profile",
            )

    def test_all_principal_curves_passing_returns_transfer_pass(self) -> None:
        rows = [
            _synthetic_evaluation_row(
                source="cpu_primary",
                requested_tokens=256,
                predicted_recompute_ms=105.0,
                actual_restore_ms=80.0,
                predicted_restore_ms=84.0,
            ),
            _synthetic_evaluation_row(
                source="secondary:filesystem",
                requested_tokens=256,
                predicted_recompute_ms=105.0,
                actual_restore_ms=120.0,
                predicted_restore_ms=126.0,
            ),
        ]

        result = self._evaluate_rows(rows)

        self.assertEqual(
            result["classification"],
            "fixed_profile_transfer_pass",
        )
        gate = result["gate"]["high_confidence"]
        self.assertEqual(gate["decision_correct"], 2)
        self.assertEqual(gate["decision_total"], 2)
        self.assertEqual(gate["decision_accuracy"], 1.0)
        self.assertAlmostEqual(gate["recompute_mape_percent"], 5.0)
        self.assertAlmostEqual(gate["cpu_restore_mape_percent"], 5.0)
        self.assertAlmostEqual(
            gate["tiered_fs_restore_mape_percent"],
            5.0,
        )
        self.assertAlmostEqual(
            gate["principal_macro_mape_percent"],
            5.0,
        )

    def test_missing_principal_high_confidence_evidence_is_insufficient(
        self,
    ) -> None:
        rows = [
            _synthetic_evaluation_row(
                source="cpu_primary",
                requested_tokens=256,
                actual_restore_ms=80.0,
                predicted_restore_ms=80.0,
                confidence="low",
            ),
            _synthetic_evaluation_row(
                source="secondary:filesystem",
                requested_tokens=256,
                actual_restore_ms=120.0,
                predicted_restore_ms=120.0,
                confidence="high",
            ),
        ]

        result = self._evaluate_rows(rows)

        self.assertEqual(
            result["classification"],
            "insufficient_evidence",
        )
        self.assertIn(
            "cpu_restore",
            result["gate"]["missing_principal_curves"],
        )

    def test_single_curve_over_twenty_percent_fails_even_if_macro_passes(
        self,
    ) -> None:
        rows = [
            _synthetic_evaluation_row(
                source="cpu_primary",
                requested_tokens=256,
                actual_restore_ms=80.0,
                predicted_restore_ms=96.8,
            ),
            _synthetic_evaluation_row(
                source="secondary:filesystem",
                requested_tokens=256,
                actual_restore_ms=120.0,
                predicted_restore_ms=120.0,
            ),
        ]

        result = self._evaluate_rows(rows)

        gate = result["gate"]["high_confidence"]
        self.assertLessEqual(
            gate["principal_macro_mape_percent"],
            15.0,
        )
        self.assertGreater(
            gate["cpu_restore_mape_percent"],
            20.0,
        )
        self.assertEqual(
            result["classification"],
            "fixed_profile_transfer_fail",
        )
        self.assertIn(
            "principal_curve_mape",
            result["gate"]["failure_reasons"],
        )

    def test_clear_margin_wrong_decision_fails_at_exact_95_percent_accuracy(
        self,
    ) -> None:
        rows = []
        for anchor in range(1, 11):
            requested = anchor * 128
            rows.append(
                _synthetic_evaluation_row(
                    source="cpu_primary",
                    requested_tokens=requested,
                    actual_restore_ms=80.0,
                    predicted_restore_ms=80.0,
                )
            )
            rows.append(
                _synthetic_evaluation_row(
                    source="secondary:filesystem",
                    requested_tokens=requested,
                    actual_restore_ms=120.0,
                    predicted_restore_ms=120.0,
                )
            )

        rows[0] = _synthetic_evaluation_row(
            source="cpu_primary",
            requested_tokens=128,
            actual_restore_ms=98.0,
            predicted_restore_ms=102.0,
        )

        result = self._evaluate_rows(rows)

        gate = result["gate"]["high_confidence"]
        self.assertEqual(gate["decision_correct"], 19)
        self.assertEqual(gate["decision_total"], 20)
        self.assertAlmostEqual(gate["decision_accuracy"], 0.95)
        self.assertEqual(
            result["classification"],
            "fixed_profile_transfer_fail",
        )
        self.assertEqual(
            result["gate"]["clear_margin_wrong_decisions"],
            1,
        )
        self.assertIn(
            "clear_margin_wrong_decision",
            result["gate"]["failure_reasons"],
        )

    def test_boundary_wrong_decision_still_counts_in_accuracy(self) -> None:
        rows = []
        for anchor in range(1, 11):
            requested = anchor * 128
            rows.append(
                _synthetic_evaluation_row(
                    source="cpu_primary",
                    requested_tokens=requested,
                    actual_restore_ms=80.0,
                    predicted_restore_ms=80.0,
                )
            )
            rows.append(
                _synthetic_evaluation_row(
                    source="secondary:filesystem",
                    requested_tokens=requested,
                    actual_restore_ms=120.0,
                    predicted_restore_ms=120.0,
                )
            )

        rows[0] = _synthetic_evaluation_row(
            source="cpu_primary",
            requested_tokens=128,
            actual_restore_ms=99.5,
            predicted_restore_ms=100.5,
        )

        result = self._evaluate_rows(rows)

        gate = result["gate"]["high_confidence"]
        self.assertEqual(gate["decision_correct"], 19)
        self.assertEqual(gate["decision_total"], 20)
        self.assertAlmostEqual(gate["decision_accuracy"], 0.95)
        self.assertEqual(
            result["gate"]["clear_margin_wrong_decisions"],
            0,
        )
        self.assertEqual(
            result["classification"],
            "fixed_profile_transfer_pass",
        )

    def test_low_confidence_rows_are_reported_but_do_not_fill_evidence(
        self,
    ) -> None:
        rows = [
            _synthetic_evaluation_row(
                source="cpu_primary",
                requested_tokens=256,
                actual_restore_ms=80.0,
                predicted_restore_ms=80.0,
                confidence="high",
            ),
            _synthetic_evaluation_row(
                source="secondary:filesystem",
                requested_tokens=256,
                actual_restore_ms=120.0,
                predicted_restore_ms=120.0,
                confidence="low",
            ),
        ]

        result = self._evaluate_rows(rows)

        self.assertEqual(
            result["classification"],
            "insufficient_evidence",
        )
        self.assertEqual(len(result["low_confidence_samples"]), 1)
        self.assertEqual(
            result["low_confidence_samples"][0]["source"],
            "secondary:filesystem",
        )
        self.assertIn(
            "tiered_fs_restore",
            result["gate"]["missing_principal_curves"],
        )


def _diagnostic_row(
    *,
    source: str,
    requested_tokens: int,
    external_tokens: int,
    actual_recompute_ms: float,
    predicted_recompute_ms: float,
    actual_restore_ms: float,
    predicted_restore_ms: float,
) -> dict:
    return {
        "source": source,
        "requested_tokens": requested_tokens,
        "external_tokens": external_tokens,
        "actual_recompute_ms": actual_recompute_ms,
        "predicted_recompute_ms": predicted_recompute_ms,
        "actual_restore_ms": actual_restore_ms,
        "predicted_restore_ms": predicted_restore_ms,
        "confidence": "high",
    }


class CurveScalingDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_distinguish_transfer_scale_and_shape(self) -> None:
        from benchmarks.cache.cost_model_generalization import (
            diagnose_curve_scaling,
        )

        evaluation = {
            "classification": "fixed_profile_transfer_fail",
            "evaluation": {
                "samples": [
                    # Recompute: a stable 5% error -> directly transferable.
                    _diagnostic_row(
                        source="cpu_primary",
                        requested_tokens=256,
                        external_tokens=232,
                        actual_recompute_ms=100.0,
                        predicted_recompute_ms=95.0,
                        actual_restore_ms=100.0,
                        predicted_restore_ms=50.0,
                    ),
                    _diagnostic_row(
                        source="secondary:filesystem",
                        requested_tokens=256,
                        external_tokens=232,
                        actual_recompute_ms=100.0,
                        predicted_recompute_ms=95.0,
                        actual_restore_ms=100.0,
                        predicted_restore_ms=100.0,
                    ),
                    _diagnostic_row(
                        source="cpu_primary",
                        requested_tokens=1024,
                        external_tokens=1024,
                        actual_recompute_ms=200.0,
                        predicted_recompute_ms=190.0,
                        actual_restore_ms=200.0,
                        predicted_restore_ms=100.0,
                    ),
                    _diagnostic_row(
                        source="secondary:filesystem",
                        requested_tokens=1024,
                        external_tokens=1024,
                        actual_recompute_ms=200.0,
                        predicted_recompute_ms=190.0,
                        actual_restore_ms=200.0,
                        predicted_restore_ms=100.0,
                    ),
                ],
            },
        }

        before = json.dumps(evaluation, sort_keys=True)
        diagnostics = diagnose_curve_scaling(evaluation)
        after = json.dumps(evaluation, sort_keys=True)

        # Diagnostics are observational only; they must not mutate the
        # fixed-profile result or replace its primary classification.
        self.assertEqual(before, after)
        self.assertEqual(
            evaluation["classification"],
            "fixed_profile_transfer_fail",
        )

        recompute = diagnostics["curves"]["recompute"]
        self.assertAlmostEqual(recompute["raw_mape_percent"], 5.0)
        self.assertAlmostEqual(
            recompute["scale"],
            100.0 / 95.0,
        )
        self.assertAlmostEqual(
            recompute["residual_mape_percent"],
            0.0,
        )
        self.assertEqual(
            recompute["classification"],
            "transferable",
        )

        cpu = diagnostics["curves"]["cpu_restore"]
        self.assertAlmostEqual(cpu["raw_mape_percent"], 50.0)
        self.assertAlmostEqual(cpu["scale"], 2.0)
        self.assertAlmostEqual(cpu["residual_mape_percent"], 0.0)
        self.assertEqual(
            cpu["classification"],
            "environment_specific_scale_candidate",
        )

        filesystem = diagnostics["curves"]["tiered_fs_restore"]
        self.assertAlmostEqual(
            filesystem["raw_mape_percent"],
            25.0,
        )
        self.assertAlmostEqual(filesystem["scale"], 1.5)
        self.assertAlmostEqual(
            filesystem["residual_mape_percent"],
            37.5,
        )
        self.assertEqual(
            filesystem["classification"],
            "curve_shape_or_missing_feature",
        )
