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
