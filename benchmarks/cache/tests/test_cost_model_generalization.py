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
