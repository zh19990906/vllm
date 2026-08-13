# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import tempfile
import unittest
from pathlib import Path

SMOKE_PATH = Path(__file__).parents[1] / "issue31_fs_capacity_smoke.py"


def load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "issue31_fs_capacity_smoke",
        SMOKE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SMOKE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Issue31FileSystemCapacitySmokeTests(unittest.TestCase):
    def test_real_filesystem_hard_capacity_contract(self) -> None:
        smoke = load_smoke_module()

        with tempfile.TemporaryDirectory() as td:
            result = smoke.run_smoke(Path(td))

        required = {
            "schema_version",
            "filesystem_provenance",
            "max_bytes",
            "block_size",
            "peak_accounted_bytes",
            "peak_reserved_bytes",
            "peak_accounted_plus_reserved_bytes",
            "temp_peak_observed",
            "runtime_eviction_observed",
            "eviction_count",
            "evicted_bytes",
            "capacity_skips",
            "restart_recovered_bytes",
            "restart_recovery_ok",
            "startup_shrink_ok",
            "ownership_conflict_rejected",
            "final_payload_apparent_bytes",
        }
        self.assertTrue(
            required.issubset(result),
            required - set(result),
        )

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(
            result["filesystem_provenance"],
            "filesystem",
        )
        self.assertLessEqual(
            result["peak_accounted_plus_reserved_bytes"],
            result["max_bytes"],
        )
        self.assertTrue(result["temp_peak_observed"])
        self.assertTrue(result["runtime_eviction_observed"])
        self.assertGreaterEqual(result["eviction_count"], 1)
        self.assertGreaterEqual(
            result["evicted_bytes"],
            result["block_size"],
        )
        self.assertTrue(result["restart_recovery_ok"])
        self.assertTrue(result["startup_shrink_ok"])
        self.assertTrue(result["ownership_conflict_rejected"])


if __name__ == "__main__":
    unittest.main()
