# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import unittest

from benchmarks.cache.metrics import _ENVIRONMENT_COMMANDS


class Issue15EnvironmentProvenanceTests(unittest.TestCase):
    def test_gpu_inventory_captures_index_and_uuid(self) -> None:
        command = _ENVIRONMENT_COMMANDS["gpu_inventory"]

        self.assertIn(
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            command,
        )


if __name__ == "__main__":
    unittest.main()
