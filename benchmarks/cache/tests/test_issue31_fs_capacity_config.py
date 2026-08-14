# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from benchmarks.cache.config import FilesystemCacheConfig, load_suite_config
from benchmarks.cache.scenarios import (
    CacheMode,
    build_execution_cases,
    build_server_command,
)


class Issue31FilesystemCapacityConfigTests(unittest.TestCase):
    def test_enabled_filesystem_requires_positive_plain_integer_max_bytes(
        self,
    ) -> None:
        with self.assertRaises(ValidationError):
            FilesystemCacheConfig(enabled=True, root_dir=Path("/tmp/cache"))

        for value in (None, 0, -1, True, 4096.0, "4096"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                FilesystemCacheConfig(
                    enabled=True,
                    root_dir=Path("/tmp/cache"),
                    max_bytes=value,
                )

    def test_disabled_filesystem_may_omit_max_bytes(self) -> None:
        config = FilesystemCacheConfig(
            enabled=False,
            root_dir=Path("/tmp/cache"),
        )
        self.assertIsNone(config.max_bytes)

    def test_enabled_filesystem_configs_have_explicit_nonbinding_capacity(self) -> None:
        config_paths = (
            Path("benchmarks/cache/configs/example-7b.yaml"),
            Path("benchmarks/cache/configs/example-70b.yaml"),
            Path("benchmarks/cache/configs/example-397b.yaml"),
            Path("benchmarks/cache/configs/local-crossover.yaml"),
            Path("benchmarks/cache/configs/issue15-7b-load-sentinel-fs.yaml"),
            Path("benchmarks/cache/configs/issue15-14b-formal-fs.yaml"),
        )

        for config_path in config_paths:
            with self.subTest(config=str(config_path)):
                config = load_suite_config(config_path)
                self.assertTrue(config.cache.filesystem.enabled)
                self.assertEqual(
                    config.cache.filesystem.max_bytes,
                    1099511627776,
                )

    def test_tiered_server_config_forwards_max_bytes(self) -> None:
        config = load_suite_config(Path("benchmarks/cache/configs/example-7b.yaml"))

        with tempfile.TemporaryDirectory() as temp_dir:
            case = next(
                case
                for case in build_execution_cases(config, Path(temp_dir))
                if case.cache_mode is CacheMode.TIERED_FS
                and case.workload_kind == "cold-unique"
            )

        command = build_server_command(case, config)
        payload = json.loads(command[command.index("--kv-transfer-config") + 1])
        fs_config = payload["kv_connector_extra_config"]["secondary_tiers"][0]

        self.assertEqual(
            fs_config["max_bytes"],
            config.cache.filesystem.max_bytes,
        )


if __name__ == "__main__":
    unittest.main()
