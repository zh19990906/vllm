# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

_HELPER_PATH = Path(__file__).with_name("test_shadow_cost_spec.py")
_SPEC = importlib.util.spec_from_file_location(
    "shadow_cost_spec_test_helpers", _HELPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPERS
_SPEC.loader.exec_module(_HELPERS)

TieringOffloadingSpec = _HELPERS.TieringOffloadingSpec


def test_runtime_tier_config_removes_cost_model_only_key() -> None:
    raw = {
        "type": "fs",
        "root_dir": "/tmp/cache",
        "cost_model_tier_key": "filesystem",
    }

    runtime = TieringOffloadingSpec._runtime_tier_config(raw)

    assert runtime == {"type": "fs", "root_dir": "/tmp/cache"}
    assert raw["cost_model_tier_key"] == "filesystem"
