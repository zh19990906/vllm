# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.cache.cost_model_calibration import (  # noqa: E402
    build_calibration_result,
    derive_calibrated_profile,
    load_issue13_dataset,
    load_profile_artifact,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--before-profile", required=True, type=Path)
    parser.add_argument(
        "--percentile",
        choices=("p50", "p95", "p99"),
        default="p95",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    dataset = load_issue13_dataset(
        args.input,
        percentile=args.percentile,
    )
    before = load_profile_artifact(args.before_profile)
    after = derive_calibrated_profile(dataset, before)
    result = build_calibration_result(dataset, before, after)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    after_agg = result["after"]["aggregate"]
    print(
        "after: "
        f"decision={after_agg['decision_correct']}/"
        f"{after_agg['decision_total']} "
        f"accuracy={after_agg['decision_accuracy']:.3f} "
        f"macro_mape="
        f"{after_agg['principal_macro_mape_percent']:.3f}% "
        f"passed={result['acceptance']['passed']}"
    )

    if args.check and not result["acceptance"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
