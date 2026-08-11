# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from benchmarks.cache.cost_model_calibration import load_profile_artifact
from benchmarks.cache.cost_model_generalization import (
    diagnose_curve_scaling,
    evaluate_frozen_condition,
    load_generalization_condition,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen cache cost-model profile against an "
            "Issue #15 holdout condition."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--percentile",
        choices=("p50", "p95", "p99"),
        default="p95",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Add observational one-scalar curve diagnostics.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Return exit code 1 unless classification is "
            "fixed_profile_transfer_pass."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    condition = load_generalization_condition(
        args.input,
        percentile=args.percentile,
    )
    profile = load_profile_artifact(args.profile)

    result = evaluate_frozen_condition(
        condition,
        profile,
        profile_identity=str(args.profile),
    )

    if args.diagnose:
        result["diagnostics"] = diagnose_curve_scaling(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gate = result["gate"]["high_confidence"]
    print(
        "condition={condition} "
        "decision={correct}/{total} "
        "accuracy={accuracy:.6f} "
        "macro_mape={macro} "
        "classification={classification}".format(
            condition=result["condition_id"],
            correct=gate["decision_correct"],
            total=gate["decision_total"],
            accuracy=gate["decision_accuracy"],
            macro=gate["principal_macro_mape_percent"],
            classification=result["classification"],
        )
    )

    if args.check and (
        result["classification"] != "fixed_profile_transfer_pass"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
