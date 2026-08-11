from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.cache.cost_model_calibration import load_profile_artifact


def test_load_profile_artifact_returns_shadow_mapping(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "fixture",
                "provenance": {"profile_role": "before_calibration"},
                "cache_cost_model": {
                    "mode": "shadow",
                    "profile": {
                        "recompute_ms": {"128": 10.0},
                        "tiers": {"cpu_primary": {"restore_ms": {"128": 12.0}}},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_profile_artifact(path)

    assert profile["mode"] == "shadow"
    assert profile["profile"]["recompute_ms"] == {"128": 10.0}


def test_load_profile_artifact_rejects_missing_cost_model(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="cache_cost_model"):
        load_profile_artifact(path)


@pytest.fixture
def issue13_artifact(tmp_path: Path) -> Path:
    artifact = {
        "schema_version": 1,
        "issue": 13,
        "scope": {"requests_per_case": 8},
        "wide_curve": [
            {
                "requested_tokens": 256,
                "recompute_ttft_ms": {"p50": 20.0, "p95": 25.0, "p99": 27.0},
                "cpu_restore_ttft_ms": {"p50": 18.0, "p95": 21.0, "p99": 22.0},
                "tiered_fs_ttft_ms": {"p50": 28.0, "p95": 31.0, "p99": 33.0},
                "external_kv_tokens": 8 * 232,
            },
            {
                "requested_tokens": 512,
                "recompute_ttft_ms": {"p50": 40.0, "p95": 45.0, "p99": 47.0},
                "cpu_restore_ttft_ms": {"p50": 19.0, "p95": 23.0, "p99": 24.0},
                "tiered_fs_ttft_ms": {"p50": 50.0, "p95": 59.0, "p99": 61.0},
                "external_kv_tokens": 8 * 512,
            },
        ],
        "cpu_crossover_points": [
            {
                "requested_tokens": 192,
                "recompute_ttft_ms": {"p50": 18.0, "p95": 22.0, "p99": 23.0},
                "cpu_restore_ttft_ms": {"p50": 19.0, "p95": 21.8, "p99": 22.8},
                "external_kv_tokens": 8 * 168,
            },
            {
                "requested_tokens": 216,
                "recompute_ttft_ms": {"p50": 23.0, "p95": 24.8, "p99": 25.3},
                "cpu_restore_ttft_ms": {"p50": 20.0, "p95": 22.1, "p99": 22.9},
                "external_kv_tokens": 8 * 192,
            },
            {
                "requested_tokens": 224,
                "recompute_ttft_ms": {"p50": 23.2, "p95": 25.2, "p99": 26.0},
                "cpu_restore_ttft_ms": {"p50": 19.9, "p95": 22.3, "p99": 23.1},
                "external_kv_tokens": 8 * 192,
            },
        ],
        "boundary_repeats": {
            "192": {"p95_delta_ms": [-0.3, -0.2, -0.1]},
            "216": {"p95_delta_ms": [-3.2, -3.0, -2.3]},
        },
        "invalid_cpu_2g_sweep": [
            {"requested_tokens": 256, "classification": "recompute_not_cpu_restore"}
        ],
        "workload_generation_failure_208": [
            {
                "cache_mode": "cpu-offload",
                "status": "benchmark_error",
                "stage": "workload",
                "type": "WorkloadGenerationError",
            }
        ],
    }
    path = tmp_path / "issue13.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_dataset_uses_p95_and_external_tokens(issue13_artifact: Path) -> None:
    from benchmarks.cache.cost_model_calibration import load_issue13_dataset

    dataset = load_issue13_dataset(issue13_artifact, percentile="p95")

    assert dataset.percentile == "p95"
    assert dataset.requests_per_case == 8
    assert len(dataset.decision_samples) == 7

    cpu_256 = next(
        sample
        for sample in dataset.decision_samples
        if sample.source == "cpu_primary" and sample.requested_tokens == 256
    )
    assert cpu_256.external_tokens == 232
    assert cpu_256.actual_recompute_ms == pytest.approx(25.0)
    assert cpu_256.actual_restore_ms == pytest.approx(21.0)


def test_dataset_retains_repeat_directions_and_exclusions(
    issue13_artifact: Path,
) -> None:
    from benchmarks.cache.cost_model_calibration import load_issue13_dataset

    dataset = load_issue13_dataset(issue13_artifact)

    assert [item["requested_tokens"] for item in dataset.repeat_direction_checks] == [
        192,
        216,
    ]
    assert all(item["all_restore_faster"] for item in dataset.repeat_direction_checks)

    assert {item["reason"] for item in dataset.excluded_samples} == {
        "invalid_cpu_restore_provenance",
        "workload_generation_failure",
    }


def test_derive_profile_aggregates_duplicate_external_tokens_by_median(
    issue13_artifact: Path,
) -> None:
    from benchmarks.cache.cost_model_calibration import (
        derive_calibrated_profile,
        load_issue13_dataset,
    )

    dataset = load_issue13_dataset(issue13_artifact)
    before = {
        "mode": "shadow",
        "ewma_alpha": 0.2,
        "sample_scale_min": 0.25,
        "sample_scale_max": 4.0,
        "profile": {
            "recompute_ms": {256: 26.0},
            "tiers": {
                "cpu_primary": {"restore_ms": {256: 20.0}},
                "filesystem": {
                    "restore_ms": {256: 30.0},
                    "promotion_ms": {256: 12.0},
                },
            },
        },
    }

    calibrated = derive_calibrated_profile(dataset, before)

    assert calibrated["profile"]["recompute_ms"][192] == pytest.approx(25.0)
    assert calibrated["profile"]["tiers"]["cpu_primary"]["restore_ms"][
        192
    ] == pytest.approx(22.2)
    assert calibrated["profile"]["tiers"]["filesystem"]["promotion_ms"] == {256: 12.0}


def test_dataset_sorts_decision_samples_by_requested_tokens_and_source(
    issue13_artifact: Path,
) -> None:
    from benchmarks.cache.cost_model_calibration import load_issue13_dataset

    dataset = load_issue13_dataset(issue13_artifact)

    keys = [
        (sample.requested_tokens, sample.source) for sample in dataset.decision_samples
    ]
    assert keys == sorted(keys)


def test_dataset_rejects_zero_external_tokens(tmp_path: Path) -> None:
    from benchmarks.cache.cost_model_calibration import load_issue13_dataset

    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "scope": {"requests_per_case": 8},
                "wide_curve": [
                    {
                        "requested_tokens": 256,
                        "recompute_ttft_ms": {"p95": 25.0},
                        "cpu_restore_ttft_ms": {"p95": 21.0},
                        "tiered_fs_ttft_ms": {"p95": 31.0},
                        "external_kv_tokens": 0,
                    }
                ],
                "cpu_crossover_points": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive integer"):
        load_issue13_dataset(path)


def test_load_profile_artifact_requires_shadow_mode(tmp_path: Path) -> None:
    from benchmarks.cache.cost_model_calibration import load_profile_artifact

    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "cache_cost_model": {
                    "mode": "enforce",
                    "profile": {
                        "recompute_ms": {"128": 10.0},
                        "tiers": {"cpu_primary": {"restore_ms": {"128": 12.0}}},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mode must be shadow"):
        load_profile_artifact(path)


def test_load_profile_artifact_requires_profile_mapping(tmp_path: Path) -> None:
    from benchmarks.cache.cost_model_calibration import load_profile_artifact

    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "cache_cost_model": {
                    "mode": "shadow",
                    "profile": None,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires cache_cost_model.profile"):
        load_profile_artifact(path)


def test_evaluate_profile_scores_costs_decisions_and_boundary(
    issue13_artifact: Path,
) -> None:
    from benchmarks.cache.cost_model_calibration import (
        evaluate_profile,
        load_issue13_dataset,
    )

    dataset = load_issue13_dataset(issue13_artifact)
    before = {
        "mode": "shadow",
        "profile": {
            "recompute_ms": {
                168: 22.0,
                192: 25.0,
                232: 25.0,
                512: 45.0,
            },
            "tiers": {
                "cpu_primary": {
                    "restore_ms": {
                        168: 21.8,
                        192: 22.2,
                        232: 21.0,
                        512: 23.0,
                    }
                },
                "filesystem": {
                    "restore_ms": {
                        232: 31.0,
                        512: 59.0,
                    }
                },
            },
        },
    }

    result = evaluate_profile(dataset, before)

    sample_192 = next(
        row
        for row in result["samples"]
        if row["source"] == "cpu_primary" and row["requested_tokens"] == 192
    )

    assert sample_192["actual_preferred"] == "restore"
    assert sample_192["predicted_preferred"] == "restore"
    assert sample_192["boundary_sensitive"] is True
    assert sample_192["actual_margin_ms"] == pytest.approx(-0.2)
    assert result["aggregate"]["decision_total"] == 7


def test_recompute_mape_counts_unique_requested_anchors_once(
    issue13_artifact: Path,
) -> None:
    from benchmarks.cache.cost_model_calibration import (
        derive_calibrated_profile,
        evaluate_profile,
        load_issue13_dataset,
    )

    dataset = load_issue13_dataset(issue13_artifact)
    profile = derive_calibrated_profile(
        dataset,
        {
            "mode": "shadow",
            "profile": {
                "recompute_ms": {256: 1.0},
                "tiers": {
                    "cpu_primary": {
                        "restore_ms": {256: 1.0},
                    },
                    "filesystem": {
                        "restore_ms": {256: 1.0},
                        "promotion_ms": {256: 1.0},
                    },
                },
            },
        },
    )

    result = evaluate_profile(dataset, profile)
    aggregate = result["aggregate"]

    # Requested anchors: 192, 216, 224, 256, 512.
    # Wide points produce both CPU and filesystem decisions, but the
    # paired recompute observation must be counted only once.
    assert aggregate["recompute_sample_count"] == 5
    assert aggregate["cpu_restore_sample_count"] == 5
    assert aggregate["tiered_fs_restore_sample_count"] == 2

    expected = (
        aggregate["recompute_mape_percent"]
        + aggregate["cpu_restore_mape_percent"]
        + aggregate["tiered_fs_restore_mape_percent"]
    ) / 3.0

    assert aggregate["principal_macro_mape_percent"] == pytest.approx(expected)


def test_build_result_reports_before_after_and_acceptance(
    issue13_artifact: Path,
) -> None:
    from benchmarks.cache.cost_model_calibration import (
        build_calibration_result,
        derive_calibrated_profile,
        load_issue13_dataset,
    )

    dataset = load_issue13_dataset(issue13_artifact)
    before = {
        "mode": "shadow",
        "ewma_alpha": 0.2,
        "sample_scale_min": 0.25,
        "sample_scale_max": 4.0,
        "profile": {
            "recompute_ms": {
                256: 25.0,
                512: 45.0,
            },
            "tiers": {
                "cpu_primary": {
                    "restore_ms": {
                        512: 60.0,
                    }
                },
                "filesystem": {
                    "restore_ms": {
                        256: 31.0,
                        512: 59.0,
                    },
                    "promotion_ms": {
                        256: 10.0,
                        512: 20.0,
                    },
                },
            },
        },
    }

    after = derive_calibrated_profile(dataset, before)

    result = build_calibration_result(
        dataset,
        before,
        after,
        mape_threshold_percent=15.0,
        decision_accuracy_threshold=0.95,
    )

    assert result["before"]["aggregate"]["decision_accuracy"] < 1.0
    assert result["after"]["aggregate"]["decision_accuracy"] == 1.0

    assert result["acceptance"]["passed"] is True
    assert result["acceptance"]["decision_total"] == 7


def test_cli_writes_deterministic_json_and_check_status(
    issue13_artifact: Path,
    tmp_path: Path,
) -> None:
    from benchmarks.cache import evaluate_cost_model

    before_path = tmp_path / "before.json"
    before_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache_cost_model": {
                    "mode": "shadow",
                    "profile": {
                        "recompute_ms": {
                            "256": 25.0,
                            "512": 45.0,
                        },
                        "tiers": {
                            "cpu_primary": {
                                "restore_ms": {
                                    "512": 23.0,
                                }
                            },
                            "filesystem": {
                                "restore_ms": {
                                    "256": 31.0,
                                    "512": 59.0,
                                },
                                "promotion_ms": {
                                    "256": 10.0,
                                    "512": 20.0,
                                },
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    args = [
        "--input",
        str(issue13_artifact),
        "--before-profile",
        str(before_path),
        "--percentile",
        "p95",
        "--output",
        str(output),
        "--check",
    ]

    rc = evaluate_cost_model.main(args)
    assert rc == 0

    first = output.read_text(encoding="utf-8")
    first_result = json.loads(first)
    assert first_result["acceptance"]["passed"] is True

    rc = evaluate_cost_model.main(args)
    assert rc == 0
    second = output.read_text(encoding="utf-8")

    assert second == first


def test_build_result_applies_configured_boundary_margin(
    issue13_artifact: Path,
) -> None:
    from benchmarks.cache.cost_model_calibration import (
        build_calibration_result,
        derive_calibrated_profile,
        load_issue13_dataset,
    )

    dataset = load_issue13_dataset(issue13_artifact)
    before = {
        "mode": "shadow",
        "profile": {
            "recompute_ms": {256: 25.0, 512: 45.0},
            "tiers": {
                "cpu_primary": {"restore_ms": {512: 23.0}},
                "filesystem": {
                    "restore_ms": {256: 31.0, 512: 59.0},
                    "promotion_ms": {256: 10.0, 512: 20.0},
                },
            },
        },
    }
    after = derive_calibrated_profile(dataset, before)

    result = build_calibration_result(
        dataset,
        before,
        after,
        boundary_margin_ms=0.1,
    )

    sample_192 = next(
        row
        for row in result["after"]["samples"]
        if row["source"] == "cpu_primary" and row["requested_tokens"] == 192
    )
    assert sample_192["boundary_sensitive"] is False


def test_cli_check_returns_one_when_acceptance_fails(
    issue13_artifact: Path,
    tmp_path: Path,
) -> None:
    from benchmarks.cache import evaluate_cost_model

    raw = json.loads(issue13_artifact.read_text(encoding="utf-8"))
    point_216 = next(
        row for row in raw["cpu_crossover_points"] if row["requested_tokens"] == 216
    )
    point_216["cpu_restore_ttft_ms"]["p95"] = 40.0

    failing_input = tmp_path / "failing-issue13.json"
    failing_input.write_text(json.dumps(raw), encoding="utf-8")

    before_path = tmp_path / "before.json"
    before_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache_cost_model": {
                    "mode": "shadow",
                    "profile": {
                        "recompute_ms": {"256": 25.0, "512": 45.0},
                        "tiers": {
                            "cpu_primary": {"restore_ms": {"512": 23.0}},
                            "filesystem": {
                                "restore_ms": {
                                    "256": 31.0,
                                    "512": 59.0,
                                },
                                "promotion_ms": {
                                    "256": 10.0,
                                    "512": 20.0,
                                },
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "result.json"
    rc = evaluate_cost_model.main(
        [
            "--input",
            str(failing_input),
            "--before-profile",
            str(before_path),
            "--percentile",
            "p95",
            "--output",
            str(output),
            "--check",
        ]
    )

    assert rc == 1
    assert (
        json.loads(output.read_text(encoding="utf-8"))["acceptance"]["passed"] is False
    )


def test_cli_prints_one_compact_after_summary(
    issue13_artifact: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from benchmarks.cache import evaluate_cost_model

    before_path = tmp_path / "before-summary.json"
    before_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache_cost_model": {
                    "mode": "shadow",
                    "profile": {
                        "recompute_ms": {
                            "256": 25.0,
                            "512": 45.0,
                        },
                        "tiers": {
                            "cpu_primary": {"restore_ms": {"512": 23.0}},
                            "filesystem": {
                                "restore_ms": {
                                    "256": 31.0,
                                    "512": 59.0,
                                },
                                "promotion_ms": {
                                    "256": 10.0,
                                    "512": 20.0,
                                },
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary-result.json"

    rc = evaluate_cost_model.main(
        [
            "--input",
            str(issue13_artifact),
            "--before-profile",
            str(before_path),
            "--percentile",
            "p95",
            "--output",
            str(output),
            "--check",
        ]
    )

    assert rc == 0
    stdout = capsys.readouterr().out
    assert stdout.count("\n") == 1
    assert stdout.startswith("after: decision=7/7 accuracy=1.000 macro_mape=")
    assert stdout.endswith("% passed=True\n")


def test_cli_creates_missing_output_parent(
    issue13_artifact: Path,
    tmp_path: Path,
) -> None:
    from benchmarks.cache import evaluate_cost_model

    before_path = tmp_path / "before-parent.json"
    before_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache_cost_model": {
                    "mode": "shadow",
                    "profile": {
                        "recompute_ms": {"256": 25.0, "512": 45.0},
                        "tiers": {
                            "cpu_primary": {"restore_ms": {"512": 23.0}},
                            "filesystem": {
                                "restore_ms": {
                                    "256": 31.0,
                                    "512": 59.0,
                                },
                                "promotion_ms": {
                                    "256": 10.0,
                                    "512": 20.0,
                                },
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "nested" / "result.json"
    rc = evaluate_cost_model.main(
        [
            "--input",
            str(issue13_artifact),
            "--before-profile",
            str(before_path),
            "--percentile",
            "p95",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.is_file()
