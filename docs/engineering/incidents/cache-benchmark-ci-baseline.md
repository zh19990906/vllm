# Incident: `benchmarks/cache` Static-Check Baseline Is Already Red

Status: **known baseline; cleanup should be isolated from feature PRs**.

Observed on latest `main` at commit:

```text
37f65141108e112a317fe4a5d8215a4c21c3c00e
```

Date observed: **2026-08-10**.

## Symptom

Repository-wide pre-commit or direct static checks report failures in
`benchmarks/cache/**`, including files untouched by the feature being validated.

This appeared during PR #7 completion and again while preparing PR #5. If treated as a
feature-local failure, it encourages large formatting/licensing diffs in otherwise focused
PRs.

## Repository rules involved

The observed pre-commit configuration includes:

- Ruff check with `--fix`;
- Ruff format;
- markdownlint-cli2 with `--fix`;
- a custom SPDX header checker.

Python/Rust/proto SPDX checking uses the repository script
`tools/pre_commit/check_spdx_header.py`.

The normal Python SPDX header format elsewhere in `benchmarks` is:

```python
# SPDX-License-Identifier: Apache-2.0
```

## Baseline reproduced on `main`

Direct checks against `benchmarks/cache` on `main@37f651411...` reported:

```text
Ruff check:        21 errors
Ruff auto-fixable: 4
Ruff format:       13 files would be reformatted
Missing SPDX:      16 Python files
```

### Ruff baseline examples

The observed errors included:

- `SIM114` and `UP037` in `benchmarks/cache/config.py`;
- `UP035` and line-length problems in `benchmarks/cache/metrics.py`;
- multiple line-length problems in `benchmarks/cache/report.py`;
- line length in `benchmarks/cache/run_suite.py`;
- line length in existing cache tests;
- `UP012` in `benchmarks/cache/workload.py`.

These are not all introduced by PR #5.

### Files Ruff format wanted to change on `main`

Observed list:

```text
benchmarks/cache/config.py
benchmarks/cache/metrics.py
benchmarks/cache/process.py
benchmarks/cache/report.py
benchmarks/cache/run_suite.py
benchmarks/cache/scenarios.py
benchmarks/cache/tests/test_metrics.py
benchmarks/cache/tests/test_process.py
benchmarks/cache/tests/test_report.py
benchmarks/cache/tests/test_run_suite.py
benchmarks/cache/tests/test_scenarios.py
benchmarks/cache/tests/test_workload.py
benchmarks/cache/workload.py
```

### Missing SPDX files on `main`

Observed list:

```text
benchmarks/cache/process.py
benchmarks/cache/report.py
benchmarks/cache/metrics.py
benchmarks/cache/workload.py
benchmarks/cache/scenarios.py
benchmarks/cache/tests/test_workload.py
benchmarks/cache/tests/test_report.py
benchmarks/cache/tests/conftest.py
benchmarks/cache/tests/test_process.py
benchmarks/cache/tests/test_config.py
benchmarks/cache/tests/test_scenarios.py
benchmarks/cache/tests/test_metrics.py
benchmarks/cache/tests/test_run_suite.py
benchmarks/cache/config.py
benchmarks/cache/__init__.py
benchmarks/cache/run_suite.py
```

PR #7's final PR description also recorded repository-wide pre-commit failure in this
benchmark area for Ruff formatting/E501, markdownlint, and SPDX headers. PR #7 deliberately
did not fold those unrelated benchmark changes into its 14-file runtime diff.

## PR #5 comparison

The observed PR #5 tree reported:

```text
Ruff check:        24 errors
Ruff auto-fixable: 5
Ruff format:       15 files would be reformatted
Missing SPDX:      18 Python files
```

Compared with the main baseline, the observed delta is:

- +3 Ruff errors;
- +2 files requiring Ruff formatting;
- +2 files missing SPDX headers.

The additional PR #5 format targets included feature files such as
`test_eviction_restore_workload.py` and `test_pressure_token_budget.py`. The additional
missing-SPDX paths included PR #5 test files such as population-result validation and
workload-search coverage.

The exact classification should be re-run after any main baseline cleanup; counts are a
snapshot, not a permanent invariant.

## Why this matters

Running:

```text
ruff format benchmarks/cache
```

inside PR #5 would mechanically modify many existing files outside the 13-file feature
boundary. The resulting PR might become statically cleaner while becoming much harder to
review and reason about.

The same problem applies to blanket SPDX and markdownlint auto-fixes.

## Recommended resolution strategy

Treat the problem as two layers.

### Layer 1: cache benchmark hygiene baseline

Create a dedicated branch from latest `main` that performs only mechanical/static cleanup
for existing `benchmarks/cache` files:

- Ruff fixes and formatting;
- SPDX headers;
- markdownlint fixes;
- no feature behavior changes.

Review the generated diff before committing because auto-fix hooks are allowed to edit
files.

### Layer 2: PR #5 feature-specific cleanup

Once the baseline is known or merged, refresh PR #5 and fix only remaining problems caused
by its own feature files. Re-run the complete cache test suite afterward.

## False conclusions to avoid

- Do not say PR #5 introduced all current cache Ruff/format/SPDX failures.
- Do not say PR #7 was functionally red because repository-wide pre-commit touched
  benchmark files outside PR #7.
- Do not make an unrelated feature PR own a directory-wide formatting migration solely to
  get one CI job green.
- Do not trust static-check counts forever; they are tied to a specific `main` and PR head.
- Do not run auto-fix hooks and immediately commit without reviewing which paths changed.

## Verification before a hygiene PR merges

A dedicated cleanup should establish:

1. diff contains only intended cache benchmark documentation/static cleanup;
2. `benchmarks/cache/tests` still pass, accounting for the known fake-executable fixture
   until it is fixed permanently;
3. `python -m compileall -q benchmarks/cache` passes;
4. Ruff check passes;
5. Ruff format check passes;
6. SPDX checker passes;
7. markdownlint passes;
8. `git diff --check` passes.

## Related records

- [`fake-vllm-newline-fixture.md`](fake-vllm-newline-fixture.md)
- [`../history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md`](../history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md)
- [`../history/2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md`](../history/2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md)
- [`../handoffs/2026-08-10-pr5-current-handoff.md`](../handoffs/2026-08-10-pr5-current-handoff.md)
