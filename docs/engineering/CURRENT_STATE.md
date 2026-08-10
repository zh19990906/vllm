# Current Engineering State

Observed: **2026-08-10**.

This file is intentionally mutable. Verify remote metadata before acting on branch heads or
PR status if this file is older than the current session.

## Repository state

- Repository: `zh19990906/vllm`
- Default branch: `main`
- Current observed `main` head:
  `37f65141108e112a317fe4a5d8215a4c21c3c00e`
- That commit merged PR #7, `Add KV offload shadow cost model`.

## Recently completed work

### PR #3: cache workload fairness and convergence

- PR: `#3`
- Branch: `fix/cache-workload-fairness`
- Final head: `4fa33f8267de7cbc4d95208886de8e63f028cb3a`
- Merge commit: `abc426ae063e90c837c48dfbe75fabe919c82575`
- State: merged on 2026-08-07.

PR #3 fixed two correctness problems in the cache benchmark workload generator:
matching cache modes no longer receive different prompt bytes because of case-specific RNG
identity, and token-length convergence no longer discards the entire suffix stream after
every correction.

See:
[`history/2026-08-07-pr3-cache-workload-fairness-convergence.md`](history/2026-08-07-pr3-cache-workload-fairness-convergence.md).

### PR #7: KV offload shadow cost model

- PR: `#7`
- Branch: `feature/kv-offload-shadow-cost-model`
- Final head: `96de0c823721c374527dbb0b3a49fdc7eccba341`
- Merge commit: `37f65141108e112a317fe4a5d8215a4c21c3c00e`
- State: merged on 2026-08-10.

The merged feature is shadow-only. It predicts restore versus recompute cost, records load
provenance and low-cardinality metrics, and calibrates secondary-tier promotion cost with
an online EWMA without changing the actual restore path. `mode: enforce` remains rejected.

Final scoped validation included 59 focused tests, compile checks, targeted Ruff check and
format check, `git diff --check`, Python 3.10-3.13 mypy CI, real scheduler integration, and
hardware anchors. The p256 filesystem case showed an expected adaptive shadow crossover
after runtime EWMA calibration; the actual path remained restore.

See:

- [`history/2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md`](history/2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md)
- [`validation/pr7-shadow-cost-model-hardware-validation.md`](validation/pr7-shadow-cost-model-hardware-validation.md)

## Active work

### PR #5: cache eviction/restore benchmark workload

Observed GitHub state on 2026-08-10:

- PR: `#5`
- Title: `Add cache eviction restore benchmark workload`
- Branch: `feature/cache-eviction-restore-benchmark`
- Head: `b91b6ee631a8145a5eabb55542ad733dfacff24a`
- State: open and draft
- Mergeable: true
- Changed files: 13
- Additions/deletions reported by GitHub: `+1522 / -17`

The PR creates an opt-in pressure workload that populates victims, adds unique fillers to
force cache pressure, then replays only victims to measure lower-tier restore behavior. It
also contains crossover sweep and population-result validation work.

A clean local PR #5 checkout reproduced 70 passing tests plus one unrelated baseline
fixture failure. After correcting only that fixture in the temporary runner workspace, the
full `benchmarks/cache/tests` suite passed **71/71** and `compileall` passed. The temporary
fixture modification was restored afterward and the worktree was clean.

Current continuation document:
[`handoffs/2026-08-10-pr5-current-handoff.md`](handoffs/2026-08-10-pr5-current-handoff.md).

## Known repository baseline: `benchmarks/cache`

The latest observed `main` already fails repository static hygiene in this directory.
Before PR #5 cleanup, the observed baseline was:

- Ruff check: 21 errors, 4 reported as auto-fixable.
- Ruff format: 13 files would be reformatted.
- SPDX: 16 Python files missing `# SPDX-License-Identifier: Apache-2.0`.
- The standard pre-commit path has also exposed markdownlint failures in the cache
  benchmark documentation.

The observed PR #5 tree had:

- Ruff check: 24 errors, 5 reported as auto-fixable.
- Ruff format: 15 files would be reformatted.
- SPDX: 18 Python files missing the required header.

The delta is therefore much smaller than the repository-wide cache baseline: three Ruff
errors, two additional format targets, and two additional missing-SPDX files were observed
on PR #5. Do not solve this by blindly formatting all of `benchmarks/cache` inside PR #5;
that would mix a broad baseline cleanup into a feature PR.

See:
[`incidents/cache-benchmark-ci-baseline.md`](incidents/cache-benchmark-ci-baseline.md).

## Known test baseline: fake `vllm` fixture

`benchmarks/cache/tests/test_run_suite.py::test_fake_executable_end_to_end` can fail because
its outer triple-quoted Python string contains a nested bytes literal with `\n`. The outer
string interprets that escape and writes an actual newline into the generated executable,
creating an unterminated bytes literal.

This is not evidence of a `run_suite.main()` regression. The generated script must be
inspected or compiled before attributing the failure to PR #5 behavior.

See:
[`incidents/fake-vllm-newline-fixture.md`](incidents/fake-vllm-newline-fixture.md).

## Current objective

Finish PR #5 without contaminating its feature scope with unrelated repository-wide cache
hygiene.

Recommended sequence:

1. Keep the PR #5 feature workspace based on the authoritative PR head.
2. Handle cache benchmark baseline hygiene as a separate workstream/branch.
3. Re-run PR #5 tests against the corrected baseline.
4. Apply only PR #5-specific static fixes after the baseline is known.
5. Update the GitHub PR branch through a GitHub-capable write path.
6. Sync GitHub to Gitee, then fetch in the Pod for final validation.
7. Mark PR #5 ready and merge only after final scoped and CI validation.

## Do not repeat without a new hypothesis

- Do not re-run the full PR #7 hardware sweep merely to reconfirm already archived
  behavior.
- Do not assume a benchmark subprocess uses feature source just because a worktree contains
  it; validate runtime module provenance. See `incidents/native-wheel-exact-overlay.md`.
- Do not interpret the p256 adaptive shadow crossover as an execution-path change.
- Do not add retry logic because of the single non-reproduced metrics collection failure.
- Do not merge PR #5 into PR #7 history; PR #7 is already merged and the branches were
  intentionally kept separate.
- Do not create a local merge commit in a Pod workspace and treat it as the authoritative
  GitHub update if the Pod cannot push to GitHub.

## Synchronization constraint

The observed development workflow uses GitHub as authoritative remote state, a Gitee
mirror for Pod fetches, and Pod workspaces for tests and hardware runs. GitHub mutations
must happen through a GitHub-capable path; mirror sync happens afterward.
