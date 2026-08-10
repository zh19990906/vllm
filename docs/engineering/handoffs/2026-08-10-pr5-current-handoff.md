# PR #5 Current Handoff

Snapshot date: **2026-08-10**.

This is a point-in-time handoff. Verify GitHub metadata and local workspace status before
performing writes.

## Read first

The active engineering goal is to finish PR #5 without mixing unrelated cache benchmark
baseline cleanup into its feature diff.

Before changing code, read:

1. [`../CURRENT_STATE.md`](../CURRENT_STATE.md)
2. [`../incidents/cache-benchmark-ci-baseline.md`](../incidents/cache-benchmark-ci-baseline.md)
3. [`../incidents/fake-vllm-newline-fixture.md`](../incidents/fake-vllm-newline-fixture.md)
4. [`../history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md`](../history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md)
5. [`../validation/cache-crossover-baseline.md`](../validation/cache-crossover-baseline.md)

PR #7 is already completed and merged. Do not restart PR #7 development unless a new
regression or new requirement explicitly requires it.

## Authoritative GitHub state

Observed through GitHub on 2026-08-10.

### `main`

```text
37f65141108e112a317fe4a5d8215a4c21c3c00e
```

This is the merge commit for PR #7.

### PR #3

```text
PR: #3
state: merged
branch: fix/cache-workload-fairness
final head: 4fa33f8267de7cbc4d95208886de8e63f028cb3a
merge commit: abc426ae063e90c837c48dfbe75fabe919c82575
```

PR #3 supplies workload fairness and tokenizer convergence fixes that PR #5 relies on.

### PR #5

```text
PR: #5
title: Add cache eviction restore benchmark workload
state: open
draft: true
mergeable: true
branch: feature/cache-eviction-restore-benchmark
head: b91b6ee631a8145a5eabb55542ad733dfacff24a
changed files: 13
GitHub diff size: +1522 / -17
```

Observed changed paths:

```text
benchmarks/cache/config.py
benchmarks/cache/configs/local-crossover.yaml
benchmarks/cache/run_suite.py
benchmarks/cache/scenarios.py
benchmarks/cache/tests/test_eviction_restore_workload.py
benchmarks/cache/tests/test_population_result_validation.py
benchmarks/cache/tests/test_pressure_token_budget.py
benchmarks/cache/tests/test_workload_search.py
benchmarks/cache/workload.py
docs/superpowers/plans/2026-08-07-cache-crossover-sweep.md
docs/superpowers/plans/2026-08-07-cache-population-result-validation.md
docs/superpowers/specs/2026-08-07-cache-crossover-sweep-design.md
docs/superpowers/specs/2026-08-07-cache-population-result-validation-design.md
```

### PR #7

```text
PR: #7
state: merged
branch: feature/kv-offload-shadow-cost-model
final head: 96de0c823721c374527dbb0b3a49fdc7eccba341
merge commit: 37f65141108e112a317fe4a5d8215a4c21c3c00e
```

PR #7 hardware and scoped validation are complete. See the dedicated validation record.

## Repository synchronization constraint

The working topology is:

```text
GitHub authoritative branches/PRs
        |
        v
Gitee mirror sync
        |
        v
Pod clone/fetch for tests and hardware
```

The Pod/local repository must not be treated as a GitHub write path. In particular:

- do not tell the Pod to push directly to GitHub;
- do not create an important local merge commit and then assume it can update the GitHub
  PR;
- perform authoritative branch/PR writes through a GitHub-capable mechanism;
- after GitHub changes, sync them to the Gitee mirror and fetch them in the Pod.

This constraint previously changed the PR #5 update strategy: a conflict-free local merge
preview was useful evidence, but the merge commit itself was intentionally not created as
the authoritative update.

## Observed local workspaces

These paths are machine-local observations from 2026-08-10. Verify before use.

### `/code/vllm-main`

Observed as a clean clone of latest main at:

```text
37f65141108e112a317fe4a5d8215a4c21c3c00e
```

It was used to reproduce the existing `benchmarks/cache` static-check baseline.

### `/code/vllm-pr5-final`

Clean PR #5 workspace created from the Gitee mirror branch.

Observed state after aborting the merge preview:

```text
branch: feature/cache-eviction-restore-benchmark
HEAD: b91b6ee631a8145a5eabb55542ad733dfacff24a
status: clean
```

This is the preferred local workspace for PR #5 scoped validation because it starts from
the authoritative PR head and does not contain the older divergent local merge history.

### `/code/vllm-cache-hygiene`

A separate workspace was created from latest main for baseline cleanup exploration:

```text
base: main@37f65141108e112a317fe4a5d8215a4c21c3c00e
branch: fix/cache-benchmark-hygiene
```

At the last confirmed observation, it had only been created and inspected for pre-commit
rules/SPDX convention. Do **not** assume a full `pre-commit --all-files` auto-fix was later
run; verify status before continuing.

### `/code/vllm`

This older local PR #5 workspace is intentionally not preferred for final cleanup.

Historical observation:

- local head included a merge commit not present on the remote PR branch;
- remote PR branch and local HEAD had diverged;
- local experimental config files were intentionally preserved.

Observed untracked experimental files included:

```text
benchmarks/cache/configs/local-pressure-cpu-hit.yaml
benchmarks/cache/configs/local-pressure.yaml
benchmarks/cache/configs/local-smoke.yaml
```

Do not delete or accidentally stage these machine-local experiment configs.

## Local merge preview already performed

In `/code/vllm-pr5-final`, latest main was fetched and a no-commit/no-ff merge preview was
performed against PR #5.

Result:

- merge had no conflicts;
- comparison against latest `main` still showed exactly the 13 PR #5 paths;
- feature boundary remained clean;
- the merge preview was aborted;
- no local merge commit was kept.

Therefore there is no known semantic merge conflict between the current PR #5 head and
`main@37f651411...` at this snapshot.

Do not repeat the preview unless remote heads have changed.

## PR #5 functional validation already performed

A temporary virtual environment was created for cache tests because the default interpreter
in the clean workspace did not have pytest installed.

Initial full run:

```text
70 passed
1 failed
```

The only failure was:

```text
benchmarks/cache/tests/test_run_suite.py::test_fake_executable_end_to_end
```

The generated fake executable was inspected and directly compiled. It contained a bytes
literal split by an outer-string newline expansion and failed with:

```text
SyntaxError: unterminated string literal
```

After changing only the fixture escape in the temporary runner workspace, the complete
suite reported:

```text
71 passed in 2.44s
pytest_rc=0
compile_rc=0
```

The fixture was automatically restored afterward and the PR #5 worktree returned clean.

Interpretation:

> At this snapshot, no PR #5 functional regression was found in the cache test suite once
> the known unrelated fake-executable fixture baseline was isolated.

Do not permanently modify or cite the fixture as a PR #5 feature fix without treating it
as an independent baseline issue.

## Static-check baseline already measured

### Latest `main`

Direct checks against `benchmarks/cache` reported:

```text
Ruff errors: 21
Ruff fixable: 4
files Ruff would format: 13
Python files missing SPDX: 16
```

The repository standard also has markdownlint failures in the cache benchmark area.

### PR #5 tree

Direct checks reported:

```text
Ruff errors: 24
Ruff fixable: 5
files Ruff would format: 15
Python files missing SPDX: 18
```

Observed feature-vs-baseline delta:

```text
+3 Ruff errors
+2 format targets
+2 missing SPDX files
```

Counts must be re-measured after any baseline cleanup; they are not permanent invariants.

## Recommended next workstream

### Phase 1: cache hygiene baseline

Use `/code/vllm-cache-hygiene` or a fresh equivalent based on the current authoritative
`main`.

Before any auto-fix:

```bash
git status --short --branch
git rev-parse HEAD
```

Then run the repository's actual hooks or their exact components. Auto-fix hooks may return
nonzero after modifying files; inspect the resulting diff before deciding what belongs.

The intended hygiene scope is mechanical only:

- Ruff check fixes;
- Ruff formatting;
- SPDX headers;
- markdownlint corrections;
- no cache benchmark behavior change.

Because the Pod is not the GitHub write path, produce and review the patch locally, then
apply the authoritative branch update through GitHub-side tooling.

### Phase 2: refresh PR #5

After the baseline hygiene strategy is resolved, update PR #5 against the latest main via a
GitHub-capable path.

Do not make a Pod-only merge commit and call the PR refreshed.

### Phase 3: PR #5-specific cleanup

Re-run Ruff/format/SPDX/markdown checks and fix only remaining feature-specific findings.
Preserve the intended 13-file boundary unless a separately reviewed baseline change is
being merged first.

### Phase 4: tests

Re-run:

```text
benchmarks/cache/tests
compileall benchmarks/cache
Ruff check
Ruff format check
SPDX check
markdownlint
git diff --check
```

Account explicitly for the fake-executable fixture depending on whether the baseline fix
has merged by then.

### Phase 5: hardware acceptance only if needed

Do not repeat the large historical crossover and PR #7 shadow sweeps without a new
hypothesis.

For PR #5, hardware work should answer only remaining benchmark correctness/acceptance
questions, such as verifying that the refreshed scenario still creates real lower-tier
pressure after cleanup.

### Phase 6: PR completion

When validation is green:

1. update PR #5 description with current dependency and validation state;
2. remove stale language saying the PR is stacked on an unmerged PR #3;
3. mark the PR ready for review;
4. verify final GitHub checks and changed-file scope;
5. merge using the repository-supported strategy.

## Expensive work that is already complete

Do not repeat these without a changed hypothesis:

- PR #7 59 focused test validation;
- PR #7 real scheduler integration: 93 passed;
- CPU-primary 1024 shadow anchor;
- filesystem 1024 shadow anchor;
- filesystem 256/512/1024/2048/4096 shadow sweep;
- p256 adaptive crossover reproduction;
- source-over-wheel runtime provenance proof;
- original cache recompute/filesystem crossover seed sweep.

See:
[`../validation/pr7-shadow-cost-model-hardware-validation.md`](../validation/pr7-shadow-cost-model-hardware-validation.md).

## Important negative knowledge

- Filesystem restore was slower than recompute for the original 256-4096 seed sweep, but
  this is machine/model-specific evidence rather than a universal rule.
- CPU-primary 1024 restore was much cheaper than recompute.
- PR #7 p256 can change shadow preference as EWMA runtime scale crosses the decision
  boundary; this is expected and did not change actual execution.
- One p4096 metrics-before connection-refused event did not reproduce; no retry was added.
- A worktree containing feature source does not prove the benchmark subprocess runs that
  source.
- The fake `vllm` test failure is a generated-script syntax defect, not evidence that
  `run_suite.main` is broken.

## Completion report expected from the next session

When PR #5 is eventually completed, update `CURRENT_STATE.md` and add a final history or
validation entry containing:

- final PR head and merge commit;
- final changed-file list;
- exact test commands and results;
- static-check results;
- hardware acceptance evidence, if any new hardware run was necessary;
- resolved and remaining risks;
- whether baseline hygiene was merged separately or incorporated by another approved
  strategy.
