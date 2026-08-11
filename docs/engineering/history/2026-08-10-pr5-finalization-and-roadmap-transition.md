# PR #5 Finalization and Roadmap Transition

Date: **2026-08-10**.

## Purpose

This record closes the final PR #5 continuation work and marks the transition from
benchmark-infrastructure completion to an issue-driven roadmap focused on adaptive KV
cache decisions.

## PR #5 finalization

PR #5, `Add cache eviction restore benchmark workload`, was completed and merged after the
feature-specific static delta was separated from repository-wide `benchmarks/cache`
baseline failures.

Final identity:

- PR: `#5`
- Branch: `feature/cache-eviction-restore-benchmark`
- Final head: `308857612f883232912cca98d9f1fdae4ec6d5c2`
- Merge commit: `f628fbafe45431bcb9579c48d818455f7197add5`
- Merged: `2026-08-10T05:35:53Z`

The final GitHub branch write before merge contained exactly five validated cleanup files
relative to the previous PR head:

```text
benchmarks/cache/config.py
benchmarks/cache/tests/test_eviction_restore_workload.py
benchmarks/cache/tests/test_population_result_validation.py
benchmarks/cache/tests/test_pressure_token_budget.py
benchmarks/cache/tests/test_workload_search.py
```

Those changes were limited to one PR-local Ruff annotation fix, Ruff formatting of two
feature-specific tests, and SPDX headers in two feature-specific tests.

## Functional verification

The final PR-local validation evidence included:

- 26 focused cache tests passing;
- a full `benchmarks/cache/tests` run passing 71/71 after temporarily correcting only the
  known unrelated fake-executable nested-newline fixture defect;
- restoration of that temporary fixture edit after the validation run;
- `python -m compileall -q benchmarks/cache` passing;
- `git diff --check` passing.

The temporary fixture correction was validation-only and was not merged as part of PR #5.
It is now tracked separately by GitHub issue #21.

## Static-check classification

The final PR #5 pre-commit workflow remained red because the workflow executes hooks with
`--all-files`.

A direct comparison against the PR #8/main baseline established that PR #5 added no new
static debt in the observed failure classes:

- 18 remaining Ruff E501 errors after Ruff auto-fix;
- 13 Ruff-format targets;
- 8 markdownlint MD060 errors at `benchmarks/cache/README.md:52`;
- 16 older cache Python files missing SPDX headers.

The line numbers of existing errors shifted in files where PR #5 added code, but the
failure pattern matched the independent main baseline. The repository-wide cleanup is now
tracked separately by issue #20.

## Merge and synchronization verification

After merge, the normal synchronization chain was exercised:

```text
GitHub main
    -> synchronization mechanism
    -> Gitee mirror
    -> Pod fetch
```

The Pod fetched:

```text
origin/main = f628fbafe45431bcb9579c48d818455f7197add5
```

The isolated `/code/vllm-pr5-final` workspace still had the same five local validation
edits on its old PR head. Comparing those five paths against merged `origin/main` returned
`compare_rc=0`, proving that the local edits were already fully represented in merged
main. The workspace was then safely reset to `origin/main` and ended with a clean status.

This closed the PR #5 development branch without discarding unmerged work.

## Why the project direction changes here

The completed work now provides the minimum experimental foundation needed to stop making
benchmark infrastructure the main deliverable:

- PR #3 made matching workload comparisons fair and token-length generation stable;
- PR #5 made real eviction and lower-tier restore measurable;
- PR #7 added a shadow-only restore-vs-recompute cost model with hardware validation;
- PR #8 created durable engineering memory for expensive validation and debugging results.

The remaining core question is no longer whether the benchmark can produce a lower-tier
restore. The core question is whether the system can **predict when restore is actually
cheaper than recompute and safely act on that prediction**.

## Issue-driven roadmap introduced

GitHub Issues were enabled and a roadmap was created so future work has explicit completion
criteria, dependency edges, and parallel workstreams.

Parent roadmap:

- #9: `[路线图] 自适应分层 KV Cache：restore / recompute / placement / eviction`

Completed archive issues:

- #10: workload fairness and token-length convergence, PR #3;
- #11: real eviction/lower-tier restore benchmark, PR #5;
- #12: shadow cost model and hardware validation, PR #7.

Core P0 chain:

```text
#13 systematic crossover measurement
  -> #14 shadow cost-model calibration
  -> #15 cross-model/concurrency/hardware generalization
  -> #16 active restore/recompute decision
```

Parallel system/maintenance work:

- #17: NUMA topology and CPU KV placement foundation;
- #20: `benchmarks/cache` static/pre-commit baseline cleanup;
- #21: fake `vllm` nested-newline fixture fix.

Later adaptive hierarchy work:

- #18: GPU/CPU/NVMe multi-tier placement;
- #19: adaptive admission/eviction policy.

## Current engineering recommendation

Start #13 as the primary research workstream. In parallel, #17 may proceed as a system
foundation, while #20 and #21 can be handled independently as maintenance.

Do not promote shadow decisions to active execution merely because the shadow model is
present. #14 and #15 should first establish prediction error, decision accuracy, and
failure boundaries using systematic real-hardware evidence.

## Related records

- [`2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md`](2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md)
- [`2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md`](2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md)
- [`../validation/cache-crossover-baseline.md`](../validation/cache-crossover-baseline.md)
- [`../incidents/cache-benchmark-ci-baseline.md`](../incidents/cache-benchmark-ci-baseline.md)
- [`../incidents/fake-vllm-newline-fixture.md`](../incidents/fake-vllm-newline-fixture.md)
- [`../handoffs/2026-08-10-post-pr5-roadmap.md`](../handoffs/2026-08-10-post-pr5-roadmap.md)
