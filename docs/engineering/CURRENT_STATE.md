# Current Engineering State

Observed: **2026-08-10**.

This file is intentionally mutable. Verify remote metadata before acting on branch heads,
PR status, or issue state if this file is older than the current session.

## Repository state

- Repository: `zh19990906/vllm`
- Default branch: `main`
- Current observed `main` head:
  `f628fbafe45431bcb9579c48d818455f7197add5`
- That commit merged PR #5, `Add cache eviction restore benchmark workload`.
- GitHub is the authoritative remote. The observed development workflow synchronizes
  GitHub branches/tags to a Gitee mirror used by Pod workspaces.

## Project objective

The long-term objective is an adaptive hierarchical KV cache system that can use GPU,
CPU/DRAM, and filesystem/NVMe tiers while making runtime decisions from measured cost
rather than fixed cache-hit rules.

The key online decisions are intended to become:

- whether a reusable KV should be restored or recomputed;
- which tier should hold a KV;
- when a KV should be admitted, retained, promoted, demoted, or evicted;
- how hardware topology, current load, KV size, and model prefill cost should influence
  those choices.

End-to-end TTFT, throughput, transfer cost, and resource utilization are the target
outcomes; cache hit rate alone is not the objective.

## Roadmap tracking

GitHub Issues are now the primary plan/status index.

- Parent roadmap: **#9**
  `[路线图] 自适应分层 KV Cache：restore / recompute / placement / eviction`

### Completed archive issues

- **#10**: workload fairness and token-length convergence, completed by PR #3.
- **#11**: real eviction/lower-tier restore benchmark, completed by PR #5.
- **#12**: KV offload shadow cost model and hardware validation, completed by PR #7.

These issues are closed with the `completed` reason and retain validation evidence and
commit identities.

### P0 core path

1. **#13**: systematically measure restore-vs-recompute crossover.
2. **#14**: calibrate the shadow cost model against real measurements.
3. **#15**: validate cost-model generalization across models, concurrency, GPU count, and
   hardware conditions.
4. **#16**: promote shadow-only decisions into a real restore/recompute execution choice
   with safe fallback.

The intended dependency chain is:

```text
#11 real eviction/restore ----+
                              +--> #13 crossover --> #14 calibration --> #15 generalization --> #16 active decision
#12 shadow cost model --------+
```

### Parallel work

- **#17**: NUMA topology discovery and CPU KV placement foundations.
- **#20**: clean the existing `benchmarks/cache` pre-commit/static baseline.
- **#21**: permanently fix the fake `vllm` nested-newline fixture defect.

Issue #17 may proceed in parallel with #13-#15. #20 and #21 are maintenance workstreams and
should not block the core restore/recompute research path.

### Later system work

- **#18**: GPU/CPU/NVMe hierarchical KV placement policy.
- **#19**: adaptive online KV admission/eviction policy.

The current intended dependency is that #18 follows sufficiently mature active
restore/recompute behavior and NUMA foundations, while #19 builds on multi-tier placement.

## Recently completed work

### PR #3: cache workload fairness and convergence

- Final head: `4fa33f8267de7cbc4d95208886de8e63f028cb3a`
- Merge commit: `abc426ae063e90c837c48dfbe75fabe919c82575`
- Archived by issue #10.

PR #3 made matching cache modes use the same workload content identity and fixed unstable
token-length convergence. This is a correctness prerequisite for comparing cache modes.

### PR #7: KV offload shadow cost model

- Final head: `96de0c823721c374527dbb0b3a49fdc7eccba341`
- Merge commit: `37f65141108e112a317fe4a5d8215a4c21c3c00e`
- Archived by issue #12.

The feature remains shadow-only: it estimates restore versus recompute cost, records
provenance and low-cardinality metrics, and calibrates secondary-tier promotion cost with
an online EWMA without changing the actual restore path. Real scheduler/hardware
validation observed an adaptive p256 crossover after calibration while execution remained
restore.

### PR #8: durable engineering documentation

- Final head: `8e45f1a97e7e0c8df589291f2466545d376fd599`
- Merge commit: `e7e4197475885fe08d8b7be153a170f31c133a82`
- Documentation only.

PR #8 established this `docs/engineering/` memory system and separated current state,
history, incidents, validation, and handoffs.

### PR #5: real cache eviction/restore benchmark

- Final head: `308857612f883232912cca98d9f1fdae4ec6d5c2`
- Merge commit: `f628fbafe45431bcb9579c48d818455f7197add5`
- Merged: `2026-08-10T05:35:53Z`
- Archived by issue #11.

PR #5 established an opt-in victim -> pressure filler -> victim replay workload so the
benchmark can measure genuine lower-tier KV restore rather than a warm GPU prefix hit. It
also added pressure token budgets, workload search, and population-result validation.

Final verification on the merged feature head included:

- 26 focused cache tests passing;
- 71/71 cache tests passing when the known unrelated fake-executable fixture defect was
  temporarily corrected for validation and then restored;
- `python -m compileall -q benchmarks/cache` passing;
- `git diff --check` passing;
- PR-local Ruff/Ruff-format/SPDX debt reduced to the repository baseline.

The GitHub pre-commit run remained red because it runs on all files. Direct comparison
against the PR #8/main baseline showed the same inherited failure classes: 18 remaining
Ruff E501 errors after auto-fix, 13 Ruff-format targets, 8 MD060 errors in
`benchmarks/cache/README.md:52`, and 16 older cache Python files missing SPDX headers.
PR #5 added no new static debt in those classes.

After merge, the GitHub -> Gitee -> Pod synchronization path was verified. A clean Pod
workspace fetched `origin/main@f628fbafe...`; its five local validation edits compared
byte-for-byte equal to merged main and the workspace was safely reset to the merge commit
with a clean status.

## Known problems that remain

### `benchmarks/cache` repository-wide static baseline

Tracked by **#20**.

The existing all-files pre-commit baseline is still red for cache benchmark Ruff/format,
README markdownlint MD060, and missing SPDX headers. This should be resolved in a separate
maintenance PR rather than mixed into core feature work.

See:
[`incidents/cache-benchmark-ci-baseline.md`](incidents/cache-benchmark-ci-baseline.md).

### Fake `vllm` end-to-end fixture

Tracked by **#21**.

`benchmarks/cache/tests/test_run_suite.py::test_fake_executable_end_to_end` still contains
the known nested-newline escaping defect. A validation-only correction proves the rest of
the cache suite can pass 71/71, but the fixture itself has not yet been permanently fixed
on `main`.

See:
[`incidents/fake-vllm-newline-fixture.md`](incidents/fake-vllm-newline-fixture.md).

## Current objective

The project should now stop treating benchmark infrastructure as the primary deliverable.
The next core objective is **#13: produce systematic, trustworthy restore-vs-recompute
crossover data on real hardware** and use it to calibrate the shadow cost model.

A useful test for new work is:

> Does this change produce better crossover evidence, reduce cost-model prediction error,
> improve online decision safety, or move the system toward adaptive tier placement?

If not, it should normally be treated as a parallel maintenance or lower-priority
workstream rather than the core path.

## Recommended parallel start

The currently useful parallel work split is:

- main research: #13 crossover measurement;
- systems foundation: #17 NUMA topology / CPU placement;
- maintenance: #20 static baseline cleanup;
- maintenance: #21 fake fixture fix.

Do not start #16 active execution merely because the shadow model exists. #14 and #15
should first establish prediction quality and failure boundaries.

## Do not repeat without a new hypothesis

- Do not re-run the full PR #7 hardware sweep only to reproduce archived evidence.
- Do not assume a subprocess uses feature source merely because a worktree contains it;
  verify runtime module provenance.
- Do not interpret a shadow crossover as an execution-path change.
- Do not fold repository-wide cache hygiene into an unrelated feature PR.
- Do not treat a red all-files pre-commit job as feature-local without comparing against
  the current main baseline.
- Do not create Pod-local commits and treat them as authoritative GitHub state when the Pod
  cannot deliver them to GitHub.

## Current continuation record

See:
[`handoffs/2026-08-10-post-pr5-roadmap.md`](handoffs/2026-08-10-post-pr5-roadmap.md).
