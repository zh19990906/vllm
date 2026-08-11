# Handoff: Post-PR #5 Roadmap and Next Workstreams

Observed: **2026-08-10**.

This is a point-in-time continuation record. Re-verify GitHub issue state, branch heads,
and current `main` before acting if this document is older than the current session.

## Authoritative repository state

- Repository: `zh19990906/vllm`
- Authoritative remote: GitHub
- Observed `main` head: `f628fbafe45431bcb9579c48d818455f7197add5`
- That commit merged PR #5.

The development environment may use a Gitee mirror for Pod fetches, but Pod-local commits
must not be treated as authoritative GitHub state unless they can be delivered through an
authorized GitHub write path.

## What is complete

### PR #3 / issue #10

Workload fairness and tokenizer-length convergence are complete. Matching cache modes can
be compared using the same workload content identity.

### PR #5 / issue #11

Real eviction/lower-tier restore benchmark infrastructure is complete and merged.

Important final evidence:

- PR head: `308857612f883232912cca98d9f1fdae4ec6d5c2`
- merge commit: `f628fbafe45431bcb9579c48d818455f7197add5`
- 26 focused tests passed;
- 71/71 cache tests passed with only the known unrelated fake-executable fixture corrected
  temporarily for validation and restored afterward;
- compileall and `git diff --check` passed;
- direct CI-baseline comparison showed zero new PR-local Ruff/Ruff-format/Markdownlint/SPDX
  debt relative to main.

The post-merge GitHub -> Gitee -> Pod synchronization path was verified, and the isolated
PR #5 workspace ended clean at the merge commit.

### PR #7 / issue #12

The shadow cost model is merged and hardware-validated. It is still shadow-only and does
not change the actual restore path.

### PR #8

The durable engineering-memory structure under `docs/engineering/` is merged and should
remain the source for current state, incidents, validation evidence, and handoffs.

## Current roadmap

Parent issue: **#9**.

### P0 core chain

```text
#13 crossover measurement
  -> #14 cost-model calibration
  -> #15 generalization validation
  -> #16 active restore/recompute decision
```

The next primary research issue is **#13**.

Issue #13 should produce systematic, structured real-hardware evidence for restore versus
recompute across a sufficiently dense token/KV-size range. The purpose is not another
benchmark feature for its own sake; the output must be usable to calibrate and judge the
shadow cost model.

### Parallel workstreams

- **#17**: NUMA topology discovery and CPU KV placement foundations.
- **#20**: clean existing `benchmarks/cache` static/pre-commit baseline.
- **#21**: permanently fix the fake `vllm` fixture newline escaping defect.

These can proceed independently of #13. #20 and #21 are maintenance and should not become
preconditions for core research unless a specific test or CI requirement makes them a
real blocker.

### Later system work

- **#18**: GPU/CPU/NVMe multi-tier placement policy.
- **#19**: adaptive online admission/eviction policy.

Do not pull these forward before #16 has established a safe active restore/recompute
boundary and #17 has provided the CPU topology foundation needed by multi-tier placement.

## Suggested execution order for #13

The first #13 implementation/design session should keep scope narrow:

1. choose one known model and one known validation machine as the baseline environment;
2. reuse the merged PR #5 pressure workload rather than inventing a new workload path;
3. define the exact measurement matrix for token/KV size and tier;
4. record recompute and real lower-tier restore cost with P50/P95/P99 where appropriate;
5. record enough transfer/cache evidence to prove the tier that served the victim;
6. compare actual measurements against the existing shadow predictions;
7. save structured outputs and a concise validation record under
   `docs/engineering/validation/`;
8. close #13 only when the result is suitable input for #14 calibration.

Avoid immediately expanding to many models or 1-8 GPUs. Establish one high-quality
baseline first, then use #15 for generalization.

## Known problems

### Issue #20: cache benchmark static baseline

The all-files pre-commit workflow currently exposes inherited cache benchmark debt. The
fresh PR #8-versus-PR #5 comparison established the same observed classes:

- 18 remaining E501 errors after Ruff auto-fix;
- 13 Ruff-format targets;
- 8 MD060 errors at `benchmarks/cache/README.md:52`;
- 16 older Python files missing SPDX headers.

Treat this as a separate maintenance branch/PR.

### Issue #21: fake executable fixture

`test_fake_executable_end_to_end` still has the nested `\n` escaping defect. A temporary
`\\n` validation correction made the full cache suite pass 71/71, but that change was
restored and is not yet on main.

Fix it separately and update the existing incident record when resolved.

## Decision rules for new work

Before starting an unplanned task, classify it against the roadmap.

A core task should answer at least one of these:

- does it improve real crossover evidence?
- does it reduce or explain cost-model prediction error?
- does it make active restore/recompute safer or more effective?
- does it establish a necessary topology/tier primitive for adaptive placement?

If none applies, it is likely maintenance or a later-priority workstream and should not
silently displace #13-#16.

## Expensive work not to repeat without a new hypothesis

- Do not repeat the full PR #7 hardware sweep merely to reconfirm archived behavior.
- Do not repeat PR #5 validation just because the documentation changed.
- Do not re-run the same all-files pre-commit failure expecting it to become green without
  changing #20 baseline files.
- Do not run a broad multi-model/multi-GPU crossover sweep before the single-environment
  measurement design is trustworthy.
- Do not treat shadow prediction changes as execution changes until #16 is implemented.

## Related records

- [`../CURRENT_STATE.md`](../CURRENT_STATE.md)
- [`../history/2026-08-10-pr5-finalization-and-roadmap-transition.md`](../history/2026-08-10-pr5-finalization-and-roadmap-transition.md)
- [`../validation/cache-crossover-baseline.md`](../validation/cache-crossover-baseline.md)
- [`../validation/pr7-shadow-cost-model-hardware-validation.md`](../validation/pr7-shadow-cost-model-hardware-validation.md)
- [`../incidents/cache-benchmark-ci-baseline.md`](../incidents/cache-benchmark-ci-baseline.md)
- [`../incidents/fake-vllm-newline-fixture.md`](../incidents/fake-vllm-newline-fixture.md)
