# PR #5: Cache Eviction/Restore Benchmark Development

Period covered: **2026-08-07 through 2026-08-10**.

Status at the end of this record: **open, draft, mergeable**.

## Purpose

PR #5 adds a benchmark workload that can create real cache pressure and then measure a
victim prefix after it has been displaced from faster cache tiers. The goal is to validate
lower-tier KV restore behavior rather than accidentally measuring a warm GPU prefix hit.

The pressure lifecycle is:

1. create and populate a small set of victim prompts;
2. create enough unique filler prompts to consume configured cache capacity;
3. execute the population phase as setup only;
4. replay only the original victims during measurement;
5. inspect cache metrics and transfer evidence to determine where the restored KV came
   from.

The feature is opt-in. `pressure_fill_requests` defaults to zero, so existing benchmark
matrices do not gain pressure cases unless explicitly configured.

## Remote identity

Observed on 2026-08-10:

- PR: `#5`
- Title: `Add cache eviction restore benchmark workload`
- Branch: `feature/cache-eviction-restore-benchmark`
- Head: `b91b6ee631a8145a5eabb55542ad733dfacff24a`
- Base branch: `main`
- State: open and draft
- Mergeable: true
- Commits reported by GitHub: 44
- Changed files: 13
- GitHub-reported diff size: `+1522 / -17`

The PR description still says it is stacked on PR #3. PR #3 has since merged, so that
historical dependency is satisfied even though PR #5 has not yet been refreshed onto the
latest `main` in authoritative GitHub history.

## Changed-file boundary

The observed GitHub PR contains exactly these 13 paths:

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

This boundary is important. Broad formatting of unrelated `benchmarks/cache` files should
not be mixed into this feature PR merely to hide repository baseline hygiene problems.

## Evolution of the workload

### Initial benchmark harness

The cache benchmark work established comparable modes for:

- no-cache / recompute;
- GPU automatic prefix caching;
- CPU offload;
- tiered filesystem offload.

PR #3 supplied the workload-fairness invariant required for those comparisons: matching
cache modes use the same content identity rather than seeding from cache-mode-specific case
identity.

### Eviction/restore pressure semantics

The pressure workload added victim/filler ordering. Victims are populated first. Unique
fillers are placed after them so they become newer cache content and create pressure that
can evict the older victim KV from GPU and, with a small enough CPU primary tier, push it
toward the configured secondary filesystem tier.

Measurement contains only victim replays. Filler requests are setup traffic and should not
be included in measured comparison rows.

### Population-result validation

A benchmark process returning success is not enough to prove that the setup population did
what the scenario required. PR #5 added validation around population result artifacts so
that failed requests, missing result fields, or incomplete setup can fail the scenario
before a misleading measurement is accepted.

This distinction later mattered in hardware work: population runs are setup-only, while
post-population victim requests are the measured evidence.

### Pressure token budget and search

A fixed filler-request count is not portable across model length, cache block size, GPU KV
capacity, CPU-primary capacity, and scenario parameters. The PR therefore grew explicit
pressure token-budget logic and workload search support so validation can reason about the
amount of unique content needed to exceed the configured cache tiers.

This also produced dedicated tests for pressure token budgets and workload search, rather
than relying only on an end-to-end GPU run.

## Tokenizer-generation issue encountered during benchmark development

Early hardware runs could fail with:

```text
unable to generate prompt with requested length 1024
```

This was not a cache restore failure. It was a workload-generation correctness problem
that was fixed in PR #3 by separating workload content identity from case identity and by
stabilizing the suffix token stream during length correction.

Later real-tokenizer preflight accepted small observed windows around requested sizes:

| Requested | Observed range |
|---:|---:|
| 256 | 254-258 |
| 512 | 510-514 |
| 1024 | 1022-1026 |
| 2048 | 2046-2050 |
| 4096 | 4094-4098 |

Matching no-cache and tiered-filesystem workload bytes were verified identical.

## Crossover benchmark outcome

The pressure workload enabled a direct comparison between recomputing a prefix and
restoring its KV from the filesystem secondary tier on the validation machine.

The measured seed values were:

| Prompt tokens | Recompute P95 TTFT | Filesystem restore P95 TTFT |
|---:|---:|---:|
| 256 | 26.414 ms | 31.119 ms |
| 512 | 44.961 ms | 56.979 ms |
| 1024 | 81.705 ms | 108.132 ms |
| 2048 | 152.461 ms | 244.266 ms |
| 4096 | 308.424 ms | 651.127 ms |

Within 256-4096 tokens, filesystem restore was slower than recompute at every measured
point. The disadvantage grew with prompt length; the 4096-token filesystem result was a
little over twice the recompute P95.

The 4096 filesystem case was checked for real external restore evidence rather than a
false warm-cache hit. Historical evidence recorded 8 secondary async lookups averaging
about 498.87 ms, 32704 external cache-hit tokens, and 1,875,378,176 bytes transferred from
CPU to GPU across 8 requests.

This crossover result directly motivated PR #7: a cache hit should not automatically imply
that restore is the cheapest action.

See:
[`../validation/cache-crossover-baseline.md`](../validation/cache-crossover-baseline.md).

## One transient metrics failure

A 4096-token run once failed while collecting `metrics_before` with a connection-refused
error. Investigation found no OOM event, no OOM kill, and no fatal server traceback.
Re-running the same case completed successfully while a watcher showed `/metrics`
remaining available after server readiness.

The incident was classified as non-reproduced and transient. No retry behavior was added
solely because of that one observation.

See:
[`../incidents/transient-metrics-before-connection-refused.md`](../incidents/transient-metrics-before-connection-refused.md).

## Relationship to PR #7

PR #5 is benchmark infrastructure. PR #7 is runtime shadow instrumentation. They were
intentionally kept out of each other's permanent branch history.

For hardware validation, disposable integration worktrees combined the two branches so the
benchmark could exercise PR #7 runtime behavior. That temporary combination was not a
reason to merge benchmark code into the runtime PR.

PR #7 is now merged. PR #5 remains independently active.

## Functional test state on 2026-08-10

A clean PR #5 workspace ran `benchmarks/cache/tests` with 70 passing tests and one failure:

```text
benchmarks/cache/tests/test_run_suite.py::test_fake_executable_end_to_end
```

The generated fake `vllm` executable was inspected directly. Its nested `\n` escape had
been expanded by the outer triple-quoted fixture string, producing an invalid bytes literal
split across two physical lines. `py_compile` reported an unterminated string literal.

Only the fixture was temporarily corrected in the runner workspace from `\n` to `\\n`.
With that baseline correction:

```text
71 passed in 2.44s
pytest_rc=0
compile_rc=0
```

The fixture modification was automatically restored and the worktree returned clean.
Therefore the observed PR #5 behavior baseline is 71/71 tests passing after isolating the
known unrelated fixture defect.

See:
[`../incidents/fake-vllm-newline-fixture.md`](../incidents/fake-vllm-newline-fixture.md).

## Static-check state on 2026-08-10

The feature branch is not yet clean under the cache-directory static checks, but the latest
`main` is also red in the same area.

Observed `main` baseline:

- 21 Ruff errors;
- 13 files would be reformatted;
- 16 Python files missing SPDX headers.

Observed PR #5 tree:

- 24 Ruff errors;
- 15 files would be reformatted;
- 18 Python files missing SPDX headers.

This indicates a small PR-specific delta layered on a larger cache benchmark baseline.
The cleanup strategy should preserve that distinction rather than mechanically formatting
the entire cache directory inside PR #5.

See:
[`../incidents/cache-benchmark-ci-baseline.md`](../incidents/cache-benchmark-ci-baseline.md).

## Local merge preview and why it was not committed

A clean PR #5 checkout successfully previewed merging the latest `main` with:

```text
git merge --no-commit --no-ff origin/main
```

The preview had no conflicts, and comparison against latest `main` still showed the same
13 PR #5 files. That proved branch compatibility without changing the authoritative GitHub
branch.

The preview was aborted rather than committed because the Pod/local environment is not an
authorized GitHub push path. Creating an authoritative merge commit locally would have
produced a commit that could not update the PR branch. The correct workflow is to make
GitHub-side branch changes through a GitHub-capable path, then sync to the Gitee mirror and
fetch from the Pod.

## Current completion path

At the end of this record, the remaining work is:

1. resolve the cache benchmark hygiene baseline separately;
2. refresh PR #5 against the authoritative latest `main` through a GitHub write path;
3. apply only PR #5-specific cleanup after the baseline is known;
4. re-run the full cache tests and static checks;
5. perform only hardware smoke/acceptance checks that answer a remaining question;
6. update the PR description, mark ready, and merge after final validation.

The exact continuation snapshot is maintained in:
[`../handoffs/2026-08-10-pr5-current-handoff.md`](../handoffs/2026-08-10-pr5-current-handoff.md).
