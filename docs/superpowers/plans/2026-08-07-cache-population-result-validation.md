# Cache Population Result Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the cache benchmark runner reject a population stage when `vllm bench serve` exits successfully but its saved native result reports missing or failed requests.

**Architecture:** Keep validation inside `benchmarks/cache/run_suite.py` because it owns stage orchestration and error categorization. Add one focused helper that reads `population-result.json`, validates `completed` and `failed`, and raises the existing `BenchmarkExecutionError`; call it immediately after `_ensure_command_success()` while the stage remains `population`.

**Tech Stack:** Python 3.11+, pytest, existing vLLM cache benchmark harness.

## Global Constraints

- Validate the native JSON result, not human-readable stdout.
- Require `completed == artifacts.num_population_prompts` and `failed == 0`.
- Missing, malformed, or incomplete population native results are population failures.
- Preserve existing subprocess timeout/non-zero handling.
- Preserve measured-run normalization semantics.
- Do not modify vLLM runtime offloading behavior, scheduler lifecycle, workload generation, case IDs, cache configuration, or metrics definitions.

---

### Task 1: Validate population native results before measurement

**Files:**

- Modify: `benchmarks/cache/tests/test_run_suite.py`
- Modify: `benchmarks/cache/run_suite.py`

**Interfaces:**

- Consumes: `population_result_path: Path`, `expected_count: int`, and the existing `BenchmarkExecutionError`.
- Produces: `_ensure_population_result_success(path: Path, expected_count: int) -> None`.

- [ ] **Step 1: Write the failing regression test**

Add an execution-level test that creates one population-bearing case, stubs the population command to return code 0 while writing `{"completed": 20, "failed": 4}`, and fails the test if a measured benchmark command is attempted. Assert the returned record has `status == "benchmark_error"`, `error.stage == "population"`, and an error message that reports expected 24, completed 20, and failed 4.

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
python -m pytest benchmarks/cache/tests/test_run_suite.py -k population_result -q
```

Expected: FAIL because the current runner accepts the return-code-0 population result and proceeds into measurement.

- [ ] **Step 3: Add the minimal result validator**

Implement in `benchmarks/cache/run_suite.py`:

```python
def _ensure_population_result_success(path: Path, expected_count: int) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkExecutionError(
            f"population native result is unavailable or invalid: {error}"
        ) from error

    if not isinstance(payload, Mapping):
        raise BenchmarkExecutionError("population native result is not a JSON object")

    completed = payload.get("completed")
    failed = payload.get("failed")
    if (
        not isinstance(completed, int)
        or isinstance(completed, bool)
        or not isinstance(failed, int)
        or isinstance(failed, bool)
        or completed != expected_count
        or failed != 0
    ):
        raise BenchmarkExecutionError(
            "population requests incomplete: "
            f"expected={expected_count}, completed={completed!r}, failed={failed!r}"
        )
```

Immediately after `_ensure_command_success(command_result, "population")`, call:

```python
_ensure_population_result_success(
    population_result_path,
    artifacts.num_population_prompts,
)
```

- [ ] **Step 4: Verify GREEN for failed and successful population paths**

Extend the tests so a return-code-0 population result with `{"completed": 24, "failed": 0}` is allowed to reach the measured benchmark path. Keep the existing non-zero-return behavior covered.

Run:

```bash
python -m pytest benchmarks/cache/tests/test_run_suite.py -k 'population_result or partial_results' -q
```

Expected: PASS.

- [ ] **Step 5: Run focused and full cache benchmark tests**

Run:

```bash
python -m pytest benchmarks/cache/tests/test_run_suite.py -q
python -m pytest benchmarks/cache/tests -q
python -m compileall -q benchmarks/cache
```

Expected: all pytest tests pass and compileall exits 0.

- [ ] **Step 6: Commit implementation**

Commit only the two code/test files for the behavior change with message:

```text
fix: validate cache population benchmark results
```

Do not mix lint-debt cleanup into this commit.
