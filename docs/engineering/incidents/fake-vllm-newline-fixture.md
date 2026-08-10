# Incident: Fake `vllm` Newline Fixture Generates Invalid Python

Status: **known baseline defect; reproduced on 2026-08-10**.

Affected test:

```text
benchmarks/cache/tests/test_run_suite.py::test_fake_executable_end_to_end
```

## Symptom

A full cache benchmark test run can report one failure even when the surrounding benchmark
logic is healthy:

```text
AssertionError: assert 1 == 0
```

The assertion occurs because `run_suite.main(...)` returns failure when it tries to start
the generated fake `vllm` executable.

Without inspecting the generated executable, this looks like an end-to-end runner
regression.

## Root cause

The test creates a Python executable by writing an outer triple-quoted Python string. Inside
that outer string is a bytes literal intended to contain a newline:

```python
body = b'vllm:prefix_cache_hit_tokens_total 10\n'
```

The outer Python string interprets `\n` before writing the child script. The resulting
file contains a physical newline inside the child's single-quoted bytes literal:

```python
body = b'vllm:prefix_cache_hit_tokens_total 10
'
```

The child script is therefore syntactically invalid.

## Direct reproduction evidence

On 2026-08-10 the generated file was preserved under a deterministic pytest base temp and
inspected with line numbers. The relevant section was:

```text
20  body = b'vllm:prefix_cache_hit_tokens_total 10
21  '
```

Compiling that generated executable directly produced:

```text
SyntaxError: unterminated string literal (detected at line 20)
```

This proves the failure occurs before the fake server can exercise `run_suite` behavior.

## Local verification workaround

For validation only, the fixture source was changed from the outer-string sequence `\n` to
`\\n`, so the generated child script contains a literal `\n` escape rather than a physical
newline inside the bytes literal.

The temporary diff affected only the fixture and was guarded by automatic restoration.
With that correction:

```text
71 passed in 2.44s
pytest_rc=0
compile_rc=0
```

After the test run, the fixture was restored and the PR #5 worktree was clean.

## Correct permanent fix

The nested string must escape the backslash at the outer-string level so the generated
Python source receives a valid escape sequence.

Conceptually:

```text
outer fixture text:       \\n
generated child source:  \n
child bytes value:        newline byte
```

Any permanent fix should include a regression assertion that the generated executable is
syntactically valid before relying on its server behavior.

## False conclusions to avoid

- Do not classify this failure as a PR #5 eviction/restore regression from the final
  `assert run_suite.main(...) == 0` alone.
- Do not debug HTTP readiness, benchmark result parsing, or cache metrics before compiling
  or directly executing the generated fixture.
- Do not use the one failing end-to-end fixture to discount the other 70 passing cache
  tests.
- Do not silently modify the fixture during a feature PR without recording that the change
  is an unrelated baseline correction.

## Diagnostic sequence for recurrence

When this exact test fails:

1. preserve pytest's temporary directory;
2. locate `*/bin/vllm`;
3. display the generated source around the `/metrics` handler;
4. run `python -m py_compile <generated-vllm>`;
5. only continue into runner/server debugging if the generated script compiles.

## Related records

- PR #3 explicitly noted this unrelated full-suite fixture issue.
- PR #5 local validation on 2026-08-10 reproduced and proved the syntax error.
- [`../history/2026-08-07-pr3-cache-workload-fairness-convergence.md`](../history/2026-08-07-pr3-cache-workload-fairness-convergence.md)
- [`../history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md`](../history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md)
