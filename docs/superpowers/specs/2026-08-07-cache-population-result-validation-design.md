# Cache Population Result Validation Design

## Context

The cache benchmark runner currently treats a population stage as successful when the `vllm bench serve` subprocess exits with return code 0. On the 4096-token tiered-fs eviction/restore reproducer, the benchmark CLI returned 0 even though only 20 of 24 population requests succeeded and 4 failed. The runner therefore continued into metrics collection and surfaced a later `/metrics` connection failure instead of reporting the population failure that actually invalidated the case.

The native benchmark command already runs with `--save-result` and writes a JSON result containing `completed` and `failed`, so the runner can validate population correctness without parsing human-readable stdout.

## Goal

Fail an execution case immediately at the `population` stage when the saved native population result does not prove that every expected population request completed successfully.

## Design

After the population subprocess returns and `_ensure_command_success()` confirms that it did not time out or exit non-zero, the runner reads `population-result.json` and validates the native counters.

A population stage is valid only when:

- the native result file exists and can be parsed as JSON;
- `completed` is an integer equal to `artifacts.num_population_prompts`;
- `failed` is an integer equal to `0`.

Any violation raises `BenchmarkExecutionError` while `stage == "population"`. The error message includes the expected population count and the observed `completed` and `failed` values when available. The runner therefore records the case as `benchmark_error` with `error.stage == "population"`, shuts the server down through the existing `finally` path, and does not collect pre-benchmark metrics or start the measured workload.

The validation is deliberately scoped to the population stage. Measured-run normalization continues to use the existing native result path and behavior; this change does not redefine measured benchmark success semantics.

## Error Handling

Malformed or missing population result data is treated as a population execution failure rather than a later parse or metrics failure. This keeps the reported stage aligned with the operation whose success could not be established.

Subprocess timeout and non-zero exit behavior remains unchanged and continues to be handled by `_ensure_command_success()` before result validation.

## Tests

Add focused runner tests covering both sides of the contract:

1. A population subprocess returns 0 and writes `{"completed": 20, "failed": 4}` for an expected 24 requests. The case must record `status == "benchmark_error"`, `error.stage == "population"`, and the measured benchmark command must not run.
2. A population subprocess returns 0 and writes `{"completed": 24, "failed": 0}` for an expected 24 requests. Execution is allowed to continue into the measured benchmark path.
3. A return-code failure remains handled by the existing command-success check, preventing the new result validation from weakening current behavior.

## Scope

This change modifies only the cache benchmark harness and its tests. It does not change vLLM runtime offloading behavior, scheduler lifecycle semantics, workload generation, case IDs, cache configuration, or metrics definitions.
