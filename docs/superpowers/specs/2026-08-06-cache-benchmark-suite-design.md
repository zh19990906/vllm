# Cache Benchmark Suite Design

- Status: Approved for implementation planning
- Date: 2026-08-06
- Base: vLLM v0.26.0 (`568afb3a13806beb53bb2e6bd518269357b237c0`)
- Branch: `feature/cache-benchmark-suite`

## 1. Background

The project will optimize vLLM's existing prefix-cache and KV-offloading stack rather than replace it. Before changing scheduler, cache-manager, transfer, or storage behavior, we need a reproducible baseline that compares the current native modes on the same model, hardware, workload, and parallelism layout.

The first delivery is therefore an external benchmark suite that wraps existing vLLM CLI capabilities. It must not modify the inference hot path.

## 2. Goals

The suite will:

1. Compare four cache modes under identical workloads:
   - APC disabled.
   - GPU APC only.
   - GPU APC plus CPU offload.
   - GPU APC plus CPU and filesystem/NVMe tiering.
2. Measure cold-cache, warm-cache, shared-prefix, mixed-prefix, concurrency, and restart-persistence behavior.
3. Produce machine-readable raw data and a human-readable comparison report.
4. Support TP sizes 1, 2, 4, and 8 through configuration rather than code changes.
5. Preserve partial results when one scenario fails.
6. Provide dry-run validation on machines without GPUs.
7. Establish the evidence needed for the next phases:
   - restore-versus-recompute cost decisions;
   - ordinary-DRAM capacity tier design.

## 3. Non-goals

The first phase will not:

- modify Scheduler, Attention, PagedAttention, KVCacheManager, BlockPool, or OffloadingConnector;
- introduce a new cache backend;
- implement NUMA placement or a restore-versus-recompute policy;
- automatically provision models, drivers, containers, NVMe filesystems, or monitoring agents;
- guarantee identical performance across different model revisions or hardware topologies;
- run the 7B, 70B, and 397B examples in CI.

## 4. Selected approach

The suite will wrap `vllm serve` and `vllm bench serve` rather than reimplement the serving benchmark client or patch the existing CLI.

Reasons:

- keeps the baseline representative of unmodified vLLM;
- minimizes maintenance and implementation risk;
- preserves native request generation and latency calculations;
- makes later comparisons between upstream behavior and our optimized behavior straightforward.

The wrapper owns orchestration, scenario generation, process lifecycle, environment capture, metrics collection, and reporting.

## 5. Repository layout

```text
benchmarks/cache/
├── README.md
├── run_suite.py
├── config.py
├── scenarios.py
├── workload.py
├── process.py
├── metrics.py
├── report.py
├── configs/
│   ├── example-7b.yaml
│   ├── example-70b.yaml
│   └── example-397b.yaml
└── tests/
    ├── test_config.py
    ├── test_scenarios.py
    ├── test_workload.py
    ├── test_metrics.py
    ├── test_report.py
    └── test_process.py
```

Each module has one responsibility:

- `config.py`: parse, normalize, and validate YAML configuration.
- `scenarios.py`: produce cache-mode server configurations.
- `workload.py`: produce deterministic benchmark invocations and request groups.
- `process.py`: start, probe, terminate, and capture server/benchmark subprocesses.
- `metrics.py`: normalize benchmark and optional system metrics into one schema.
- `report.py`: create JSONL, CSV, and Markdown summaries.
- `run_suite.py`: coordinate the end-to-end state machine.

## 6. Configuration model

A single YAML file describes the machine, model, server, cache tiers, workload, and output policy.

Required logical sections:

```yaml
schema_version: 1

model:
  id: /models/example
  served_name: example
  dtype: auto
  max_model_len: 32768
  trust_remote_code: false

parallelism:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1

server:
  host: 127.0.0.1
  port: 8100
  startup_timeout_seconds: 900
  shutdown_timeout_seconds: 60
  extra_args: []
  env: {}

cache:
  gpu_memory_utilization: 0.90
  cpu_bytes_to_use: 68719476736
  offload_block_size: 64
  eviction_policy: lru
  filesystem:
    enabled: true
    root_dir: /mnt/nvme/vllm-kv-cache
    read_threads: 32
    write_threads: 16

workload:
  seed: 1
  tokenizer: /models/example
  prompt_tokens: [1024, 4096, 8192]
  output_tokens: 128
  concurrency: [1, 8, 32]
  request_rate: [inf]
  requests_per_case: 32
  shared_prefix_ratios: [0.0, 0.5, 0.9]
  warmup_requests: 2

results:
  root_dir: results/cache
  keep_server_logs: true
  fail_fast: false
```

Validation rules:

- schema version must be supported;
- model id, served name, output directory, and tokenizer must be non-empty;
- TP and PP values must be positive integers;
- ports and timeouts must be valid positive values;
- CPU bytes must be positive for CPU and tiered scenarios;
- filesystem root must be configured when the filesystem scenario is enabled;
- offload block size must be a positive integer;
- workload lengths, concurrency, request counts, and ratios must be within valid ranges;
- unknown keys are rejected by default to prevent silent misspellings;
- secrets are not written into result files.

Example configuration values are illustrative and must not be treated as production defaults.

## 7. Scenario matrix

### 7.1 Cache modes

The suite generates four server modes:

| ID | APC | CPU offload | Filesystem tier |
| --- | --- | --- | --- |
| `no-cache` | off | off | off |
| `gpu-apc` | on | off | off |
| `cpu-offload` | on | on | off |
| `tiered-fs` | on | on | on |

The tiered mode uses the native `OffloadingConnector` and `TieringOffloadingSpec`; CPU remains the primary tier and filesystem is secondary.

### 7.2 Workload cases

Each enabled cache mode runs these logical cases:

1. `cold-unique`: unique prompts after cache reset.
2. `warm-exact-prefix`: repeat identical prompts to measure exact-prefix reuse.
3. `shared-prefix`: many requests share a configured prefix ratio while suffixes differ.
4. `mixed-prefix`: combine warm, cold, short-prefix, and long-prefix requests.
5. `concurrency-sweep`: repeat representative prefix cases at configured concurrency levels.
6. `request-rate-sweep`: repeat representative cases at configured rates.
7. `restart-persistence`: populate cache, stop the server, restart it, and repeat requests. This case is enabled only for filesystem tiering.

The complete case identity is deterministic and includes cache mode, workload type, prompt length, prefix ratio, concurrency, request rate, seed, and repetition.

## 8. Workload generation

The suite will generate token-length-controlled prompts using the configured tokenizer. The generator must be deterministic for the same seed and configuration.

Shared-prefix prompts are constructed as:

```text
shared prefix tokens + request-specific suffix tokens
```

Requirements:

- generated prompts must meet requested token lengths within a documented tolerance;
- identical-prefix cases must use byte-identical request payloads for the shared section;
- request-specific suffixes must differ;
- datasets and prompts may be persisted under the run directory for reproducibility;
- workload generation must not require a running vLLM server;
- the benchmark invocation uses native `vllm bench serve` options wherever available.

## 9. Execution flow

For each scenario:

1. Validate the scenario configuration.
2. Create an isolated scenario result directory.
3. Prepare cache state:
   - no-cache/GPU/CPU scenarios rely on fresh server lifecycle;
   - filesystem scenarios use a per-run path unless the restart case explicitly reuses it.
4. Build and persist the exact server command and sanitized environment.
5. Start `vllm serve` as a managed subprocess.
6. Poll a health/model endpoint until ready or timeout.
7. Run configured warmup requests, excluded from final statistics.
8. Execute benchmark cases with `vllm bench serve`.
9. Capture stdout, stderr, exit code, timestamps, and parsed metrics.
10. Stop the server gracefully; force termination only after timeout.
11. Persist the scenario result immediately.
12. Continue to the next scenario unless `fail_fast` is enabled.

A suite interrupted after one or more completed scenarios remains reportable.

## 10. Metrics and evidence

### 10.1 Required benchmark metrics

When emitted by native vLLM benchmark output, normalize:

- successful and failed request counts;
- request throughput;
- output-token throughput;
- total-token throughput;
- TTFT mean, median, P95, and P99;
- TPOT mean, median, P95, and P99;
- inter-token latency statistics;
- end-to-end request latency statistics.

### 10.2 Cache and transfer metrics

Collect from available vLLM Prometheus metrics or structured logs:

- prefix-cache hit and query token counts;
- CPU cache usage, read usage, and write usage;
- CPU allocation size;
- skipped stores when configured;
- secondary-tier synchronous and asynchronous lookup delay;
- offload read/write bytes and operations when exposed;
- failed or dropped transfers when exposed.

Unavailable metrics are recorded as `null` with a reason, not fabricated as zero.

### 10.3 Environment evidence

Capture without failing the run when commands are unavailable:

- vLLM and Python versions;
- Git commit and dirty-state indicator where available;
- model and tokenizer identifiers;
- full sanitized server and benchmark commands;
- CUDA runtime and driver information;
- GPU names, memory, topology, and visible device list;
- CPU model, socket/NUMA layout, and total memory;
- configured filesystem path and basic device information;
- UTC timestamps and elapsed durations.

## 11. Result format

Each run creates:

```text
results/<run-id>/
├── manifest.json
├── environment.json
├── scenarios.json
├── scenario-results.jsonl
├── summary.csv
├── report.md
├── workloads/
└── raw/
    └── <scenario-id>/
        ├── server-command.json
        ├── benchmark-command.json
        ├── server.stdout.log
        ├── server.stderr.log
        ├── benchmark.stdout.log
        ├── benchmark.stderr.log
        ├── native-result.json
        └── normalized-result.json
```

`scenario-results.jsonl` is append-only during execution. `summary.csv` and `report.md` are regenerated from JSONL, so reports can be rebuilt after interruption.

## 12. Report behavior

The Markdown report compares every cache mode against both `no-cache` and `gpu-apc` for matching workload dimensions.

It will show:

- absolute metrics;
- TTFT and throughput deltas;
- cache-hit evidence;
- CPU/NVMe resource costs where available;
- failures and missing metrics;
- command/config fingerprints needed to reproduce the case.

The report must avoid claiming a cache improvement when the compared cases do not share the same workload identity.

## 13. Error handling

Errors are categorized as:

- `configuration_error`;
- `server_start_error`;
- `server_timeout`;
- `benchmark_error`;
- `parse_error`;
- `metrics_error`;
- `shutdown_error`;
- `interrupted`.

Every failed scenario writes a normalized error record containing stage, exit code, concise message, log paths, start/end timestamps, and retryability. Process cleanup runs in `finally` paths. The suite never silently discards a failed case.

## 14. Safety and isolation

- Default bind address is loopback.
- Shell commands are represented as argument arrays and executed without `shell=True`.
- Environment output is sanitized by an explicit denylist for tokens, passwords, credentials, and secrets.
- Filesystem cache paths must be descendants of configured roots; the suite will not recursively delete an arbitrary user-provided directory.
- Each ordinary run receives a unique filesystem cache subdirectory.
- Cache reset is explicit and logged.
- A dry run never starts processes or deletes cache data.

## 15. Testing strategy

### 15.1 Unit tests without GPU

- valid and invalid YAML parsing;
- unknown-key rejection;
- deterministic scenario matrix generation;
- correct native CLI argument construction for all four cache modes;
- deterministic workload construction and shared-prefix ratios;
- metric parsing with complete, partial, malformed, and version-varied samples;
- report generation from synthetic JSONL;
- process timeout, graceful shutdown, forced shutdown, and partial-result behavior using fake subprocesses;
- environment secret sanitization;
- safe cache-directory validation.

### 15.2 Lightweight integration tests

- dry-run end-to-end command and manifest generation;
- a fake HTTP server readiness probe;
- a fake benchmark executable producing representative native JSON;
- report rebuilding from an interrupted run.

### 15.3 Hardware validation

Hardware validation is manual in phase one:

- one small model on TP=1;
- one representative multi-GPU model;
- CPU-only offload and CPU+filesystem tiering;
- restart-persistence on local NVMe.

Actual 70B and 397B executions are deployment validation, not CI gates.

## 16. Acceptance criteria

Phase A is complete when:

1. All configuration, generation, parsing, reporting, and process-management tests pass without a GPU.
2. `--dry-run` emits valid, reproducible commands for all enabled cache modes.
3. The suite supports TP=1/2/4/8 by configuration.
4. One failed scenario does not erase or prevent reporting of completed scenarios unless fail-fast is requested.
5. A completed run produces manifest, environment, JSONL, CSV, Markdown, commands, and raw logs.
6. Comparison logic only compares matching workload identities.
7. At least one real GPU smoke test successfully completes `no-cache`, `gpu-apc`, and `cpu-offload`.
8. At least one local-NVMe smoke test successfully completes `tiered-fs` and restart-persistence.
9. No inference-core source file is modified.

## 17. Future extension boundaries

The benchmark output schema must be extensible so phases B and C can add fields without rewriting historical results.

Phase B will consume measured prefill latency, transfer latency, cache size, batch state, and hit length to design restore-versus-recompute decisions.

Phase C will use the same scenarios to compare pinned-memory-only behavior with a small pinned staging layer plus large ordinary-DRAM capacity layer.

These future features are intentionally excluded from this implementation cycle.
