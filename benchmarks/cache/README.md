# vLLM Cache Benchmark Suite

This suite compares native vLLM v0.26.0 prefix-cache and KV-offloading modes without changing the inference hot path. It orchestrates `vllm serve` and `vllm bench serve`, generates deterministic exact-prefix workloads, captures cache and resource evidence, and produces recoverable JSONL, CSV, and Markdown results.

## Prerequisites

- Linux with Python 3.10 or newer.
- A working vLLM v0.26.0 installation from this repository.
- A locally available model and tokenizer; replace the `/models/replace-with-*` paths in the examples.
- Enough GPU memory for the selected tensor-parallel layout.
- Enough host memory for `cpu_bytes_to_use`.
- A writable local NVMe path for the filesystem tier.

The examples are starting points, not production defaults. Confirm model context length, host-memory headroom, GPU topology, NUMA placement, and NVMe capacity before a real run.

## Commands

Validate configuration and print a complete scenario plan without loading a model, contacting a server, creating filesystem-cache directories, or invoking GPU tools:

```bash
python benchmarks/cache/run_suite.py \
  --config benchmarks/cache/configs/example-7b.yaml \
  --dry-run
```

Run the full selected configuration:

```bash
python benchmarks/cache/run_suite.py \
  --config /path/to/machine-model.yaml
```

Rebuild reports from the append-only journal after an interrupted run:

```bash
python benchmarks/cache/run_suite.py \
  --rebuild-report results/cache/<run-id>
```

A dry run writes `scenarios.json`. Copy one or more `case_id` values and restrict a real run with repeated flags:

```bash
python benchmarks/cache/run_suite.py \
  --config /path/to/machine-model.yaml \
  --case-id '<case-id-1>' \
  --case-id '<case-id-2>'
```

## Cache modes

| Mode | Prefix cache | Host tier | Secondary tier |
| --- | --- | --- | --- |
| `no-cache` | Disabled | None | None |
| `gpu-apc` | GPU APC | None | None |
| `cpu-offload` | GPU APC | Pinned CPU memory | None |
| `tiered-fs` | GPU APC | Pinned CPU memory | Filesystem/NVMe |

The filesystem tier is implemented by vLLM's native `OffloadingConnector` with `TieringOffloadingSpec`. GPU↔filesystem movement is staged through the CPU primary tier. Tiered runs force `PYTHONHASHSEED=0` so identical token prefixes map to stable block keys.

## Workloads

- **cold-unique**: unique prompts after a fresh server start.
- **warm-exact-prefix**: populate exact prompts, then measure the same prompts.
- **shared-prefix**: populate one representative prefix, then measure unique suffixes sharing that prefix.
- **mixed-prefix**: combines cold, exact-warm, 50% shared-prefix, and 90% shared-prefix requests.
- **restart-persistence**: populate the filesystem tier, stop the server, restart with the same cache directory, then measure. Only the post-restart benchmark is included in comparisons.

Population commands are setup only. Their latency numbers are not treated as measured results.

## Output layout

Each run creates:

```text
<results-root>/<run-id>/
├── .vllm-cache-benchmark-owned
├── manifest.json
├── environment.json
├── scenarios.json
├── scenario-results.jsonl
├── summary.csv
├── report.md
├── workloads/
└── raw/<case-id>/
    ├── .vllm-cache-benchmark-owned
    ├── measure.jsonl
    ├── populate.jsonl              # when required
    ├── metadata.json
    ├── native-result.json
    ├── server.stdout.log
    ├── server.stderr.log
    ├── benchmark.stdout.log
    └── benchmark.stderr.log
```

`scenario-results.jsonl` is flushed and synchronized after every case. Completed cases therefore remain reportable when a later case fails or the run is interrupted. `summary.csv` and `report.md` can always be regenerated from this journal.

## Safety behavior

The suite only regards a directory as removable or reusable when it is below the configured results or filesystem root and contains `.vllm-cache-benchmark-owned`. It never recursively deletes an arbitrary operator path. Dry-run mode does not create the NVMe cache root.

Environment keys containing `TOKEN`, `PASSWORD`, `SECRET`, `CREDENTIAL`, `API_KEY`, `ACCESS_KEY`, or `PRIVATE_KEY` are redacted in persisted configuration evidence.

## Metrics and missing data

Native benchmark JSON supplies throughput and latency metrics. The suite also snapshots `/metrics` before and after the measured command and samples process-tree RSS, system memory, and GPU process memory. vLLM builds and model paths may expose different metric names. A missing metric is written as `null` with a reason such as `metric_not_exposed`; it is never converted to zero.

Comparisons are made only when workload kind, prompt length, prefix ratio, concurrency, request rate, repetition, model identity, and TP/PP layout match exactly. The report shows both signed TTFT deltas and positive TTFT improvement percentages.

## Parallelism guidance

- **TP=1**: establish a small-model baseline and validate commands first.
- **TP=2/4**: verify GPU interconnect and CPU/NVMe NUMA locality before interpreting transfer latency.
- **TP=8**: keep the same visible-device ordering across runs and confirm that aggregate CPU-tier capacity leaves substantial host-memory headroom.
- Pipeline parallelism is configurable, but phase-A validation focuses on PP=1.

## Manual smoke-test checklist

1. Run `--dry-run` and inspect all server and benchmark commands.
2. Run one `no-cache` case on a small model.
3. Run the matching `gpu-apc` case and verify prefix-cache metrics or an explicit missing-metric reason.
4. Run the matching `cpu-offload` case and inspect CPU usage plus TTFT.
5. Confirm the NVMe root is dedicated and writable, then run `tiered-fs`.
6. Run `restart-persistence` and verify the second server lifecycle uses the same filesystem path.
7. Repeat representative cases at TP=1/2/4/8 as hardware permits.
8. Compare only rows with matching workload identity and review raw logs for every failure.
