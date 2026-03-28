# vLLM Latency Benchmark Report

**Date**: 2026-03-27 16:29:26 UTC
**Server**: http://localhost:8000
**Model**: adapter
**Guided JSON**: No
**Images**: No
**Iterations**: 10 (warmup: 3)
**Batch sizes**: [4, 8, 16]

## GPU Utilization

| Metric | Value |
|--------|-------|
| Peak memory | 7753 MB / 8188 MB |
| Avg utilization | 92.1% |
| Samples collected | 82 |

## Prompts Tested

| Prompt | Length (chars) |
|--------|--------------|
| V0 (short) | 185 |
| V0+ (long) | 1332 |

## Cold vs Warm Cache Latency

### V0 (short)

| Metric | Cold (1st req) | Warm p50 | Warm p95 | Warm p99 | Speedup |
|--------|---------------|----------|----------|----------|---------|
| TTFT (ms) | 1875.1 | 43.8 | 71.4 | 85.1 | 42.8x |
| Total latency (ms) | 2721.1 | 929.1 | 1037.3 | 1070.4 | 2.9x |
| Tokens/sec | 60.3 | 57.4 | — | — | — |
| Per-token latency (ms) | 17.28 | 17.83 | — | — | — |
| Prompt tokens | 61 | — | — | — | — |
| Completion tokens | 51 | 51 (avg) | — | — | — |

### V0+ (long)

| Metric | Cold (1st req) | Warm p50 | Warm p95 | Warm p99 | Speedup |
|--------|---------------|----------|----------|----------|---------|
| TTFT (ms) | 80.0 | 43.7 | 45.0 | 45.6 | 1.8x |
| Total latency (ms) | 951.4 | 930.2 | 1000.1 | 1041.4 | 1.0x |
| Tokens/sec | 58.5 | 58.0 | — | — | — |
| Per-token latency (ms) | 17.40 | 17.60 | — | — | — |
| Prompt tokens | 377 | — | — | — | — |
| Completion tokens | 51 | 52 (avg) | — | — | — |

## Batch Scaling (Latency per Batch vs per Request)

### V0 (short)

| Concurrency | Wall clock (ms) | p50 latency (ms) | p95 latency (ms) | p99 latency (ms) | p50 TTFT (ms) | Throughput (tok/s) | Failed |
|-------------|----------------|------------------|------------------|------------------|---------------|-------------------|--------|
| 1 | 9453 | 929.1 | 1037.3 | 1070.4 | 43.8 | 57.4 | 0 |
| 4 | 3894 | 954.1 | 1028.6 | 1028.7 | 63.8 | 57.9 | 0 |
| 8 | 2205 | 1081.9 | 1197.2 | 1197.2 | 155.1 | 57.7 | 0 |
| 16 | 1212 | 1193.8 | 1209.2 | 1209.5 | 291.0 | 56.3 | 0 |

### V0+ (long)

| Concurrency | Wall clock (ms) | p50 latency (ms) | p95 latency (ms) | p99 latency (ms) | p50 TTFT (ms) | Throughput (tok/s) | Failed |
|-------------|----------------|------------------|------------------|------------------|---------------|-------------------|--------|
| 1 | 9403 | 930.2 | 1000.1 | 1041.4 | 43.7 | 58.0 | 0 |
| 4 | 3985 | 972.5 | 1023.8 | 1087.3 | 64.3 | 56.4 | 0 |
| 8 | 2183 | 1032.6 | 1089.7 | 1172.9 | 102.4 | 55.0 | 0 |
| 16 | 1294 | 1164.4 | 1211.2 | 1276.3 | 168.6 | 51.4 | 0 |

## Token Generation Throughput

| Prompt | Warm tok/s (single) | Warm per-token (ms) | Avg completion tokens |
|--------|--------------------|--------------------|----------------------|
| V0 (short) | 57.4 | 17.83 | 51 |
| V0+ (long) | 58.0 | 17.60 | 52 |

## Raw Request Data

<details>
<summary>Click to expand per-request metrics</summary>

### V0 (short)

**Cold**: TTFT=1875.1ms, total=2721.1ms, tok/s=60.3, prompt_tok=61, comp_tok=51

**Warm sequential requests:**

p50 TTFT=43.8ms, p50 latency=929.1ms

**Batch (concurrency=4)**: wall=3894ms, p50=954.1ms, p95=1028.6ms
**Batch (concurrency=8)**: wall=2205ms, p50=1081.9ms, p95=1197.2ms
**Batch (concurrency=16)**: wall=1212ms, p50=1193.8ms, p95=1209.2ms

### V0+ (long)

**Cold**: TTFT=80.0ms, total=951.4ms, tok/s=58.5, prompt_tok=377, comp_tok=51

**Warm sequential requests:**

p50 TTFT=43.7ms, p50 latency=930.2ms

**Batch (concurrency=4)**: wall=3985ms, p50=972.5ms, p95=1023.8ms
**Batch (concurrency=8)**: wall=2183ms, p50=1032.6ms, p95=1089.7ms
**Batch (concurrency=16)**: wall=1294ms, p50=1164.4ms, p95=1211.2ms

</details>
