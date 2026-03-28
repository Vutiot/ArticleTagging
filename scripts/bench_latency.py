"""vLLM KV caching latency benchmark for ArticleTagging prompts.

Measures TTFT, per-token latency, throughput, GPU usage, and batch scaling
across different prompt variants (V0 short, V0+ long with enum values).

Usage:
    # Start vLLM server first, then:
    python scripts/bench_latency.py

    # With guided JSON decoding
    python scripts/bench_latency.py --guided

    # Custom iterations and batch sizes
    python scripts/bench_latency.py --iterations 20 --batch-sizes 2,4,8

    # Text-only (no images)
    python scripts/bench_latency.py --no-images
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class RequestMetrics:
    """Metrics from a single inference request."""

    ttft_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    token_latencies_ms: list[float]
    tokens_per_sec: float


@dataclass
class BatchMetrics:
    """Aggregated metrics from a batch of requests."""

    concurrency: int
    num_requests: int
    wall_clock_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    avg_tokens_per_sec: float
    avg_per_token_latency_ms: float
    avg_completion_tokens: float
    failed_requests: int = 0


@dataclass
class GpuSnapshot:
    """Single GPU metrics reading."""

    timestamp: float
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one prompt variant."""

    prompt_label: str
    prompt_char_count: int
    cold_request: RequestMetrics | None
    warm_single: BatchMetrics | None
    warm_batches: dict[int, BatchMetrics] = field(default_factory=dict)
    gpu_snapshots: list[GpuSnapshot] = field(default_factory=list)


# ─── Percentile helper ──────────────────────────────────────────────────────


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) via linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


# ─── GPU monitoring ─────────────────────────────────────────────────────────


class GpuMonitor:
    """Context manager that polls nvidia-smi in a background thread."""

    def __init__(self, interval_s: float = 0.5):
        self._interval = interval_s
        self._stop_event = threading.Event()
        self._snapshots: list[GpuSnapshot] = []
        self._thread: threading.Thread | None = None
        self._available = True

    def __enter__(self) -> GpuMonitor:
        self._stop_event.clear()
        self._snapshots.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 3:
                        self._snapshots.append(
                            GpuSnapshot(
                                timestamp=time.time(),
                                utilization_pct=float(parts[0].strip()),
                                memory_used_mb=float(parts[1].strip()),
                                memory_total_mb=float(parts[2].strip()),
                            )
                        )
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                if self._available:
                    logger.warning("nvidia-smi not available, GPU monitoring disabled")
                    self._available = False
                return
            self._stop_event.wait(self._interval)

    @property
    def snapshots(self) -> list[GpuSnapshot]:
        return list(self._snapshots)

    @property
    def peak_memory_mb(self) -> float:
        return max((s.memory_used_mb for s in self._snapshots), default=0.0)

    @property
    def avg_utilization_pct(self) -> float:
        if not self._snapshots:
            return 0.0
        return sum(s.utilization_pct for s in self._snapshots) / len(self._snapshots)

    @property
    def total_memory_mb(self) -> float:
        return self._snapshots[-1].memory_total_mb if self._snapshots else 0.0


# ─── Prompt builders (same as eval_baseline.py) ─────────────────────────────


def build_v0_prompt(schema):
    """V0: just list attribute names."""
    attr_names = ", ".join(a.name for a in schema.attributes)
    return (
        "You extract product attributes from the title and image. "
        "Respond with valid JSON only.\n"
        f"Attributes to extract: {attr_names}"
    )


def build_v0_plus_prompt(schema):
    """V0+: list attribute names with valid enum values."""
    lines = [
        "You extract product attributes from the title and image.",
        "Respond with valid JSON only. Use ONLY values from the lists below.",
        "",
    ]
    for attr in schema.attributes:
        if attr.type == "enum" and attr.values:
            lines.append(f"{attr.name} (valid values): {', '.join(attr.values)}")
        else:
            lines.append(f"{attr.name}: free text string")
    return "\n".join(lines)


# ─── Core measurement ───────────────────────────────────────────────────────


async def measure_request(
    client,
    messages: list[dict],
    model_name: str = "adapter",
    guided_json: dict | None = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> RequestMetrics:
    """Measure a single streaming request, capturing TTFT and per-token timing."""
    extra_body: dict = {}
    if guided_json is not None:
        extra_body["guided_json"] = guided_json

    t_start = time.perf_counter()
    t_first_token: float | None = None
    token_timestamps: list[float] = []
    completion_tokens = 0
    prompt_tokens = 0

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body if extra_body else None,
        timeout=timeout,
    )

    async for chunk in response:
        now = time.perf_counter()

        # Extract usage from final chunk
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

        if chunk.choices and chunk.choices[0].delta.content:
            if t_first_token is None:
                t_first_token = now
            token_timestamps.append(now)

    t_end = time.perf_counter()

    if t_first_token is None:
        t_first_token = t_end

    # Compute inter-token latencies
    token_latencies_ms = []
    for i in range(1, len(token_timestamps)):
        token_latencies_ms.append((token_timestamps[i] - token_timestamps[i - 1]) * 1000)

    # Fallback: count chunks if usage wasn't reported
    if completion_tokens == 0:
        completion_tokens = len(token_timestamps)

    generation_time = t_end - t_first_token
    tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0.0

    return RequestMetrics(
        ttft_ms=(t_first_token - t_start) * 1000,
        total_latency_ms=(t_end - t_start) * 1000,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        token_latencies_ms=token_latencies_ms,
        tokens_per_sec=tokens_per_sec,
    )


def _aggregate_metrics(
    request_metrics: list[RequestMetrics],
    concurrency: int,
    wall_clock_ms: float,
    failed: int = 0,
) -> BatchMetrics:
    """Aggregate individual request metrics into batch-level percentiles."""
    latencies = [m.total_latency_ms for m in request_metrics]
    ttfts = [m.ttft_ms for m in request_metrics]
    tps_values = [m.tokens_per_sec for m in request_metrics]
    all_token_lats = [lat for m in request_metrics for lat in m.token_latencies_ms]
    comp_tokens = [m.completion_tokens for m in request_metrics]

    return BatchMetrics(
        concurrency=concurrency,
        num_requests=len(request_metrics),
        wall_clock_ms=wall_clock_ms,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        p99_latency_ms=percentile(latencies, 99),
        p50_ttft_ms=percentile(ttfts, 50),
        p95_ttft_ms=percentile(ttfts, 95),
        p99_ttft_ms=percentile(ttfts, 99),
        avg_tokens_per_sec=sum(tps_values) / len(tps_values) if tps_values else 0,
        avg_per_token_latency_ms=(
            sum(all_token_lats) / len(all_token_lats) if all_token_lats else 0
        ),
        avg_completion_tokens=sum(comp_tokens) / len(comp_tokens) if comp_tokens else 0,
        failed_requests=failed,
    )


# ─── Message builder ────────────────────────────────────────────────────────


def build_messages(
    record: dict,
    system_prompt: str,
    use_images: bool = True,
) -> list[dict]:
    """Build chat messages from a test record."""
    # Extract title from chat-format or raw records
    if "messages" in record:
        msgs = record["messages"]
        user_content = msgs[1]["content"]
        if isinstance(user_content, list):
            title = next(
                (b["text"] for b in user_content if b.get("type") == "text"), ""
            )
            img_path = next(
                (b.get("image") for b in user_content if b.get("type") == "image"),
                None,
            )
        else:
            title = user_content
            img_path = None
    else:
        title = record.get("title", "")
        image_urls = record.get("image_urls", [])
        img_path = image_urls[0] if image_urls else None

    user_text = f'Title: "{title}"' if not title.startswith('Title: "') else title

    if use_images and img_path and Path(img_path).exists():
        from article_tagging.dataset.image_processing import (
            image_to_base64,
            preprocess_image,
        )

        img = preprocess_image(Path(img_path))
        data_uri = image_to_base64(img)
        user_content_final: str | list[dict] = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": user_text},
        ]
    else:
        user_content_final = user_text

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content_final},
    ]


# ─── Benchmark orchestrator ─────────────────────────────────────────────────


async def run_benchmark(
    server_url: str,
    test_records: list[dict],
    schema,
    prompt_configs: list[tuple[str, str]],
    guided_json: dict | None,
    model_name: str,
    num_warmup: int = 3,
    num_iterations: int = 10,
    batch_sizes: list[int] | None = None,
    num_batch_records: int = 16,
    use_images: bool = True,
    gpu_poll_interval: float = 0.5,
) -> list[BenchmarkResult]:
    """Run the full benchmark protocol for all prompt variants."""
    from openai import AsyncOpenAI

    if batch_sizes is None:
        batch_sizes = [4, 8, 16]

    client = AsyncOpenAI(base_url=f"{server_url}/v1", api_key="not-needed")
    results: list[BenchmarkResult] = []

    from rich.console import Console

    console = Console()

    for label, system_prompt in prompt_configs:
        console.print(f"\n[bold]{'=' * 60}[/bold]")
        console.print(f"[bold]Benchmarking: {label}[/bold] ({len(system_prompt)} chars)")
        console.print(f"[bold]{'=' * 60}[/bold]")

        bench_result = BenchmarkResult(
            prompt_label=label,
            prompt_char_count=len(system_prompt),
            cold_request=None,
            warm_single=None,
        )

        with GpuMonitor(interval_s=gpu_poll_interval) as gpu:
            # ── Phase A: Cold cache ──────────────────────────────────────
            console.print("\n[cyan]Phase A:[/cyan] Cold cache (1st request)...")
            first_record = test_records[0]
            messages = build_messages(first_record, system_prompt, use_images)

            try:
                cold = await measure_request(
                    client, messages, model_name, guided_json
                )
                bench_result.cold_request = cold
                console.print(
                    f"  Cold TTFT: [yellow]{cold.ttft_ms:.1f}ms[/yellow], "
                    f"Total: [yellow]{cold.total_latency_ms:.1f}ms[/yellow], "
                    f"Tokens/sec: {cold.tokens_per_sec:.1f}"
                )
            except Exception as e:
                console.print(f"  [red]Cold request failed: {e}[/red]")

            # ── Phase B: Warmup ──────────────────────────────────────────
            console.print(f"\n[cyan]Phase B:[/cyan] Warmup ({num_warmup} requests)...")
            for i in range(num_warmup):
                rec = test_records[i % len(test_records)]
                msgs = build_messages(rec, system_prompt, use_images)
                try:
                    await measure_request(client, msgs, model_name, guided_json)
                except Exception:
                    pass

            # ── Phase C: Warm single requests ────────────────────────────
            console.print(
                f"\n[cyan]Phase C:[/cyan] Warm sequential ({num_iterations} requests)..."
            )
            warm_metrics: list[RequestMetrics] = []
            warm_failed = 0
            t_warm_start = time.perf_counter()

            for i in range(num_iterations):
                rec = test_records[(i + num_warmup) % len(test_records)]
                msgs = build_messages(rec, system_prompt, use_images)
                try:
                    m = await measure_request(client, msgs, model_name, guided_json)
                    warm_metrics.append(m)
                except Exception:
                    warm_failed += 1

            t_warm_end = time.perf_counter()

            if warm_metrics:
                bench_result.warm_single = _aggregate_metrics(
                    warm_metrics,
                    concurrency=1,
                    wall_clock_ms=(t_warm_end - t_warm_start) * 1000,
                    failed=warm_failed,
                )
                ws = bench_result.warm_single
                console.print(
                    f"  p50 TTFT: [green]{ws.p50_ttft_ms:.1f}ms[/green], "
                    f"p95: {ws.p95_ttft_ms:.1f}ms, "
                    f"p99: {ws.p99_ttft_ms:.1f}ms"
                )
                console.print(
                    f"  p50 latency: [green]{ws.p50_latency_ms:.1f}ms[/green], "
                    f"p95: {ws.p95_latency_ms:.1f}ms, "
                    f"p99: {ws.p99_latency_ms:.1f}ms"
                )
                console.print(
                    f"  Throughput: {ws.avg_tokens_per_sec:.1f} tok/s, "
                    f"Per-token: {ws.avg_per_token_latency_ms:.2f}ms"
                )

            # ── Phase D: Batch scaling ───────────────────────────────────
            for conc in batch_sizes:
                n_reqs = max(conc, num_batch_records)
                console.print(
                    f"\n[cyan]Phase D:[/cyan] Batch concurrency={conc} ({n_reqs} requests)..."
                )

                sem = asyncio.Semaphore(conc)
                batch_metrics_list: list[RequestMetrics] = []
                batch_failed = 0

                async def _measure_one(idx: int) -> RequestMetrics | None:
                    nonlocal batch_failed
                    async with sem:
                        rec = test_records[idx % len(test_records)]
                        msgs = build_messages(rec, system_prompt, use_images)
                        try:
                            return await measure_request(
                                client, msgs, model_name, guided_json
                            )
                        except Exception:
                            batch_failed += 1
                            return None

                t_batch_start = time.perf_counter()
                tasks = [_measure_one(i) for i in range(n_reqs)]
                raw_results = await asyncio.gather(*tasks)
                t_batch_end = time.perf_counter()

                batch_metrics_list = [r for r in raw_results if r is not None]

                if batch_metrics_list:
                    bm = _aggregate_metrics(
                        batch_metrics_list,
                        concurrency=conc,
                        wall_clock_ms=(t_batch_end - t_batch_start) * 1000,
                        failed=batch_failed,
                    )
                    bench_result.warm_batches[conc] = bm
                    console.print(
                        f"  Wall clock: {bm.wall_clock_ms:.0f}ms, "
                        f"p50 latency: {bm.p50_latency_ms:.1f}ms, "
                        f"p95: {bm.p95_latency_ms:.1f}ms"
                    )

            bench_result.gpu_snapshots = gpu.snapshots

        results.append(bench_result)

    return results


# ─── Report generator ────────────────────────────────────────────────────────


def generate_report(
    results: list[BenchmarkResult],
    gpu_monitor: GpuMonitor,
    config: dict,
) -> str:
    """Generate a Markdown latency benchmark report."""
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("# vLLM Latency Benchmark Report")
    lines.append("")
    lines.append(f"**Date**: {ts}")
    lines.append(f"**Server**: {config['server_url']}")
    lines.append(f"**Model**: {config['model_name']}")
    lines.append(f"**Guided JSON**: {'Yes' if config.get('guided') else 'No'}")
    lines.append(f"**Images**: {'No' if config.get('no_images') else 'Yes'}")
    lines.append(f"**Iterations**: {config['iterations']} (warmup: {config['warmup']})")
    lines.append(f"**Batch sizes**: {config['batch_sizes']}")
    lines.append("")

    # GPU summary
    lines.append("## GPU Utilization")
    lines.append("")
    if gpu_monitor.snapshots:
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(
            f"| Peak memory | {gpu_monitor.peak_memory_mb:.0f} MB "
            f"/ {gpu_monitor.total_memory_mb:.0f} MB |"
        )
        lines.append(f"| Avg utilization | {gpu_monitor.avg_utilization_pct:.1f}% |")
        lines.append(
            f"| Samples collected | {len(gpu_monitor.snapshots)} |"
        )
    else:
        lines.append("*GPU monitoring unavailable (nvidia-smi not found)*")
    lines.append("")

    # Prompt summary
    lines.append("## Prompts Tested")
    lines.append("")
    lines.append("| Prompt | Length (chars) |")
    lines.append("|--------|--------------|")
    for r in results:
        lines.append(f"| {r.prompt_label} | {r.prompt_char_count} |")
    lines.append("")

    # ── Cold vs Warm ─────────────────────────────────────────────────────
    lines.append("## Cold vs Warm Cache Latency")
    lines.append("")

    for r in results:
        lines.append(f"### {r.prompt_label}")
        lines.append("")

        if r.cold_request and r.warm_single:
            cold = r.cold_request
            warm = r.warm_single
            ttft_speedup = cold.ttft_ms / warm.p50_ttft_ms if warm.p50_ttft_ms > 0 else 0
            lat_speedup = (
                cold.total_latency_ms / warm.p50_latency_ms
                if warm.p50_latency_ms > 0
                else 0
            )

            lines.append(
                "| Metric | Cold (1st req) | Warm p50 | Warm p95 | Warm p99 | Speedup |"
            )
            lines.append(
                "|--------|---------------|----------|----------|----------|---------|"
            )
            lines.append(
                f"| TTFT (ms) | {cold.ttft_ms:.1f} | {warm.p50_ttft_ms:.1f} | "
                f"{warm.p95_ttft_ms:.1f} | {warm.p99_ttft_ms:.1f} | "
                f"{ttft_speedup:.1f}x |"
            )
            lines.append(
                f"| Total latency (ms) | {cold.total_latency_ms:.1f} | "
                f"{warm.p50_latency_ms:.1f} | {warm.p95_latency_ms:.1f} | "
                f"{warm.p99_latency_ms:.1f} | {lat_speedup:.1f}x |"
            )
            lines.append(
                f"| Tokens/sec | {cold.tokens_per_sec:.1f} | "
                f"{warm.avg_tokens_per_sec:.1f} | — | — | — |"
            )
            lines.append(
                f"| Per-token latency (ms) | "
                f"{sum(cold.token_latencies_ms) / len(cold.token_latencies_ms):.2f}"
                if cold.token_latencies_ms
                else "| Per-token latency (ms) | —"
            )
            lines[-1] += (
                f" | {warm.avg_per_token_latency_ms:.2f} | — | — | — |"
            )
            lines.append(
                f"| Prompt tokens | {cold.prompt_tokens} | — | — | — | — |"
            )
            lines.append(
                f"| Completion tokens | {cold.completion_tokens} | "
                f"{warm.avg_completion_tokens:.0f} (avg) | — | — | — |"
            )
        else:
            lines.append("*No data collected*")
        lines.append("")

    # ── Batch Scaling ────────────────────────────────────────────────────
    lines.append("## Batch Scaling (Latency per Batch vs per Request)")
    lines.append("")

    for r in results:
        lines.append(f"### {r.prompt_label}")
        lines.append("")

        all_concs: list[tuple[int, BatchMetrics]] = []
        if r.warm_single:
            all_concs.append((1, r.warm_single))
        for conc in sorted(r.warm_batches.keys()):
            all_concs.append((conc, r.warm_batches[conc]))

        if all_concs:
            lines.append(
                "| Concurrency | Wall clock (ms) | p50 latency (ms) | "
                "p95 latency (ms) | p99 latency (ms) | p50 TTFT (ms) | "
                "Throughput (tok/s) | Failed |"
            )
            lines.append(
                "|-------------|----------------|------------------|"
                "------------------|------------------|---------------|"
                "-------------------|--------|"
            )
            for conc, bm in all_concs:
                lines.append(
                    f"| {conc} | {bm.wall_clock_ms:.0f} | {bm.p50_latency_ms:.1f} | "
                    f"{bm.p95_latency_ms:.1f} | {bm.p99_latency_ms:.1f} | "
                    f"{bm.p50_ttft_ms:.1f} | {bm.avg_tokens_per_sec:.1f} | "
                    f"{bm.failed_requests} |"
                )
        else:
            lines.append("*No data collected*")
        lines.append("")

    # ── Token throughput summary ─────────────────────────────────────────
    lines.append("## Token Generation Throughput")
    lines.append("")
    lines.append(
        "| Prompt | Warm tok/s (single) | Warm per-token (ms) | "
        "Avg completion tokens |"
    )
    lines.append(
        "|--------|--------------------|--------------------|"
        "----------------------|"
    )
    for r in results:
        if r.warm_single:
            ws = r.warm_single
            lines.append(
                f"| {r.prompt_label} | {ws.avg_tokens_per_sec:.1f} | "
                f"{ws.avg_per_token_latency_ms:.2f} | {ws.avg_completion_tokens:.0f} |"
            )
    lines.append("")

    # ── Raw data (collapsed) ─────────────────────────────────────────────
    lines.append("## Raw Request Data")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Click to expand per-request metrics</summary>")
    lines.append("")

    for r in results:
        lines.append(f"### {r.prompt_label}")
        lines.append("")

        if r.cold_request:
            c = r.cold_request
            lines.append(
                f"**Cold**: TTFT={c.ttft_ms:.1f}ms, "
                f"total={c.total_latency_ms:.1f}ms, "
                f"tok/s={c.tokens_per_sec:.1f}, "
                f"prompt_tok={c.prompt_tokens}, "
                f"comp_tok={c.completion_tokens}"
            )
            lines.append("")

        if r.warm_single:
            lines.append("**Warm sequential requests:**")
            lines.append("")
            # Note: we don't store individual metrics in BatchMetrics,
            # but we have them in the JSON output
            lines.append(
                f"p50 TTFT={r.warm_single.p50_ttft_ms:.1f}ms, "
                f"p50 latency={r.warm_single.p50_latency_ms:.1f}ms"
            )
            lines.append("")

        for conc in sorted(r.warm_batches.keys()):
            bm = r.warm_batches[conc]
            lines.append(
                f"**Batch (concurrency={conc})**: "
                f"wall={bm.wall_clock_ms:.0f}ms, "
                f"p50={bm.p50_latency_ms:.1f}ms, "
                f"p95={bm.p95_latency_ms:.1f}ms"
            )
        lines.append("")

    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


# ─── JSON serializer ────────────────────────────────────────────────────────


def results_to_json(results: list[BenchmarkResult], config: dict) -> dict:
    """Convert benchmark results to a JSON-serializable dict."""
    ts = datetime.now(timezone.utc).isoformat()

    def _metrics_dict(m: RequestMetrics | None) -> dict | None:
        if m is None:
            return None
        return asdict(m)

    def _batch_dict(b: BatchMetrics | None) -> dict | None:
        if b is None:
            return None
        return asdict(b)

    return {
        "timestamp": ts,
        "config": config,
        "results": [
            {
                "prompt_label": r.prompt_label,
                "prompt_char_count": r.prompt_char_count,
                "cold_request": _metrics_dict(r.cold_request),
                "warm_single": _batch_dict(r.warm_single),
                "warm_batches": {
                    str(k): _batch_dict(v) for k, v in r.warm_batches.items()
                },
                "gpu_snapshots_count": len(r.gpu_snapshots),
            }
            for r in results
        ],
    }


# ─── Health check ────────────────────────────────────────────────────────────


async def check_server_health(server_url: str) -> bool:
    """Check if the vLLM server is reachable."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{server_url}/health", timeout=5)
            return resp.status_code == 200
    except Exception:
        return False


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM KV caching latency for ArticleTagging prompts"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="adapter",
        help="Model name for API requests (default: adapter)",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/processed/fashion/test_50_seed42.jsonl"),
        help="Path to test JSONL",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("configs/schemas/fashion.yaml"),
        help="Path to schema YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for benchmark report",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of measured iterations per config (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="4,8,16",
        help="Comma-separated concurrency levels to test (default: 4,8,16)",
    )
    parser.add_argument(
        "--guided",
        action="store_true",
        default=False,
        help="Enable guided JSON decoding (adds FSM overhead)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        default=False,
        help="Skip image encoding (text-only requests)",
    )
    parser.add_argument(
        "--gpu-poll-interval",
        type=float,
        default=0.5,
        help="GPU monitoring poll interval in seconds (default: 0.5)",
    )
    args = parser.parse_args()

    from rich.console import Console

    console = Console()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Load schema
    from article_tagging.inference.schema_generator import generate_json_schema, load_schema

    schema = load_schema(args.schema)
    guided_json = generate_json_schema(schema) if args.guided else None

    # Load test data
    console.print(f"[bold]Loading test data[/bold] from {args.test_data}...")
    records: list[dict] = []
    with args.test_data.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    console.print(f"  {len(records)} test samples loaded")

    # Build prompt configs
    prompt_configs = [
        ("V0 (short)", build_v0_prompt(schema)),
        ("V0+ (long)", build_v0_plus_prompt(schema)),
    ]
    for label, prompt in prompt_configs:
        console.print(f"  {label}: {len(prompt)} chars")

    # Check server health
    console.print(f"\n[bold]Checking server[/bold] at {args.server_url}...")
    healthy = asyncio.run(check_server_health(args.server_url))
    if not healthy:
        console.print(
            f"[red bold]Server not reachable at {args.server_url}[/red bold]\n"
            "Start the vLLM server first:\n"
            "  python -m article_tagging.cli.main serve "
            "--config configs/serving_fashion_adapter.yaml"
        )
        sys.exit(1)
    console.print("  [green]Server is healthy[/green]")

    config = {
        "server_url": args.server_url,
        "model_name": args.model_name,
        "guided": args.guided,
        "no_images": args.no_images,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "batch_sizes": batch_sizes,
    }

    # Run global GPU monitor for the report summary
    gpu_global = GpuMonitor(interval_s=args.gpu_poll_interval)

    with gpu_global:
        results = asyncio.run(
            run_benchmark(
                server_url=args.server_url,
                test_records=records,
                schema=schema,
                prompt_configs=prompt_configs,
                guided_json=guided_json,
                model_name=args.model_name,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                batch_sizes=batch_sizes,
                num_batch_records=max(batch_sizes) if batch_sizes else 16,
                use_images=not args.no_images,
                gpu_poll_interval=args.gpu_poll_interval,
            )
        )

    # Generate report
    report = generate_report(results, gpu_global, config)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "latency_benchmark.md"
    report_path.write_text(report, encoding="utf-8")

    json_path = args.output_dir / "latency_benchmark.json"
    json_data = results_to_json(results, config)
    json_path.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    console.print(f"\n[bold green]Benchmark complete![/bold green]")
    console.print(f"  Report: [cyan]{report_path}[/cyan]")
    console.print(f"  Raw data: [cyan]{json_path}[/cyan]")


if __name__ == "__main__":
    main()
