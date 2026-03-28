"""Benchmark Qwen3-VL-8B on the fashion dataset (Lightning AI / GPU VM).

Runs the full benchmark pipeline: train, evaluate, and latency test.
Produces reports in the same format as reports/v2_guided/ for comparison.

Usage:
    # Run everything
    python scripts/benchmark_qwen3vl_8b.py --phase all

    # Individual phases
    python scripts/benchmark_qwen3vl_8b.py --phase train
    python scripts/benchmark_qwen3vl_8b.py --phase eval
    python scripts/benchmark_qwen3vl_8b.py --phase latency

    # Eval with custom adapter path
    python scripts/benchmark_qwen3vl_8b.py --phase eval --adapter models/fashion-qwen3vl-8b/adapter

    # Quick training test (10 steps)
    python scripts/benchmark_qwen3vl_8b.py --phase train --max-steps 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPORT_DIR = Path("reports/v3_qwen3vl_8b")
TRAINING_CONFIG = Path("configs/training_fashion_qwen3vl_8b.yaml")
SERVING_CONFIG = Path("configs/serving_fashion_qwen3vl_8b.yaml")
DATASET_DIR = Path("data/processed/fashion")
TEST_DATA = Path("data/processed/fashion/test_500_seed42.jsonl")
SCHEMA_PATH = Path("configs/schemas/fashion.yaml")
V2_RESULTS = Path("reports/v2_guided/eval_result.json")


# ─── Phase A: Training ───────────────────────────────────────────────────────


def run_train(max_steps: int | None = None) -> Path:
    """Fine-tune Qwen3-VL-8B on the fashion dataset."""
    from rich.console import Console

    from article_tagging.configs.models import TrainingConfig, load_config
    from article_tagging.training.data import load_training_dataset
    from article_tagging.training.export import export_model
    from article_tagging.training.trainer import run_training

    console = Console()
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print("[bold]Phase A: Training Qwen3-VL-8B[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]\n")

    # Load config
    config = load_config(TRAINING_CONFIG, TrainingConfig)
    if max_steps is not None:
        config = TrainingConfig(**{**config.model_dump(), "max_steps": max_steps})

    console.print(f"  Model: [cyan]{config.model_name}[/cyan]")
    console.print(f"  Output: [cyan]{config.output_dir}[/cyan]")
    console.print(f"  Epochs: {config.epochs}, LR: {config.learning_rate}")
    if max_steps is not None:
        console.print(f"  Max steps override: {max_steps}")

    # Print GPU info
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        console.print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        console.print("[red]No GPU detected! Training will be very slow.[/red]")

    # Load dataset
    console.print("\n[bold]Loading dataset...[/bold]")
    train_path = DATASET_DIR / "train.jsonl"
    val_path = DATASET_DIR / "val.jsonl"
    train_ds, val_ds = load_training_dataset(
        train_path,
        val_path if val_path.exists() else None,
    )
    console.print(f"  train: {len(train_ds)}, val: {len(val_ds) if val_ds else 0}")

    # Train
    t0 = time.time()
    output_dir, model, tokenizer = run_training(config, train_ds, val_ds)
    train_time = time.time() - t0
    console.print(f"\n[bold green]Training complete![/bold green] ({train_time / 3600:.1f} hours)")

    # Export
    export_model(model, tokenizer, config)

    # Save training metadata
    meta = {
        "model": config.model_name,
        "train_time_seconds": train_time,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
        "epochs": config.epochs,
        "max_steps": config.max_steps,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = REPORT_DIR / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    console.print(f"  Training metadata: [cyan]{meta_path}[/cyan]")

    return output_dir


# ─── Phase B: Evaluation (direct model.generate) ─────────────────────────────


def parse_response(response: str) -> dict:
    """Extract JSON dict from model response, handling thinking tags and markdown."""
    resp = response.strip()
    if "</think>" in resp:
        resp = resp.split("</think>")[-1].strip()
    if "```json" in resp:
        resp = resp.split("```json")[1].split("```")[0].strip()
    elif "```" in resp:
        resp = resp.split("```")[1].split("```")[0].strip()
    start = resp.find("{")
    end = resp.rfind("}") + 1
    if start >= 0 and end > start:
        resp = resp[start:end]
    return json.loads(resp)


def run_eval(adapter_path: str | None = None) -> None:
    """Evaluate the fine-tuned model using direct model.generate()."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from rich.console import Console
    from rich.table import Table

    from article_tagging.evaluation.metrics import (
        compute_metrics,
        load_eval_result,
        save_eval_result,
    )
    from article_tagging.evaluation.report import generate_comparison, save_report
    from article_tagging.inference.schema_generator import load_schema

    console = Console()
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print("[bold]Phase B: Evaluation (direct inference)[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]\n")

    # Resolve adapter path
    if adapter_path is None:
        adapter_path = "models/fashion-qwen3vl-8b/adapter"
    adapter_dir = Path(adapter_path)

    if not adapter_dir.exists():
        console.print(f"[red]Adapter not found at {adapter_dir}[/red]")
        console.print("Run --phase train first, or specify --adapter path")
        sys.exit(1)

    # Load schema and test data
    schema = load_schema(SCHEMA_PATH)
    attr_names = [a.name for a in schema.attributes]

    console.print(f"Loading test data from {TEST_DATA}...")
    records = []
    with TEST_DATA.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    console.print(f"  {len(records)} test samples")

    # Build system prompt (same as used in training/evaluation)
    from article_tagging.dataset.formatter import build_system_prompt

    system_prompt = build_system_prompt(
        schema,
        "You extract product attributes from the title and image. "
        "Respond with valid JSON only.",
    )

    # Load model with LoRA adapter
    console.print(f"\nLoading model with adapter from {adapter_dir}...")
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=str(adapter_dir),
        load_in_4bit=True,
        max_seq_length=2048,
    )
    FastVisionModel.for_inference(model)

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**3
        console.print(f"  VRAM after loading: {vram:.1f} GB")

    # Run predictions
    console.print(f"\n[bold]Running predictions[/bold] on {len(records)} samples...")
    predictions = []
    ground_truths = []
    errors = 0
    total_time = 0.0

    for i, record in enumerate(records):
        msgs = record["messages"]
        gt = json.loads(msgs[2]["content"])
        ground_truths.append(gt)

        # Extract title and image from user message
        user_content = msgs[1]["content"]
        title = ""
        img_path = None
        if isinstance(user_content, list):
            for block in user_content:
                if block.get("type") == "text":
                    title = block["text"]
                elif block.get("type") == "image":
                    img_path = block.get("image")
        else:
            title = user_content

        # Build inference messages
        inf_messages = [{"role": "system", "content": system_prompt}]
        user_blocks = []
        if img_path and Path(img_path).exists():
            img = Image.open(img_path).convert("RGB").resize((384, 384))
            user_blocks.append({"type": "image", "image": img})
        user_blocks.append({"type": "text", "text": title})
        inf_messages.append({"role": "user", "content": user_blocks})

        text = processor.apply_chat_template(
            inf_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(inf_messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
        ).to(model.device)

        try:
            t0 = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=200, do_sample=False
                )
            total_time += time.time() - t0

            generated = output_ids[0][inputs.input_ids.shape[1]:]
            response = processor.decode(generated, skip_special_tokens=True)
            pred = parse_response(response)
            predictions.append(pred)
        except Exception as e:
            predictions.append({})
            errors += 1
            if errors <= 3:
                console.print(f"  [yellow]Error on sample {i}: {e}[/yellow]")

        if (i + 1) % 10 == 0:
            console.print(f"  {i + 1}/{len(records)} done...")

        del inputs, output_ids
        torch.cuda.empty_cache()

    console.print(
        f"\n  Errors: {errors}/{len(records)}, "
        f"Avg time: {total_time / len(records):.1f}s/sample"
    )

    # Compute metrics
    result = compute_metrics(
        predictions, ground_truths, attr_names, category_field="masterCategory"
    )

    # Print results table
    table = Table(title=f"V3 Qwen3-VL-8B Results ({len(records)} samples)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Exact Match", f"{result.exact_match:.1%}")
    table.add_row("Total Samples", str(result.total_samples))
    table.add_section()
    for attr in attr_names:
        acc = result.per_attribute.get(attr, 0.0)
        table.add_row(f"  {attr}", f"{acc:.1%}")
    if result.category_breakdown:
        table.add_section()
        for cat, em in sorted(result.category_breakdown.items()):
            table.add_row(f"  [{cat}]", f"{em:.1%}")
    console.print(table)

    # Save results
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = REPORT_DIR / "eval_result.json"
    save_eval_result(result, result_path)
    console.print(f"  Results: [cyan]{result_path}[/cyan]")

    # Save per-sample predictions
    predictions_path = REPORT_DIR / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for pred, gt in zip(predictions, ground_truths):
            fh.write(
                json.dumps(
                    {"prediction": pred, "ground_truth": gt}, ensure_ascii=False
                )
                + "\n"
            )
    console.print(f"  Predictions: [cyan]{predictions_path}[/cyan]")

    # Comparison with V2 if available
    if V2_RESULTS.exists():
        console.print("\n[bold]Comparison with V2 (Qwen3-VL-2B):[/bold]")
        v2_result = load_eval_result(V2_RESULTS)
        comparison = generate_comparison([
            ("V2 (Qwen3-VL-2B)", v2_result),
            ("V3 (Qwen3-VL-8B)", result),
        ])
        report_path = REPORT_DIR / "comparison_v2_v3.md"
        save_report(comparison, report_path)
        console.print(f"  Comparison report: [cyan]{report_path}[/cyan]")

        # Quick delta summary
        delta_table = Table(title="V2 vs V3 Delta")
        delta_table.add_column("Metric", style="bold")
        delta_table.add_column("V2 (2B)", justify="right")
        delta_table.add_column("V3 (8B)", justify="right")
        delta_table.add_column("Delta", justify="right")
        delta = result.exact_match - v2_result.exact_match
        color = "green" if delta > 0 else "red" if delta < 0 else "white"
        delta_table.add_row(
            "Exact Match",
            f"{v2_result.exact_match:.1%}",
            f"{result.exact_match:.1%}",
            f"[{color}]{delta:+.1%}[/{color}]",
        )
        delta_table.add_section()
        for attr in attr_names:
            v2_acc = v2_result.per_attribute.get(attr, 0.0)
            v3_acc = result.per_attribute.get(attr, 0.0)
            d = v3_acc - v2_acc
            c = "green" if d > 0 else "red" if d < 0 else "white"
            delta_table.add_row(
                f"  {attr}", f"{v2_acc:.1%}", f"{v3_acc:.1%}", f"[{c}]{d:+.1%}[/{c}]"
            )
        console.print(delta_table)


# ─── Phase C: vLLM Latency Benchmark ─────────────────────────────────────────


def run_latency() -> None:
    """Run vLLM latency benchmark with the fine-tuned adapter."""
    import subprocess

    from rich.console import Console

    console = Console()
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print("[bold]Phase C: vLLM Latency Benchmark[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]\n")

    if not SERVING_CONFIG.exists():
        console.print(f"[red]Serving config not found: {SERVING_CONFIG}[/red]")
        sys.exit(1)

    adapter_dir = Path("models/fashion-qwen3vl-8b/adapter")
    if not adapter_dir.exists():
        console.print(f"[red]Adapter not found at {adapter_dir}[/red]")
        console.print("Run --phase train first")
        sys.exit(1)

    # Launch vLLM server
    from article_tagging.configs.models import ServingConfig, load_config
    from article_tagging.inference.server import launch_server

    serving_config = load_config(SERVING_CONFIG, ServingConfig)
    console.print(f"Launching vLLM server (model: {serving_config.model_path})...")
    server_process = launch_server(serving_config)

    # Wait for server to be ready
    import httpx

    server_url = f"http://{serving_config.host}:{serving_config.port}"
    console.print("Waiting for server to be ready...")
    for i in range(120):
        try:
            resp = httpx.get(f"{server_url}/health", timeout=5)
            if resp.status_code == 200:
                console.print(f"  [green]Server ready after {i + 1}s[/green]")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        console.print("[red]Server failed to start within 120s[/red]")
        server_process.terminate()
        sys.exit(1)

    try:
        # Run the latency benchmark
        bench_args = [
            sys.executable,
            "scripts/bench_latency.py",
            "--server-url", server_url,
            "--model-name", "adapter",
            "--test-data", str(TEST_DATA),
            "--schema", str(SCHEMA_PATH),
            "--output-dir", str(REPORT_DIR),
            "--iterations", "10",
            "--warmup", "3",
            "--batch-sizes", "4,8,16",
        ]
        console.print(f"\nRunning: {' '.join(bench_args)}")
        subprocess.run(bench_args, check=True)

    finally:
        console.print("\n[yellow]Shutting down vLLM server...[/yellow]")
        server_process.terminate()
        server_process.wait(timeout=30)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-VL-8B on fashion dataset"
    )
    parser.add_argument(
        "--phase",
        choices=["train", "eval", "latency", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter (for eval phase, default: models/fashion-qwen3vl-8b/adapter)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps (for quick testing)",
    )
    args = parser.parse_args()

    from rich.console import Console

    console = Console()
    console.print("[bold]Qwen3-VL-8B Fashion Benchmark[/bold]")
    console.print(f"  Phase: {args.phase}")
    console.print(f"  Reports: {REPORT_DIR}")

    if args.phase in ("train", "all"):
        run_train(max_steps=args.max_steps)

    if args.phase in ("eval", "all"):
        run_eval(adapter_path=args.adapter)

    if args.phase in ("latency", "all"):
        run_latency()

    console.print("\n[bold green]Benchmark complete![/bold green]")
    console.print(f"  Reports saved to: [cyan]{REPORT_DIR}[/cyan]")


if __name__ == "__main__":
    main()
