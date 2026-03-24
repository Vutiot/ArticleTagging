"""Reproduce V0 and V0+ baseline evaluations.

Usage:
    # Run both V0 and V0+ on 50 samples (default)
    python scripts/eval_baseline.py

    # Run only V0+ on 100 samples
    python scripts/eval_baseline.py --run v0+ --samples 100

    # Run on full test set (slow!)
    python scripts/eval_baseline.py --samples 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


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


def run_eval(
    model,
    processor,
    samples: list[dict],
    system_prompt: str,
    attr_names: list[str],
    label: str,
) -> dict:
    """Run evaluation on samples and return eval result dict."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    predictions = []
    ground_truths = []
    errors = 0
    total_time = 0.0

    for i, record in enumerate(samples):
        msgs = record["messages"]
        gt = json.loads(msgs[2]["content"])
        ground_truths.append(gt)

        # Extract title and image path from user message
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
                output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            total_time += time.time() - t0

            generated = output_ids[0][inputs.input_ids.shape[1] :]
            response = processor.decode(generated, skip_special_tokens=True)
            pred = parse_response(response)
            predictions.append(pred)
        except Exception:
            predictions.append({})
            errors += 1

        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i + 1}/{len(samples)} done...")

        del inputs, output_ids
        torch.cuda.empty_cache()

    # Compute metrics
    from article_tagging.evaluation.metrics import compute_metrics

    result = compute_metrics(predictions, ground_truths, attr_names, category_field="masterCategory")

    print(f"\n  [{label}] Errors: {errors}/{len(samples)}, Avg time: {total_time / len(samples):.1f}s/sample")
    return result


def main():
    parser = argparse.ArgumentParser(description="Reproduce V0/V0+ baseline evaluations")
    parser.add_argument(
        "--run",
        choices=["v0", "v0+", "both"],
        default="both",
        help="Which run to execute (default: both)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of test samples (0 = full test set, default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/processed/fashion/test.jsonl"),
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
        help="Output directory for results",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from article_tagging.evaluation.metrics import save_eval_result
    from article_tagging.inference.schema_generator import load_schema

    # Load schema
    schema = load_schema(args.schema)
    attr_names = [a.name for a in schema.attributes]

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    records = []
    with args.test_data.open() as f:
        for line in f:
            records.append(json.loads(line.strip()))

    if args.samples > 0:
        random.seed(args.seed)
        samples = random.sample(records, min(args.samples, len(records)))
    else:
        samples = records
    print(f"Using {len(samples)} samples (seed={args.seed})")

    # Load model
    print("\nLoading Qwen3-VL-2B (FP16)...")
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB\n")

    results = {}

    # V0
    if args.run in ("v0", "both"):
        print("=" * 60)
        print("Running V0 (naive prompt)...")
        v0_prompt = build_v0_prompt(schema)
        v0_result = run_eval(model, processor, samples, v0_prompt, attr_names, "V0")
        save_eval_result(v0_result, args.output_dir / "v0_baseline" / "eval_result.json")
        results["V0"] = v0_result

    # V0+
    if args.run in ("v0+", "both"):
        print("=" * 60)
        print("Running V0+ (enum values in prompt)...")
        v0p_prompt = build_v0_plus_prompt(schema)
        v0p_result = run_eval(model, processor, samples, v0p_prompt, attr_names, "V0+")
        save_eval_result(v0p_result, args.output_dir / "v0_plus_prompt" / "eval_result.json")
        results["V0+"] = v0p_result

    # Print comparison
    print("\n" + "=" * 60)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Baseline Results ({len(samples)} samples, seed={args.seed})")
    table.add_column("Metric", style="bold")
    for name in results:
        table.add_column(name, justify="right")
    if len(results) == 2:
        table.add_column("Delta", justify="right")

    vals = list(results.values())
    table.add_row(
        "Exact Match",
        *[f"{r.exact_match:.1%}" for r in vals],
        *(
            [f"[green]+{vals[1].exact_match - vals[0].exact_match:.1%}[/green]"]
            if len(vals) == 2
            else []
        ),
    )
    table.add_section()

    for attr in attr_names:
        row = [f"  {attr}"]
        for r in vals:
            row.append(f"{r.per_attribute.get(attr, 0):.1%}")
        if len(vals) == 2:
            d = vals[1].per_attribute.get(attr, 0) - vals[0].per_attribute.get(attr, 0)
            color = "green" if d > 0 else "red" if d < 0 else "white"
            row.append(f"[{color}]{d:+.1%}[/{color}]")
        table.add_row(*row)

    console.print(table)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
