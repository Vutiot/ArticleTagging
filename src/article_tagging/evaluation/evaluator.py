"""Batch evaluation pipeline.

Runs predictions on the test set via the inference client, computes metrics,
and displays results as a formatted table.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from article_tagging.evaluation.metrics import (
    EvalResult,
    compute_metrics,
    save_eval_result,
)

if TYPE_CHECKING:
    from article_tagging.configs.models import EvalConfig

logger = logging.getLogger(__name__)
console = Console()


async def run_evaluation(config: EvalConfig) -> EvalResult:
    """Run the full evaluation pipeline.

    1. Load test data JSONL
    2. Load schema for guided decoding
    3. Run batch predictions against vLLM server
    4. Compute exact match + per-attribute metrics
    5. Print results table and save to JSON

    Args:
        config: Evaluation configuration.

    Returns:
        The computed :class:`EvalResult`.
    """
    from article_tagging.dataset.formatter import build_system_prompt
    from article_tagging.inference.client import predict_batch
    from article_tagging.inference.schema_generator import load_schema

    # Load test data
    console.print(f"[bold]Loading test data[/bold] from {config.test_data_path} ...")
    records: list[dict] = []
    with config.test_data_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    console.print(f"  {len(records)} test samples")

    # Load schema
    schema = load_schema(config.schema_path)
    attr_names = [a.name for a in schema.attributes]

    system_prompt = build_system_prompt(
        schema,
        "You extract product attributes from the title and image. "
        "Respond with valid JSON only.",
    )

    # Extract ground truth from test records
    ground_truth: list[dict] = []
    test_inputs: list[dict] = []
    for record in records:
        # Test data may be in chat format (from prepare) or raw format
        if "messages" in record:
            # Chat format — extract title from user message and GT from assistant
            msgs = record["messages"]
            gt = json.loads(msgs[-1]["content"])  # assistant message
            user_content = msgs[1]["content"]
            title = user_content if isinstance(user_content, str) else ""
            if isinstance(user_content, list):
                for block in user_content:
                    if block.get("type") == "text":
                        title = block["text"]
                        break
            test_inputs.append({"title": title, "image_urls": []})
            ground_truth.append(gt)
        else:
            # Raw format
            test_inputs.append(record)
            ground_truth.append(record.get("attributes", {}))

    # Run predictions
    console.print(f"[bold]Running predictions[/bold] against {config.server_url} ...")
    predictions = await predict_batch(
        test_inputs,
        schema,
        config.server_url,
        system_prompt=system_prompt,
        concurrency=config.batch_concurrency,
    )

    # Compute metrics
    result = compute_metrics(
        predictions,
        ground_truth,
        attr_names,
        category_field=schema.category_field,
    )

    # Display results
    _print_results_table(result, attr_names)

    # Save
    config.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = config.output_dir / "eval_result.json"
    save_eval_result(result, result_path)

    # Save per-sample predictions for error analysis
    predictions_path = config.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for pred, gt in zip(predictions, ground_truth):
            fh.write(json.dumps({"prediction": pred, "ground_truth": gt}, ensure_ascii=False) + "\n")

    console.print(f"\nResults saved to [cyan]{config.output_dir}[/cyan]")
    return result


def _print_results_table(result: EvalResult, attr_names: list[str]) -> None:
    """Print a formatted results table to the console."""
    table = Table(title="Evaluation Results")
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
