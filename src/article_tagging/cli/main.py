"""CLI entry point for the ArticleTagging pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console

from article_tagging.configs.models import SiteConfig, load_config

console = Console()


@click.group()
def cli() -> None:
    """ArticleTagging — fine-tune VLMs for structured attribute extraction."""


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to site YAML config.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/raw"), show_default=True, help="Root output directory.")
@click.option("--no-images", is_flag=True, default=False, help="Skip image downloading.")
@click.option("--max-listings", type=int, default=None, help="Override max listings from config.")
def scrape(config: Path, output_dir: Path, no_images: bool, max_listings: int | None) -> None:
    """Scrape listings from a configured site."""
    from article_tagging.scraping.orchestrator import run_scrape

    site_config = load_config(config, SiteConfig)
    if max_listings is not None:
        site_config = SiteConfig(**{**site_config.model_dump(), "max_listings": max_listings})

    asyncio.run(run_scrape(site_config, output_dir, download_images_flag=not no_images))


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to dataset YAML config.")
@click.option("--raw-data", type=click.Path(exists=True, path_type=Path), required=True, help="Path to raw JSONL from scraping.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/processed"), show_default=True, help="Output directory for formatted splits.")
@click.option("--image-dir", type=click.Path(exists=True, path_type=Path), default=None, help="Directory containing downloaded images.")
def prepare(config: Path, raw_data: Path, output_dir: Path, image_dir: Path | None) -> None:
    """Prepare a fine-tuning dataset from scraped data."""
    from article_tagging.configs.models import DatasetConfig
    from article_tagging.dataset.cleaning import clean_listings, load_raw_jsonl
    from article_tagging.dataset.formatter import format_dataset
    from article_tagging.dataset.splitting import split_dataset
    from article_tagging.inference.schema_generator import load_schema

    dataset_config = load_config(config, DatasetConfig)
    schema = load_schema(dataset_config.schema_path)

    # Load and clean
    console.print(f"[bold]Loading[/bold] {raw_data} ...")
    listings = load_raw_jsonl(raw_data)
    console.print(f"  Loaded {len(listings)} raw listings")

    cleaned, stats = clean_listings(
        listings,
        schema,
        deduplicate=dataset_config.deduplicate,
        require_images=not dataset_config.text_only,
    )
    console.print(
        f"  Cleaned: {stats.kept}/{stats.total} kept "
        f"(-{stats.dropped_empty_title} empty, -{stats.dropped_invalid_attrs} invalid, "
        f"-{stats.dropped_duplicates} dupes, -{stats.dropped_missing_images} no image)"
    )

    if stats.kept < dataset_config.min_samples:
        console.print(f"[red]Only {stats.kept} samples after cleaning (min: {dataset_config.min_samples}). Aborting.[/red]")
        raise SystemExit(1)

    # Split
    train, val, test = split_dataset(
        cleaned,
        dataset_config.split_ratio,
        dataset_config.category_field,
    )
    console.print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Format into chat conversations
    console.print("[bold]Formatting[/bold] into chat conversations ...")
    train_fmt = format_dataset(train, schema, dataset_config.system_prompt, text_only=dataset_config.text_only, image_dir=image_dir)
    val_fmt = format_dataset(val, schema, dataset_config.system_prompt, text_only=dataset_config.text_only, image_dir=image_dir)
    test_fmt = format_dataset(test, schema, dataset_config.system_prompt, text_only=dataset_config.text_only, image_dir=image_dir)

    # Save
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_fmt), ("val", val_fmt), ("test", test_fmt)]:
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in data:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    console.print(f"\n[bold green]Done![/bold green] Saved to {output_dir}/")
    console.print(f"  train: {len(train_fmt)}, val: {len(val_fmt)}, test: {len(test_fmt)}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to training YAML config.")
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), required=True, help="Path to prepared dataset directory (with train.jsonl, val.jsonl).")
@click.option("--text-only", is_flag=True, default=False, help="Skip image loading for text-only training.")
@click.option("--max-steps", type=int, default=None, help="Override max training steps (overrides epochs).")
@click.option("--run-name", type=str, default=None, help="Override run name.")
@click.option("--wandb", "use_wandb", is_flag=True, default=False, help="Enable W&B logging.")
def train(config: Path, dataset: Path, text_only: bool, max_steps: int | None, run_name: str | None, use_wandb: bool) -> None:
    """Fine-tune a VLM on the prepared dataset."""
    from article_tagging.configs.models import TrainingConfig
    from article_tagging.training.data import load_training_dataset
    from article_tagging.training.export import export_model
    from article_tagging.training.trainer import run_training

    training_config = load_config(config, TrainingConfig)

    # Override from CLI flags
    overrides: dict = {}
    if max_steps is not None:
        overrides["max_steps"] = max_steps
    if run_name is not None:
        overrides["run_name"] = run_name
    if use_wandb:
        overrides["use_wandb"] = True
    if overrides:
        training_config = TrainingConfig(**{**training_config.model_dump(), **overrides})

    train_path = dataset / "train.jsonl"
    val_path = dataset / "val.jsonl"

    console.print(f"[bold]Loading dataset[/bold] from {dataset} ...")
    train_ds, val_ds = load_training_dataset(
        train_path,
        val_path if val_path.exists() else None,
        text_only=text_only,
    )
    console.print(f"  train: {len(train_ds)}, val: {len(val_ds) if val_ds else 0}")

    output_dir, model, tokenizer = run_training(training_config, train_ds, val_ds)
    export_model(model, tokenizer, training_config)


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to serving YAML config.")
@click.option("--port", type=int, default=None, help="Override port from config.")
def serve(config: Path, port: int | None) -> None:
    """Start the vLLM inference server."""
    from article_tagging.configs.models import ServingConfig
    from article_tagging.inference.server import launch_server

    serving_config = load_config(config, ServingConfig)
    if port is not None:
        serving_config = ServingConfig(**{**serving_config.model_dump(), "port": port})

    process = launch_server(serving_config)
    try:
        process.wait()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down server...[/yellow]")
        process.terminate()
        process.wait()


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to evaluation YAML config.")
@click.option("--compare-with", type=click.Path(exists=True, path_type=Path), multiple=True, help="Previous eval result JSONs to compare.")
def evaluate(config: Path, compare_with: tuple[Path, ...]) -> None:
    """Evaluate model predictions against ground truth."""
    from article_tagging.configs.models import EvalConfig
    from article_tagging.evaluation.evaluator import run_evaluation

    eval_config = load_config(config, EvalConfig)
    if compare_with:
        eval_config = EvalConfig(**{
            **eval_config.model_dump(),
            "compare_with": list(compare_with),
        })

    result = asyncio.run(run_evaluation(eval_config))

    if eval_config.compare_with:
        from article_tagging.evaluation.metrics import load_eval_result
        from article_tagging.evaluation.report import generate_comparison, save_report

        all_results = [("current", result)]
        for i, prev_path in enumerate(eval_config.compare_with):
            name = prev_path.stem.replace("eval_result", f"run_{i}")
            all_results.append((name, load_eval_result(prev_path)))

        report = generate_comparison(all_results)
        report_path = eval_config.output_dir / "comparison.md"
        save_report(report, report_path)
        console.print(f"Comparison report: [cyan]{report_path}[/cyan]")


@cli.command()
@click.option("--title", type=str, required=True, help="Product title text to classify.")
@click.option("--image", type=click.Path(exists=True, path_type=Path), default=None, help="Optional product image path.")
@click.option("--schema", "schema_path", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the dataset schema YAML.")
@click.option("--server-url", type=str, default="http://localhost:8000", show_default=True, help="vLLM server URL.")
def predict(title: str, image: Path | None, schema_path: Path, server_url: str) -> None:
    """Run a single prediction against a running server."""
    from article_tagging.inference.client import predict as predict_fn
    from article_tagging.inference.schema_generator import load_schema

    schema = load_schema(schema_path)
    result = asyncio.run(predict_fn(title, schema, server_url, image_path=image))
    console.print_json(data=result)
