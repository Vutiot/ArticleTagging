"""CLI entry point for the ArticleTagging pipeline."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
def cli() -> None:
    """ArticleTagging — fine-tune VLMs for structured attribute extraction."""


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to pipeline YAML config.")
def scrape(config: Path) -> None:
    """Scrape listings from a configured site."""
    console.print(f"[yellow]scrape[/yellow] is not yet implemented. Config: {config}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to pipeline YAML config.")
def prepare(config: Path) -> None:
    """Prepare a fine-tuning dataset from scraped data."""
    console.print(f"[yellow]prepare[/yellow] is not yet implemented. Config: {config}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to pipeline YAML config.")
def train(config: Path) -> None:
    """Fine-tune a VLM on the prepared dataset."""
    console.print(f"[yellow]train[/yellow] is not yet implemented. Config: {config}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to pipeline YAML config.")
@click.option("--port", type=int, default=8000, show_default=True, help="Port for the inference server.")
def serve(config: Path, port: int) -> None:
    """Start the vLLM inference server."""
    console.print(f"[yellow]serve[/yellow] is not yet implemented. Config: {config}, port: {port}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to pipeline YAML config.")
def evaluate(config: Path) -> None:
    """Evaluate model predictions against ground truth."""
    console.print(f"[yellow]evaluate[/yellow] is not yet implemented. Config: {config}")


@cli.command()
@click.option("--title", type=str, required=True, help="Product title text to classify.")
@click.option("--image", type=click.Path(exists=True, path_type=Path), default=None, help="Optional product image path.")
@click.option("--schema", "schema_path", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the dataset schema YAML.")
def predict(title: str, image: Path | None, schema_path: Path) -> None:
    """Run a single prediction against a running server."""
    console.print(f"[yellow]predict[/yellow] is not yet implemented. Title: {title!r}, image: {image}, schema: {schema_path}")
