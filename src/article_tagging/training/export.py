"""Model export and LoRA merging utilities.

Saves LoRA adapter weights separately and optionally merges them back into
the base model for standalone vLLM deployment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from article_tagging.configs.models import TrainingConfig

logger = logging.getLogger(__name__)
console = Console()


def export_adapter(model: Any, tokenizer: Any, output_dir: Path) -> Path:
    """Save LoRA adapter weights and tokenizer.

    Args:
        model: The LoRA-adapted model.
        tokenizer: The tokenizer/processor.
        output_dir: Base output directory.

    Returns:
        Path to the saved adapter directory.
    """
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    console.print(f"  Adapter saved to [cyan]{adapter_dir}[/cyan]")
    return adapter_dir


def export_merged(model: Any, tokenizer: Any, output_dir: Path) -> Path:
    """Merge LoRA weights into the base model and save for standalone serving.

    Uses Unsloth's ``save_pretrained_merged`` for efficient merging.

    Args:
        model: The LoRA-adapted model.
        tokenizer: The tokenizer/processor.
        output_dir: Base output directory.

    Returns:
        Path to the merged model directory.
    """
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merged_dir), tokenizer)
    console.print(f"  Merged model saved to [cyan]{merged_dir}[/cyan]")
    return merged_dir


def export_model(model: Any, tokenizer: Any, config: TrainingConfig) -> Path:
    """Export the trained model based on configuration.

    Always saves the LoRA adapter. If ``config.merge_on_export`` is ``True``,
    also merges the adapter into the base model for standalone deployment.

    Args:
        model: The LoRA-adapted model after training.
        tokenizer: The tokenizer/processor.
        config: Training configuration with output settings.

    Returns:
        Path to the output directory.
    """
    output_dir = config.output_dir
    console.print("[bold]Exporting model[/bold] ...")

    export_adapter(model, tokenizer, output_dir)

    if config.merge_on_export:
        export_merged(model, tokenizer, output_dir)

    console.print(f"[bold green]Export complete![/bold green] Output: {output_dir}")
    return output_dir
