"""Training loop for VLM fine-tuning with Unsloth + SFTTrainer.

Orchestrates model loading, dataset preparation, and LoRA training with
early stopping.  All heavy imports are lazy so the module can be imported
without a GPU present.
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

_TRL_INSTALL_MSG = (
    "trl is required for training but is not installed.\n"
    "Install the training extras with:  pip install article-tagging[training]"
)


def run_training(
    config: TrainingConfig,
    train_dataset: Any,
    val_dataset: Any | None = None,
    model: Any = None,
    tokenizer: Any = None,
) -> Path:
    """Run LoRA fine-tuning with SFTTrainer.

    If *model* and *tokenizer* are not provided, they are loaded via
    :func:`~article_tagging.training.model.load_model`.

    Args:
        config: Training hyper-parameters and output settings.
        train_dataset: HuggingFace Dataset with ``messages`` column.
        val_dataset: Optional validation Dataset.
        model: Pre-loaded model (skips loading if provided).
        tokenizer: Pre-loaded tokenizer/processor.

    Returns:
        Tuple of ``(output_dir, model, tokenizer)`` so the caller can
        run export (merge) after training.

    Raises:
        ImportError: If trl or transformers are not installed.
        RuntimeError: On CUDA out-of-memory (with a helpful suggestion).
    """
    # ── Load model if not provided ────────────────────────────────────
    if model is None or tokenizer is None:
        from article_tagging.training.model import load_model

        console.print("[bold]Loading model[/bold] ...")
        model, tokenizer = load_model(config)

    # ── Lazy imports ──────────────────────────────────────────────────
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        raise ImportError(_TRL_INSTALL_MSG) from None

    from transformers import EarlyStoppingCallback, TrainerCallback

    from article_tagging.training.data import get_vision_data_collator

    # ── Data collator ─────────────────────────────────────────────────
    data_collator = get_vision_data_collator(model, tokenizer)

    # ── Progress logging callback ─────────────────────────────────────
    class ProgressCallback(TrainerCallback):
        """Logs training progress to console and a log file."""

        def __init__(self, log_path: Path) -> None:
            self.log_path = log_path
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        def _write(self, msg: str) -> None:
            console.print(msg)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
            if logs:
                step = state.global_step
                total = state.max_steps
                pct = step / total * 100 if total > 0 else 0
                loss = logs.get("loss", logs.get("eval_loss", ""))
                lr = logs.get("learning_rate", "")
                parts = [f"Step {step}/{total} ({pct:.0f}%)"]
                if loss != "":
                    parts.append(f"loss={loss:.4f}")
                if lr != "":
                    parts.append(f"lr={lr:.2e}")
                if "eval_loss" in logs:
                    parts.append(f"eval_loss={logs['eval_loss']:.4f}")
                self._write("  " + " | ".join(parts))

        def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
            self._write(f"[bold]Training started[/bold] — {state.max_steps} steps")

        def on_epoch_begin(self, args, state, control, **kwargs):  # noqa: ARG002
            self._write(f"\n[bold]Epoch {int(state.epoch) + 1}/{args.num_train_epochs}[/bold]")

    # ── Callbacks ─────────────────────────────────────────────────────
    output_dir = str(config.output_dir)
    progress_log = Path(output_dir) / "training.log"
    callbacks = [ProgressCallback(progress_log)]
    if val_dataset is not None and config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        )

    # ── W&B ───────────────────────────────────────────────────────────
    report_to = "none"
    if config.use_wandb:
        try:
            import wandb  # noqa: F401

            report_to = "wandb"
        except ImportError:
            logger.warning("wandb not installed — disabling W&B logging")

    # ── Training arguments ────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=config.eval_steps if val_dataset is not None else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        load_best_model_at_end=val_dataset is not None,
        logging_steps=10,
        logging_dir=output_dir + "/logs",
        report_to=report_to,
        run_name=config.run_name,
        remove_unused_columns=False,
        fp16=False,
        bf16=True,
        disable_tqdm=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────
    console.print("[bold]Starting training[/bold] ...")
    console.print(
        f"  epochs={config.epochs}, batch_size={config.batch_size}, "
        f"grad_accum={config.gradient_accumulation_steps}, lr={config.learning_rate}"
    )

    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            console.print(
                "\n[bold red]CUDA out of memory![/bold red]\n"
                "Try reducing batch_size to 1, enabling gradient_checkpointing='unsloth', "
                "or reducing max_seq_length.\n"
            )
        raise

    console.print("[bold green]Training complete![/bold green]")

    return Path(output_dir), model, tokenizer
