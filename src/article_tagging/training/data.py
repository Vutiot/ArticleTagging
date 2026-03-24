"""Dataset loading for SFTTrainer with Unsloth vision support.

Bridges the JSONL chat format produced by the ``prepare`` command and Unsloth's
``UnslothVisionDataCollator``, resolving image file paths to PIL Image objects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATASETS_INSTALL_MSG = (
    "The 'datasets' library is required for training data loading.\n"
    "Install the training extras with:  pip install article-tagging[training]"
)

_UNSLOTH_INSTALL_MSG = (
    "Unsloth is required for the vision data collator.\n"
    "Install the training extras with:  pip install article-tagging[training]"
)


# ─── Public API ───────────────────────────────────────────────────────────────


def load_training_dataset(
    train_path: Path,
    val_path: Path | None = None,
    *,
    text_only: bool = False,
) -> tuple[Any, Any | None]:
    """Load prepared JSONL files into HuggingFace Datasets for SFTTrainer.

    When ``text_only`` is ``False``, image paths in the chat messages are
    replaced with PIL Image objects (required by ``UnslothVisionDataCollator``).

    Args:
        train_path: Path to the training JSONL file.
        val_path: Optional path to the validation JSONL file.
        text_only: If ``True``, skip image loading.

    Returns:
        A tuple of ``(train_dataset, val_dataset)`` where ``val_dataset``
        may be ``None``.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(_DATASETS_INSTALL_MSG) from None

    train_records = _load_jsonl(train_path, resolve_images=not text_only)
    train_dataset = Dataset.from_list(train_records)
    logger.info("Loaded training dataset: %d samples from %s", len(train_dataset), train_path)

    val_dataset = None
    if val_path is not None and val_path.exists():
        val_records = _load_jsonl(val_path, resolve_images=not text_only)
        val_dataset = Dataset.from_list(val_records)
        logger.info("Loaded validation dataset: %d samples from %s", len(val_dataset), val_path)

    return train_dataset, val_dataset


def get_vision_data_collator(model: Any, tokenizer: Any) -> Any:
    """Create an ``UnslothVisionDataCollator`` for SFTTrainer.

    Args:
        model: The loaded VLM model.
        tokenizer: The corresponding tokenizer/processor.

    Returns:
        An ``UnslothVisionDataCollator`` instance.

    Raises:
        ImportError: If Unsloth is not installed.
    """
    try:
        from unsloth.trainer import UnslothVisionDataCollator
    except ImportError:
        raise ImportError(_UNSLOTH_INSTALL_MSG) from None

    return UnslothVisionDataCollator(model, tokenizer)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _load_jsonl(path: Path, *, resolve_images: bool = False) -> list[dict]:
    """Load a JSONL file and optionally resolve image paths to PIL objects."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                record = json.loads(stripped)
                if resolve_images:
                    _resolve_images_in_messages(record)
                records.append(record)
    return records


def _resolve_images_in_messages(record: dict) -> None:
    """Walk messages and replace image path strings with PIL Image objects.

    Modifies the record in place.  If an image path doesn't exist or can't be
    opened, it is removed from the content list with a warning.
    """
    from article_tagging.dataset.image_processing import preprocess_image

    messages = record.get("messages", [])
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        cleaned_content: list[dict] = []
        for block in content:
            if block.get("type") == "image":
                image_ref = block.get("image", "")
                image_path = Path(image_ref)
                if image_path.exists():
                    try:
                        pil_image = preprocess_image(image_path)
                        cleaned_content.append({"type": "image", "image": pil_image})
                        continue
                    except Exception:  # noqa: BLE001
                        logger.warning("Failed to load image: %s", image_path)
                else:
                    logger.warning("Image not found: %s", image_ref)
                # Skip broken images
                continue
            cleaned_content.append(block)

        msg["content"] = cleaned_content
