"""Dataset loading for SFTTrainer with Unsloth vision support.

Loads JSONL chat-format files and converts image paths to PIL Image objects
for the ``UnslothVisionDataCollator``.  Uses plain Python lists (not Arrow-
backed HuggingFace Datasets) because PIL objects can't be serialised to Arrow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
) -> tuple[list[dict], list[dict] | None]:
    """Load prepared JSONL files for SFTTrainer.

    When ``text_only`` is ``False``, image paths in the chat messages are
    replaced with PIL Image objects (required by ``UnslothVisionDataCollator``).

    Returns plain Python lists — ``UnslothVisionDataCollator`` accepts these
    directly and PIL images cannot be serialised into Arrow tables.

    Args:
        train_path: Path to the training JSONL file.
        val_path: Optional path to the validation JSONL file.
        text_only: If ``True``, skip image loading.

    Returns:
        A tuple of ``(train_data, val_data)`` where ``val_data``
        may be ``None``.
    """
    train_data = _load_jsonl(train_path, resolve_images=not text_only)
    logger.info("Loaded training data: %d samples from %s", len(train_data), train_path)

    val_data = None
    if val_path is not None and val_path.exists():
        val_data = _load_jsonl(val_path, resolve_images=not text_only)
        logger.info("Loaded validation data: %d samples from %s", len(val_data), val_path)

    return train_data, val_data


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
    skipped = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                record = json.loads(stripped)
                if resolve_images:
                    ok = _resolve_images_in_messages(record)
                    if not ok:
                        skipped += 1
                        continue
                records.append(record)
    if skipped:
        logger.warning("Skipped %d records with missing/broken images", skipped)
    return records


def _resolve_images_in_messages(record: dict) -> bool:
    """Walk messages and replace image path strings with PIL Image objects.

    Modifies the record in place.

    Returns:
        ``True`` if the record is valid (all images resolved or no images needed),
        ``False`` if it should be skipped.
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
                        return False
                else:
                    logger.warning("Image not found: %s", image_ref)
                    return False
            cleaned_content.append(block)

        msg["content"] = cleaned_content
    return True
