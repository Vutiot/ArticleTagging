"""Chat-format conversation builder for VLM fine-tuning.

Converts cleaned listing dicts into the chat-style messages format required by
Unsloth / SFTTrainer (system / user / assistant roles).  The system prompt is
deterministic and static per schema to maximise vLLM V1 prefix-cache hits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from article_tagging.inference.schema_generator import DatasetSchema

logger = logging.getLogger(__name__)


# ─── System prompt builder ────────────────────────────────────────────────────


def build_system_prompt(schema: DatasetSchema, base_prompt: str) -> str:
    """Build a deterministic system prompt from a schema.

    The prompt includes the list of attributes to extract so that it is
    identical for every record in the same schema — enabling vLLM V1 prefix
    caching to reuse the KV cache across requests.

    Args:
        schema: Dataset schema with attribute definitions.
        base_prompt: The base instruction text (e.g. from ``DatasetConfig.system_prompt``).

    Returns:
        A static system prompt string.
    """
    attr_names = ", ".join(attr.name for attr in schema.attributes)
    return f"{base_prompt}\nAttributes to extract: {attr_names}"


# ─── Record formatter ────────────────────────────────────────────────────────


def format_record(
    record: dict,
    schema: DatasetSchema,
    system_prompt: str,
    *,
    text_only: bool = False,
    image_dir: Path | None = None,
) -> dict:
    """Format a single listing record into a chat conversation.

    Args:
        record: A cleaned listing dict with ``title``, ``image_urls``, and
            ``attributes`` keys.
        schema: Dataset schema (used to select which attributes go in the
            assistant response).
        system_prompt: Pre-built system prompt (from :func:`build_system_prompt`).
        text_only: If ``True``, omit image content from the user message.
        image_dir: Optional local directory to resolve image paths against.
            When set, the first image filename is looked up under this directory.

    Returns:
        A dict with a ``"messages"`` key containing the conversation list.
    """
    title = record.get("title", "")

    # ── User message content ──────────────────────────────────────────
    user_text = f'Title: "{title}"'

    image_path = _resolve_image(record, image_dir)

    if not text_only and image_path:
        user_content: str | list[dict] = [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": user_text},
        ]
    else:
        user_content = user_text

    # ── Assistant message (ground truth) ──────────────────────────────
    attrs = record.get("attributes", {})
    schema_attr_names = {attr.name for attr in schema.attributes}
    ground_truth = {k: v for k, v in attrs.items() if k in schema_attr_names}
    assistant_content = json.dumps(ground_truth, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ─── Dataset formatter ────────────────────────────────────────────────────────


def format_dataset(
    listings: list[dict],
    schema: DatasetSchema,
    base_prompt: str,
    *,
    text_only: bool = False,
    image_dir: Path | None = None,
) -> list[dict]:
    """Format all listings into chat conversations for SFTTrainer.

    Builds the system prompt once (ensuring it's identical across all records
    for prefix caching), then formats each record.

    Args:
        listings: Cleaned listing dicts.
        schema: Dataset schema defining target attributes.
        base_prompt: Base instruction text for the system prompt.
        text_only: Omit images from user messages.
        image_dir: Directory containing downloaded images.

    Returns:
        List of conversation dicts, each with a ``"messages"`` key.
    """
    system_prompt = build_system_prompt(schema, base_prompt)

    formatted = []
    for record in listings:
        formatted.append(
            format_record(
                record,
                schema,
                system_prompt,
                text_only=text_only,
                image_dir=image_dir,
            )
        )

    logger.info("Formatted %d records into chat conversations", len(formatted))
    return formatted


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_image(record: dict, image_dir: Path | None) -> str | None:
    """Get the first usable image path/URL from a record."""
    image_urls = record.get("image_urls", [])
    if not image_urls:
        return None

    first = image_urls[0]

    # If image_dir is provided, try to find a local file
    if image_dir is not None:
        # Images may have been downloaded to image_dir/{index}/0.jpg
        # or the URL itself may be a local path
        local = Path(first)
        if local.exists():
            return str(local)
        # Try under image_dir
        candidate = image_dir / Path(first).name
        if candidate.exists():
            return str(candidate)

    return first
