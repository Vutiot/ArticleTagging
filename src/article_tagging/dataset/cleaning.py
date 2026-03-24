"""Data cleaning pipeline for raw scraped listings.

Validates attributes against a dataset schema, normalises text, deduplicates,
and filters rows with missing required fields or images.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from article_tagging.inference.schema_generator import DatasetSchema

logger = logging.getLogger(__name__)

# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass
class CleaningStats:
    """Counts produced by :func:`clean_listings`."""

    total: int
    kept: int
    dropped_empty_title: int = 0
    dropped_invalid_attrs: int = 0
    dropped_duplicates: int = 0
    dropped_missing_images: int = 0


# ─── Helpers ──────────────────────────────────────────────────────────────────

_MULTI_SPACE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Strip, collapse whitespace, and decode HTML entities."""
    return _MULTI_SPACE.sub(" ", html.unescape(text).strip())


def _content_hash(record: dict) -> str:
    """Deterministic hash of title + attributes for deduplication."""
    title = record.get("title", "")
    attrs = record.get("attributes", {})
    key = title + "|" + "|".join(f"{k}={v}" for k, v in sorted(attrs.items()))
    return hashlib.sha256(key.encode()).hexdigest()


def _validate_against_schema(record: dict, schema: DatasetSchema) -> bool:
    """Return True if all required attributes are present and valid."""
    attrs = record.get("attributes", {})

    for attr_def in schema.attributes:
        value = attrs.get(attr_def.name)

        # Required attribute missing?
        if attr_def.required and not value:
            return False

        # Enum value not in allowed list?
        if value and attr_def.type == "enum" and attr_def.values:
            normalised = value.strip().lower()
            allowed = {v.strip().lower() for v in attr_def.values}
            if normalised not in allowed:
                return False

    return True


# ─── Public API ───────────────────────────────────────────────────────────────


def load_raw_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed dicts, one per non-empty line.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def clean_listings(
    listings: list[dict],
    schema: DatasetSchema,
    *,
    deduplicate: bool = True,
    require_images: bool = False,
) -> tuple[list[dict], CleaningStats]:
    """Clean and validate raw listings against a dataset schema.

    Steps applied in order:

    1. Normalise text (strip, collapse whitespace, decode HTML entities)
    2. Drop rows with empty titles
    3. Validate attributes against schema (required fields, enum values)
    4. Deduplicate by title + attributes hash
    5. Optionally filter rows with no images

    Args:
        listings: Raw listing dicts (as loaded from JSONL).
        schema: Dataset schema for validation.
        deduplicate: Remove duplicate rows (default True).
        require_images: Drop rows with empty ``image_urls`` (default False).

    Returns:
        Tuple of ``(cleaned_listings, stats)``.
    """
    stats = CleaningStats(total=len(listings), kept=0)
    cleaned: list[dict] = []
    seen_hashes: set[str] = set()

    for record in listings:
        # ── Normalise ─────────────────────────────────────────────────
        record["title"] = _normalize_text(record.get("title", ""))
        if "attributes" in record:
            record["attributes"] = {
                k: _normalize_text(str(v)) for k, v in record["attributes"].items()
            }

        # ── Empty title ───────────────────────────────────────────────
        if not record["title"]:
            stats.dropped_empty_title += 1
            continue

        # ── Schema validation ─────────────────────────────────────────
        if not _validate_against_schema(record, schema):
            stats.dropped_invalid_attrs += 1
            continue

        # ── Deduplication ─────────────────────────────────────────────
        if deduplicate:
            h = _content_hash(record)
            if h in seen_hashes:
                stats.dropped_duplicates += 1
                continue
            seen_hashes.add(h)

        # ── Image filter ──────────────────────────────────────────────
        if require_images:
            images = record.get("image_urls", [])
            if not images:
                stats.dropped_missing_images += 1
                continue

        cleaned.append(record)

    stats.kept = len(cleaned)

    logger.info(
        "Cleaning: %d total → %d kept (-%d empty title, -%d invalid attrs, "
        "-%d duplicates, -%d missing images)",
        stats.total,
        stats.kept,
        stats.dropped_empty_title,
        stats.dropped_invalid_attrs,
        stats.dropped_duplicates,
        stats.dropped_missing_images,
    )

    return cleaned, stats
