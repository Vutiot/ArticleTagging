"""CSV and JSON dataset importers for pre-existing labeled data.

Allows users who already have labeled data (Kaggle datasets, database exports,
manual labeling) to skip scraping and import directly into the same
:class:`~article_tagging.scraping.base.RawListing` format produced by the
scraping pipeline.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from article_tagging.scraping.base import RawListing

# ─── Mapping configuration ───────────────────────────────────────────────────


@dataclass
class ImportMapping:
    """Defines how CSV/JSON columns map to :class:`RawListing` fields.

    Attributes:
        title_field: Column name for the listing title.
        image_field: Column name for image URL or path (optional).
        attribute_fields: Maps target attribute name to source column name.
            If ``None``, all columns not mapped to title/image/url are treated
            as attributes.
        url_field: Column name for the source URL (optional).
    """

    title_field: str
    image_field: str | None = None
    attribute_fields: dict[str, str] | None = None
    url_field: str | None = None


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _mapped_fields(mapping: ImportMapping) -> set[str]:
    """Return the set of source column names that are explicitly mapped."""
    fields = {mapping.title_field}
    if mapping.image_field is not None:
        fields.add(mapping.image_field)
    if mapping.url_field is not None:
        fields.add(mapping.url_field)
    if mapping.attribute_fields is not None:
        fields.update(mapping.attribute_fields.values())
    return fields


def _record_to_listing(record: dict, mapping: ImportMapping) -> RawListing:
    """Convert a single record (dict) into a :class:`RawListing`.

    Handles both flat records (CSV-style, all string values) and the native
    nested format produced by :func:`export_jsonl` (with ``image_urls`` list
    and ``attributes`` dict).
    """
    # Native format: already has the right structure from export_jsonl
    if "attributes" in record and isinstance(record["attributes"], dict):
        raw_images = record.get("image_urls", [])
        if isinstance(raw_images, str):
            raw_images = [raw_images] if raw_images else []
        return RawListing(
            url=record.get("url", ""),
            title=record.get("title", ""),
            image_urls=raw_images,
            attributes={k: str(v) for k, v in record["attributes"].items() if v},
        )

    # Flat format: map columns via ImportMapping
    title = str(record.get(mapping.title_field, ""))

    url = ""
    if mapping.url_field is not None:
        url = str(record.get(mapping.url_field, ""))

    image_urls: list[str] = []
    if mapping.image_field is not None:
        raw_image = str(record.get(mapping.image_field, ""))
        if raw_image:
            image_urls = [raw_image]

    if mapping.attribute_fields is not None:
        attributes = {
            target: str(record[source])
            for target, source in mapping.attribute_fields.items()
            if source in record and record[source]
        }
    else:
        # All columns not explicitly mapped become attributes.
        reserved = _mapped_fields(mapping)
        attributes = {
            key: str(value)
            for key, value in record.items()
            if key not in reserved and value
        }

    return RawListing(
        url=url,
        title=title,
        image_urls=image_urls,
        attributes=attributes,
    )


# ─── Public API ───────────────────────────────────────────────────────────────


def import_csv(path: Path, mapping: ImportMapping) -> list[RawListing]:
    """Import a CSV file into a list of :class:`RawListing`.

    Args:
        path: Path to the CSV file.
        mapping: Describes how columns map to ``RawListing`` fields.

    Returns:
        List of parsed :class:`RawListing` objects.
    """
    listings: list[RawListing] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            listings.append(_record_to_listing(row, mapping))
    return listings


def import_json(path: Path, mapping: ImportMapping) -> list[RawListing]:
    """Import a JSON or JSONL file into a list of :class:`RawListing`.

    Auto-detects the format: tries ``json.load()`` (JSON array of objects)
    first; if that fails, falls back to line-by-line ``json.loads()`` (JSONL).

    Args:
        path: Path to the JSON or JSONL file.
        mapping: Describes how fields map to ``RawListing`` fields.

    Returns:
        List of parsed :class:`RawListing` objects.
    """
    text = path.read_text(encoding="utf-8")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        # Parsed as a JSON array of objects.
        pass
    else:
        # Fall back to JSONL (one JSON object per line).
        data = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                data.append(json.loads(stripped))

    return [_record_to_listing(record, mapping) for record in data]


def export_jsonl(listings: list[RawListing], path: Path) -> None:
    """Write listings as JSONL (one JSON object per line).

    Creates parent directories if they do not exist.

    Args:
        listings: List of :class:`RawListing` to export.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for listing in listings:
            fh.write(json.dumps(asdict(listing), ensure_ascii=False))
            fh.write("\n")
