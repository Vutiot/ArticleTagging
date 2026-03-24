"""Train/validation/test splitting with optional stratification by category."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def split_dataset(
    listings: list[dict],
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    category_field: str | None = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split listings into train/val/test sets.

    When *category_field* is given, performs stratified splitting so that each
    split has proportional representation of every category value.  Falls back
    to random splitting when any category has fewer than 2 samples.

    Args:
        listings: Cleaned listing dicts.
        split_ratio: ``(train, val, test)`` fractions summing to 1.0.
        category_field: Attribute name used for stratification (optional).
        seed: Random seed for reproducibility.

    Returns:
        ``(train, val, test)`` lists.
    """
    rng = random.Random(seed)

    if not listings:
        return [], [], []

    train_r, val_r, _ = split_ratio

    if category_field is not None:
        return _stratified_split(listings, train_r, val_r, category_field, rng)

    return _random_split(listings, train_r, val_r, rng)


def save_splits(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    output_dir: Path,
) -> dict[str, int]:
    """Save train/val/test splits as JSONL files.

    Args:
        train: Training set.
        val: Validation set.
        test: Test set.
        output_dir: Directory to write ``{train,val,test}.jsonl`` into.

    Returns:
        Stats dict with counts per split and total.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in data:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "total": len(train) + len(val) + len(test),
    }

    logger.info(
        "Splits saved to %s — train: %d, val: %d, test: %d (total: %d)",
        output_dir,
        stats["train"],
        stats["val"],
        stats["test"],
        stats["total"],
    )

    return stats


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _random_split(
    items: list[dict],
    train_r: float,
    val_r: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle and split by ratio."""
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_r)
    val_end = train_end + int(n * val_r)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def _stratified_split(
    items: list[dict],
    train_r: float,
    val_r: float,
    category_field: str,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split proportionally within each category group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        cat = item.get("attributes", {}).get(category_field, "_unknown_")
        groups[cat].append(item)

    # Check feasibility — need at least 2 per category for a meaningful split
    small_cats = [cat for cat, members in groups.items() if len(members) < 2]
    if small_cats:
        logger.warning(
            "Categories with <2 samples (%d categories) — falling back to random split",
            len(small_cats),
        )
        return _random_split(items, train_r, val_r, rng)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for members in groups.values():
        rng.shuffle(members)
        t, v, te = _random_split(members, train_r, val_r, rng)
        train.extend(t)
        val.extend(v)
        test.extend(te)

    # Shuffle again so categories aren't grouped together
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test
