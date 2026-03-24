"""Evaluation metrics for article tagging predictions.

Provides exact match, per-attribute accuracy, and category breakdown metrics
with JSON serialization for report storage and comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class EvalResult:
    """Container for evaluation results."""

    exact_match: float  # 0.0 to 1.0
    per_attribute: dict[str, float]  # attribute_name -> accuracy
    category_breakdown: dict[str, float] | None  # category -> exact_match
    total_samples: int
    timestamp: str  # ISO format


def _normalize(value: object) -> str:
    """Normalize a value for case-insensitive, whitespace-stripped comparison."""
    return str(value).strip().lower()


def exact_match(predictions: list[dict], ground_truth: list[dict]) -> float:
    """Compute exact match accuracy across all attributes.

    A prediction counts as correct only when **every** key-value pair in the
    corresponding ground truth entry matches (case-insensitive, stripped).

    Args:
        predictions: List of predicted attribute dicts.
        ground_truth: List of ground truth attribute dicts (same length).

    Returns:
        Fraction of samples where all attributes match exactly.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(predictions) != len(ground_truth):
        msg = (
            f"predictions ({len(predictions)}) and ground_truth "
            f"({len(ground_truth)}) must have the same length"
        )
        raise ValueError(msg)

    if not predictions:
        return 0.0

    matches = 0
    for pred, truth in zip(predictions, ground_truth):
        all_keys = set(truth.keys())
        if set(pred.keys()) != all_keys:
            continue
        if all(_normalize(pred[k]) == _normalize(truth[k]) for k in all_keys):
            matches += 1

    return matches / len(predictions)


def per_attribute_accuracy(
    predictions: list[dict],
    ground_truth: list[dict],
    attributes: list[str],
) -> dict[str, float]:
    """Compute accuracy independently for each attribute.

    Missing keys in a prediction are counted as incorrect.

    Args:
        predictions: List of predicted attribute dicts.
        ground_truth: List of ground truth attribute dicts (same length).
        attributes: Attribute names to evaluate.

    Returns:
        Dict mapping each attribute name to its accuracy (0.0 to 1.0).

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(predictions) != len(ground_truth):
        msg = (
            f"predictions ({len(predictions)}) and ground_truth "
            f"({len(ground_truth)}) must have the same length"
        )
        raise ValueError(msg)

    if not predictions:
        return {attr: 0.0 for attr in attributes}

    counts: dict[str, int] = {attr: 0 for attr in attributes}

    for pred, truth in zip(predictions, ground_truth):
        for attr in attributes:
            truth_val = truth.get(attr)
            pred_val = pred.get(attr)
            if truth_val is not None and pred_val is not None:
                if _normalize(pred_val) == _normalize(truth_val):
                    counts[attr] += 1

    return {attr: counts[attr] / len(predictions) for attr in attributes}


def compute_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    attributes: list[str],
    category_field: str | None = None,
) -> EvalResult:
    """Compute all evaluation metrics.

    Combines exact match, per-attribute accuracy, and an optional per-category
    breakdown into a single :class:`EvalResult`.

    Args:
        predictions: List of predicted attribute dicts.
        ground_truth: List of ground truth attribute dicts (same length).
        attributes: Attribute names to evaluate.
        category_field: If provided, compute exact match per category value.

    Returns:
        An :class:`EvalResult` with all computed metrics.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    em = exact_match(predictions, ground_truth)
    pa = per_attribute_accuracy(predictions, ground_truth, attributes)

    category_breakdown: dict[str, float] | None = None
    if category_field is not None:
        # Group indices by category value from ground truth
        groups: dict[str, tuple[list[dict], list[dict]]] = {}
        for pred, truth in zip(predictions, ground_truth):
            cat = str(truth.get(category_field, "unknown"))
            if cat not in groups:
                groups[cat] = ([], [])
            groups[cat][0].append(pred)
            groups[cat][1].append(truth)

        category_breakdown = {
            cat: exact_match(preds, truths)
            for cat, (preds, truths) in sorted(groups.items())
        }

    return EvalResult(
        exact_match=em,
        per_attribute=pa,
        category_breakdown=category_breakdown,
        total_samples=len(predictions),
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


def save_eval_result(result: EvalResult, path: Path) -> None:
    """Serialize an EvalResult to a JSON file.

    Args:
        result: The evaluation result to save.
        path: Destination file path (parent directories are created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


def load_eval_result(path: Path) -> EvalResult:
    """Deserialize an EvalResult from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialized :class:`EvalResult`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalResult(**data)
