"""Comparison report generator for multiple evaluation runs.

Produces side-by-side Markdown tables comparing exact match and per-attribute
accuracy across different model versions or configurations.
"""

from __future__ import annotations

import logging
from pathlib import Path

from article_tagging.evaluation.metrics import EvalResult, load_eval_result

logger = logging.getLogger(__name__)


def generate_comparison(
    results: list[tuple[str, EvalResult]],
) -> str:
    """Generate a Markdown comparison report from multiple eval results.

    Args:
        results: List of ``(name, EvalResult)`` tuples to compare.

    Returns:
        Markdown-formatted comparison report as a string.
    """
    if not results:
        return "No results to compare.\n"

    lines: list[str] = []
    lines.append("# Evaluation Comparison Report\n")

    # ── Exact match table ─────────────────────────────────────────────
    lines.append("## Exact Match\n")
    header = "| Run | Exact Match | Samples |"
    sep = "|-----|------------|---------|"
    lines.append(header)
    lines.append(sep)
    for name, result in results:
        lines.append(f"| {name} | {result.exact_match:.1%} | {result.total_samples} |")
    lines.append("")

    # ── Per-attribute table ───────────────────────────────────────────
    all_attrs: list[str] = []
    for _, result in results:
        for attr in result.per_attribute:
            if attr not in all_attrs:
                all_attrs.append(attr)

    if all_attrs:
        lines.append("## Per-Attribute Accuracy\n")
        attr_header = "| Attribute | " + " | ".join(name for name, _ in results) + " |"
        attr_sep = "|-----------|" + "|".join("----------" for _ in results) + "|"
        lines.append(attr_header)
        lines.append(attr_sep)
        for attr in all_attrs:
            values = []
            for _, result in results:
                val = result.per_attribute.get(attr, 0.0)
                values.append(f"{val:.1%}")
            lines.append(f"| {attr} | " + " | ".join(values) + " |")
        lines.append("")

    # ── Category breakdown ────────────────────────────────────────────
    has_categories = any(r.category_breakdown for _, r in results)
    if has_categories:
        all_cats: list[str] = []
        for _, result in results:
            if result.category_breakdown:
                for cat in result.category_breakdown:
                    if cat not in all_cats:
                        all_cats.append(cat)

        lines.append("## Per-Category Exact Match\n")
        cat_header = "| Category | " + " | ".join(name for name, _ in results) + " |"
        cat_sep = "|----------|" + "|".join("----------" for _ in results) + "|"
        lines.append(cat_header)
        lines.append(cat_sep)
        for cat in sorted(all_cats):
            values = []
            for _, result in results:
                val = (result.category_breakdown or {}).get(cat, 0.0)
                values.append(f"{val:.1%}")
            lines.append(f"| {cat} | " + " | ".join(values) + " |")
        lines.append("")

    return "\n".join(lines)


def load_and_compare(result_paths: list[tuple[str, Path]]) -> str:
    """Load eval result JSONs and generate a comparison report.

    Args:
        result_paths: List of ``(name, path_to_eval_result.json)`` tuples.

    Returns:
        Markdown comparison report.
    """
    results = [(name, load_eval_result(path)) for name, path in result_paths]
    return generate_comparison(results)


def save_report(report: str, path: Path) -> None:
    """Save a Markdown report to a file.

    Args:
        report: Markdown content.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", path)
