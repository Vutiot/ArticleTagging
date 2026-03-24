"""Scraping orchestrator — coordinates scraper, image downloader, and JSONL export."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from article_tagging.scraping.base import create_scraper
from article_tagging.scraping.importers import export_jsonl

if TYPE_CHECKING:
    from article_tagging.configs.models import SiteConfig

logger = logging.getLogger(__name__)
console = Console()


async def run_scrape(
    config: SiteConfig,
    output_dir: Path,
    *,
    download_images_flag: bool = True,
    max_image_size: int = 1024,
) -> Path:
    """Run the full scraping pipeline for a site.

    1. Create scraper from config
    2. Scrape all listings (paginate + detail pages)
    3. Export listings as JSONL
    4. Optionally download and deduplicate images

    Args:
        config: Site configuration.
        output_dir: Root output directory (files go to ``output_dir/{site_name}/``).
        download_images_flag: Whether to download images (default True).
        max_image_size: Max image dimension in pixels for resizing.

    Returns:
        Path to the generated JSONL file.
    """
    site_dir = output_dir / config.name
    jsonl_path = site_dir / "listings.jsonl"

    # ── Step 1: Scrape listings ───────────────────────────────────────────
    console.print(f"[bold]Scraping[/bold] {config.base_url} ...")
    scraper = create_scraper(config)
    try:
        listings = await scraper.scrape_listings()
    finally:
        await scraper.close()

    if not listings:
        console.print("[yellow]No listings found.[/yellow]")
        return jsonl_path

    console.print(f"  Scraped [green]{len(listings)}[/green] listings")

    # ── Step 2: Export JSONL ──────────────────────────────────────────────
    export_jsonl(listings, jsonl_path)
    console.print(f"  Saved to [cyan]{jsonl_path}[/cyan]")

    # ── Step 3: Download images ───────────────────────────────────────────
    has_images = any(listing.image_urls for listing in listings)

    if download_images_flag and has_images:
        from article_tagging.scraping.images import download_images, save_manifest

        images_dir = site_dir / "images"
        console.print("  Downloading images ...")
        manifests = await download_images(
            listings,
            images_dir,
            max_size=max_image_size,
        )
        manifest_path = site_dir / "image_manifest.json"
        save_manifest(manifests, manifest_path)

        total_images = sum(len(m.local_paths) for m in manifests)
        console.print(f"  Downloaded [green]{total_images}[/green] images to [cyan]{images_dir}[/cyan]")

    # ── Summary ───────────────────────────────────────────────────────────
    console.print(f"\n[bold green]Done![/bold green] {len(listings)} listings in {jsonl_path}")
    return jsonl_path
