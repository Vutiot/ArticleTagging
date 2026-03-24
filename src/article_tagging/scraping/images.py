"""Image downloader with SHA-256 deduplication for scraped listings.

Downloads images referenced by :class:`~article_tagging.scraping.base.RawListing`
objects, deduplicates by content hash, optionally resizes, and persists a
JSON manifest that maps each listing to its local image paths and hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio

    import httpx
    from PIL import Image

    from article_tagging.scraping.base import RawListing

logger = logging.getLogger(__name__)

# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ImageManifest:
    """Tracks downloaded images for a single listing."""

    listing_id: str
    local_paths: list[Path] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _validate_and_open(data: bytes) -> Image.Image:
    """Validate raw bytes as a real image and return a Pillow ``Image``.

    Handles animated GIFs (takes first frame), converts palette / CMYK / RGBA
    images to RGB, and raises ``ValueError`` for truncated or invalid data.
    """
    from PIL import Image as PILImage

    # Allow Pillow to load truncated images instead of raising
    PILImage.LOAD_TRUNCATED_IMAGES = True

    try:
        img = PILImage.open(io.BytesIO(data))
    except Exception as exc:
        msg = f"Cannot open image data: {exc}"
        raise ValueError(msg) from exc

    # Animated GIF / APNG — seek to first frame (already there, but explicit)
    try:
        img.seek(0)
    except EOFError:
        pass

    # Convert colour modes
    if img.mode in ("CMYK", "YCbCr", "LAB"):
        img = img.convert("RGB")
    elif img.mode in ("RGBA", "LA", "PA"):
        # Composite over white background to drop transparency
        background = PILImage.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode == "P":
        img = img.convert("RGB")
    elif img.mode in ("L", "1"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def _resize_if_needed(img: Image.Image, max_size: int) -> Image.Image:
    """Proportionally resize *img* so its longest side is at most *max_size*."""
    from PIL import Image as PILImage

    w, h = img.size
    longest = max(w, h)
    if longest <= max_size:
        return img

    scale = max_size / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), PILImage.LANCZOS)


async def _download_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    url: str,
    dest: Path,
    max_size: int,
    seen_hashes: set[str],
) -> tuple[Path, str] | None:
    """Download a single image URL, deduplicate, resize, and save as JPEG.

    Returns ``(local_path, sha256_hex)`` on success, or ``None`` if the image
    was skipped (error, duplicate, invalid).
    """
    import httpx as _httpx  # noqa: F811 — lazy import

    async with sem:
        try:
            resp = await client.get(url, timeout=5.0, follow_redirects=True)
            resp.raise_for_status()
        except (_httpx.HTTPStatusError, _httpx.TimeoutException, _httpx.RequestError) as exc:
            logger.warning("Skipping %s: %s", url, exc)
            return None

        data = resp.content

        # Content-type sanity check (best-effort)
        content_type = resp.headers.get("content-type", "")
        if content_type and not content_type.startswith("image/"):
            logger.warning("Skipping %s: non-image content-type %r", url, content_type)
            return None

    # SHA-256 deduplication
    digest = hashlib.sha256(data).hexdigest()
    if digest in seen_hashes:
        logger.debug("Duplicate image skipped (hash %s): %s", digest[:12], url)
        return None
    seen_hashes.add(digest)

    # Validate / convert with Pillow
    try:
        img = _validate_and_open(data)
        img = _resize_if_needed(img, max_size)
    except (ValueError, OSError) as exc:
        logger.warning("Skipping %s: image processing failed: %s", url, exc)
        return None

    # Save as JPEG
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        img.save(dest, format="JPEG", quality=90)
    except OSError as exc:
        logger.warning("Failed to save %s: %s", dest, exc)
        return None

    return dest, digest


# ─── Public API ───────────────────────────────────────────────────────────────


async def download_images(
    listings: list[RawListing],
    output_dir: Path,
    max_size: int = 1024,
    concurrency: int = 8,
) -> list[ImageManifest]:
    """Download all images from *listings* into *output_dir*.

    Each listing's images are saved under ``output_dir/{listing_index}/`` as
    numbered JPEG files.  Duplicate images (by SHA-256 hash) across the entire
    batch are downloaded only once and omitted from subsequent manifests.

    Args:
        listings: Scraped listings whose ``image_urls`` should be fetched.
        output_dir: Root directory for downloaded images.
        max_size: Maximum pixel length of the longest side.  Images exceeding
            this are resized proportionally.
        concurrency: Maximum number of simultaneous HTTP requests.

    Returns:
        A list of :class:`ImageManifest` objects — one per listing — recording
        which images were successfully downloaded.
    """
    import asyncio as _asyncio

    import httpx as _httpx

    sem = _asyncio.Semaphore(concurrency)
    seen_hashes: set[str] = set()
    manifests: list[ImageManifest] = []

    async with _httpx.AsyncClient() as client:
        for listing_idx, listing in enumerate(listings):
            listing_dir = output_dir / str(listing_idx)
            manifest = ImageManifest(listing_id=str(listing_idx))

            tasks: list[_asyncio.Task[tuple[Path, str] | None]] = []
            for img_idx, url in enumerate(listing.image_urls):
                dest = listing_dir / f"{img_idx}.jpg"
                task = _asyncio.create_task(
                    _download_one(client, sem, url, dest, max_size, seen_hashes)
                )
                tasks.append(task)

            results = await _asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, BaseException):
                    logger.warning("Unexpected error downloading image: %s", result)
                    continue
                if result is None:
                    continue
                path, digest = result
                manifest.local_paths.append(path)
                manifest.hashes.append(digest)

            manifests.append(manifest)

    logger.info(
        "Downloaded %d unique images for %d listings",
        sum(len(m.local_paths) for m in manifests),
        len(manifests),
    )
    return manifests


# ─── Manifest persistence ────────────────────────────────────────────────────


def save_manifest(manifests: list[ImageManifest], path: Path) -> None:
    """Serialize a list of :class:`ImageManifest` objects to a JSON file.

    Args:
        manifests: Manifests to persist.
        path: Destination file path (will be created / overwritten).
    """
    serializable = []
    for m in manifests:
        d = asdict(m)
        # Convert Path objects to strings for JSON compatibility
        d["local_paths"] = [str(p) for p in m.local_paths]
        serializable.append(d)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> list[ImageManifest]:
    """Deserialize a list of :class:`ImageManifest` objects from a JSON file.

    Args:
        path: Path to a manifest JSON previously written by :func:`save_manifest`.

    Returns:
        Reconstructed list of :class:`ImageManifest` objects.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    manifests: list[ImageManifest] = []
    for entry in raw:
        manifests.append(
            ImageManifest(
                listing_id=entry["listing_id"],
                local_paths=[Path(p) for p in entry["local_paths"]],
                hashes=entry["hashes"],
            )
        )
    return manifests
