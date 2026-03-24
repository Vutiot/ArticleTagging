"""Image preprocessing utilities for VLM training.

Handles resizing, colour-space conversion, animated GIFs, and base64 encoding
for images used in fine-tuning chat conversations.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

_PILLOW_INSTALL_MSG = (
    "Pillow is required for image preprocessing but is not installed.\n"
    "Install core extras with:  pip install article-tagging"
)


def _load_pillow() -> type:
    """Return the ``PIL.Image`` module, raising a clear error if missing."""
    try:
        from PIL import Image as _Image
    except ImportError:
        raise ImportError(_PILLOW_INSTALL_MSG) from None
    return _Image


def preprocess_image(path: Path, max_size: int = 1024) -> Image.Image:
    """Load and preprocess an image for VLM training.

    Handles animated GIFs (first frame), CMYK/RGBA/palette colour spaces,
    and proportional resizing when the longest side exceeds *max_size*.
    Qwen3-VL supports dynamic resolution, so we just cap the maximum.

    Args:
        path: Path to the image file.
        max_size: Maximum dimension in pixels for the longest side.

    Returns:
        A preprocessed RGB ``PIL.Image.Image``.

    Raises:
        ImportError: If Pillow is not installed.
        FileNotFoundError: If *path* does not exist.
    """
    PILImage = _load_pillow()

    img = PILImage.open(path)

    # Animated GIFs — use first frame
    try:
        img.seek(0)
    except EOFError:
        pass

    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Proportional resize if too large
    w, h = img.size
    longest = max(w, h)
    if longest > max_size:
        scale = max_size / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), PILImage.LANCZOS)

    return img


def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL Image as a base64 data URI string.

    Args:
        image: The PIL Image to encode.
        fmt: Image format for encoding (default ``"JPEG"``).

    Returns:
        A ``data:image/{mime};base64,...`` string suitable for embedding
        in chat message content.
    """
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = fmt.lower()
    if mime == "jpeg":
        mime = "jpeg"
    return f"data:image/{mime};base64,{b64}"


def validate_image(path: Path) -> bool:
    """Check whether a file is a valid image that Pillow can open.

    Args:
        path: Path to the file to validate.

    Returns:
        ``True`` if the file is a valid image, ``False`` otherwise.
    """
    PILImage = _load_pillow()
    try:
        with PILImage.open(path) as img:
            img.verify()
        return True
    except Exception:  # noqa: BLE001
        return False
