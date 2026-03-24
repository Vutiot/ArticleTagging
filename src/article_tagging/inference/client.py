"""Async inference client for the vLLM server.

Sends prediction requests with guided JSON decoding, handling retries and
concurrent batch processing.  The system prompt is kept identical across
requests to maximise vLLM V1 prefix-cache hits.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from article_tagging.inference.schema_generator import DatasetSchema

logger = logging.getLogger(__name__)

_OPENAI_INSTALL_MSG = (
    "The 'openai' library is required for inference.\n"
    "Install the serving extras with:  pip install article-tagging[serving]"
)


async def predict(
    title: str,
    schema: DatasetSchema,
    server_url: str = "http://localhost:8000",
    *,
    image_path: Path | None = None,
    system_prompt: str | None = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> dict[str, Any]:
    """Run a single prediction against a vLLM server.

    Args:
        title: Product title text.
        schema: Dataset schema for guided decoding and prompt building.
        server_url: Base URL of the vLLM server (no trailing ``/v1``).
        image_path: Optional local image path (encoded as base64 data URI).
        system_prompt: Override system prompt. If ``None``, built from schema.
        timeout: Request timeout in seconds.
        retries: Number of retry attempts on failure.

    Returns:
        Dict of predicted attribute values.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(_OPENAI_INSTALL_MSG) from None

    from article_tagging.dataset.formatter import build_system_prompt
    from article_tagging.inference.schema_generator import generate_json_schema

    if system_prompt is None:
        system_prompt = build_system_prompt(
            schema,
            "You extract product attributes from the title and image. "
            "Respond with valid JSON only.",
        )

    # Build user message
    user_text = f'Title: "{title}"'
    if image_path is not None and image_path.exists():
        from article_tagging.dataset.image_processing import image_to_base64, preprocess_image

        img = preprocess_image(image_path)
        data_uri = image_to_base64(img)
        user_content: str | list[dict] = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": user_text},
        ]
    else:
        user_content = user_text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    guided_schema = generate_json_schema(schema)

    client = AsyncOpenAI(base_url=f"{server_url}/v1", api_key="not-needed")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model="default",
                messages=messages,
                extra_body={"guided_json": guided_schema},
                timeout=timeout,
            )
            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("Attempt %d failed (%s), retrying in %ds", attempt + 1, e, wait)
                await asyncio.sleep(wait)

    logger.error("All %d attempts failed", retries)
    raise RuntimeError(f"Prediction failed after {retries} attempts: {last_error}")


async def predict_batch(
    records: list[dict],
    schema: DatasetSchema,
    server_url: str = "http://localhost:8000",
    *,
    system_prompt: str | None = None,
    concurrency: int = 8,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Run predictions on a batch of records concurrently.

    Args:
        records: List of listing dicts with ``title`` and optional ``image_urls``.
        schema: Dataset schema.
        server_url: vLLM server base URL.
        system_prompt: Override system prompt.
        concurrency: Max concurrent requests.
        timeout: Per-request timeout.

    Returns:
        List of predicted attribute dicts (same order as input).
    """
    sem = asyncio.Semaphore(concurrency)

    async def _predict_one(record: dict) -> dict[str, Any]:
        async with sem:
            title = record.get("title", "")
            image_urls = record.get("image_urls", [])
            image_path = Path(image_urls[0]) if image_urls else None
            # Only use local files for images
            if image_path and not image_path.exists():
                image_path = None

            return await predict(
                title,
                schema,
                server_url,
                image_path=image_path,
                system_prompt=system_prompt,
                timeout=timeout,
            )

    tasks = [_predict_one(r) for r in records]
    return await asyncio.gather(*tasks)
