"""Static site scraper using httpx and BeautifulSoup.

Implements :class:`BaseScraper` for server-rendered pages that do not require
JavaScript execution.  Uses ``httpx.AsyncClient`` for HTTP requests.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from article_tagging.scraping.base import BaseScraper

if TYPE_CHECKING:
    from article_tagging.configs.models import SiteConfig

logger = logging.getLogger(__name__)


class StaticScraper(BaseScraper):
    """Scraper for static (server-rendered) websites.

    Uses :mod:`httpx` to fetch pages and relies on the base class for all HTML
    parsing, pagination, and field extraction.

    Args:
        config: Site configuration driving the scraping behaviour.

    Raises:
        ImportError: If ``httpx`` is not installed.
    """

    def __init__(self, config: SiteConfig) -> None:
        super().__init__(config)

        try:
            import httpx  # noqa: F811
        except ImportError:
            raise ImportError(
                "httpx is required for the static scraper but is not installed. "
                "Install it with: pip install article-tagging[scraping]"
            ) from None

        self._client = httpx.AsyncClient(
            headers=config.headers,
            follow_redirects=True,
            timeout=30.0,
        )

    async def fetch_page(self, url: str) -> str:
        """Fetch raw HTML from *url* via an HTTP GET request.

        Args:
            url: Absolute URL to fetch.

        Returns:
            The response body as text.

        Raises:
            httpx.HTTPStatusError: If the response status is not 2xx.
        """
        logger.debug("Fetching %s", url)
        response = await self._client.get(url)
        response.raise_for_status()
        return response.text

    async def close(self) -> None:
        """Close the underlying httpx client and release connections."""
        await self._client.aclose()
