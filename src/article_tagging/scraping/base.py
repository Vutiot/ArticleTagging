"""Base scraper interface and factory for config-driven web scraping.

Provides the abstract ``BaseScraper`` class that all concrete scrapers extend,
data classes for scraper output, a CSS selector parser supporting the ``@attr``
convention, and a factory function that picks the right backend.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from article_tagging.configs.models import PaginationType

if TYPE_CHECKING:
    from article_tagging.configs.models import SiteConfig

logger = logging.getLogger(__name__)

# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class RawListing:
    """A single scraped listing with its extracted attributes."""

    url: str
    title: str
    image_urls: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class ScrapedPage:
    """Result of scraping one listing/index page."""

    listing_urls: list[str]
    next_page_url: str | None = None


# ─── Selector parsing ────────────────────────────────────────────────────────

_ATTR_RE = re.compile(r"^(.+?)@(\w+)$")


def parse_selector(raw: str) -> tuple[str, str | None]:
    """Split a CSS selector with optional ``@attr`` suffix.

    Args:
        raw: A selector string like ``"div.gallery img@src"`` or ``"h1.title"``.

    Returns:
        A tuple of ``(css_selector, attribute_name | None)``.
        If no ``@attr`` suffix, the second element is ``None`` (extract text).

    Examples:
        >>> parse_selector("div.foo img@src")
        ('div.foo img', 'src')
        >>> parse_selector("h1.title")
        ('h1.title', None)
    """
    m = _ATTR_RE.match(raw.strip())
    if m:
        return m.group(1).strip(), m.group(2)
    return raw.strip(), None


def extract_with_selector(soup: BeautifulSoup, raw_selector: str) -> str | None:
    """Apply a single selector (with optional ``@attr``) to a BeautifulSoup tree.

    Args:
        soup: Parsed HTML document.
        raw_selector: CSS selector, optionally ending with ``@attr``.

    Returns:
        Extracted text or attribute value, or ``None`` if not found.
    """
    css, attr = parse_selector(raw_selector)
    tag = soup.select_one(css)
    if tag is None:
        return None
    if attr:
        return tag.get(attr)  # type: ignore[return-value]
    return tag.get_text(strip=True)


def extract_all_with_selector(soup: BeautifulSoup, raw_selector: str) -> list[str]:
    """Apply a selector to all matching elements and return a list of values.

    Args:
        soup: Parsed HTML document.
        raw_selector: CSS selector, optionally ending with ``@attr``.

    Returns:
        List of extracted text/attribute values (empty strings filtered out).
    """
    css, attr = parse_selector(raw_selector)
    tags = soup.select(css)
    results: list[str] = []
    for tag in tags:
        val = tag.get(attr) if attr else tag.get_text(strip=True)
        if val:
            results.append(str(val))
    return results


# ─── Abstract base scraper ───────────────────────────────────────────────────


class BaseScraper(ABC):
    """Abstract base class for all scrapers.

    Concrete subclasses only need to implement :meth:`fetch_page` — all HTML
    parsing, pagination, and field extraction logic is shared in the base class.

    Args:
        config: Site configuration driving the scraping behaviour.
    """

    def __init__(self, config: SiteConfig) -> None:
        self.config = config
        self._last_request_time: float = 0.0

    @abstractmethod
    async def fetch_page(self, url: str) -> str:
        """Fetch raw HTML content from *url*.

        Subclasses implement this with their specific HTTP backend
        (httpx, Playwright, etc.).
        """

    async def close(self) -> None:
        """Release any resources held by the scraper (override if needed)."""

    # ── Public API ────────────────────────────────────────────────────────

    async def scrape_listings(self) -> list[RawListing]:
        """Scrape all listings by paginating through index pages and detail pages.

        Returns:
            List of :class:`RawListing` objects with extracted attributes.
        """
        all_listings: list[RawListing] = []
        url: str | None = self.config.base_url
        pages_scraped = 0

        while url is not None:
            if pages_scraped >= self.config.pagination.max_pages:
                logger.info("Reached max_pages limit (%d)", self.config.pagination.max_pages)
                break

            await self._rate_limit()
            html = await self.fetch_page(url)
            page = self._parse_listing_page(html, url)
            pages_scraped += 1

            logger.info(
                "Page %d: found %d listing URLs (total so far: %d)",
                pages_scraped,
                len(page.listing_urls),
                len(all_listings),
            )

            for listing_url in page.listing_urls:
                if self.config.max_listings and len(all_listings) >= self.config.max_listings:
                    logger.info("Reached max_listings limit (%d)", self.config.max_listings)
                    return all_listings

                await self._rate_limit()
                listing = await self.scrape_detail(listing_url)
                all_listings.append(listing)

            url = self._resolve_next_page(page, url, pages_scraped)

        logger.info("Scraping complete: %d listings collected", len(all_listings))
        return all_listings

    async def scrape_detail(self, url: str) -> RawListing:
        """Fetch a detail page and extract all fields via ``detail_selectors``.

        Args:
            url: Absolute URL of the detail page.

        Returns:
            A :class:`RawListing` with the extracted data.
        """
        html = await self.fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")

        attributes: dict[str, str] = {}
        for name, selector in self.config.detail_selectors.items():
            value = extract_with_selector(soup, selector)
            if value:
                attributes[name] = value

        title = attributes.pop("title", "")
        image_urls: list[str] = []
        if "image" in attributes:
            raw_img = attributes.pop("image")
            image_urls = [urljoin(url, raw_img)]
        if "images" in self.config.detail_selectors:
            image_urls = [
                urljoin(url, u)
                for u in extract_all_with_selector(soup, self.config.detail_selectors["images"])
            ]

        return RawListing(
            url=url,
            title=title,
            image_urls=image_urls,
            attributes=attributes,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _parse_listing_page(self, html: str, base_url: str) -> ScrapedPage:
        """Extract listing URLs and next-page link from a listing/index page."""
        soup = BeautifulSoup(html, "html.parser")

        listing_urls = [
            urljoin(base_url, u) for u in extract_all_with_selector(soup, self.config.listing_selector)
        ]

        next_page_url: str | None = None
        pagination = self.config.pagination
        if pagination.type == PaginationType.NEXT_LINK and pagination.selector:
            raw = extract_with_selector(soup, pagination.selector)
            if raw:
                next_page_url = urljoin(base_url, raw)

        return ScrapedPage(listing_urls=listing_urls, next_page_url=next_page_url)

    def _resolve_next_page(
        self, page: ScrapedPage, current_url: str, pages_scraped: int
    ) -> str | None:
        """Determine the next page URL based on pagination type."""
        pagination = self.config.pagination

        if pagination.type == PaginationType.NEXT_LINK:
            return page.next_page_url

        if pagination.type == PaginationType.PAGE_NUMBER:
            if pagination.url_pattern is None:
                return None
            return pagination.url_pattern.format(page=pages_scraped + 1)

        if pagination.type == PaginationType.INFINITE_SCROLL:
            raise NotImplementedError(
                "INFINITE_SCROLL pagination must be handled by a Playwright-based scraper"
            )

        return None

    async def _rate_limit(self) -> None:
        """Enforce the configured rate limit between requests."""
        if self.config.rate_limit > 0:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self.config.rate_limit:
                await asyncio.sleep(self.config.rate_limit - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()


# ─── Factory ──────────────────────────────────────────────────────────────────


def create_scraper(config: SiteConfig) -> BaseScraper:
    """Instantiate the appropriate scraper backend for a site configuration.

    Args:
        config: Site configuration. If ``use_playwright`` is ``True``, returns
            a Playwright-based scraper; otherwise an httpx + BeautifulSoup one.

    Returns:
        A concrete :class:`BaseScraper` instance.

    Raises:
        ImportError: If the required backend package is not installed.
    """
    if config.use_playwright:
        from article_tagging.scraping.dynamic_scraper import PlaywrightScraper

        return PlaywrightScraper(config)

    from article_tagging.scraping.static_scraper import StaticScraper

    return StaticScraper(config)
