"""Playwright-based scraper for JavaScript-rendered and infinite-scroll sites.

Extends :class:`~article_tagging.scraping.base.BaseScraper` with a headless
Chromium browser managed by Playwright.  The browser is lazily initialised on
first use so importing this module never triggers a Playwright dependency check.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from article_tagging.configs.models import PaginationType
from article_tagging.scraping.base import BaseScraper, RawListing

if TYPE_CHECKING:
    from article_tagging.configs.models import SiteConfig

logger = logging.getLogger(__name__)


class PlaywrightScraper(BaseScraper):
    """Scraper that uses a headless Chromium browser via Playwright.

    Suitable for sites that require JavaScript rendering or use infinite-scroll
    pagination.  The browser is created lazily — no Playwright dependency is
    needed until :meth:`fetch_page` is actually called.

    Args:
        config: Site configuration driving the scraping behaviour.
    """

    def __init__(self, config: SiteConfig) -> None:
        super().__init__(config)
        self._playwright: object | None = None
        self._browser: object | None = None
        self._page: object | None = None

    async def _ensure_browser(self) -> None:
        """Lazily launch Playwright and create a browser page.

        Raises:
            ImportError: If ``playwright`` is not installed or Chromium has not
                been set up.
        """
        if self._page is not None:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright is required for dynamic scraping but is not installed. "
                "Install it with:\n"
                "  pip install article-tagging[scraping]\n"
                "  playwright install chromium"
            ) from None

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)  # type: ignore[union-attr]
        self._page = await self._browser.new_page()  # type: ignore[union-attr]

        if self.config.headers:
            await self._page.set_extra_http_headers(self.config.headers)  # type: ignore[union-attr]

        logger.debug("Playwright browser launched (headless Chromium)")

    async def fetch_page(self, url: str) -> str:
        """Navigate to *url* in the headless browser and return the rendered HTML.

        If :pyattr:`config.wait_for_selector` is set, the method waits up to
        10 seconds for that element to appear in the DOM before returning.

        Args:
            url: Absolute URL to fetch.

        Returns:
            Fully rendered HTML content of the page.
        """
        await self._ensure_browser()

        await self._page.goto(url, wait_until="domcontentloaded")  # type: ignore[union-attr]

        if self.config.wait_for_selector:
            await self._page.wait_for_selector(  # type: ignore[union-attr]
                self.config.wait_for_selector,
                timeout=10_000,
            )

        return await self._page.content()  # type: ignore[union-attr]

    async def close(self) -> None:
        """Close the browser and stop the Playwright instance if they were started."""
        if self._browser is not None:
            await self._browser.close()  # type: ignore[union-attr]
            self._browser = None
            self._page = None

        if self._playwright is not None:
            await self._playwright.stop()  # type: ignore[union-attr]
            self._playwright = None

        logger.debug("Playwright browser closed")

    # ── Infinite-scroll support ──────────────────────────────────────────

    async def scrape_listings(self) -> list[RawListing]:
        """Scrape all listings, handling ``INFINITE_SCROLL`` pagination.

        For infinite-scroll sites the method repeatedly scrolls to the bottom
        of the page, waits for new content, and collects listing URLs until no
        new items appear or ``max_pages`` scroll iterations are reached.

        For all other pagination types the call is delegated to the base class.

        Returns:
            List of :class:`RawListing` objects with extracted attributes.
        """
        if self.config.pagination.type != PaginationType.INFINITE_SCROLL:
            return await super().scrape_listings()

        await self._ensure_browser()
        await self._rate_limit()

        await self._page.goto(  # type: ignore[union-attr]
            self.config.base_url,
            wait_until="domcontentloaded",
        )

        if self.config.wait_for_selector:
            await self._page.wait_for_selector(  # type: ignore[union-attr]
                self.config.wait_for_selector,
                timeout=10_000,
            )

        all_listing_urls: list[str] = []
        max_scrolls = self.config.pagination.max_pages
        scrolls = 0

        while scrolls < max_scrolls:
            # Collect currently visible listing URLs
            html = await self._page.content()  # type: ignore[union-attr]
            page = self._parse_listing_page(html, self.config.base_url)
            new_urls = [u for u in page.listing_urls if u not in all_listing_urls]

            if not new_urls:
                logger.info("No new listings after scroll %d — stopping", scrolls)
                break

            all_listing_urls.extend(new_urls)
            scrolls += 1

            logger.info(
                "Scroll %d: found %d new URLs (total: %d)",
                scrolls,
                len(new_urls),
                len(all_listing_urls),
            )

            if self.config.max_listings and len(all_listing_urls) >= self.config.max_listings:
                all_listing_urls = all_listing_urls[: self.config.max_listings]
                logger.info("Reached max_listings limit (%d)", self.config.max_listings)
                break

            # Scroll to bottom and wait for potential new content
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")  # type: ignore[union-attr]
            await self._page.wait_for_timeout(1_000)  # type: ignore[union-attr]

        # Scrape detail pages for each collected URL
        all_listings: list[RawListing] = []
        for listing_url in all_listing_urls:
            if self.config.max_listings and len(all_listings) >= self.config.max_listings:
                break

            await self._rate_limit()
            listing = await self.scrape_detail(listing_url)
            all_listings.append(listing)

        logger.info("Scraping complete: %d listings collected", len(all_listings))
        return all_listings
