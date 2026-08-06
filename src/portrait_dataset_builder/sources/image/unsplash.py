"""Unsplash image search provider (no API key needed)."""

from __future__ import annotations

import asyncio

import httpx

from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import image_source
from portrait_dataset_builder.sources.image.base import ImageResult, ImageSource

logger = get_logger("source.unsplash")

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_UNSPLASH_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://unsplash.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": _UA,
}


@image_source("unsplash")
class UnsplashImageSource(ImageSource):
    """Search Unsplash for portrait photos (web scraping, no API key)."""

    provider_name = "unsplash"

    async def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[ImageResult]:
        results: list[ImageResult] = []
        page = 1
        per_page = 30

        while len(results) < max_results and page <= 10:
            try:
                url = (
                    "https://unsplash.com/napi/search/photos"
                    f"?query={query}&page={page}&per_page={per_page}"
                )
                batch = await asyncio.get_running_loop().run_in_executor(
                    None, lambda u=url: self._sync_fetch(u)
                )
                if not batch:
                    break
                results.extend(batch)
                logger.info("Unsplash page {}: {} results", page, len(batch))
            except Exception as e:
                logger.warning("Unsplash search failed page {}: {}", page, e)
                break
            page += 1
            await asyncio.sleep(2.0)

        seen = set()
        unique = []
        for r in results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)

        logger.info("Unsplash: {} unique results for '{}'", len(unique), query)
        return unique[:max_results]

    def _sync_fetch(self, url: str) -> list[ImageResult]:
        results: list[ImageResult] = []
        try:
            with httpx.Client(
                timeout=15,
                follow_redirects=True,
                headers=_UNSPLASH_HEADERS,
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("results", []):
                    urls = item.get("urls", {})
                    raw_url = urls.get("regular") or urls.get("raw") or ""
                    if not raw_url:
                        continue
                    results.append(
                        ImageResult(
                            url=raw_url,
                            title=item.get("alt_description") or "",
                            source_provider=self.provider_name,
                            width=item.get("width", 0),
                            height=item.get("height", 0),
                        )
                    )
        except Exception as e:
            logger.warning("Unsplash sync fetch failed: {}", e)
        return results
