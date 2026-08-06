"""Pexels image search provider (web scraping, no API key)."""

from __future__ import annotations

import asyncio
import re

import httpx

from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import image_source
from portrait_dataset_builder.sources.image.base import ImageResult, ImageSource

logger = get_logger("source.pexels")

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


@image_source("pexels")
class PexelsImageSource(ImageSource):
    """Search Pexels for portrait photos (web scraping)."""

    provider_name = "pexels"

    async def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[ImageResult]:
        results: list[ImageResult] = []
        page = 1

        while len(results) < max_results and page <= 10:
            try:
                url = (
                    f"https://www.pexels.com/search/{query}/"
                    f"?page={page}"
                )
                batch = await asyncio.get_running_loop().run_in_executor(
                    None, lambda u=url: self._sync_fetch(u)
                )
                if not batch:
                    break
                results.extend(batch)
                logger.info("Pexels page {}: {} results", page, len(batch))
            except Exception as e:
                logger.warning("Pexels search failed page {}: {}", page, e)
                break
            page += 1
            await asyncio.sleep(2.0)

        seen = set()
        unique = []
        for r in results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)

        logger.info("Pexels: {} unique results for '{}'", len(unique), query)
        return unique[:max_results]

    def _sync_fetch(self, url: str) -> list[ImageResult]:
        results: list[ImageResult] = []
        try:
            with httpx.Client(
                timeout=15,
                follow_redirects=True,
                headers={
                    "User-Agent": _UA,
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                html = resp.text

                img_pattern = re.compile(
                    r'src="(https://images\.pexels\.com/photos/[^"]+)"'
                )
                for match in img_pattern.finditer(html):
                    img_url = match.group(1)
                    if img_url.startswith("http"):
                        cleaned = img_url.split("?")[0]
                        results.append(
                            ImageResult(
                                url=cleaned,
                                title="",
                                source_provider=self.provider_name,
                            )
                        )

        except Exception as e:
            logger.warning("Pexels sync fetch failed: {}", e)
        return results
