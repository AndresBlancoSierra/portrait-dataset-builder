"""Bing Images search provider."""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup

from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import image_source
from portrait_dataset_builder.sources.image.base import ImageResult, ImageSource

logger = get_logger("source.bing")


@image_source("bing")
class BingImageSource(ImageSource):
    """Search Bing Images for portrait reference photos."""

    provider_name = "bing"
    BASE_URL = "https://www.bing.com/images/search"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    async def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[ImageResult]:
        results: list[ImageResult] = []
        async with httpx.AsyncClient(
            headers=self.HEADERS, timeout=30.0, follow_redirects=True
        ) as client:
            search_query = f"{query} portrait face photo"

            for offset in range(0, max_results, 35):
                params = {
                    "q": search_query,
                    "first": offset + 1,
                    "count": 35,
                    "qft": "+filterui:photo-portrait",
                }

                try:
                    resp = await client.get(self.BASE_URL, params=params)
                    resp.raise_for_status()
                except httpx.HTTPError as e:
                    logger.warning("Bing search failed at offset {}: {}", offset, e)
                    continue

                batch = self._parse_results(resp.text)
                results.extend(batch)

                if len(batch) == 0:
                    break
                if len(results) >= max_results:
                    break

        logger.info("Bing Images: found {} results for '{}'", len(results), query)
        return results[:max_results]

    def _parse_results(self, html: str) -> list[ImageResult]:
        results: list[ImageResult] = []
        soup = BeautifulSoup(html, "html.parser")

        for item in soup.select("a.iusc"):
            m_attr = item.get("m", "")
            if not m_attr:
                continue

            import json

            try:
                data = json.loads(m_attr)
            except (json.JSONDecodeError, TypeError):
                continue

            img_url = data.get("murl", "")
            if not img_url:
                continue

            title = data.get("t", "")
            data.get("purl", 0)

            results.append(
                ImageResult(
                    url=img_url,
                    title=title,
                    source_provider=self.provider_name,
                    thumbnail_url=data.get("turl", ""),
                )
            )

        for script_tag in soup.find_all("script"):
            if script_tag.string and "murl" in (script_tag.string or ""):
                pattern = re.compile(r'"murl"\s*:\s*"(https?://[^"]+)"')
                for match in pattern.finditer(script_tag.string):
                    url = match.group(1).replace("\\u0026", "&")
                    results.append(
                        ImageResult(
                            url=url,
                            source_provider=self.provider_name,
                        )
                    )

        return results
