"""Wikimedia Commons search provider."""

from __future__ import annotations

import httpx

from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import image_source
from portrait_dataset_builder.sources.image.base import ImageResult, ImageSource

logger = get_logger("source.wikimedia")


@image_source("wikimedia")
class WikimediaImageSource(ImageSource):
    """Search Wikimedia Commons for freely licensed portrait photos."""

    provider_name = "wikimedia"
    API_URL = "https://commons.wikimedia.org/w/api.php"

    async def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[ImageResult]:
        results: list[ImageResult] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            search_query = f"{query} portrait"

            for offset in range(0, max_results, 50):
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": search_query,
                    "srnamespace": "6",
                    "srlimit": min(50, max_results - offset),
                    "sroffset": offset,
                    "format": "json",
                }

                try:
                    resp = await client.get(self.API_URL, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                except (httpx.HTTPError, ValueError) as e:
                    logger.warning("Wikimedia API failed at offset {}: {}", offset, e)
                    continue

                search_results = data.get("query", {}).get("search", [])
                if not search_results:
                    break

                for item in search_results:
                    title = item.get("title", "")
                    image_url = await self._get_image_url(client, title)
                    if image_url:
                        results.append(
                            ImageResult(
                                url=image_url,
                                title=title,
                                source_provider=self.provider_name,
                                source_url=(
                                    "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_")
                                ),
                            )
                        )

                if len(results) >= max_results:
                    break

        logger.info("Wikimedia Commons: found {} results for '{}'", len(results), query)
        return results[:max_results]

    async def _get_image_url(self, client: httpx.AsyncClient, title: str) -> str | None:
        params = {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 2000,
            "format": "json",
        }

        try:
            resp = await client.get(self.API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                imageinfo = page.get("imageinfo", [{}])[0]
                url = imageinfo.get("thumburl") or imageinfo.get("url")
                if url and any(
                    url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]
                ):
                    return url
        except (httpx.HTTPError, ValueError, KeyError):
            pass

        return None
