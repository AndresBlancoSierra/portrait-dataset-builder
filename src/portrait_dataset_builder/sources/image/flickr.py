"""Flickr public feeds image provider."""

from __future__ import annotations

import asyncio

import httpx

from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import image_source
from portrait_dataset_builder.sources.image.base import ImageResult, ImageSource

logger = get_logger("source.flickr")

FEED_URL = "https://api.flickr.com/services/feeds/photos_public.gne"


@image_source("flickr")
class FlickrImageSource(ImageSource):
    """Search Flickr public feeds for portrait reference photos."""

    provider_name = "flickr"
    SEARCH_DELAY = 1.0

    async def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[ImageResult]:
        tag_variations = [
            f"{query}",
            f"{query} photo",
            f"{query} face",
            f"{query} portrait",
            f"{query} headshot",
            f"{query} red carpet",
        ]

        all_results: list[ImageResult] = []

        async with httpx.AsyncClient(timeout=15.0) as client:
            for tags in tag_variations:
                if len(all_results) >= max_results:
                    break
                try:
                    batch = await self._fetch_feed(client, tags)
                    all_results.extend(batch)
                    logger.info(
                        "Flickr batch '{}': {} results (total raw: {})",
                        tags, len(batch), len(all_results),
                    )
                except Exception as e:
                    logger.warning("Flickr feed failed for '{}': {}", tags, e)

                await asyncio.sleep(self.SEARCH_DELAY)

        seen: set[str] = set()
        unique: list[ImageResult] = []
        for r in all_results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)

        logger.info("Flickr: {} total unique results for '{}'", len(unique), query)
        return unique[:max_results]

    async def _fetch_feed(
        self,
        client: httpx.AsyncClient,
        tags: str,
    ) -> list[ImageResult]:
        params = {
            "tags": tags,
            "tagmode": "any",
            "nojsoncallback": "1",
            "format": "json",
        }
        resp = await client.get(FEED_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        results: list[ImageResult] = []
        for item in data.get("items", []):
            media = item.get("media", {})
            url = media.get("m", "")
            if not url:
                continue
            url = url.replace("_m.jpg", "_b.jpg")

            title = item.get("title", "")
            source_url = item.get("link", "")
            author = item.get("author", "")
            published = item.get("published", "")

            results.append(
                ImageResult(
                    url=url,
                    title=title,
                    source_provider=self.provider_name,
                    source_url=source_url,
                    metadata={
                        "author": author,
                        "published": published,
                        "tags": item.get("tags", ""),
                    },
                )
            )
        return results
