"""Wikipedia image search provider."""

from __future__ import annotations

import httpx

from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import image_source
from portrait_dataset_builder.sources.image.base import ImageResult, ImageSource

logger = get_logger("source.wikipedia")


@image_source("wikipedia")
class WikipediaImageSource(ImageSource):
    """Search Wikipedia for portrait images from infoboxes."""

    provider_name = "wikipedia"
    API_URL = "https://en.wikipedia.org/w/api.php"

    async def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[ImageResult]:
        results: list[ImageResult] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            page_title = await self._search_page(client, query)
            if not page_title:
                logger.warning("Wikipedia: no page found for '{}'", query)
                return results

            results = await self._get_images(client, page_title)

        logger.info("Wikipedia: found {} results for '{}'", len(results), query)
        return results[:max_results]

    async def _search_page(self, client: httpx.AsyncClient, query: str) -> str | None:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        }
        try:
            resp = await client.get(self.API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            search_results = data.get("query", {}).get("search", [])
            if search_results:
                return search_results[0].get("title")
        except (httpx.HTTPError, ValueError):
            pass
        return None

    async def _get_images(self, client: httpx.AsyncClient, page_title: str) -> list[ImageResult]:
        results: list[ImageResult] = []
        params = {
            "action": "query",
            "titles": page_title,
            "prop": "images",
            "imlimit": "50",
            "format": "json",
        }
        try:
            resp = await client.get(self.API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                images = page.get("images", [])
                for img_info in images:
                    title = img_info.get("title", "")
                    is_image = any(
                        title.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".tiff"]
                    )
                    is_relevant = any(
                        kw in title.lower()
                        for kw in ["photo", "portrait", "headshot", "face", "201", "202"]
                    ) or not any(
                        kw in title.lower() for kw in ["logo", "icon", "flag", "map", "diagram"]
                    )
                    if is_image and is_relevant:
                        image_url = await self._get_image_info(client, title)
                        if image_url:
                            results.append(
                                ImageResult(
                                    url=image_url,
                                    title=title,
                                    source_provider=self.provider_name,
                                    source_url=(
                                        "https://en.wikipedia.org/wiki/"
                                        + page_title.replace(" ", "_")
                                    ),
                                )
                            )
        except (httpx.HTTPError, ValueError):
            pass

        return results

    async def _get_image_info(self, client: httpx.AsyncClient, file_title: str) -> str | None:
        params = {
            "action": "query",
            "titles": file_title,
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
                imageinfo = page.get("imageinfo", [{}])
                if imageinfo:
                    return imageinfo[0].get("thumburl") or imageinfo[0].get("url")
        except (httpx.HTTPError, ValueError):
            pass
        return None
