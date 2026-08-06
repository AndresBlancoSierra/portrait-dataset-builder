"""IMDb image search provider."""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup

from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import image_source
from portrait_dataset_builder.sources.image.base import ImageResult, ImageSource

logger = get_logger("source.imdb")


@image_source("imdb")
class IMDbImageSource(ImageSource):
    """Search IMDb for official portrait/headshot images."""

    provider_name = "imdb"
    BASE_URL = "https://www.imdb.com"
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
            person_id = await self._find_person_id(client, query)
            if not person_id:
                logger.warning("IMDb: could not find person ID for '{}'", query)
                return results

            media_url = f"{self.BASE_URL}/name/{person_id}/mediaindex"
            try:
                resp = await client.get(media_url)
                resp.raise_for_status()
                results = self._parse_media_page(resp.text, person_id)
            except httpx.HTTPError as e:
                logger.warning("IMDb media page failed: {}", e)

        logger.info("IMDb: found {} results for '{}'", len(results), query)
        return results[:max_results]

    async def _find_person_id(self, client: httpx.AsyncClient, name: str) -> str | None:
        params = {"q": name, "s": "nm", "exact": "true"}
        try:
            resp = await client.get(f"{self.BASE_URL}/find", params=params)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            link = soup.select_one("a.ipc-metadata-list-summary-item__t")
            if link:
                href = link.get("href", "")
                match = re.search(r"/nm(\d+)", href)
                if match:
                    return f"nm{match.group(1)}"
        except httpx.HTTPError:
            pass

        params = {"q": name, "s": "nm"}
        try:
            resp = await client.get(f"{self.BASE_URL}/find", params=params)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.select("a.ipc-metadata-list-summary-item__t")
            if links:
                href = links[0].get("href", "")
                match = re.search(r"/nm(\d+)", href)
                if match:
                    return f"nm{match.group(1)}"
        except httpx.HTTPError:
            pass

        return None

    def _parse_media_page(self, html: str, person_id: str) -> list[ImageResult]:
        results: list[ImageResult] = []
        soup = BeautifulSoup(html, "html.parser")

        for img in soup.select("img.ipc-image"):
            src = img.get("src", "")
            if src and ("media-imdb" in src or "m.media-amazon" in src):
                clean_url = re.sub(r"\._.*?_\.", ".", src)
                results.append(
                    ImageResult(
                        url=clean_url,
                        source_provider=self.provider_name,
                        source_url=f"{self.BASE_URL}/name/{person_id}",
                    )
                )

        return results
