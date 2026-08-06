"""Google Images search provider."""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup

from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import image_source
from portrait_dataset_builder.sources.image.base import ImageResult, ImageSource

logger = get_logger("source.google")


@image_source("google")
class GoogleImageSource(ImageSource):
    """Search Google Images for portrait reference photos."""

    provider_name = "google"
    BASE_URL = "https://www.google.com/search"
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
            search_query = f"{query} portrait photo face"
            num_batches = min((max_results // 20) + 1, 10)

            for start in range(0, max_results, 20):
                if start >= num_batches * 20:
                    break

                params = {
                    "q": search_query,
                    "tbm": "isch",
                    "start": start,
                    "ijn": "0",
                }

                try:
                    resp = await client.get(self.BASE_URL, params=params)
                    resp.raise_for_status()
                except httpx.HTTPError as e:
                    logger.warning("Google search batch failed at offset {}: {}", start, e)
                    continue

                page_results = self._parse_results(resp.text, start)
                results.extend(page_results)

                if len(page_results) == 0:
                    break

                if len(results) >= max_results:
                    break

        logger.info("Google Images: found {} results for '{}'", len(results), query)
        return results[:max_results]

    def _parse_results(self, html: str, offset: int) -> list[ImageResult]:
        results: list[ImageResult] = []

        img_data_pattern = re.compile(r'\["(https?://[^"]+)",\d+,\d+\]')
        matches = img_data_pattern.findall(html)

        seen_urls: set[str] = set()
        for url in matches:
            if url in seen_urls:
                continue
            if not any(
                url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]
            ) and ("encrypted" in url or "gstatic" in url):
                continue
            seen_urls.add(url)
            results.append(
                ImageResult(
                    url=url,
                    source_provider=self.provider_name,
                    source_url=f"google_search_offset_{offset}",
                )
            )

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if "/imgres?" in href:
                img_match = re.search(r"imgurl=([^&]+)", href)
                if img_match:
                    import urllib.parse

                    img_url = urllib.parse.unquote(img_match.group(1))
                    if img_url not in seen_urls:
                        seen_urls.add(img_url)
                        w_match = re.search(r"imgw=(\d+)", href)
                        h_match = re.search(r"imgh=(\d+)", href)
                        results.append(
                            ImageResult(
                                url=img_url,
                                source_provider=self.provider_name,
                                width=int(w_match.group(1)) if w_match else 0,
                                height=int(h_match.group(1)) if h_match else 0,
                            )
                        )

        return results
