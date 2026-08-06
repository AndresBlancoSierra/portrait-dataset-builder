"""DuckDuckGo Images search provider."""

from __future__ import annotations

import asyncio
from functools import partial

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import image_source
from portrait_dataset_builder.sources.image.base import ImageResult, ImageSource

logger = get_logger("source.duckduckgo")


@image_source("duckduckgo")
class DuckDuckGoImageSource(ImageSource):
    """Search DuckDuckGo Images for portrait reference photos."""

    provider_name = "duckduckgo"
    SEARCH_DELAY = 3.5

    async def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[ImageResult]:
        results: list[ImageResult] = []
        name_parts = query.split()
        first_name = name_parts[0] if name_parts else query
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        queries = [
            f"{query} portrait",
            f"{query} face photo",
            f"{query} headshot",
            f"{query} red carpet",
            f"{query} movie premiere",
            f"{query} interview photo",
            f"{query} magazine cover",
            f"{query} photoshoot",
            f"{query} red carpet 2024",
            f"{query} candid photo",
            f"{query} press conference",
            f"{query} film festival",
            f"{query} red carpet 2023",
            f"{query} studio portrait",
            f"{query} editorial photo",
            f"{query} close up face",
            f"{first_name} {last_name} actor portrait" if last_name else f"{query} actor portrait",
            f"{query} black and white portrait",
            f"{query} smiling photo",
            f"{query} looking at camera",
            f"{query} high resolution photo",
            f"{query} professional photo",
            f"{query} event photo 2024",
            f"{query} event photo 2023",
            f"{query} event photo 2022",
            f"{query} award ceremony",
            f"{query} movie still",
            f"{query} film promotion",
            f"{query} poster photo",
            f"{query} fashion photo",
            f"{query} Vanity Fair",
            f"{query} GQ magazine",
            f"{query} portrait photography",
            f"{query} face closeup",
            f"{query} smiling face",
            f"{query} serious face",
            f"{query} formal portrait",
            f"{query} casual photo",
            f"{query} outdoors photo",
            f"{query} studio headshot",
            f"{query} official photo",
            f"{query} publicity photo",
            f"{query} red carpet 2022",
            f"{query} red carpet 2021",
            f"{query} Cannes",
            f"{query} Golden Globes",
            f"{query} Academy Awards",
            f"{query} BAFTA",
        ]

        batch_size = 50

        for q in queries:
            if len(results) >= max_results:
                break
            try:
                loop = asyncio.get_running_loop()
                search_fn = partial(self._sync_search, q, batch_size)
                batch = await loop.run_in_executor(None, search_fn)
                results.extend(batch)
                logger.info(
                    "DuckDuckGo batch '{}': {} results (total raw: {})",
                    q, len(batch), len(results),
                )
            except Exception as e:
                logger.warning("DuckDuckGo search failed for '{}': {}", q, e)

            await asyncio.sleep(self.SEARCH_DELAY)

        seen = set()
        unique = []
        for r in results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)

        logger.info("DuckDuckGo: {} total unique results for '{}'", len(unique), query)
        return unique[:max_results]

    def _sync_search(self, query: str, max_results: int) -> list[ImageResult]:
        results: list[ImageResult] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.images(query, max_results=max_results, safesearch="off"):
                    url = r.get("image", "")
                    if not url:
                        continue
                    results.append(
                        ImageResult(
                            url=url,
                            title=r.get("title", ""),
                            source_provider=self.provider_name,
                            thumbnail_url=r.get("thumbnail", ""),
                            width=r.get("width", 0),
                            height=r.get("height", 0),
                        )
                    )
        except Exception as e:
            logger.warning("DuckDuckGo sync search failed: {}", e)
        return results
