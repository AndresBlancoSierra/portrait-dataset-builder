"""URL Safety Filter pipeline stage — keyword blocking + source trust scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.logging import get_logger

if TYPE_CHECKING:
    from portrait_dataset_builder.sources.image.base import ImageResult

logger = get_logger("stage.url_safety_filter")

BLOCKED_PATTERNS: list[str] = [
    "porn", "xxx", "sex", "nude", "naked", "erotic",
    "onlyfans", "nsfw", "adult", "playboy", "penthouse",
]


def get_trust_score(url: str, provider: str, trust_map: dict[str, float]) -> float:
    """Compute a source trust score from the URL domain and provider name."""
    url_lower = url.lower()
    for key in ("official", "editorial", "wikimedia", "wikipedia", "flickr", "imdb"):
        if key in url_lower or key in provider.lower():
            return trust_map.get(key, 0.5)
    if "duckduckgo" in provider.lower():
        return trust_map.get("duckduckgo", 0.5)
    if any(d in url_lower for d in (".gov", ".edu", "bbc.com", "reuters.com")):
        return 1.0
    return trust_map.get("unknown", 0.3)


class URLSafetyFilterStage(PipelineStage):
    """Filter image results before download using URL keyword blocking and source trust scoring."""

    def __init__(self) -> None:
        super().__init__("url_safety_filter")

    async def should_run(self, context: PipelineContext) -> bool:
        results = context.metadata.get("image_results", [])
        return len(results) > 0

    async def execute(self, context: PipelineContext) -> StageResult:
        image_results: list[ImageResult] = context.metadata.get("image_results", [])
        if not image_results:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.SKIPPED,
            )

        blocked_keywords = list(context.settings.safety.blocked_url_keywords)
        trust_map = dict(context.settings.safety.source_trust_scores)

        allowed: list[ImageResult] = []
        blocked_count = 0
        low_trust_count = 0
        blocked_keywords_found: list[str] = []

        for result in image_results:
            url_lower = result.url.lower()
            is_blocked = any(kw in url_lower for kw in blocked_keywords)
            if is_blocked:
                blocked_count += 1
                blocked_keywords_found.append(result.url[:80])
                continue

            trust_score = get_trust_score(result.url, result.source_provider, trust_map)
            result.metadata["source_trust_score"] = trust_score
            if trust_score < 0.3:
                low_trust_count += 1
                continue

            allowed.append(result)

        context.metadata["image_results"] = allowed

        logger.info(
            "URL Safety: {} allowed, {} blocked by keywords, {} low trust",
            len(allowed), blocked_count, low_trust_count,
        )

        if blocked_keywords_found:
            logger.debug("Sample blocked URLs: {}", blocked_keywords_found[:5])

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(image_results),
            items_succeeded=len(allowed),
            items_rejected=blocked_count + low_trust_count,
            metadata={
                "blocked_by_keywords": blocked_count,
                "low_trust": low_trust_count,
                "allowed": len(allowed),
            },
        )
