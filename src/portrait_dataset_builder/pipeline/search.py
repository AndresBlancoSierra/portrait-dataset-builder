"""Search pipeline stage — queries all enabled image/video sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import PluginRegistry

if TYPE_CHECKING:
    from portrait_dataset_builder.sources.image.base import ImageResult

logger = get_logger("stage.search")


class SearchStage(PipelineStage):
    """Search all enabled image sources for the identity."""

    def __init__(self) -> None:
        super().__init__("search")

    async def should_run(self, context: PipelineContext) -> bool:
        return True

    async def execute(self, context: PipelineContext) -> StageResult:
        all_results: list[ImageResult] = []
        errors: list[str] = []
        enabled = context.settings.search.enabled_sources

        _progress = context.metadata.get("_progress_task")
        total = len(enabled)

        for i, source_name in enumerate(enabled):
            if _progress:
                _progress["items_processed"] = i
                _progress["items_total"] = total
            try:
                source_cls = PluginRegistry.get_image_source(source_name)
                source = source_cls()
                await source.setup()
                results = await source.search(
                    context.identity,
                    max_results=context.settings.search.max_results_per_source,
                )
                all_results.extend(results)
                logger.info("Source {}: {} results", source_name, len(results))
                await source.teardown()
            except KeyError:
                logger.warning("Unknown source '{}', skipping", source_name)
                errors.append(f"Unknown source: {source_name}")
            except Exception as e:
                logger.error("Source '{}' failed: {}", source_name, e)
                errors.append(f"{source_name}: {e}")

        context.metadata["image_results"] = all_results

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED if not errors else StageStatus.COMPLETED,
            items_processed=len(all_results),
            items_succeeded=len(all_results),
            items_failed=len(errors),
            errors=errors,
            metadata={"total_urls": len(all_results)},
        )
