"""Cleanup pipeline stage — removes rejected/duplicate/no_face image files from disk."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session
from portrait_dataset_builder.database.models import Image
from portrait_dataset_builder.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = get_logger("stage.cleanup")

TERMINAL_FAILED_STATES = ["rejected", "duplicate", "no_face"]


class CleanupStage(PipelineStage):
    """Delete image files from disk that are in terminal failed states."""

    def __init__(self) -> None:
        super().__init__("cleanup")

    async def should_run(self, context: PipelineContext) -> bool:
        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            result = await session.execute(
                select(Image.id)
                .where(Image.pipeline_state.in_(TERMINAL_FAILED_STATES))
                .limit(1)
            )
            return result.first() is not None

    async def execute(self, context: PipelineContext) -> StageResult:
        engine = get_engine(context.db_path)
        images_dir = context.resolve_images_dir()

        if not images_dir.exists():
            logger.info("Images directory does not exist: {}", images_dir)
            return StageResult(
                stage_name=self.name,
                status=StageStatus.COMPLETED,
                items_processed=0,
            )

        async with get_session(engine) as session:
            result = await session.execute(
                select(Image)
                .where(Image.pipeline_state.in_(TERMINAL_FAILED_STATES))
            )
            failed_images: list[Image] = list(result.scalars().all())

        total = len(failed_images)
        deleted_count = 0
        skipped_count = 0

        for img in failed_images:
            if not img.local_path:
                skipped_count += 1
                continue

            path = Path(img.local_path)
            if path.exists():
                try:
                    path.unlink()
                    deleted_count += 1
                except OSError as e:
                    logger.warning("Failed to delete {}: {}", path, e)
                    skipped_count += 1
            else:
                skipped_count += 1

        logger.info(
            "Cleanup: {} files deleted, {} skipped (no path / not found) out of {} failed images",
            deleted_count,
            skipped_count,
            total,
        )

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=total,
            items_succeeded=deleted_count,
            items_failed=total - deleted_count - skipped_count,
            items_skipped=skipped_count,
        )
