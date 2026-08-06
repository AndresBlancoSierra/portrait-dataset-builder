"""Pipeline orchestrator that runs stages sequentially with checkpointing."""

from __future__ import annotations

import time
from collections import Counter

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session, init_db
from portrait_dataset_builder.database.engine import _configure_sqlite
from portrait_dataset_builder.database.repository import (
    ClassificationRepository,
    ImageRepository,
    ProcessingLogRepository,
    QualityRepository,
)
from portrait_dataset_builder.logging import get_logger

logger = get_logger("orchestrator")


class PipelineOrchestrator:
    """Runs a sequence of pipeline stages with resumability."""

    def __init__(
        self,
        stages: list[PipelineStage],
        context: PipelineContext,
        build_id: str = "",
    ) -> None:
        self.stages = stages
        self.context = context
        self.build_id = build_id
        self._completed_stages: set[str] = set()

    async def _load_checkpoint(self) -> None:
        """Load completed stages from the processing log."""
        engine = get_engine(self.context.db_path)
        async with get_session(engine) as session:
            repo = ProcessingLogRepository(session)
            logs = await repo.get_completed_stages(self.context.identity, self.build_id)
            self._completed_stages = set(logs)
        logger.info("Loaded checkpoint: {} stages already completed", len(self._completed_stages))

    async def _save_checkpoint(self, stage_name: str, result: StageResult) -> None:
        """Save a completed stage to the processing log."""
        engine = get_engine(self.context.db_path)
        async with get_session(engine) as session:
            repo = ProcessingLogRepository(session)
            await repo.log_stage_completion(
                identity=self.context.identity,
                build_id=self.build_id,
                stage=stage_name,
                status=result.status.value,
                items_processed=result.items_processed,
                duration_ms=result.duration_ms,
            )

    async def _check_early_stop(self) -> bool:
        """Check if target images reached with sufficient coverage and quality."""
        settings = self.context.settings
        target = settings.pipeline.target_images
        early_stop_coverage = settings.pipeline.early_stop_coverage
        early_stop_quality = settings.pipeline.early_stop_quality

        engine = get_engine(self.context.db_path)
        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            q_repo = QualityRepository(session)
            cls_repo = ClassificationRepository(session)

            verified = await img_repo.get_by_state("verified", limit=100000)
            verified_count = len(verified)

            if verified_count < target:
                return False

            scores = []
            expressions = []
            angles = []
            for img in verified:
                q = await q_repo.get_by_image_id(img.id)
                if q:
                    scores.append(q.final_score)
                c = await cls_repo.get_by_image_id(img.id)
                if c:
                    if c.expression:
                        expressions.append(c.expression)
                    if c.angle:
                        angles.append(c.angle)

            avg_quality = sum(scores) / len(scores) if scores else 0

            expr_counter = Counter(expressions)
            expr_diversity = len(expr_counter) / max(1, len(expressions))
            angle_counter = Counter(angles)
            angle_diversity = len(angle_counter) / max(1, len(angles))
            coverage_score = (expr_diversity + angle_diversity) / 2

            logger.info(
                "Early stop check: {} verified (target {}), quality {:.2f}, coverage {:.2f}",
                verified_count,
                target,
                avg_quality,
                coverage_score,
            )

            if (
                verified_count >= target
                and avg_quality >= early_stop_quality
                and coverage_score >= early_stop_coverage
            ):
                logger.info(
                    "Early stop triggered: {} images >= {} target, "
                    "quality {:.2f} >= {}, coverage {:.2f} >= {}",
                    verified_count,
                    target,
                    avg_quality,
                    early_stop_quality,
                    coverage_score,
                    early_stop_coverage,
                )
                return True

            return False

    async def run(self) -> list[StageResult]:
        """Run all stages sequentially, skipping completed ones."""
        self.context.output_dir.mkdir(parents=True, exist_ok=True)

        engine = get_engine(self.context.db_path)
        await init_db(engine, self.context.db_path)
        await _configure_sqlite(self.context.db_path)

        from portrait_dataset_builder.compute import resolve_device, log_device_info

        resolved = resolve_device(self.context.settings.effective_device)
        log_device_info(resolved)

        if self.build_id and self.context.settings.pipeline.checkpoint_enabled:
            await self._load_checkpoint()

        results: list[StageResult] = []

        for stage in self.stages:
            if stage.name in self._completed_stages:
                logger.info("Skipping already completed stage: {}", stage.name)
                continue

            if not await stage.should_run(self.context):
                logger.info("Stage {} has no work, skipping", stage.name)
                continue

            logger.info("Starting stage: {}", stage.name)
            await stage.setup(self.context)

            start = time.monotonic()
            try:
                result = await stage.execute(self.context)
                result.duration_ms = (time.monotonic() - start) * 1000
            except Exception as exc:
                result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.FAILED,
                    duration_ms=(time.monotonic() - start) * 1000,
                    errors=[str(exc)],
                )
                logger.error("Stage {} failed: {}", stage.name, exc)
            finally:
                await stage.teardown(self.context)

            self.context.set_stage_result(result)
            results.append(result)

            if result.status == StageStatus.COMPLETED:
                self._completed_stages.add(stage.name)
                if self.build_id and self.context.settings.pipeline.checkpoint_enabled:
                    await self._save_checkpoint(stage.name, result)

            logger.info(
                "Stage {} completed: {}/{} items succeeded in {:.1f}s",
                stage.name,
                result.items_succeeded,
                result.items_processed,
                result.duration_ms / 1000,
            )

            if stage.name == "classification" and await self._check_early_stop():
                logger.info("Early stopping triggered after classification stage")
                break

        self._print_summary(results)
        return results

    def _print_summary(self, results: list[StageResult]) -> None:
        logger.info("=" * 60)
        logger.info("Pipeline Summary for '{}'", self.context.identity)
        logger.info("=" * 60)
        for r in results:
            status_icon = "✓" if r.status == StageStatus.COMPLETED else "✗"
            logger.info(
                "  {} {} — {}/{} items ({:.1f}s)",
                status_icon,
                r.stage_name,
                r.items_succeeded,
                r.items_processed,
                r.duration_ms / 1000,
            )
        logger.info("=" * 60)
