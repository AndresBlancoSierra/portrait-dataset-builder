"""Test pipeline core components."""

from __future__ import annotations

import pytest

from portrait_dataset_builder.config.settings import Settings
from portrait_dataset_builder.core.orchestrator import PipelineOrchestrator
from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)


class TestStageStatus:
    def test_values(self) -> None:
        assert StageStatus.PENDING == "pending"
        assert StageStatus.RUNNING == "running"
        assert StageStatus.COMPLETED == "completed"
        assert StageStatus.FAILED == "failed"
        assert StageStatus.SKIPPED == "skipped"


class TestStageResult:
    def test_success_rate_empty(self) -> None:
        result = StageResult(stage_name="test", status=StageStatus.COMPLETED)
        assert result.success_rate == 0.0

    def test_success_rate(self) -> None:
        result = StageResult(
            stage_name="test",
            status=StageStatus.COMPLETED,
            items_processed=10,
            items_succeeded=7,
        )
        assert result.success_rate == 0.7

    def test_default_values(self) -> None:
        result = StageResult(stage_name="test", status=StageStatus.COMPLETED)
        assert result.items_processed == 0
        assert result.items_succeeded == 0
        assert result.errors == []
        assert result.metadata == {}


class MockStage(PipelineStage):
    def __init__(self, name: str, should_run: bool = True) -> None:
        super().__init__(name)
        self._should_run = should_run
        self.executed = False

    async def should_run(self, context: PipelineContext) -> bool:
        return self._should_run

    async def execute(self, context: PipelineContext) -> StageResult:
        self.executed = True
        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=10,
            items_succeeded=8,
        )


class FailingStage(PipelineStage):
    def __init__(self) -> None:
        super().__init__("failing")

    async def should_run(self, context: PipelineContext) -> bool:
        return True

    async def execute(self, context: PipelineContext) -> StageResult:
        raise ValueError("Test error")


class TestPipelineContext:
    def test_set_get_stage_result(self) -> None:
        settings = Settings(identity="test")
        from pathlib import Path

        context = PipelineContext(
            identity="test",
            output_dir=Path("/tmp/test"),
            db_path=Path("/tmp/test/test.db"),
            settings=settings,
        )

        result = StageResult(
            stage_name="search",
            status=StageStatus.COMPLETED,
            items_processed=10,
            items_succeeded=8,
        )
        context.set_stage_result(result)

        found = context.get_stage_result("search")
        assert found is not None
        assert found.items_succeeded == 8

    def test_get_nonexistent_result(self) -> None:
        settings = Settings(identity="test")
        from pathlib import Path

        context = PipelineContext(
            identity="test",
            output_dir=Path("/tmp/test"),
            db_path=Path("/tmp/test/test.db"),
            settings=settings,
        )
        assert context.get_stage_result("nonexistent") is None

    def test_total_images_processed(self) -> None:
        settings = Settings(identity="test")
        from pathlib import Path

        context = PipelineContext(
            identity="test",
            output_dir=Path("/tmp/test"),
            db_path=Path("/tmp/test/test.db"),
            settings=settings,
        )
        context.set_stage_result(
            StageResult(stage_name="a", status=StageStatus.COMPLETED, items_processed=5)
        )
        context.set_stage_result(
            StageResult(stage_name="b", status=StageStatus.COMPLETED, items_processed=10)
        )
        assert context.total_images_processed == 15


class TestPipelineOrchestrator:
    @pytest.mark.asyncio
    async def test_runs_stages(self) -> None:
        settings = Settings(identity="test")
        settings.pipeline.checkpoint_enabled = False
        from pathlib import Path

        context = PipelineContext(
            identity="test",
            output_dir=Path("/tmp/test_orch"),
            db_path=Path("/tmp/test_orch/test.db"),
            settings=settings,
        )

        stage = MockStage("mock")
        orchestrator = PipelineOrchestrator([stage], context)
        results = await orchestrator.run()

        assert len(results) == 1
        assert results[0].status == StageStatus.COMPLETED
        assert stage.executed is True

    @pytest.mark.asyncio
    async def test_skips_when_should_not_run(self) -> None:
        settings = Settings(identity="test")
        settings.pipeline.checkpoint_enabled = False
        from pathlib import Path

        context = PipelineContext(
            identity="test",
            output_dir=Path("/tmp/test_orch2"),
            db_path=Path("/tmp/test_orch2/test.db"),
            settings=settings,
        )

        stage = MockStage("skipped", should_run=False)
        orchestrator = PipelineOrchestrator([stage], context)
        results = await orchestrator.run()

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_handles_stage_failure(self) -> None:
        settings = Settings(identity="test")
        settings.pipeline.checkpoint_enabled = False
        from pathlib import Path

        context = PipelineContext(
            identity="test",
            output_dir=Path("/tmp/test_orch3"),
            db_path=Path("/tmp/test_orch3/test.db"),
            settings=settings,
        )

        stage = FailingStage()
        orchestrator = PipelineOrchestrator([stage], context)
        results = await orchestrator.run()

        assert len(results) == 1
        assert results[0].status == StageStatus.FAILED
        assert len(results[0].errors) > 0
