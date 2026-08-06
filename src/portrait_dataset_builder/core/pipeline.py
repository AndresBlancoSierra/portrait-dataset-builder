"""Base classes and interfaces for the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from portrait_dataset_builder.logging import get_logger

logger = get_logger("core")

T = TypeVar("T")


class StageStatus(StrEnum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result from a pipeline stage execution."""

    stage_name: str
    status: StageStatus
    items_processed: int = 0
    items_succeeded: int = 0
    items_failed: int = 0
    items_skipped: int = 0
    items_rejected: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.items_processed == 0:
            return 0.0
        return self.items_succeeded / self.items_processed


class PipelineStage(ABC):
    """Abstract base class for a pipeline stage."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    async def execute(self, context: PipelineContext) -> StageResult:
        """Execute the stage and return a result."""

    @abstractmethod
    async def should_run(self, context: PipelineContext) -> bool:
        """Return True if this stage has work to do."""

    async def setup(self, context: PipelineContext) -> None:  # noqa: B027
        """Optional setup before execution."""

    async def teardown(self, context: PipelineContext) -> None:  # noqa: B027
        """Optional cleanup after execution."""


@dataclass
class PipelineContext:
    """Shared context passed through all pipeline stages."""

    identity: str
    output_dir: Path
    db_path: Path
    settings: Any
    seed_images: list[Path] = field(default_factory=list)
    stage_results: dict[str, StageResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def set_stage_result(self, result: StageResult) -> None:
        self.stage_results[result.stage_name] = result

    def get_stage_result(self, stage_name: str) -> StageResult | None:
        return self.stage_results.get(stage_name)

    def resolve_images_dir(self) -> Path:
        return self.output_dir / "images"

    def resolve_videos_dir(self) -> Path:
        return self.output_dir / "videos"

    def resolve_frames_dir(self) -> Path:
        return self.output_dir / "frames"

    def resolve_seeds_dir(self) -> Path:
        candidate = Path(self.settings.seed_dir)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        output_seeds = self.output_dir / "seeds"
        if output_seeds.exists():
            return output_seeds
        parent_seeds = self.output_dir.parent / "seeds" / self.identity.lower().replace(" ", "_")
        if parent_seeds.exists():
            return parent_seeds
        return candidate

    @property
    def total_images_processed(self) -> int:
        return sum(r.items_processed for r in self.stage_results.values())

    @property
    def total_images_accepted(self) -> int:
        return sum(r.items_succeeded for r in self.stage_results.values())
