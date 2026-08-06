"""Persistent FIFO build queue worker."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from portrait_dataset_builder.config.settings import load_settings
from portrait_dataset_builder.database.engine import (
    _configure_sqlite,
    _engine_cache,
    _migrate_db,
    get_engine,
    get_session,
)
from portrait_dataset_builder.database.models import Base, BuildJob, Identity
from portrait_dataset_builder.database.repository import (
    BuildJobRepository,
    IdentityRepository,
    ImageRepository,
)

logger = logging.getLogger(__name__)

_data_root = Path("data")


class BuildQueueWorker:
    """Persistent FIFO build queue processor.

    Each person has their own portrait.db. This worker scans all library DBs
    to find queued jobs across the distributed databases.
    """

    def __init__(self) -> None:
        self._running = False
        self._paused = False
        self._worker_task: asyncio.Task[None] | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._current_job_id: int | None = None
        self._current_identity: str | None = None

    async def start(self) -> None:
        """Start the queue worker. Called on server startup."""
        if self._running:
            return
        self._running = True
        await self._recover_abandoned()
        self._worker_task = asyncio.create_task(self._process_loop())
        logger.info("Build queue worker started")

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Build queue worker stopped")

    # ── Library DB discovery ────────────────────────────────────────────

    def _find_library_dbs(self) -> list[tuple[str, Path]]:
        """Find all (identity, db_path) pairs under data root."""
        if not _data_root.exists():
            return []
        libs = []
        for d in sorted(_data_root.iterdir()):
            db = d / "portrait.db"
            if db.exists():
                libs.append((d.name, db))
        return libs

    def _ensure_engine(self, db_path: Path, migrate: bool = True) -> Any:
        """Get or create engine for a DB, optionally running migration."""
        if migrate:
            _migrate_db(db_path)
        return get_engine(db_path)

    # ── Queue operations ────────────────────────────────────────────────

    async def enqueue(self, identity: str) -> BuildJob:
        """Create a library and queue a build job. Returns the BuildJob."""
        root = _data_root / identity
        db_path = root / "portrait.db"

        root.mkdir(parents=True, exist_ok=True)
        _migrate_db(db_path)
        engine = get_engine(db_path)
        await _configure_sqlite(db_path)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with get_session(engine) as session:
            id_repo = IdentityRepository(session)
            bj_repo = BuildJobRepository(session)

            existing = await id_repo.get_by_name(identity)
            if not existing:
                await id_repo.get_or_create(identity, status="building")
            else:
                active = await bj_repo.get_active_by_identity(identity)
                if active:
                    return active

            position = await self._global_next_position()
            job = BuildJob(
                identity=identity,
                status="pending",
                stage_label="Queued",
                queue_status="queued",
                queue_position=position,
                queued_at=datetime.utcnow(),
            )
            job = await bj_repo.add(job)
            await session.commit()

        logger.info("Enqueued build: %s (position %s)", identity, position)
        return job

    async def _global_next_position(self) -> int:
        """Find the next queue position across ALL library DBs."""
        max_pos = 0
        for identity, db_path in self._find_library_dbs():
            engine = self._ensure_engine(db_path, migrate=False)
            async with get_session(engine) as session:
                bj_repo = BuildJobRepository(session)
                queued = await bj_repo.get_non_terminal_queue()
                for j in queued:
                    if j.queue_position and j.queue_position > max_pos:
                        max_pos = j.queue_position
        return max_pos + 1

    async def enqueue_batch(self, names: list[str]) -> dict[str, Any]:
        """Enqueue multiple names. Returns summary of what was added."""
        normalized = self._normalize_names(names)
        added: list[str] = []
        already_exists: list[dict[str, str]] = []

        for name in normalized:
            root = _data_root / name
            db_path = root / "portrait.db"

            exists_in_fs = db_path.exists()
            exists_in_queue = False

            if exists_in_fs:
                engine = self._ensure_engine(db_path, migrate=False)
                async with get_session(engine) as session:
                    bj_repo = BuildJobRepository(session)
                    active = await bj_repo.get_active_by_identity(name)
                    if active:
                        exists_in_queue = True

            if exists_in_fs and not exists_in_queue:
                engine = self._ensure_engine(db_path, migrate=False)
                async with get_session(engine) as session:
                    id_repo = IdentityRepository(session)
                    ident = await id_repo.get_by_name(name)
                    if ident and ident.status in ("building", "queued"):
                        already_exists.append({"name": name, "status": ident.status})
                        continue
                    elif ident and ident.status in ("ready", "identity_established"):
                        already_exists.append({"name": name, "status": ident.status})
                        continue

            if exists_in_queue:
                status = "building"
                if exists_in_fs:
                    engine = self._ensure_engine(db_path, migrate=False)
                    async with get_session(engine) as session:
                        id_repo = IdentityRepository(session)
                        ident = await id_repo.get_by_name(name)
                        if ident:
                            status = ident.status
                already_exists.append({"name": name, "status": status})
            else:
                await self.enqueue(name)
                added.append(name)

        return {
            "added": added,
            "already_exists": already_exists,
            "queued": len(added),
        }

    async def _find_job_engine(self, job_id: int) -> tuple[Any, BuildJob | None]:
        """Search all library DBs for a job by ID. Returns (engine, job)."""
        for identity, db_path in self._find_library_dbs():
            engine = self._ensure_engine(db_path, migrate=False)
            async with get_session(engine) as session:
                bj_repo = BuildJobRepository(session)
                job = await bj_repo.get_by_id(job_id)
                if job:
                    return engine, job
        return None, None

    async def cancel_job(self, job_id: int) -> bool:
        """Cancel a queued or running job."""
        engine, job = await self._find_job_engine(job_id)
        if not job:
            return False

        async with get_session(engine) as session:
            bj_repo = BuildJobRepository(session)
            if job.queue_status == "queued":
                await bj_repo.mark_queue_cancelled(job_id)
                await self._reorder_positions(session)
                return True

            if job.queue_status == "running" and self._current_job_id == job_id:
                if self._current_task and not self._current_task.done():
                    self._current_task.cancel()
                return True

        return False

    async def retry_job(self, job_id: int) -> bool:
        """Requeue a failed job."""
        engine, job = await self._find_job_engine(job_id)
        if not job or job.queue_status != "failed":
            return False
        if job.retry_count >= job.max_retries:
            return False

        async with get_session(engine) as session:
            bj_repo = BuildJobRepository(session)
            await bj_repo.increment_retry(job_id)
            await bj_repo.requeue(job_id)
        return True

    async def remove_job(self, job_id: int) -> bool:
        """Remove a queued (not running) job."""
        engine, job = await self._find_job_engine(job_id)
        if not job or job.queue_status != "queued":
            return False

        async with get_session(engine) as session:
            bj_repo = BuildJobRepository(session)
            await bj_repo.mark_queue_cancelled(job_id)
            await self._reorder_positions(session)
        return True

    async def pause(self) -> None:
        """After current build finishes, don't start next."""
        self._paused = True
        logger.info("Build queue paused")

    async def resume(self) -> None:
        """Resume processing the queue."""
        self._paused = False
        logger.info("Build queue resumed")

    async def get_status(self) -> dict[str, Any]:
        """Return queue overview aggregated from ALL library DBs."""
        all_jobs: list[dict[str, Any]] = []

        for identity, db_path in self._find_library_dbs():
            engine = self._ensure_engine(db_path, migrate=False)
            async with get_session(engine) as session:
                bj_repo = BuildJobRepository(session)
                jobs = await bj_repo.get_queue_overview()
                for j in jobs:
                    entry: dict[str, Any] = {
                        "id": j.id,
                        "name": j.identity,
                        "status": j.queue_status,
                        "position": j.queue_position,
                    }
                    if j.queue_status == "running":
                        entry["stage"] = j.current_stage
                        entry["stage_label"] = j.stage_label
                        entry["processed"] = j.items_processed
                        entry["total"] = j.items_total
                        entry["error"] = j.error
                    all_jobs.append(entry)

        all_jobs.sort(key=lambda x: (x.get("position") or 9999))

        return {
            "jobs": all_jobs,
            "active_job": self._current_job_id,
            "active_identity": self._current_identity,
            "queue_paused": self._paused,
            "max_concurrent": 1,
        }

    # ── Worker loop ─────────────────────────────────────────────────────

    async def _process_loop(self) -> None:
        """Main loop: scan all DBs, dequeue next, start build, wait, repeat."""
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(2)
                    continue

                if self._current_task and not self._current_task.done():
                    await asyncio.sleep(2)
                    continue

                job, engine = await self._find_next_queued()
                if not job:
                    await asyncio.sleep(2)
                    continue

                await self._start_build(job, engine)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Queue worker error")
                await asyncio.sleep(5)

    async def _find_next_queued(self) -> tuple[BuildJob | None, Any]:
        """Scan all library DBs for the next queued job by position."""
        best_job: BuildJob | None = None
        best_engine: Any = None

        for identity, db_path in self._find_library_dbs():
            engine = self._ensure_engine(db_path, migrate=False)
            async with get_session(engine) as session:
                bj_repo = BuildJobRepository(session)
                job = await bj_repo.get_next_queued()
                if job:
                    if best_job is None or (
                        (job.queue_position or 9999) < (best_job.queue_position or 9999)
                    ):
                        best_job = job
                        best_engine = engine

        return best_job, best_engine

    async def _start_build(self, job: BuildJob, engine: Any) -> None:
        """Execute a single build."""
        identity = job.identity
        job_id = job.id
        root = _data_root / identity
        db_path = root / "portrait.db"

        async with get_session(engine) as session:
            bj_repo = BuildJobRepository(session)
            await bj_repo.mark_queue_running(job_id)

        self._current_job_id = job_id
        self._current_identity = identity

        async def _run() -> None:
            try:
                await self._execute_pipeline(identity, job_id, db_path, engine)
            except asyncio.CancelledError:
                async with get_session(engine) as sess:
                    id_repo = IdentityRepository(sess)
                    ident = await id_repo.get_by_name(identity)
                    if ident:
                        ident.status = "cancelled"
                        await id_repo.update(ident)
                    bj_repo = BuildJobRepository(sess)
                    await bj_repo.mark_queue_cancelled(job_id)
            except Exception as e:
                logger.error("Build failed for %s: %s", identity, e)
                async with get_session(engine) as sess:
                    id_repo = IdentityRepository(sess)
                    ident = await id_repo.get_by_name(identity)
                    if ident:
                        ident.status = "failed"
                        await id_repo.update(ident)
                    bj_repo = BuildJobRepository(sess)
                    await bj_repo.mark_queue_failed(job_id)
            finally:
                self._current_task = None
                self._current_job_id = None
                self._current_identity = None

        self._current_task = asyncio.create_task(_run())
        await self._current_task

    async def _execute_pipeline(
        self,
        identity: str,
        job_id: int,
        db_path: Path,
        engine: Any,
    ) -> None:
        """Run the full pipeline for a single identity."""
        build_id = str(uuid.uuid4())[:8]

        async with get_session(engine) as session:
            bj_repo = BuildJobRepository(session)
            await bj_repo.update_status(job_id, "running", stage_label="Starting")

        settings = load_settings(identity)
        from portrait_dataset_builder.core.orchestrator import PipelineOrchestrator
        from portrait_dataset_builder.core.pipeline import PipelineContext
        from portrait_dataset_builder.logging import setup_logging
        from portrait_dataset_builder.pipeline.classification import ClassificationStage
        from portrait_dataset_builder.pipeline.cleanup import CleanupStage
        from portrait_dataset_builder.pipeline.download import DownloadStage
        from portrait_dataset_builder.pipeline.duplicates import DuplicateDetectionStage
        from portrait_dataset_builder.pipeline.export import ExportStage
        from portrait_dataset_builder.pipeline.face_detection import FaceDetectionStage
        from portrait_dataset_builder.pipeline.face_verification import FaceVerificationStage
        from portrait_dataset_builder.pipeline.identity_bootstrap import IdentityBootstrapStage
        from portrait_dataset_builder.pipeline.quality import QualityStage
        from portrait_dataset_builder.pipeline.safety_gate import SafetyGateStage
        from portrait_dataset_builder.pipeline.search import SearchStage
        from portrait_dataset_builder.pipeline.semantic_filter import SemanticFilterStage
        from portrait_dataset_builder.pipeline.url_safety_filter import URLSafetyFilterStage

        setup_logging(settings.log_level)

        context = PipelineContext(
            identity=identity,
            output_dir=settings.resolve_data_dir(),
            db_path=settings.resolve_db_path(),
            settings=settings,
        )

        stage_map = {
            "search": SearchStage,
            "url_safety_filter": URLSafetyFilterStage,
            "identity_bootstrap": IdentityBootstrapStage,
            "download": DownloadStage,
            "safety_gate": SafetyGateStage,
            "face_detection": FaceDetectionStage,
            "face_verification": FaceVerificationStage,
            "semantic_filter": SemanticFilterStage,
            "quality": QualityStage,
            "duplicates": DuplicateDetectionStage,
            "classification": ClassificationStage,
            "export": ExportStage,
            "cleanup": CleanupStage,
        }

        stages = []
        for stage_name in settings.pipeline.stages:
            if stage_name in stage_map:
                stages.append(stage_map[stage_name]())

        async def _update_job(**kwargs: Any) -> None:
            eng = get_engine(db_path)
            async with get_session(eng) as sess:
                r = BuildJobRepository(sess)
                await r.update_status(job_id, **kwargs)

        from portrait_dataset_builder.api.routes import STAGE_LABELS

        for stage in stages:
            orig_execute = stage.execute

            async def _patched_execute(
                ctx: PipelineContext,
                _orig: Any = orig_execute,
                _sname: str = stage.name,
            ) -> Any:
                await _update_job(
                    status="running",
                    current_stage=_sname,
                    stage_label=STAGE_LABELS.get(_sname, _sname),
                )
                result = await _orig(ctx)
                await _update_job(status="running")
                return result

            stage.execute = _patched_execute  # type: ignore

        orchestrator = PipelineOrchestrator(stages, context, build_id=build_id)
        await orchestrator.run()

        verified_count = 0
        async with get_session(engine) as sess:
            img_repo = ImageRepository(sess)
            verified = await img_repo.get_by_state("verified", limit=1)
            verified_count = len(verified)

        async with get_session(engine) as sess:
            id_repo = IdentityRepository(sess)
            bj_repo = BuildJobRepository(sess)
            ident = await id_repo.get_by_name(identity)

            if verified_count == 0:
                if ident and ident.status not in ("identity_established", "identity_unverified"):
                    ident.status = "empty"
                    await id_repo.update(ident)
                await bj_repo.update_status(
                    job_id, "completed", error=None, stage_label="No valid images"
                )
                await bj_repo.mark_queue_completed(job_id)
            else:
                if ident:
                    ident.status = "ready"
                    await id_repo.update(ident)
                await bj_repo.update_status(job_id, "completed", stage_label="Complete")
                await bj_repo.mark_queue_completed(job_id)

    # ── Helpers ─────────────────────────────────────────────────────────

    async def _recover_abandoned(self) -> None:
        """On startup: scan all DBs and mark abandoned running jobs as failed."""
        for identity, db_path in self._find_library_dbs():
            engine = self._ensure_engine(db_path, migrate=False)
            async with get_session(engine) as session:
                bj_repo = BuildJobRepository(session)
                running = await bj_repo.get_queue_running()
                if running:
                    logger.warning(
                        "Recovering abandoned build: %s (job %s)",
                        running.identity,
                        running.id,
                    )
                    await bj_repo.mark_queue_failed(running.id)
                    id_repo = IdentityRepository(session)
                    ident = await id_repo.get_by_name(running.identity)
                    if ident and ident.status == "building":
                        ident.status = "failed"
                        await id_repo.update(ident)

    async def _reorder_positions(self, session: Any) -> None:
        """Reassign queue positions after removal."""
        bj_repo = BuildJobRepository(session)
        queued = await bj_repo.get_non_terminal_queue()
        for i, job in enumerate(queued):
            if job.queue_status == "queued":
                job.queue_position = i + 1
        await session.flush()

    @staticmethod
    def _normalize_names(names: list[str]) -> list[str]:
        """Clean and deduplicate a list of names."""
        import re

        seen: set[str] = set()
        result: list[str] = []
        for raw in names:
            name = raw.strip()
            name = re.sub(r"^\d+[\.\)\-\s]+", "", name)
            name = re.sub(r"^[-•*]\s+", "", name)
            name = name.strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(name)
        return result


_worker: BuildQueueWorker | None = None


def get_queue_worker() -> BuildQueueWorker:
    global _worker
    if _worker is None:
        _worker = BuildQueueWorker()
    return _worker
