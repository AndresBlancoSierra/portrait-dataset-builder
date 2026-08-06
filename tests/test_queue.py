"""Tests for the build queue system."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from portrait_dataset_builder.core.queue_worker import BuildQueueWorker
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
)


@pytest.fixture
async def db_setup(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    _migrate_db(db_path)
    engine = get_engine(db_path)
    await _configure_sqlite(db_path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine, db_path


def _make_worker_with_data_root(data_root: Path):
    """Create a worker that scans a specific data root."""
    w = BuildQueueWorker()
    w._running = False
    return w


class TestNormalizeNames:
    def test_basic_names(self):
        result = BuildQueueWorker._normalize_names(["Brad Pitt", "Emma Stone"])
        assert result == ["Brad Pitt", "Emma Stone"]

    def test_numbered_list(self):
        result = BuildQueueWorker._normalize_names([
            "1. Brad Pitt",
            "2. Leonardo DiCaprio",
            "3. Christian Bale",
        ])
        assert result == ["Brad Pitt", "Leonardo DiCaprio", "Christian Bale"]

    def test_bullets(self):
        result = BuildQueueWorker._normalize_names([
            "- Brad Pitt",
            "• Leonardo DiCaprio",
            "* Christian Bale",
        ])
        assert result == ["Brad Pitt", "Leonardo DiCaprio", "Christian Bale"]

    def test_whitespace(self):
        result = BuildQueueWorker._normalize_names([
            "  Brad Pitt  ",
            "\tEmma Stone\t",
        ])
        assert result == ["Brad Pitt", "Emma Stone"]

    def test_empty_lines(self):
        result = BuildQueueWorker._normalize_names([
            "Brad Pitt",
            "",
            "  ",
            "Emma Stone",
        ])
        assert result == ["Brad Pitt", "Emma Stone"]

    def test_duplicates(self):
        result = BuildQueueWorker._normalize_names([
            "Brad Pitt",
            "brad pitt",
            "BRAD PITT",
            "Emma Stone",
        ])
        assert result == ["Brad Pitt", "Emma Stone"]

    def test_mixed_format(self):
        result = BuildQueueWorker._normalize_names([
            "1. Brad Pitt",
            "- Leonardo DiCaprio",
            "• Christian Bale",
            "4) Joaquin Phoenix",
            "Emma Stone",
        ])
        assert result == [
            "Brad Pitt",
            "Leonardo DiCaprio",
            "Christian Bale",
            "Joaquin Phoenix",
            "Emma Stone",
        ]


class TestQueueWorkerEnqueue:
    @pytest.mark.asyncio
    async def test_enqueue_creates_job(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()

        with patch("portrait_dataset_builder.core.queue_worker._data_root", db_path.parent):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                job = await worker.enqueue("Test Person")

                assert job.identity == "Test Person"
                assert job.queue_status == "queued"
                assert job.status == "pending"
                assert job.queue_position is not None

    @pytest.mark.asyncio
    async def test_enqueue_assigns_position(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()

        with patch("portrait_dataset_builder.core.queue_worker._data_root", db_path.parent):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                job1 = await worker.enqueue("Person A")
                job2 = await worker.enqueue("Person B")
                job3 = await worker.enqueue("Person C")

                assert job1.queue_position == 1
                assert job2.queue_position == 2
                assert job3.queue_position == 3

    @pytest.mark.asyncio
    async def test_enqueue_returns_active_if_exists(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()

        with patch("portrait_dataset_builder.core.queue_worker._data_root", db_path.parent):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                job1 = await worker.enqueue("Same Person")
                job2 = await worker.enqueue("Same Person")

                assert job1.id == job2.id


class TestQueueWorkerBatch:
    @pytest.mark.asyncio
    async def test_batch_enqueue(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()
        data_root = db_path.parent / "data"

        with patch("portrait_dataset_builder.core.queue_worker._data_root", data_root):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                result = await worker.enqueue_batch([
                    "Brad Pitt",
                    "Emma Stone",
                    "Matt Damon",
                ])

                assert result["queued"] == 3
                assert len(result["added"]) == 3
                assert len(result["already_exists"]) == 0

    @pytest.mark.asyncio
    async def test_batch_deduplicates(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()
        data_root = db_path.parent / "data"

        with patch("portrait_dataset_builder.core.queue_worker._data_root", data_root):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                result = await worker.enqueue_batch([
                    "Brad Pitt",
                    "brad pitt",
                    "Emma Stone",
                ])

                assert result["queued"] == 2
                assert len(result["added"]) == 2


class TestQueueWorkerPauseResume:
    @pytest.mark.asyncio
    async def test_pause_resume(self, db_setup):
        worker = BuildQueueWorker()
        assert worker._paused is False

        await worker.pause()
        assert worker._paused is True

        await worker.resume()
        assert worker._paused is False


class TestQueueWorkerStatus:
    @pytest.mark.asyncio
    async def test_get_status_empty_queue(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()

        with patch("portrait_dataset_builder.core.queue_worker._data_root", db_path.parent):
            status = await worker.get_status()
            assert status["jobs"] == []
            assert status["queue_paused"] is False
            assert status["max_concurrent"] == 1


def _setup_library(tmp_path: Path, name: str, engine: AsyncEngine):
    """Create a library directory structure under tmp_path."""
    lib_dir = tmp_path / name
    lib_dir.mkdir(exist_ok=True)
    db = lib_dir / "portrait.db"
    return lib_dir, db


class TestQueueWorkerCancel:
    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()
        data_root = db_path.parent

        with patch("portrait_dataset_builder.core.queue_worker._data_root", data_root):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                job = await worker.enqueue("Cancel Test")

                result = await worker.cancel_job(job.id)
                assert result is True

                async with get_session(engine) as session:
                    bj_repo = BuildJobRepository(session)
                    updated = await bj_repo.get_by_id(job.id)
                    assert updated.queue_status == "cancelled"


class TestQueueWorkerRetry:
    @pytest.mark.asyncio
    async def test_retry_failed_job(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()
        data_root = db_path.parent

        with patch("portrait_dataset_builder.core.queue_worker._data_root", data_root):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                job = await worker.enqueue("Retry Test")

        async with get_session(engine) as session:
            bj_repo = BuildJobRepository(session)
            await bj_repo.mark_queue_failed(job.id)

        with patch("portrait_dataset_builder.core.queue_worker._data_root", data_root):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                result = await worker.retry_job(job.id)
                assert result is True

        async with get_session(engine) as session:
            bj_repo = BuildJobRepository(session)
            updated = await bj_repo.get_by_id(job.id)
            assert updated.queue_status == "queued"
            assert updated.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_non_failed_job_fails(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()
        data_root = db_path.parent

        with patch("portrait_dataset_builder.core.queue_worker._data_root", data_root):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                job = await worker.enqueue("No Retry Test")

                result = await worker.retry_job(job.id)
                assert result is False


class TestQueueWorkerRemove:
    @pytest.mark.asyncio
    async def test_remove_queued_job(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()
        data_root = db_path.parent

        with patch("portrait_dataset_builder.core.queue_worker._data_root", data_root):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                job = await worker.enqueue("Remove Test")

                result = await worker.remove_job(job.id)
                assert result is True

                async with get_session(engine) as session:
                    bj_repo = BuildJobRepository(session)
                    updated = await bj_repo.get_by_id(job.id)
                    assert updated.queue_status == "cancelled"


class TestRecoveryAbandoned:
    @pytest.mark.asyncio
    async def test_recover_running_job(self, db_setup):
        engine, db_path = db_setup
        worker = BuildQueueWorker()
        data_root = db_path.parent

        with patch("portrait_dataset_builder.core.queue_worker._data_root", data_root):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                job = await worker.enqueue("Recover Test")

        async with get_session(engine) as session:
            bj_repo = BuildJobRepository(session)
            job_ref = await bj_repo.get_by_id(job.id)
            job_ref.queue_status = "running"
            job_ref.status = "running"
            await session.commit()

            id_repo = IdentityRepository(session)
            await id_repo.get_or_create("Recover Test", status="building")
            await session.commit()

        with patch("portrait_dataset_builder.core.queue_worker._data_root", data_root):
            with patch("portrait_dataset_builder.core.queue_worker.get_engine", return_value=engine):
                await worker._recover_abandoned()

        async with get_session(engine) as session:
            bj_repo = BuildJobRepository(session)
            updated = await bj_repo.get_by_id(job.id)
            assert updated.queue_status == "failed"


class TestMigrationV4:
    def test_migration_adds_queue_columns(self, tmp_path):
        db_path = tmp_path / "test.db"
        _migrate_db(db_path)

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(build_jobs)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "queue_status" in cols
        assert "queue_position" in cols
        assert "queued_at" in cols
        assert "retry_count" in cols
        assert "max_retries" in cols
