from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from portrait_dataset_builder.database.models import Base

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_engine_cache: dict[str, AsyncEngine] = {}

CURRENT_SCHEMA_VERSION = 4


def get_engine(db_path: Path) -> AsyncEngine:
    key = str(db_path.resolve())
    if key not in _engine_cache:
        _engine_cache[key] = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
        )
    return _engine_cache[key]


async def _configure_sqlite(db_path: Path) -> None:
    """Enable WAL mode and busy timeout for the SQLite database."""
    engine = get_engine(db_path)
    async with engine.begin() as conn:
        await conn.execute(__import__("sqlalchemy").text("PRAGMA journal_mode=WAL"))
        await conn.execute(__import__("sqlalchemy").text("PRAGMA busy_timeout=30000"))


def dispose_engine(db_path: Path) -> None:
    key = str(db_path.resolve())
    _engine_cache.pop(key, None)


async def dispose_all_engines() -> None:
    engines = list(_engine_cache.values())
    _engine_cache.clear()
    for engine in engines:
        await engine.dispose()


def _get_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def get_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    factory = _get_session_factory(engine)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _get_schema_version(conn: sqlite3.Connection) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
    if not cursor.fetchone():
        return 0
    cursor.execute("SELECT MAX(version) FROM schema_version")
    row = cursor.fetchone()
    return row[0] if row and row[0] else 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, CURRENT_TIMESTAMP)",
        (version,),
    )


def _migrate_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        current = _get_schema_version(conn)

        if current < 1:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='identities'"
            )
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(identities)")
                cols = {row[1] for row in cursor.fetchall()}
                if "status" not in cols:
                    cursor.execute(
                        "ALTER TABLE identities ADD COLUMN status VARCHAR NOT NULL DEFAULT 'ready'"
                    )
                if "updated_at" not in cols:
                    cursor.execute("ALTER TABLE identities ADD COLUMN updated_at DATETIME")

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='build_jobs'"
            )
            if not cursor.fetchone():
                cursor.execute(
                    "CREATE TABLE build_jobs ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  identity VARCHAR NOT NULL,"
                    "  status VARCHAR NOT NULL DEFAULT 'pending',"
                    "  current_stage VARCHAR,"
                    "  stage_label VARCHAR,"
                    "  items_processed INTEGER NOT NULL DEFAULT 0,"
                    "  items_total INTEGER NOT NULL DEFAULT 0,"
                    "  error TEXT,"
                    "  started_at DATETIME,"
                    "  completed_at DATETIME,"
                    "  created_at DATETIME"
                    " NOT NULL DEFAULT CURRENT_TIMESTAMP,"
                    "  updated_at DATETIME"
                    " NOT NULL DEFAULT CURRENT_TIMESTAMP"
                    ")"
                )
                cursor.execute("CREATE INDEX idx_build_jobs_identity ON build_jobs(identity)")

        if current < 2:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if not cursor.fetchone():
                cursor.execute(
                    "CREATE TABLE schema_version ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  version INTEGER NOT NULL,"
                    "  applied_at DATETIME"
                    " NOT NULL DEFAULT CURRENT_TIMESTAMP"
                    ")"
                )

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='processing_logs'"
            )
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(processing_logs)")
                cols = {row[1] for row in cursor.fetchall()}
                if "build_id" not in cols:
                    cursor.execute(
                        "ALTER TABLE processing_logs"
                        " ADD COLUMN build_id VARCHAR NOT NULL"
                        " DEFAULT ''"
                    )
                    cursor.execute(
                        "CREATE INDEX idx_processing_logs_build_id ON processing_logs(build_id)"
                    )

        if current < 3:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='classifications'"
            )
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(classifications)")
                cols = {row[1] for row in cursor.fetchall()}
                if "horizontal_pose" not in cols:
                    cursor.execute(
                        "ALTER TABLE classifications ADD COLUMN horizontal_pose VARCHAR"
                    )
                if "vertical_pose" not in cols:
                    cursor.execute(
                        "ALTER TABLE classifications ADD COLUMN vertical_pose VARCHAR"
                    )

        if current < 4:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='build_jobs'"
            )
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(build_jobs)")
                cols = {row[1] for row in cursor.fetchall()}
                if "queue_status" not in cols:
                    cursor.execute(
                        "ALTER TABLE build_jobs ADD COLUMN queue_status VARCHAR NOT NULL"
                        " DEFAULT 'queued'"
                    )
                if "queue_position" not in cols:
                    cursor.execute(
                        "ALTER TABLE build_jobs ADD COLUMN queue_position INTEGER"
                    )
                if "queued_at" not in cols:
                    cursor.execute(
                        "ALTER TABLE build_jobs ADD COLUMN queued_at DATETIME"
                    )
                if "retry_count" not in cols:
                    cursor.execute(
                        "ALTER TABLE build_jobs ADD COLUMN retry_count INTEGER NOT NULL"
                        " DEFAULT 0"
                    )
                if "max_retries" not in cols:
                    cursor.execute(
                        "ALTER TABLE build_jobs ADD COLUMN max_retries INTEGER NOT NULL"
                        " DEFAULT 3"
                    )
                cursor.execute(
                    "UPDATE build_jobs SET queue_status = 'completed'"
                    " WHERE status IN ('completed', 'failed', 'cancelled')"
                )
                cursor.execute(
                    "UPDATE build_jobs SET queue_status = 'running',"
                    " queued_at = created_at WHERE status = 'running'"
                )
                cursor.execute(
                    "UPDATE build_jobs SET queue_status = 'queued',"
                    " queued_at = created_at WHERE status = 'pending'"
                )

        _set_schema_version(conn, CURRENT_SCHEMA_VERSION)
        conn.commit()
    finally:
        conn.close()


async def init_db(engine: AsyncEngine, db_path: Path | None = None) -> None:
    if db_path:
        _migrate_db(db_path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
