"""REST API for WHO? frontend."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from portrait_dataset_builder.api.routes import _build_tasks, router
from portrait_dataset_builder.database import dispose_all_engines

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()
    await worker.start()
    yield
    await worker.stop()
    for task in _build_tasks.values():
        task.cancel()
    if _build_tasks:
        await asyncio.gather(*_build_tasks.values(), return_exceptions=True)
    await dispose_all_engines()
    logger.info("Server shutdown: engines disposed, tasks cancelled")


def create_app() -> FastAPI:
    app = FastAPI(title="WHO?", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")
    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
