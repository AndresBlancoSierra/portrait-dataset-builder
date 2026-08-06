"""Download pipeline stage — downloads images found by the search stage."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

import httpx
from PIL import Image as PILImage

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session, init_db
from portrait_dataset_builder.database.models import Image
from portrait_dataset_builder.database.repository import ImageRepository
from portrait_dataset_builder.logging import get_logger

if TYPE_CHECKING:
    from portrait_dataset_builder.sources.image.base import ImageResult

logger = get_logger("stage.download")

MAX_RETRIES = 3


class DownloadStage(PipelineStage):
    """Download images from URLs found during the search stage."""

    def __init__(self) -> None:
        super().__init__("download")

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

        images_dir = context.resolve_images_dir()
        images_dir.mkdir(parents=True, exist_ok=True)

        engine = get_engine(context.db_path)
        await init_db(engine)

        succeeded = 0
        failed = 0
        skipped = 0
        errors: list[str] = []

        _progress = context.metadata.get("_progress_task")
        total = len(image_results)

        sem = asyncio.Semaphore(context.settings.download.max_concurrent)

        async with httpx.AsyncClient(
            timeout=context.settings.download.timeout,
            follow_redirects=True,
            headers={"User-Agent": context.settings.search.user_agent},
        ) as client:

            async def _download_one(idx: int, result: ImageResult) -> None:
                nonlocal succeeded, failed, skipped
                async with sem:
                    try:
                        content = await self._fetch_image(client, result.url)
                        if content is None:
                            skipped += 1
                            return

                        content_hash = hashlib.sha256(content).hexdigest()

                        async with get_session(engine) as session:
                            repo = ImageRepository(session)
                            if await repo.exists(content_hash):
                                skipped += 1
                                return

                        ext = self._guess_extension(result.mime_type, content)
                        local_path = images_dir / f"{content_hash}{ext}"
                        local_path.write_bytes(content)

                        try:
                            with PILImage.open(local_path) as img:
                                w, h = img.size
                        except Exception:
                            w, h = 0, 0

                        async with get_session(engine) as session:
                            repo = ImageRepository(session)
                            image = Image(
                                uri=result.url,
                                local_path=str(local_path),
                                source_type="image_search",
                                source_provider=result.source_provider,
                                content_hash=content_hash,
                                width=w,
                                height=h,
                                file_size=len(content),
                                mime_type=result.mime_type or ext.lstrip("."),
                                pipeline_state="downloaded",
                            )
                            await repo.add(image)

                        succeeded += 1

                    except Exception as e:
                        failed += 1
                        errors.append(f"{result.url}: {e}")
                    finally:
                        if _progress:
                            _progress["items_processed"] = succeeded + failed + skipped
                            _progress["items_total"] = total

            tasks = [_download_one(i, r) for i, r in enumerate(image_results)]
            await asyncio.gather(*tasks)

        logger.info(
            "Downloaded: {} succeeded, {} skipped, {} failed",
            succeeded,
            skipped,
            failed,
        )

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(image_results),
            items_succeeded=succeeded,
            items_failed=failed,
            items_skipped=skipped,
            errors=errors[:50],
        )

    async def _fetch_image(self, client: httpx.AsyncClient, url: str) -> bytes | None:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if "image" not in content_type and not any(
                    url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
                ):
                    return None
                return resp.content
            except (httpx.HTTPError, httpx.TimeoutException):
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1)
        return None

    def _guess_extension(self, mime_type: str, content: bytes) -> str:
        mime_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
        }
        if mime_type in mime_map:
            return mime_map[mime_type]

        if content[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if content[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
            return ".webp"
        return ".jpg"
