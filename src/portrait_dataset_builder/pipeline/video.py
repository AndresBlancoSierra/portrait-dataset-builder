"""Video pipeline stage — searches, downloads, and processes videos."""

from __future__ import annotations

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session, init_db
from portrait_dataset_builder.database.models import Video
from portrait_dataset_builder.database.repository import VideoRepository
from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import PluginRegistry

logger = get_logger("stage.video")


class VideoStage(PipelineStage):
    """Search and download videos from enabled video sources."""

    def __init__(self) -> None:
        super().__init__("video")
        self._identity = ""

    async def should_run(self, context: PipelineContext) -> bool:
        return len(context.settings.video.enabled_sources) > 0

    async def execute(self, context: PipelineContext) -> StageResult:
        self._identity = context.identity
        videos_dir = context.resolve_videos_dir()
        videos_dir.mkdir(parents=True, exist_ok=True)

        engine = get_engine(context.db_path)
        await init_db(engine)

        all_videos = []
        errors: list[str] = []

        for source_name in context.settings.video.enabled_sources:
            try:
                source_cls = PluginRegistry.get_video_source(source_name)
                source = source_cls()
                await source.setup()

                videos = await source.search(
                    context.identity,
                    max_results=context.settings.video.max_videos,
                )

                for video_result in videos:
                    if not self._is_relevant(video_result):
                        continue
                    if video_result.duration > context.settings.video.max_duration:
                        continue
                    if video_result.duration < context.settings.video.min_duration:
                        continue

                    async with get_session(engine) as session:
                        repo = VideoRepository(session)
                        existing = await repo.get_by_url(video_result.url)
                        if existing:
                            continue

                        video = Video(
                            url=video_result.url,
                            title=video_result.title,
                            source=source_name,
                            duration=video_result.duration,
                            pipeline_state="pending",
                        )
                        await repo.add(video)
                        all_videos.append((video, source))

                await source.teardown()
            except Exception as e:
                logger.error("Video source '{}' failed: {}", source_name, e)
                errors.append(f"{source_name}: {e}")

        downloaded = 0
        for video, source in all_videos:
            local_path = await source.download(
                video.url,
                str(videos_dir),
                quality=context.settings.video.download_quality,
            )
            if local_path:
                async with get_session(engine) as session:
                    repo = VideoRepository(session)
                    await repo.update_local_path(video.id, local_path)
                downloaded += 1

        context.metadata["video_count"] = len(all_videos)
        context.metadata["video_downloaded"] = downloaded

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(all_videos),
            items_succeeded=downloaded,
            items_failed=len(errors),
            errors=errors,
        )

    def _is_relevant(self, video_result) -> bool:
        identity_lower = self._identity.lower()
        if not identity_lower:
            return True
        title = (video_result.title or "").lower()
        desc = (video_result.description or "").lower()
        channel = (video_result.channel or "").lower()

        identity_words = identity_lower.split()
        has_name = any(w in title or w in desc or w in channel for w in identity_words)
        return has_name
