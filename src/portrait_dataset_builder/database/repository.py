from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from portrait_dataset_builder.database.models import (
    BuildJob,
    Classification,
    EmbeddingIndex,
    Face,
    Frame,
    Identity,
    IdentityImage,
    Image,
    ProcessingLog,
    QualityScore,
    ReviewQueue,
    SafetyScore,
    Video,
)


class ImageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, image: Image) -> Image:
        self.session.add(image)
        await self.session.flush()
        return image

    async def get_by_id(self, entity_id: int) -> Image | None:
        return await self.session.get(Image, entity_id)

    async def get_by_hash(self, content_hash: str) -> Image | None:
        result = await self.session.execute(select(Image).where(Image.content_hash == content_hash))
        return result.scalar_one_or_none()

    async def exists(self, content_hash: str) -> bool:
        result = await self.session.execute(
            select(func.count()).where(Image.content_hash == content_hash)
        )
        return result.scalar_one() > 0

    async def update_state(self, image_id: int, state: str) -> None:
        image = await self.session.get(Image, image_id)
        if image:
            image.pipeline_state = state
            await self.session.flush()

    async def get_pending(self, limit: int = 100) -> list[Image]:
        result = await self.session.execute(
            select(Image).where(Image.pipeline_state == "pending").limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_state(self, state: str, limit: int = 100) -> list[Image]:
        result = await self.session.execute(
            select(Image).where(Image.pipeline_state == state).limit(limit)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Image))
        return result.scalar_one()

    async def count_by_state(self, state: str) -> int:
        result = await self.session.execute(
            select(func.count()).where(Image.pipeline_state == state)
        )
        return result.scalar_one()

    async def get_random_by_state(self, state: str, limit: int = 10) -> list[Image]:
        result = await self.session.execute(
            select(Image)
            .where(Image.pipeline_state == state)
            .order_by(func.random())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete(self, entity_id: int) -> bool:
        image = await self.session.get(Image, entity_id)
        if image:
            await self.session.delete(image)
            await self.session.flush()
            return True
        return False


class FaceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, face: Face) -> Face:
        self.session.add(face)
        await self.session.flush()
        return face

    async def get_by_id(self, entity_id: int) -> Face | None:
        return await self.session.get(Face, entity_id)

    async def get_by_image_id(self, image_id: int) -> list[Face]:
        result = await self.session.execute(select(Face).where(Face.image_id == image_id))
        return list(result.scalars().all())

    async def get_by_identity_id(self, identity_id: int) -> list[Face]:
        result = await self.session.execute(
            select(Face)
            .join(IdentityImage, Face.image_id == IdentityImage.image_id)
            .where(IdentityImage.identity_id == identity_id)
        )
        return list(result.scalars().all())

    async def delete(self, entity_id: int) -> bool:
        face = await self.session.get(Face, entity_id)
        if face:
            await self.session.delete(face)
            await self.session.flush()
            return True
        return False

    async def get_image_ids_with_faces(self, image_ids: list[int]) -> list[int]:
        if not image_ids:
            return []
        result = await self.session.execute(
            select(Face.image_id).where(Face.image_id.in_(image_ids)).distinct()
        )
        return [row[0] for row in result.all()]

    async def get_best_by_image_ids(self, image_ids: list[int]) -> dict[int, Face]:
        """Batch fetch the highest-confidence face per image_id.

        Returns {image_id: Face} with only the best face per image.
        """
        if not image_ids:
            return {}
        result = await self.session.execute(
            select(Face).where(Face.image_id.in_(image_ids))
        )
        all_faces = list(result.scalars().all())
        best_by_image: dict[int, Face] = {}
        for face in all_faces:
            existing = best_by_image.get(face.image_id)
            if existing is None or (face.confidence or 0) > (existing.confidence or 0):
                best_by_image[face.image_id] = face
        return best_by_image

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Face))
        return result.scalar_one()


class IdentityRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, identity: Identity) -> Identity:
        self.session.add(identity)
        await self.session.flush()
        return identity

    async def get_by_id(self, entity_id: int) -> Identity | None:
        return await self.session.get(Identity, entity_id)

    async def get_by_name(self, name: str) -> Identity | None:
        result = await self.session.execute(select(Identity).where(Identity.name == name))
        return result.scalar_one_or_none()

    async def get_or_create(self, name: str, status: str = "building") -> Identity:
        identity = await self.get_by_name(name)
        if identity is None:
            identity = Identity(name=name, status=status)
            await self.add(identity)
        return identity

    async def update_status(self, name: str, status: str) -> None:
        identity = await self.get_by_name(name)
        if identity:
            identity.status = status
            await self.session.flush()

    async def list_all(self, limit: int = 100) -> list[Identity]:
        result = await self.session.execute(select(Identity).limit(limit))
        return list(result.scalars().all())

    async def update(self, identity: Identity) -> Identity:
        self.session.add(identity)
        await self.session.flush()
        return identity

    async def delete(self, entity_id: int) -> bool:
        identity = await self.session.get(Identity, entity_id)
        if identity:
            await self.session.delete(identity)
            await self.session.flush()
            return True
        return False

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Identity))
        return result.scalar_one()


class VideoRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, video: Video) -> Video:
        self.session.add(video)
        await self.session.flush()
        return video

    async def get_by_id(self, entity_id: int) -> Video | None:
        return await self.session.get(Video, entity_id)

    async def get_by_url(self, url: str) -> Video | None:
        result = await self.session.execute(select(Video).where(Video.url == url))
        return result.scalar_one_or_none()

    async def get_pending(self, limit: int = 100) -> list[Video]:
        result = await self.session.execute(
            select(Video).where(Video.pipeline_state == "pending").limit(limit)
        )
        return list(result.scalars().all())

    async def get_downloaded(self) -> list[Video]:
        result = await self.session.execute(
            select(Video).where(Video.pipeline_state == "downloaded")
        )
        return list(result.scalars().all())

    async def update_state(self, video_id: int, state: str) -> None:
        video = await self.session.get(Video, video_id)
        if video:
            video.pipeline_state = state
            await self.session.flush()

    async def update_local_path(self, video_id: int, local_path: str) -> None:
        video = await self.session.get(Video, video_id)
        if video:
            video.local_path = local_path
            video.pipeline_state = "downloaded"
            await self.session.flush()

    async def list_all(self, limit: int = 100) -> list[Video]:
        result = await self.session.execute(select(Video).limit(limit))
        return list(result.scalars().all())

    async def delete(self, entity_id: int) -> bool:
        video = await self.session.get(Video, entity_id)
        if video:
            await self.session.delete(video)
            await self.session.flush()
            return True
        return False

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Video))
        return result.scalar_one()


class QualityRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, score: QualityScore) -> QualityScore:
        self.session.add(score)
        await self.session.flush()
        return score

    async def get_by_id(self, entity_id: int) -> QualityScore | None:
        return await self.session.get(QualityScore, entity_id)

    async def get_by_image_id(self, image_id: int) -> QualityScore | None:
        result = await self.session.execute(
            select(QualityScore).where(QualityScore.image_id == image_id)
        )
        return result.scalars().first()

    async def update(self, score: QualityScore) -> QualityScore:
        self.session.add(score)
        await self.session.flush()
        return score

    async def delete(self, entity_id: int) -> bool:
        score = await self.session.get(QualityScore, entity_id)
        if score:
            await self.session.delete(score)
            await self.session.flush()
            return True
        return False

    async def get_by_image_ids(self, image_ids: list[int]) -> dict[int, QualityScore]:
        """Batch fetch quality scores for multiple images.

        Returns {image_id: QualityScore}.
        """
        if not image_ids:
            return {}
        result = await self.session.execute(
            select(QualityScore).where(QualityScore.image_id.in_(image_ids))
        )
        return {q.image_id: q for q in result.scalars().all()}

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(QualityScore))
        return result.scalar_one()


class ClassificationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, classification: Classification) -> Classification:
        self.session.add(classification)
        await self.session.flush()
        return classification

    async def get_by_id(self, entity_id: int) -> Classification | None:
        return await self.session.get(Classification, entity_id)

    async def get_by_image_id(self, image_id: int) -> Classification | None:
        result = await self.session.execute(
            select(Classification).where(Classification.image_id == image_id)
        )
        return result.scalars().first()

    async def update(self, classification: Classification) -> Classification:
        self.session.add(classification)
        await self.session.flush()
        return classification

    async def delete(self, entity_id: int) -> bool:
        classification = await self.session.get(Classification, entity_id)
        if classification:
            await self.session.delete(classification)
            await self.session.flush()
            return True
        return False

    async def get_by_image_ids(self, image_ids: list[int]) -> dict[int, Classification]:
        """Batch fetch classifications for multiple images.

        Returns {image_id: Classification}.
        """
        if not image_ids:
            return {}
        result = await self.session.execute(
            select(Classification).where(Classification.image_id.in_(image_ids))
        )
        return {c.image_id: c for c in result.scalars().all()}

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Classification))
        return result.scalar_one()


class ProcessingLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, log: ProcessingLog) -> ProcessingLog:
        self.session.add(log)
        await self.session.flush()
        return log

    async def get_by_id(self, entity_id: int) -> ProcessingLog | None:
        return await self.session.get(ProcessingLog, entity_id)

    async def get_by_entity(self, entity_type: str, entity_id: int) -> list[ProcessingLog]:
        result = await self.session.execute(
            select(ProcessingLog)
            .where(
                ProcessingLog.entity_type == entity_type,
                ProcessingLog.entity_id == entity_id,
            )
            .order_by(ProcessingLog.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_stage(self, stage: str, limit: int = 100) -> list[ProcessingLog]:
        result = await self.session.execute(
            select(ProcessingLog)
            .where(ProcessingLog.stage == stage)
            .order_by(ProcessingLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete(self, entity_id: int) -> bool:
        log = await self.session.get(ProcessingLog, entity_id)
        if log:
            await self.session.delete(log)
            await self.session.flush()
            return True
        return False

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(ProcessingLog))
        return result.scalar_one()

    async def get_completed_stages(self, identity: str, build_id: str) -> list[str]:
        result = await self.session.execute(
            select(ProcessingLog.stage)
            .where(
                ProcessingLog.entity_type == "pipeline",
                ProcessingLog.build_id == build_id,
                ProcessingLog.status == "completed",
            )
            .distinct()
        )
        return [row[0] for row in result.all()]

    async def log_stage_completion(
        self,
        identity: str,
        build_id: str,
        stage: str,
        status: str,
        items_processed: int = 0,
        duration_ms: float = 0.0,
    ) -> ProcessingLog:
        log = ProcessingLog(
            entity_type="pipeline",
            entity_id=0,
            build_id=build_id,
            stage=stage,
            status=status,
            message=f"Processed {items_processed} items in {duration_ms:.0f}ms",
            duration_ms=duration_ms,
        )
        self.session.add(log)
        await self.session.flush()
        return log


class IdentityImageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, entry: IdentityImage) -> IdentityImage:
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_by_identity(self, identity_id: int) -> list[IdentityImage]:
        result = await self.session.execute(
            select(IdentityImage).where(IdentityImage.identity_id == identity_id)
        )
        return list(result.scalars().all())

    async def get_by_image(self, image_id: int) -> list[IdentityImage]:
        result = await self.session.execute(
            select(IdentityImage).where(IdentityImage.image_id == image_id)
        )
        return list(result.scalars().all())

    async def delete(self, identity_id: int, image_id: int) -> bool:
        result = await self.session.execute(
            select(IdentityImage).where(
                IdentityImage.identity_id == identity_id,
                IdentityImage.image_id == image_id,
            )
        )
        entry = result.scalar_one_or_none()
        if entry:
            await self.session.delete(entry)
            await self.session.flush()
            return True
        return False


class FrameRepository:
    def __init__(self, session: AsyncSession | None) -> None:
        self.session = session

    async def add(self, frame: Frame) -> Frame:
        self.session.add(frame)
        await self.session.flush()
        return frame

    async def get_by_id(self, entity_id: int) -> Frame | None:
        return await self.session.get(Frame, entity_id)

    async def get_by_video_id(self, video_id: int) -> list[Frame]:
        result = await self.session.execute(select(Frame).where(Frame.video_id == video_id))
        return list(result.scalars().all())

    async def count_pending(self) -> int:
        result = await self.session.execute(
            select(func.count()).where(Frame.selected == True)  # noqa: E712
        )
        return result.scalar_one()

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Frame))
        return result.scalar_one()


class ReviewQueueRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, entry: ReviewQueue) -> ReviewQueue:
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_pending(self, limit: int = 100) -> list[ReviewQueue]:
        result = await self.session.execute(
            select(ReviewQueue)
            .where(ReviewQueue.reviewed == False)  # noqa: E712
            .order_by(ReviewQueue.variance.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_id(self, entity_id: int) -> ReviewQueue | None:
        return await self.session.get(ReviewQueue, entity_id)

    async def mark_reviewed(self, entry_id: int, answer: str) -> None:
        entry = await self.session.get(ReviewQueue, entry_id)
        if entry:
            entry.reviewed = True
            entry.user_answer = answer
            await self.session.flush()

    async def count_pending(self) -> int:
        result = await self.session.execute(
            select(func.count()).where(ReviewQueue.reviewed == False)  # noqa: E712
        )
        return result.scalar_one()

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(ReviewQueue))
        return result.scalar_one()


class EmbeddingIndexRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, entry: EmbeddingIndex) -> EmbeddingIndex:
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_by_face_id(self, face_id: int) -> EmbeddingIndex | None:
        result = await self.session.execute(
            select(EmbeddingIndex).where(EmbeddingIndex.face_id == face_id)
        )
        return result.scalar_one_or_none()

    async def list_all(self, limit: int = 10000) -> list[EmbeddingIndex]:
        result = await self.session.execute(select(EmbeddingIndex).limit(limit))
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(EmbeddingIndex))
        return result.scalar_one()


class SafetyScoreRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def table_exists(self) -> bool:
        try:
            result = await self.session.execute(select(func.count()).select_from(SafetyScore))
            result.scalar_one()
            return True
        except Exception:
            return False

    async def add(self, score: SafetyScore) -> SafetyScore:
        self.session.add(score)
        await self.session.flush()
        return score

    async def get_by_image_id(self, image_id: int) -> SafetyScore | None:
        result = await self.session.execute(
            select(SafetyScore).where(SafetyScore.image_id == image_id)
        )
        return result.scalars().first()

    async def update(self, score: SafetyScore) -> SafetyScore:
        self.session.add(score)
        await self.session.flush()
        return score

    async def get_rejected(self, limit: int = 100) -> list[SafetyScore]:
        result = await self.session.execute(
            select(SafetyScore)
            .where(SafetyScore.rejection_reason.isnot(None))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_image_ids(self, image_ids: list[int]) -> dict[int, SafetyScore]:
        """Batch fetch safety scores for multiple images.

        Returns {image_id: SafetyScore}.
        """
        if not image_ids:
            return {}
        result = await self.session.execute(
            select(SafetyScore).where(SafetyScore.image_id.in_(image_ids))
        )
        return {s.image_id: s for s in result.scalars().all()}

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(SafetyScore))
        return result.scalar_one()


class BuildJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, job: BuildJob) -> BuildJob:
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_by_id(self, job_id: int) -> BuildJob | None:
        return await self.session.get(BuildJob, job_id)

    async def get_active_by_identity(self, identity: str) -> BuildJob | None:
        result = await self.session.execute(
            select(BuildJob)
            .where(
                BuildJob.identity == identity,
                BuildJob.status.in_(["pending", "running"]),
            )
            .order_by(BuildJob.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_latest_by_identity(self, identity: str) -> BuildJob | None:
        result = await self.session.execute(
            select(BuildJob)
            .where(BuildJob.identity == identity)
            .order_by(BuildJob.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_active(self) -> list[BuildJob]:
        result = await self.session.execute(
            select(BuildJob)
            .where(BuildJob.status.in_(["pending", "running"]))
            .order_by(BuildJob.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_all(self, limit: int = 50) -> list[BuildJob]:
        result = await self.session.execute(
            select(BuildJob).order_by(BuildJob.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        job_id: int,
        status: str,
        current_stage: str | None = None,
        stage_label: str | None = None,
        items_processed: int | None = None,
        items_total: int | None = None,
        error: str | None = None,
    ) -> None:
        job = await self.session.get(BuildJob, job_id)
        if not job:
            return
        job.status = status
        if current_stage is not None:
            job.current_stage = current_stage
        if stage_label is not None:
            job.stage_label = stage_label
        if items_processed is not None:
            job.items_processed = items_processed
        if items_total is not None:
            job.items_total = items_total
        if error is not None:
            job.error = error
        if status == "running" and not job.started_at:
            from datetime import datetime
            job.started_at = datetime.utcnow()
        if status in ("completed", "failed", "cancelled"):
            from datetime import datetime
            job.completed_at = datetime.utcnow()
        await self.session.flush()

    async def cancel_by_identity(self, identity: str) -> bool:
        job = await self.get_active_by_identity(identity)
        if job:
            await self.update_status(job.id, "cancelled")
            return True
        return False

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(BuildJob))
        return result.scalar_one()

    # ── Queue methods ─────────────────────────────────────────────────────

    async def get_next_queued(self) -> BuildJob | None:
        result = await self.session.execute(
            select(BuildJob)
            .where(BuildJob.queue_status == "queued")
            .order_by(BuildJob.queue_position.asc().nullslast(), BuildJob.created_at.asc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_queue_running(self) -> BuildJob | None:
        result = await self.session.execute(
            select(BuildJob)
            .where(BuildJob.queue_status == "running")
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_queue_overview(self) -> list[BuildJob]:
        result = await self.session.execute(
            select(BuildJob)
            .where(BuildJob.queue_status.in_(["queued", "running"]))
            .order_by(
                BuildJob.queue_position.asc().nullslast(),
                BuildJob.created_at.asc(),
            )
        )
        return list(result.scalars().all())

    async def count_queued(self) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(BuildJob)
            .where(BuildJob.queue_status == "queued")
        )
        return result.scalar_one()

    async def next_queue_position(self) -> int:
        result = await self.session.execute(
            select(func.coalesce(func.max(BuildJob.queue_position), 0))
            .where(BuildJob.queue_status.in_(["queued", "running"]))
        )
        return result.scalar_one() + 1

    async def mark_queued(self, job_id: int) -> None:
        job = await self.session.get(BuildJob, job_id)
        if not job:
            return
        from datetime import datetime
        job.queue_status = "queued"
        job.status = "pending"
        job.queued_at = job.queued_at or datetime.utcnow()
        if job.queue_position is None:
            job.queue_position = await self.next_queue_position()
        await self.session.flush()

    async def mark_queue_running(self, job_id: int) -> None:
        job = await self.session.get(BuildJob, job_id)
        if not job:
            return
        from datetime import datetime
        job.queue_status = "running"
        job.status = "running"
        if not job.started_at:
            job.started_at = datetime.utcnow()
        await self.session.flush()

    async def mark_queue_completed(self, job_id: int) -> None:
        job = await self.session.get(BuildJob, job_id)
        if not job:
            return
        from datetime import datetime
        job.queue_status = "completed"
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        await self.session.flush()

    async def mark_queue_failed(self, job_id: int) -> None:
        job = await self.session.get(BuildJob, job_id)
        if not job:
            return
        from datetime import datetime
        job.queue_status = "failed"
        job.status = "failed"
        job.completed_at = datetime.utcnow()
        await self.session.flush()

    async def mark_queue_cancelled(self, job_id: int) -> None:
        job = await self.session.get(BuildJob, job_id)
        if not job:
            return
        from datetime import datetime
        job.queue_status = "cancelled"
        job.status = "cancelled"
        job.completed_at = datetime.utcnow()
        await self.session.flush()

    async def increment_retry(self, job_id: int) -> int:
        job = await self.session.get(BuildJob, job_id)
        if not job:
            return 0
        job.retry_count += 1
        await self.session.flush()
        return job.retry_count

    async def requeue(self, job_id: int) -> None:
        job = await self.session.get(BuildJob, job_id)
        if not job:
            return
        from datetime import datetime
        job.queue_status = "queued"
        job.status = "pending"
        job.queue_position = await self.next_queue_position()
        job.queued_at = datetime.utcnow()
        job.error = None
        await self.session.flush()

    async def get_non_terminal_queue(self) -> list[BuildJob]:
        result = await self.session.execute(
            select(BuildJob)
            .where(BuildJob.queue_status.in_(["queued", "running"]))
            .order_by(
                BuildJob.queue_position.asc().nullslast(),
                BuildJob.created_at.asc(),
            )
        )
        return list(result.scalars().all())
