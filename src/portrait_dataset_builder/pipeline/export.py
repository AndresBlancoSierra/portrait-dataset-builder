"""Export pipeline stage — organizes verified images into dataset structures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session
from portrait_dataset_builder.database.repository import (
    ClassificationRepository,
    FaceRepository,
    ImageRepository,
    QualityRepository,
)
from portrait_dataset_builder.logging import get_logger

if TYPE_CHECKING:
    from portrait_dataset_builder.database.models import (
        Classification,
        Face,
        Image,
        QualityScore,
    )

logger = get_logger("stage.export")


class ExportStage(PipelineStage):
    """Export verified images into organized dataset structures."""

    def __init__(self) -> None:
        super().__init__("export")

    async def should_run(self, context: PipelineContext) -> bool:
        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            count = await repo.count_by_state("verified")
        return count > 0

    async def execute(self, context: PipelineContext) -> StageResult:
        engine = get_engine(context.db_path)
        export_base = context.output_dir / "export"
        export_base.mkdir(parents=True, exist_ok=True)

        await self._clean_stale_exports(export_base, engine)

        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            images = await img_repo.get_by_state("verified", limit=100000)

        exported = 0
        errors: list[str] = []

        flat_dir = export_base / "flat"
        flat_dir.mkdir(exist_ok=True)

        min_quality = context.settings.quality.min_quality_score

        for image_record in images:
            if not image_record.local_path or not Path(image_record.local_path).exists():
                continue

            try:
                async with get_session(engine) as session:
                    q_repo = QualityRepository(session)
                    quality = await q_repo.get_by_image_id(image_record.id)
                    if quality and quality.final_score < min_quality:
                        continue

                    face_repo = FaceRepository(session)
                    faces = await face_repo.get_by_image_id(image_record.id)
                    if not faces:
                        continue

                    best_face = max(faces, key=lambda f: f.confidence or 0)
                    if best_face.face_width < 30 or best_face.face_height < 30:
                        continue

                    cls_repo = ClassificationRepository(session)
                    classification = await cls_repo.get_by_image_id(image_record.id)

                    src = Path(image_record.local_path)
                    dst = flat_dir / f"{image_record.content_hash}{src.suffix}"
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    exported += 1

                    metadata = self._build_metadata(
                        image_record, classification, quality, faces, context.identity
                    )

                    json_path = flat_dir / f"{image_record.content_hash}.json"
                    json_path.write_text(json.dumps(metadata, indent=2, default=str))

            except Exception as e:
                errors.append(f"Image {image_record.id}: {e}")

        if "by_angle" in context.settings.export.formats:
            await self._export_by_angle(images, export_base, engine, context.identity)

        if "top_quality" in context.settings.export.formats:
            await self._export_top_quality(images, export_base, engine, context.identity)

        logger.info("Export: {} images exported to {}", exported, export_base)

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(images),
            items_succeeded=exported,
            items_failed=len(errors),
            errors=errors[:50],
        )

    async def _clean_stale_exports(self, export_base: Path, engine) -> None:
        """Remove exported files for images that are no longer in 'verified' state."""
        from sqlalchemy import select

        from portrait_dataset_builder.database.models import Image

        async with get_session(engine) as session:
            result = await session.execute(
                select(Image.content_hash).where(Image.pipeline_state == "verified")
            )
            verified_hashes = {row[0] for row in result.all()}

        for subdir in ["flat", "by_angle", "top_quality"]:
            dir_path = export_base / subdir
            if not dir_path.exists():
                continue
            for f in dir_path.iterdir():
                if f.suffix in (".json", ".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    content_hash = f.stem
                else:
                    continue
                if content_hash not in verified_hashes:
                    f.unlink(missing_ok=True)
                    logger.debug("Removed stale export: {}", f.name)

    async def _export_by_angle(
        self, images: list, export_base: Path, engine, identity: str
    ) -> None:
        angles = [
            "frontal",
            "profile_left",
            "profile_right",
            "three_quarter_left",
            "three_quarter_right",
            "looking_up",
            "looking_down",
        ]

        async with get_session(engine) as session:
            cls_repo = ClassificationRepository(session)
            face_repo = FaceRepository(session)

            # Batch fetch all classifications and faces (eliminates N+1)
            image_ids = [img.id for img in images]
            cls_map = await cls_repo.get_by_image_ids(image_ids)
            best_faces = await face_repo.get_best_by_image_ids(image_ids)

        for angle in angles:
            angle_dir = export_base / "by_angle" / angle
            angle_dir.mkdir(parents=True, exist_ok=True)

            for image_record in images:
                classification = cls_map.get(image_record.id)
                best_face = best_faces.get(image_record.id)
                if not best_face:
                    continue
                if best_face.face_width < 30 or best_face.face_height < 30:
                    continue
                if (
                    classification
                    and classification.angle == angle
                    and image_record.local_path
                    and Path(image_record.local_path).exists()
                ):
                    src = Path(image_record.local_path)
                    dst = angle_dir / f"{image_record.content_hash}{src.suffix}"
                    if not dst.exists():
                        shutil.copy2(src, dst)

    async def _export_top_quality(
        self, images: list, export_base: Path, engine, identity: str
    ) -> None:
        top_dir = export_base / "top_quality"
        top_dir.mkdir(parents=True, exist_ok=True)

        async with get_session(engine) as session:
            q_repo = QualityRepository(session)
            face_repo = FaceRepository(session)

            # Batch fetch (eliminates N+1)
            image_ids = [img.id for img in images]
            quality_map = await q_repo.get_by_image_ids(image_ids)
            best_faces = await face_repo.get_best_by_image_ids(image_ids)

        scored_images: list[tuple[Image, float]] = []
        for image_record in images:
            quality = quality_map.get(image_record.id)
            best_face = best_faces.get(image_record.id)
            if not best_face:
                continue
            if best_face.face_width < 30 or best_face.face_height < 30:
                continue
            if quality and quality.final_score >= 0.7:
                scored_images.append((image_record, quality.final_score))

        scored_images.sort(key=lambda x: x[1], reverse=True)

        for image_record, _score in scored_images[:500]:
            if image_record.local_path and Path(image_record.local_path).exists():
                src = Path(image_record.local_path)
                dst = top_dir / f"{image_record.content_hash}{src.suffix}"
                if not dst.exists():
                    shutil.copy2(src, dst)

    def _build_metadata(
        self,
        image: Image,
        classification: Classification | None,
        quality: QualityScore | None,
        faces: list[Face],
        identity: str,
    ) -> dict:
        best_face = max(faces, key=lambda f: f.confidence or 0) if faces else None

        return {
            "identity": identity,
            "source": image.source_provider,
            "source_type": image.source_type,
            "uri": image.uri,
            "content_hash": image.content_hash,
            "width": image.width,
            "height": image.height,
            "file_size": image.file_size,
            "face": (
                {
                    "yaw": best_face.yaw if best_face else None,
                    "pitch": best_face.pitch if best_face else None,
                    "roll": best_face.roll if best_face else None,
                    "confidence": best_face.confidence if best_face else None,
                }
                if best_face
                else None
            ),
            "classification": (
                {
                    "angle": classification.angle if classification else None,
                    "horizontal_pose": classification.horizontal_pose if classification else None,
                    "vertical_pose": classification.vertical_pose if classification else None,
                    "expression": classification.expression if classification else None,
                    "accessories": classification.accessories if classification else None,
                    "age_group": classification.age_group if classification else None,
                    "lighting": classification.lighting if classification else None,
                }
                if classification
                else None
            ),
            "quality": (
                {
                    "final": quality.final_score if quality else None,
                    "sharpness": quality.sharpness_score if quality else None,
                    "resolution": quality.resolution_score if quality else None,
                    "blur": quality.blur_score if quality else None,
                }
                if quality
                else None
            ),
        }
