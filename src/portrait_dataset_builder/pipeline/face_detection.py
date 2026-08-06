"""Face detection pipeline stage using InsightFace."""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2

from portrait_dataset_builder.compute import (
    ResolvedDevice,
    get_ctx_id,
    get_onnx_providers,
    log_device_info,
    resolve_device,
)
from portrait_dataset_builder.compute.resize import resize_for_inference
from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session
from portrait_dataset_builder.database.models import Face
from portrait_dataset_builder.database.repository import FaceRepository, ImageRepository
from portrait_dataset_builder.logging import get_logger

logger = get_logger("stage.face_detection")


class FaceDetectionStage(PipelineStage):
    """Detect faces in all downloaded images using InsightFace."""

    def __init__(self) -> None:
        super().__init__("face_detection")
        self._analysis_app = None
        self._resolved: ResolvedDevice | None = None

    async def should_run(self, context: PipelineContext) -> bool:
        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            count = await repo.count_by_state("downloaded")
        return count > 0

    def _init_models(self, context: PipelineContext) -> None:
        if self._analysis_app is not None:
            return

        try:
            from insightface.app import FaceAnalysis

            self._resolved = resolve_device(context.settings.effective_device)
            providers = get_onnx_providers(self._resolved)
            ctx_id = get_ctx_id(self._resolved)

            log_device_info(self._resolved)
            logger.info(
                "InsightFace model: {} | Providers: {} | ctx_id: {}",
                context.settings.face_detection.model_name,
                providers,
                ctx_id,
            )

            self._analysis_app = FaceAnalysis(
                name=context.settings.face_detection.model_name,
                providers=providers,
            )
            self._analysis_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully")
        except ImportError:
            logger.error("insightface not installed. Install with: pip install insightface")
            raise

    async def execute(self, context: PipelineContext) -> StageResult:
        self._init_models(context)

        max_dim = context.settings.compute.inference_max_dimension

        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            face_repo = FaceRepository(session)
            images = await repo.get_by_state("downloaded", limit=10000)

        image_ids = [img.id for img in images]
        async with get_session(engine) as session:
            face_repo = FaceRepository(session)
            existing_face_ids = await face_repo.get_image_ids_with_faces(image_ids)

        skip_set = set(existing_face_ids)
        images = [img for img in images if img.id not in skip_set]

        logger.info("Face detection: {} images to process", len(images))

        _progress = context.metadata.get("_progress_task")
        total = len(images)

        detected = 0
        no_face = 0
        failed = 0
        errors: list[str] = []

        for i, image_record in enumerate(images):
            if i % 10 == 0:
                await asyncio.sleep(0)
            if _progress:
                _progress["items_processed"] = i
                _progress["items_total"] = total
            if not image_record.local_path or not Path(image_record.local_path).exists():
                continue

            try:
                img = cv2.imread(image_record.local_path)
                if img is None:
                    failed += 1
                    continue

                resized_img, scale = resize_for_inference(img, max_dim=max_dim)
                faces = self._analysis_app.get(resized_img)

                async with get_session(engine) as session:
                    face_repo = FaceRepository(session)
                    img_repo = ImageRepository(session)

                    for face in faces:
                        bbox = face.bbox.astype(float)
                        kps = face.kps
                        embedding = face.normed_embedding

                        if scale != 1.0:
                            bbox[0] *= scale
                            bbox[1] *= scale
                            bbox[2] *= scale
                            bbox[3] *= scale
                            kps = kps * scale

                        face_w = int(bbox[2] - bbox[0])
                        face_h = int(bbox[3] - bbox[1])
                        min_size = context.settings.face_detection.min_face_size
                        if face_w < min_size or face_h < min_size:
                            continue

                        face_record = Face(
                            image_id=image_record.id,
                            bbox_x=float(bbox[0]),
                            bbox_y=float(bbox[1]),
                            bbox_w=float(bbox[2] - bbox[0]),
                            bbox_h=float(bbox[3] - bbox[1]),
                            landmark_left_eye_x=float(kps[0][0]),
                            landmark_left_eye_y=float(kps[0][1]),
                            landmark_right_eye_x=float(kps[1][0]),
                            landmark_right_eye_y=float(kps[1][1]),
                            landmark_nose_x=float(kps[2][0]),
                            landmark_nose_y=float(kps[2][1]),
                            landmark_left_mouth_x=float(kps[3][0]),
                            landmark_left_mouth_y=float(kps[3][1]),
                            landmark_right_mouth_x=float(kps[4][0]),
                            landmark_right_mouth_y=float(kps[4][1]),
                            yaw=float(face["pose"][0]),
                            pitch=float(face["pose"][1]),
                            roll=float(face["pose"][2]),
                            embedding=embedding.tobytes(),
                            confidence=float(face.det_score),
                            face_width=face_w,
                            face_height=face_h,
                        )
                        await face_repo.add(face_record)
                        detected += 1

                    if len(faces) > 0:
                        await img_repo.update_state(image_record.id, "face_detected")
                    else:
                        no_face += 1
                        await img_repo.update_state(image_record.id, "no_face")

            except Exception as e:
                failed += 1
                errors.append(f"Image {image_record.id}: {type(e).__name__}: {e}")
                if len(errors) <= 5:
                    logger.error("Face detection error on image {}: {}", image_record.id, e)
                if len(errors) > 10:
                    break

        logger.info(
            "Face detection: {} faces found, {} no face, {} failed",
            detected,
            no_face,
            failed,
        )

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(images),
            items_succeeded=detected,
            items_failed=failed,
            items_skipped=no_face,
            errors=errors,
            metadata={
                "faces_detected": detected,
                "no_face_images": no_face,
            },
        )

    async def teardown(self, context: PipelineContext) -> None:
        self._analysis_app = None
