"""Quality assessment pipeline stage."""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session
from portrait_dataset_builder.database.models import Face, Image, QualityScore
from portrait_dataset_builder.database.repository import (
    FaceRepository,
    ImageRepository,
    QualityRepository,
)
from portrait_dataset_builder.logging import get_logger

logger = get_logger("stage.quality")


class QualityStage(PipelineStage):
    """Assess image quality and compute composite quality scores."""

    def __init__(self) -> None:
        super().__init__("quality")

    async def should_run(self, context: PipelineContext) -> bool:
        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            count = await repo.count_by_state("verified")
        return count > 0

    async def execute(self, context: PipelineContext) -> StageResult:
        engine = get_engine(context.db_path)
        weights = context.settings.quality.weights

        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            images = await img_repo.get_by_state("verified", limit=100000)

        scored = 0
        failed = 0
        errors: list[str] = []

        _progress = context.metadata.get("_progress_task")
        total = len(images)

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

                async with get_session(engine) as session:
                    face_repo = FaceRepository(session)
                    faces = await face_repo.get_by_image_id(image_record.id)

                if not faces:
                    failed += 1
                    continue

                best_face = max(faces, key=lambda f: f.confidence or 0)
                scores = self._compute_scores(img, best_face, image_record)

                final_score = sum(scores.get(k, 0.0) * v for k, v in weights.items())

                async with get_session(engine) as session:
                    q_repo = QualityRepository(session)
                    quality = QualityScore(
                        image_id=image_record.id,
                        resolution_score=scores.get("resolution", 0.0),
                        sharpness_score=scores.get("sharpness", 0.0),
                        blur_score=scores.get("blur", 0.0),
                        noise_score=scores.get("noise", 0.0),
                        lighting_score=scores.get("lighting", 0.0),
                        occlusion_score=scores.get("occlusion", 0.0),
                        face_size_score=scores.get("face_size", 0.0),
                        frontal_score=scores.get("frontal", 0.0),
                        jpeg_score=scores.get("jpeg", 0.0),
                        final_score=final_score,
                    )
                    await q_repo.add(quality)

                scored += 1

            except Exception as e:
                failed += 1
                errors.append(f"Image {image_record.id}: {e}")

        logger.info("Quality: {} scored, {} failed", scored, failed)

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(images),
            items_succeeded=scored,
            items_failed=failed,
            errors=errors[:50],
        )

    def _compute_scores(self, img: np.ndarray, face: Face, image: Image) -> dict[str, float]:
        h, w = img.shape[:2]
        min_res = self._get_min_res()
        resolution = min(w, h) / max(min_res, 1)
        resolution_score = min(1.0, resolution)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500.0)

        blur_score = min(1.0, laplacian_var / 100.0)

        noise_score = self._estimate_noise(gray)

        lighting_score = self._assess_lighting(gray)

        face_w = face.face_width
        face_h = face.face_height
        face_ratio = (face_w * face_h) / max(w * h, 1)
        face_size_score = min(1.0, face_ratio * 10)

        yaw = abs(face.yaw or 0)
        pitch = abs(face.pitch or 0)
        frontal_score = max(0.0, 1.0 - (yaw / 90.0) - (pitch / 90.0))

        occlusion_score = self._estimate_occlusion(face)

        jpeg_score = self._estimate_jpeg_quality(gray)

        return {
            "resolution": resolution_score,
            "sharpness": sharpness_score,
            "blur": blur_score,
            "noise": noise_score,
            "lighting": lighting_score,
            "occlusion": occlusion_score,
            "face_size": face_size_score,
            "frontal": frontal_score,
            "jpeg": jpeg_score,
        }

    def _get_min_res(self) -> int:
        return 300

    def _estimate_noise(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.5
        small = cv2.resize(gray, (min(256, w), min(256, h)))
        laplacian = cv2.Laplacian(small, cv2.CV_64F)
        noise_var = laplacian.var()
        noise_score = max(0.0, 1.0 - noise_var / 2000.0)
        return noise_score

    def _assess_lighting(self, gray: np.ndarray) -> float:
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
        contrast_score = min(1.0, std_brightness / 64.0)

        return 0.5 * brightness_score + 0.5 * contrast_score

    def _estimate_occlusion(self, face: Face) -> float:
        landmarks = np.array(
            [
                [face.landmark_left_eye_x, face.landmark_left_eye_y],
                [face.landmark_right_eye_x, face.landmark_right_eye_y],
                [face.landmark_nose_x, face.landmark_nose_y],
                [face.landmark_left_mouth_x, face.landmark_left_mouth_y],
                [face.landmark_right_mouth_x, face.landmark_right_mouth_y],
            ]
        )

        spread_x = landmarks[:, 0].max() - landmarks[:, 0].min()
        spread_y = landmarks[:, 1].max() - landmarks[:, 1].min()

        bbox_area = face.bbox_w * face.bbox_h
        if bbox_area <= 0:
            return 0.0

        landmark_area = spread_x * spread_y
        coverage = landmark_area / bbox_area

        occlusion = max(0.0, 1.0 - min(1.0, coverage / 0.3))
        return occlusion

    def _estimate_jpeg_quality(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        if h < 16 or w < 16:
            return 0.5

        blockiness_h = 0.0
        blockiness_v = 0.0

        for i in range(8, h - 8, 8):
            diff = np.abs(gray[i, :].astype(float) - gray[i - 1, :].astype(float))
            blockiness_h += np.mean(diff)

        for j in range(8, w - 8, 8):
            diff = np.abs(gray[:, j].astype(float) - gray[:, j - 1].astype(float))
            blockiness_v += np.mean(diff)

        n_blocks_h = max(1, (h - 16) // 8)
        n_blocks_v = max(1, (w - 16) // 8)

        blockiness_h /= n_blocks_h
        blockiness_v /= n_blocks_v

        avg_blockiness = (blockiness_h + blockiness_v) / 2.0

        score = max(0.0, 1.0 - min(1.0, avg_blockiness / 5.0))
        return score
