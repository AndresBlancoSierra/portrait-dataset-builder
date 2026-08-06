"""Automatic classification pipeline stage with continuous values."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session
from portrait_dataset_builder.database.models import Classification, Face, Image
from portrait_dataset_builder.database.repository import (
    ClassificationRepository,
    FaceRepository,
    ImageRepository,
)
from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.taxonomy import (
    ExpressionLabel,
    HorizontalPose,
    LightingLabel,
    VerticalPose,
)

logger = get_logger("stage.classification")


class ClassificationStage(PipelineStage):
    """Classify images by angle, expression, accessories, age, and lighting."""

    def __init__(self) -> None:
        super().__init__("classification")

    async def should_run(self, context: PipelineContext) -> bool:
        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            count = await repo.count_by_state("verified")
        return count > 0

    async def execute(self, context: PipelineContext) -> StageResult:
        engine = get_engine(context.db_path)
        settings = context.settings.classification

        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            images = await img_repo.get_by_state("verified", limit=100000)

        classified = 0
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
            try:
                async with get_session(engine) as session:
                    face_repo = FaceRepository(session)
                    faces = await face_repo.get_by_image_id(image_record.id)

                if not faces:
                    failed += 1
                    continue

                best_face = max(faces, key=lambda f: f.confidence or 0)

                horizontal_pose = self._classify_horizontal_pose(best_face, settings)
                vertical_pose = self._classify_vertical_pose(best_face, settings)
                angle = self._derive_angle(horizontal_pose, vertical_pose)
                expression = self._classify_expression(best_face)
                accessories = self._classify_accessories(best_face, image_record)
                age_group = self._classify_age(best_face)
                self._classify_gender(best_face)
                lighting = self._classify_lighting(image_record, best_face)
                self._compute_continuous(best_face)

                async with get_session(engine) as session:
                    cls_repo = ClassificationRepository(session)
                    classification = Classification(
                        image_id=image_record.id,
                        angle=angle,
                        horizontal_pose=horizontal_pose,
                        vertical_pose=vertical_pose,
                        expression=expression,
                        accessories=accessories,
                        age_group=age_group,
                        lighting=lighting,
                    )
                    await cls_repo.add(classification)

                classified += 1

            except Exception as e:
                failed += 1
                errors.append(f"Image {image_record.id}: {type(e).__name__}: {e}")
                if len(errors) <= 5:
                    logger.error("Classification error on image {}: {}", image_record.id, e)

        logger.info("Classification: {} classified, {} failed", classified, failed)

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(images),
            items_succeeded=classified,
            items_failed=failed,
            errors=errors[:50],
        )

    def _classify_horizontal_pose(self, face: Face, settings) -> str:
        """Classify horizontal pose based on yaw angle.

        Returns one of: frontal, three_quarter_left, three_quarter_right,
        profile_left, profile_right.
        """
        yaw = face.yaw or 0

        if abs(yaw) <= settings.yaw_frontal_threshold:
            return HorizontalPose.FRONTAL

        if abs(yaw) >= settings.yaw_profile_threshold:
            return HorizontalPose.PROFILE_LEFT if yaw > 0 else HorizontalPose.PROFILE_RIGHT

        return HorizontalPose.QUARTER_LEFT if yaw > 0 else HorizontalPose.QUARTER_RIGHT

    def _classify_vertical_pose(self, face: Face, settings) -> str:
        """Classify vertical pose based on pitch angle.

        Returns one of: neutral, looking_up, looking_down.
        """
        pitch = face.pitch or 0

        if pitch < settings.pitch_up_threshold:
            return VerticalPose.LOOKING_UP
        if pitch > settings.pitch_down_threshold:
            return VerticalPose.LOOKING_DOWN
        return VerticalPose.NEUTRAL

    def _derive_angle(self, horizontal_pose: str, vertical_pose: str) -> str:
        """Derive legacy angle field from horizontal and vertical pose.

        For backward compatibility. When vertical pose is not neutral,
        returns the vertical pose. Otherwise returns the horizontal pose.
        """
        if vertical_pose != VerticalPose.NEUTRAL:
            return vertical_pose
        return horizontal_pose

    def _classify_expression(self, face: Face) -> str:
        left_eye = np.array([face.landmark_left_eye_x, face.landmark_left_eye_y])
        right_eye = np.array([face.landmark_right_eye_x, face.landmark_right_eye_y])
        nose = np.array([face.landmark_nose_x, face.landmark_nose_y])
        mouth_left = np.array([face.landmark_left_mouth_x, face.landmark_left_mouth_y])
        mouth_right = np.array([face.landmark_right_mouth_x, face.landmark_right_mouth_y])

        eye_distance = np.linalg.norm(left_eye - right_eye)
        if eye_distance == 0:
            return ExpressionLabel.NEUTRAL

        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        mouth_ratio = mouth_width / eye_distance

        mouth_center = (mouth_left + mouth_right) / 2
        nose_to_mouth = np.linalg.norm(nose - mouth_center)
        vertical_ratio = nose_to_mouth / eye_distance

        if mouth_ratio > 0.8:
            return ExpressionLabel.LAUGH
        if mouth_ratio > 0.55:
            return ExpressionLabel.SMILE
        if vertical_ratio > 0.8:
            return ExpressionLabel.SPEAKING

        return ExpressionLabel.NEUTRAL

    def _classify_accessories(self, face: Face, image: Image) -> dict:
        glasses = self._detect_glasses_heuristic(face)
        beard = self._detect_beard_heuristic(face)

        return {
            "glasses": glasses,
            "hat": False,
            "beard": beard,
            "mustache": False,
            "headphones": False,
        }

    def _detect_glasses_heuristic(self, face: Face) -> bool:
        left_eye = np.array([face.landmark_left_eye_x, face.landmark_left_eye_y])
        right_eye = np.array([face.landmark_right_eye_x, face.landmark_right_eye_y])
        nose = np.array([face.landmark_nose_x, face.landmark_nose_y])

        eye_midpoint = (left_eye + right_eye) / 2
        eye_to_nose = np.linalg.norm(eye_midpoint - nose)

        eye_distance = np.linalg.norm(left_eye - right_eye)
        if eye_distance == 0:
            return False

        ratio = eye_to_nose / eye_distance
        return bool(0.6 < ratio < 1.2)

    def _detect_beard_heuristic(self, face: Face) -> bool:
        mouth_left = np.array([face.landmark_left_mouth_x, face.landmark_left_mouth_y])
        mouth_right = np.array([face.landmark_right_mouth_x, face.landmark_right_mouth_y])
        chin_y = face.bbox_y + face.bbox_h
        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2

        lower_face_ratio = (chin_y - mouth_center_y) / max(face.bbox_h, 1)
        return bool(lower_face_ratio > 0.25)

    def _classify_age(self, face: Face) -> str:
        age = getattr(face, "age", None)
        if age is None:
            return "adult"
        if age < 18:
            return "child"
        if age < 30:
            return "young_adult"
        if age < 50:
            return "adult"
        return "senior"

    def _classify_gender(self, face: Face) -> str:
        gender = getattr(face, "gender", None)
        if gender is None:
            return "unknown"
        if gender == 0:
            return "M"
        if gender == 1:
            return "F"
        return "unknown"

    def _classify_lighting(self, image: Image, face: Face) -> str:
        if not image.local_path:
            return LightingLabel.BALANCED
        try:
            img = cv2.imread(image.local_path)
            if img is None:
                return LightingLabel.BALANCED

            x = max(0, int(face.bbox_x))
            y = max(0, int(face.bbox_y))
            w = min(img.shape[1] - x, int(face.bbox_w))
            h = min(img.shape[0] - y, int(face.bbox_h))

            if w <= 0 or h <= 0:
                return LightingLabel.BALANCED

            face_roi = img[y : y + h, x : x + w]
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)

            if mean_brightness < 80:
                return LightingLabel.DARK
            if mean_brightness > 180:
                return LightingLabel.BRIGHT
            return LightingLabel.BALANCED
        except Exception:
            return LightingLabel.BALANCED

    def _compute_continuous(self, face: Face) -> dict:
        yaw = face.yaw or 0
        pitch = face.pitch or 0
        roll = face.roll or 0

        left_eye = np.array([face.landmark_left_eye_x, face.landmark_left_eye_y])
        right_eye = np.array([face.landmark_right_eye_x, face.landmark_right_eye_y])
        np.array([face.landmark_nose_x, face.landmark_nose_y])
        mouth_left = np.array([face.landmark_left_mouth_x, face.landmark_left_mouth_y])
        mouth_right = np.array([face.landmark_right_mouth_x, face.landmark_right_mouth_y])

        eye_distance = np.linalg.norm(left_eye - right_eye)
        mouth_ratio = 0.0
        if eye_distance > 0:
            mouth_width = np.linalg.norm(mouth_left - mouth_right)
            mouth_ratio = mouth_width / eye_distance

        eye_openness = 1.0

        return {
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "roll": round(roll, 1),
            "mouth_ratio": round(mouth_ratio, 3),
            "eye_openness": round(eye_openness, 3),
        }

    def _detect_glasses_heuristic(self, face: Face) -> bool:
        left_eye = np.array([face.landmark_left_eye_x, face.landmark_left_eye_y])
        right_eye = np.array([face.landmark_right_eye_x, face.landmark_right_eye_y])
        nose = np.array([face.landmark_nose_x, face.landmark_nose_y])

        eye_midpoint = (left_eye + right_eye) / 2
        eye_to_nose = np.linalg.norm(eye_midpoint - nose)

        eye_distance = np.linalg.norm(left_eye - right_eye)
        if eye_distance == 0:
            return False

        ratio = eye_to_nose / eye_distance
        return bool(0.6 < ratio < 1.2)

    def _detect_beard_heuristic(self, face: Face) -> bool:
        mouth_left = np.array([face.landmark_left_mouth_x, face.landmark_left_mouth_y])
        mouth_right = np.array([face.landmark_right_mouth_x, face.landmark_right_mouth_y])
        chin_y = face.bbox_y + face.bbox_h
        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2

        lower_face_ratio = (chin_y - mouth_center_y) / max(face.bbox_h, 1)
        return bool(lower_face_ratio > 0.25)
