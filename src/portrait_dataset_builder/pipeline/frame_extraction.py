"""Intelligent frame extraction from videos with face awareness."""

from __future__ import annotations

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
from portrait_dataset_builder.database.models import Frame, Image
from portrait_dataset_builder.database.repository import FrameRepository, ImageRepository
from portrait_dataset_builder.logging import get_logger

logger = get_logger("stage.frame_extraction")


class FrameExtractionStage(PipelineStage):
    """Extract diverse frames from videos using face-aware differencing."""

    def __init__(self) -> None:
        super().__init__("frame_extraction")
        self._face_app = None

    async def should_run(self, context: PipelineContext) -> bool:
        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = FrameRepository(session)
            count = await repo.count_pending()
        return count > 0

    def _init_face_model(self, context: PipelineContext) -> None:
        if self._face_app is not None:
            return
        try:
            from insightface.app import FaceAnalysis

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if context.settings.device != "cpu"
                else ["CPUExecutionProvider"]
            )
            ctx_id = 0 if context.settings.device == "cuda" else -1

            self._face_app = FaceAnalysis(
                name=context.settings.face_detection.model_name,
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )
            self._face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("InsightFace loaded for frame extraction")
        except ImportError:
            logger.warning("InsightFace not available, using pixel-only differencing")

    async def execute(self, context: PipelineContext) -> StageResult:
        self._init_face_model(context)
        engine = get_engine(context.db_path)
        frames_dir = context.resolve_frames_dir()
        frames_dir.mkdir(parents=True, exist_ok=True)

        from portrait_dataset_builder.database.repository import VideoRepository

        async with get_session(engine) as session:
            v_repo = VideoRepository(session)
            videos = await v_repo.get_downloaded()

        total_extracted = 0
        total_skipped = 0
        errors: list[str] = []

        for video in videos:
            if not video.local_path or not Path(video.local_path).exists():
                continue

            try:
                extracted, skipped = await self._extract_frames(
                    video_id=video.id,
                    video_path=Path(video.local_path),
                    frames_dir=frames_dir,
                    engine=engine,
                    settings=context.settings.frame_extraction,
                )
                total_extracted += extracted
                total_skipped += skipped
            except Exception as e:
                logger.error("Frame extraction failed for video {}: {}", video.id, e)
                errors.append(f"Video {video.id}: {e}")

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=total_extracted + total_skipped,
            items_succeeded=total_extracted,
            items_skipped=total_skipped,
            errors=errors,
        )

    async def _extract_frames(
        self,
        video_id: int,
        video_path: Path,
        frames_dir: Path,
        engine,  # noqa: ANN001
        settings,  # noqa: ANN001
    ) -> tuple[int, int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_interval = max(1, int(fps / settings.sample_fps))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = min(
            settings.max_frames_per_video,
            total_frames // max(1, frame_interval),
        )

        extracted = 0
        skipped = 0
        frame_idx = 0
        last_saved_data: _FrameData | None = None

        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps
            current_data = self._analyze_frame(frame)

            if last_saved_data is None:
                is_first = True
                should_save = True
            else:
                is_first = False
                change_score = self._compute_change_score(
                    last_saved_data, current_data, settings
                )
                should_save = change_score >= settings.min_difference

            if should_save:
                frame_filename = f"v{video_id}_f{frame_idx:08d}.jpg"
                frame_path = frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

                async with get_session(engine) as session:
                    frame_repo = FrameRepository(session)
                    image_repo = ImageRepository(session)

                    content_hash = self._compute_hash(frame_path)
                    existing = await image_repo.get_by_hash(content_hash)
                    if existing:
                        skipped += 1
                        frame_idx += 1
                        continue

                    image = Image(
                        uri=str(frame_path),
                        local_path=str(frame_path),
                        source_type="video_frame",
                        source_provider="video",
                        content_hash=content_hash,
                        width=frame.shape[1],
                        height=frame.shape[0],
                        file_size=frame_path.stat().st_size,
                        mime_type="image/jpeg",
                        pipeline_state="downloaded",
                    )
                    await image_repo.add(image)

                    change_score = (
                        0.0
                        if is_first
                        else self._compute_change_score(last_saved_data, current_data, settings)
                    )

                    frame_record = Frame(
                        video_id=video_id,
                        timestamp=timestamp,
                        image_id=image.id,
                        difference_score=change_score,
                        selected=True,
                    )
                    await frame_repo.add(frame_record)

                last_saved_data = current_data
                extracted += 1
            else:
                skipped += 1

            frame_idx += 1

        cap.release()
        return extracted, skipped

    def _analyze_frame(self, frame: np.ndarray) -> _FrameData:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (128, 128))

        prev_hist = cv2.calcHist([small], [0], None, [64], [0, 256])
        cv2.normalize(prev_hist, prev_hist)

        yaw = 0.0
        pitch = 0.0
        embedding: np.ndarray | None = None
        landmarks: np.ndarray | None = None
        has_face = False

        if self._face_app is not None:
            faces = self._face_app.get(frame)
            if faces:
                best = max(faces, key=lambda f: f.det_score)
                has_face = True
                yaw = float(best.pose[0])
                pitch = float(best.pose[1])
                embedding = best.normed_embedding
                landmarks = best.kps

        return _FrameData(
            gray_small=small,
            histogram=prev_hist,
            yaw=yaw,
            pitch=pitch,
            embedding=embedding,
            landmarks=landmarks,
            has_face=has_face,
        )

    def _compute_change_score(
        self, prev: _FrameData, curr: _FrameData, settings: object
    ) -> float:
        points = 0.0

        if curr.has_face and prev.has_face:
            yaw_diff = abs(curr.yaw - prev.yaw)
            if yaw_diff > getattr(settings, "pose_change_threshold", 10.0):
                points += 1.0

            pitch_diff = abs(curr.pitch - prev.pitch)
            if pitch_diff > getattr(settings, "pitch_change_threshold", 8.0):
                points += 1.0

            if curr.landmarks is not None and prev.landmarks is not None:
                landmark_shift = np.max(np.abs(curr.landmarks - prev.landmarks))
                if landmark_shift > getattr(settings, "landmark_shift_threshold", 5.0):
                    points += 1.0

            if curr.embedding is not None and prev.embedding is not None:
                cos_dist = 1.0 - float(
                    np.dot(curr.embedding, prev.embedding)
                    / (np.linalg.norm(curr.embedding) * np.linalg.norm(prev.embedding) + 1e-8)
                )
                if cos_dist > getattr(settings, "embedding_distance_threshold", 0.3):
                    points += 1.0

        if prev.histogram is not None and curr.histogram is not None:
            hist_corr = cv2.compareHist(prev.histogram, curr.histogram, cv2.HISTCMP_CORREL)
            ssim_approx = 1.0 - hist_corr
            if ssim_approx > (1.0 - getattr(settings, "ssim_threshold", 0.7)):
                points += 1.0

        mouth_ratio_prev = (
            self._mouth_openness(prev.landmarks) if prev.landmarks is not None else 0.0
        )
        mouth_ratio_curr = (
            self._mouth_openness(curr.landmarks) if curr.landmarks is not None else 0.0
        )
        if abs(mouth_ratio_curr - mouth_ratio_prev) > getattr(
            settings, "expression_change_threshold", 0.2
        ):
            points += 1.0

        if not curr.has_face and prev.has_face:
            points += 2.0

        max_points = 6.0
        return min(1.0, points / max_points)

    def _mouth_openness(self, landmarks: np.ndarray | None) -> float:
        if landmarks is None or len(landmarks) < 5:
            return 0.0
        mouth_left = landmarks[3]
        mouth_right = landmarks[4]
        nose = landmarks[2]
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        nose_to_mouth = np.linalg.norm(nose - (mouth_left + mouth_right) / 2)
        if mouth_width == 0:
            return 0.0
        return nose_to_mouth / mouth_width

    def _compute_hash(self, path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    async def teardown(self, context: PipelineContext) -> None:
        self._face_app = None


class _FrameData:
    __slots__ = (
        "gray_small",
        "histogram",
        "yaw",
        "pitch",
        "embedding",
        "landmarks",
        "has_face",
    )

    def __init__(
        self,
        gray_small: np.ndarray,
        histogram: np.ndarray,
        yaw: float,
        pitch: float,
        embedding: np.ndarray | None,
        landmarks: np.ndarray | None,
        has_face: bool,
    ) -> None:
        self.gray_small = gray_small
        self.histogram = histogram
        self.yaw = yaw
        self.pitch = pitch
        self.embedding = embedding
        self.landmarks = landmarks
        self.has_face = has_face
