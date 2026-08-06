from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.utcnow()


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uri: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    local_path: Mapped[str | None] = mapped_column(String, nullable=True)
    source_type: Mapped[str] = mapped_column(String, default="image_search")
    source_provider: Mapped[str] = mapped_column(String, default="unknown")
    content_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    p_hash: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    d_hash: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    pipeline_state: Mapped[str] = mapped_column(String, default="pending")


class Face(Base):
    __tablename__ = "faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("images.id"), nullable=False, index=True
    )
    bbox_x: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_y: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_w: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_h: Mapped[float] = mapped_column(Float, default=0.0)
    landmark_left_eye_x: Mapped[float] = mapped_column(Float, default=0.0)
    landmark_left_eye_y: Mapped[float] = mapped_column(Float, default=0.0)
    landmark_right_eye_x: Mapped[float] = mapped_column(Float, default=0.0)
    landmark_right_eye_y: Mapped[float] = mapped_column(Float, default=0.0)
    landmark_nose_x: Mapped[float] = mapped_column(Float, default=0.0)
    landmark_nose_y: Mapped[float] = mapped_column(Float, default=0.0)
    landmark_left_mouth_x: Mapped[float] = mapped_column(Float, default=0.0)
    landmark_left_mouth_y: Mapped[float] = mapped_column(Float, default=0.0)
    landmark_right_mouth_x: Mapped[float] = mapped_column(Float, default=0.0)
    landmark_right_mouth_y: Mapped[float] = mapped_column(Float, default=0.0)
    yaw: Mapped[float] = mapped_column(Float, default=0.0)
    pitch: Mapped[float] = mapped_column(Float, default=0.0)
    roll: Mapped[float] = mapped_column(Float, default=0.0)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    face_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    face_height: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Identity(Base):
    __tablename__ = "identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="building")
    seed_embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, onupdate=_utcnow)


class IdentityImage(Base):
    __tablename__ = "identity_images"

    identity_id: Mapped[int] = mapped_column(Integer, ForeignKey("identities.id"), primary_key=True)
    image_id: Mapped[int] = mapped_column(Integer, ForeignKey("images.id"), primary_key=True)


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String, unique=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, default="youtube")
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    local_path: Mapped[str | None] = mapped_column(String, nullable=True)
    pipeline_state: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("videos.id"), nullable=False, index=True
    )
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    image_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("images.id"), nullable=True)
    difference_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)


class QualityScore(Base):
    __tablename__ = "quality_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("images.id"), unique=True, nullable=False
    )
    resolution_score: Mapped[float] = mapped_column(Float, default=0.0)
    sharpness_score: Mapped[float] = mapped_column(Float, default=0.0)
    blur_score: Mapped[float] = mapped_column(Float, default=0.0)
    noise_score: Mapped[float] = mapped_column(Float, default=0.0)
    lighting_score: Mapped[float] = mapped_column(Float, default=0.0)
    occlusion_score: Mapped[float] = mapped_column(Float, default=0.0)
    face_size_score: Mapped[float] = mapped_column(Float, default=0.0)
    frontal_score: Mapped[float] = mapped_column(Float, default=0.0)
    jpeg_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_score: Mapped[float] = mapped_column(Float, default=0.0)


class Classification(Base):
    __tablename__ = "classifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("images.id"), nullable=False, index=True
    )
    angle: Mapped[str | None] = mapped_column(String, nullable=True)
    horizontal_pose: Mapped[str | None] = mapped_column(String, nullable=True)
    vertical_pose: Mapped[str | None] = mapped_column(String, nullable=True)
    expression: Mapped[str | None] = mapped_column(String, nullable=True)
    accessories: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    age_group: Mapped[str | None] = mapped_column(String, nullable=True)
    lighting: Mapped[str | None] = mapped_column(String, nullable=True)


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    build_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="success")
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class SchemaVersion(Base):
    __tablename__ = "schema_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    face_id: Mapped[int] = mapped_column(Integer, ForeignKey("faces.id"), nullable=False)
    image_id: Mapped[int] = mapped_column(Integer, ForeignKey("images.id"), nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    variance: Mapped[float] = mapped_column(Float, default=0.0)
    best_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    user_answer: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class EmbeddingIndex(Base):
    __tablename__ = "embedding_index"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("images.id"), nullable=False, index=True
    )
    embedding_type: Mapped[str] = mapped_column(String, nullable=False)


class SafetyScore(Base):
    __tablename__ = "safety_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("images.id"), unique=True, nullable=False, index=True
    )
    is_nsfw: Mapped[bool] = mapped_column(Boolean, default=False)
    nsfw_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_ai_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_probability: Mapped[float] = mapped_column(Float, default=0.0)
    real_photo_score: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String, default="unknown")
    source_trust_score: Mapped[float] = mapped_column(Float, default=0.5)
    rejection_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    pipeline_state: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class BuildJob(Base):
    __tablename__ = "build_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    current_stage: Mapped[str | None] = mapped_column(String, nullable=True)
    stage_label: Mapped[str | None] = mapped_column(String, nullable=True)
    items_processed: Mapped[int] = mapped_column(Integer, default=0)
    items_total: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    queue_status: Mapped[str] = mapped_column(String, default="queued")
    queue_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
