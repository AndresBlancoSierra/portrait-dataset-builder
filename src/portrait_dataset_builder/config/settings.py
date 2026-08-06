"""Application settings using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource


class DatabaseSettings(BaseModel):
    """SQLite database configuration."""

    path: str = "data/{identity}/portrait.db"
    echo: bool = False


class SearchSettings(BaseModel):
    """Image search source configuration."""

    max_results_per_source: int = 500
    enabled_sources: list[str] = Field(
        default_factory=lambda: ["google", "bing", "duckduckgo", "wikimedia"]
    )
    search_delay: float = 2.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    bing_api_key: str = ""


class DownloadSettings(BaseModel):
    """Image download configuration."""

    max_concurrent: int = 5
    timeout: int = 30
    retries: int = 3
    min_image_size: int = 200
    max_image_size: int = 20000
    allowed_formats: list[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    )


class VideoSettings(BaseModel):
    """Video source configuration."""

    enabled_sources: list[str] = Field(default_factory=lambda: ["youtube"])
    max_videos: int = 20
    max_duration: int = 3600
    min_duration: int = 10
    download_quality: str = "best[height<=1080]"
    output_format: str = "mp4"


class FrameExtractionSettings(BaseModel):
    """Intelligent frame extraction configuration."""

    min_difference: float = 0.15
    max_frames_per_video: int = 500
    sample_fps: float = 1.0
    detect_change_types: list[str] = Field(
        default_factory=lambda: [
            "pose",
            "expression",
            "lighting",
            "head_orientation",
            "scene",
        ]
    )
    ssim_threshold: float = 0.7
    histogram_threshold: float = 0.3
    landmark_shift_threshold: float = 5.0
    pose_change_threshold: float = 10.0
    pitch_change_threshold: float = 8.0
    embedding_distance_threshold: float = 0.3
    expression_change_threshold: float = 0.2
    min_points_to_keep: int = 2


class FaceDetectionSettings(BaseModel):
    """Face detection configuration."""

    detector: str = "insightface"
    model_name: str = "buffalo_l"
    min_confidence: float = 0.5
    min_face_size: int = 30
    max_faces_per_image: int = 10


class FaceVerificationSettings(BaseModel):
    """Face verification / identity matching configuration."""

    recognizer: str = "arcface"
    model_name: str = "buffalo_l"
    strict_threshold: float = 0.45
    normal_threshold: float = 0.35
    permissive_threshold: float = 0.28
    mode: str = "normal"
    seed_images_min: int = 3
    seed_images_max: int = 20


class QualitySettings(BaseModel):
    """Image quality assessment configuration."""

    min_resolution: int = 300
    min_quality_score: float = 0.3
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "resolution": 0.10,
            "sharpness": 0.20,
            "blur": 0.15,
            "noise": 0.10,
            "lighting": 0.10,
            "occlusion": 0.10,
            "face_size": 0.10,
            "frontal": 0.10,
            "jpeg": 0.05,
        }
    )


class DuplicateSettings(BaseModel):
    """Duplicate detection configuration."""

    phash_threshold: int = 8
    dhash_threshold: int = 8
    embedding_similarity_threshold: float = 0.95
    ssim_threshold: float = 0.92
    enabled_methods: list[str] = Field(default_factory=lambda: ["phash", "dhash", "embedding"])


class ClassificationSettings(BaseModel):
    """Automatic classification configuration."""

    angle_bins: list[str] = Field(
        default_factory=lambda: [
            "frontal",
            "profile_left",
            "profile_right",
            "three_quarter_left",
            "three_quarter_right",
            "looking_up",
            "looking_down",
        ]
    )
    expression_labels: list[str] = Field(
        default_factory=lambda: [
            "neutral",
            "smile",
            "laugh",
            "speaking",
        ]
    )
    yaw_frontal_threshold: float = 15.0
    yaw_profile_threshold: float = 60.0
    pitch_up_threshold: float = -15.0
    pitch_down_threshold: float = 15.0


class ExportSettings(BaseModel):
    """Dataset export configuration."""

    formats: list[str] = Field(
        default_factory=lambda: ["flat", "by_angle", "by_expression", "top_quality"]
    )
    image_format: str = "jpg"
    jpeg_quality: int = 95
    max_export_size: int = 2000
    include_metadata: bool = True
    export_structure: str = "flat"


class SemanticFilterSettings(BaseModel):
    """CLIP-based semantic filtering configuration."""

    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    pos_threshold: float = 0.22
    neg_threshold: float = 0.27


class SafetySettings(BaseModel):
    """Content safety and filtering configuration."""

    nsfw_model: str = "Falconsai/nsfw_image_detection"
    nsfw_threshold: float = 0.20
    clip_safety_threshold: float = 0.50
    ai_mode: str = "strict"
    ai_metadata_threshold: float = 0.70
    blocked_url_keywords: list[str] = Field(
        default_factory=lambda: [
            "porn",
            "xxx",
            "sex",
            "nude",
            "naked",
            "erotic",
            "onlyfans",
            "nsfw",
            "adult",
            "playboy",
            "penthouse",
        ]
    )
    source_trust_scores: dict[str, float] = Field(
        default_factory=lambda: {
            "official": 1.0,
            "editorial": 1.0,
            "wikimedia": 0.95,
            "wikipedia": 0.95,
            "flickr": 0.90,
            "imdb": 0.90,
            "duckduckgo": 0.70,
            "google": 0.70,
            "bing": 0.70,
            "unknown": 0.30,
        }
    )
    fail_closed: bool = True


class IdentityBootstrapSettings(BaseModel):
    """Identity bootstrap configuration — auto-discover seeds from search results."""

    enabled: bool = True
    candidate_count: int = 50
    min_candidates: int = 5
    target_seeds: int = 12
    max_seeds: int = 20
    similarity_threshold: float = 0.4
    min_cluster_size: int = 3
    min_identity_confidence: float = 0.75
    min_face_quality: float = 0.3
    min_face_size: int = 40


class PipelineSettings(BaseModel):
    """Pipeline orchestration settings."""

    stages: list[str] = Field(
        default_factory=lambda: [
            "search",
            "url_safety_filter",
            "identity_bootstrap",
            "download",
            "safety_gate",
            "face_detection",
            "face_verification",
            "semantic_filter",
            "quality",
            "duplicates",
            "classification",
            "export",
            "cleanup",
        ]
    )
    checkpoint_enabled: bool = True
    max_retries: int = 3
    batch_size: int = 50
    target_images: int = 200
    max_candidate_pool: int = 500
    early_stop_coverage: float = 0.85
    early_stop_quality: float = 0.75
    max_concurrent_builds: int = 1


class ComputeSettings(BaseModel):
    """GPU/CPU compute configuration."""

    device: str = "auto"
    gpu_batch_size: int = 4
    inference_max_dimension: int = 1600
    cpu_workers: int = 2
    gpu_semaphore: int = 1


class Settings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_prefix="PDB_",
        env_nested_delimiter="__",
        yaml_file="configs/default.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    identity: str = "default"
    output_dir: str = "data/{identity}"
    log_level: str = "INFO"
    log_file: str = "data/{identity}/logs/portrait-builder.log"
    seed_dir: str = "seeds/{identity}"
    device: str = "auto"

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    frame_extraction: FrameExtractionSettings = Field(default_factory=FrameExtractionSettings)
    face_detection: FaceDetectionSettings = Field(default_factory=FaceDetectionSettings)
    face_verification: FaceVerificationSettings = Field(default_factory=FaceVerificationSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    duplicates: DuplicateSettings = Field(default_factory=DuplicateSettings)
    classification: ClassificationSettings = Field(default_factory=ClassificationSettings)
    semantic_filter: SemanticFilterSettings = Field(default_factory=SemanticFilterSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    identity_bootstrap: IdentityBootstrapSettings = Field(default_factory=IdentityBootstrapSettings)

    @property
    def effective_device(self) -> str:
        """Return the device to use: compute.device takes precedence over legacy device field."""
        return self.compute.device if self.compute.device != "auto" else self.device

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    @field_validator("output_dir", "seed_dir", "log_file", mode="before")
    @classmethod
    def _resolve_identity_path(cls, v: str, info: dict) -> str:  # type: ignore[type-arg]
        identity = info.data.get("identity", "default")
        return v.replace("{identity}", identity)

    def resolve_db_path(self) -> Path:
        """Return the resolved database file path."""
        return Path(self.database.path.replace("{identity}", self.identity))

    def resolve_data_dir(self) -> Path:
        """Return the resolved data directory path."""
        return Path(self.output_dir)

    def resolve_images_dir(self) -> Path:
        """Return the resolved images directory path."""
        return self.resolve_data_dir() / "images"

    def resolve_videos_dir(self) -> Path:
        """Return the resolved videos directory path."""
        return self.resolve_data_dir() / "videos"

    def resolve_frames_dir(self) -> Path:
        """Return the resolved frames directory path."""
        return self.resolve_data_dir() / "frames"

    def resolve_seeds_dir(self) -> Path:
        """Return the resolved seeds directory path."""
        return Path(self.seed_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()


def load_settings(identity: str, config_path: str | None = None) -> Settings:
    """Load settings for a specific identity."""
    overrides: dict[str, str | None] = {"identity": identity}
    if config_path:
        overrides["yaml_file"] = config_path
    return Settings(**overrides)
