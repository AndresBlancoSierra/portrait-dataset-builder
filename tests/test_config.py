"""Test configuration settings."""

from __future__ import annotations

from portrait_dataset_builder.config.settings import Settings, load_settings


class TestSettings:
    """Tests for the Settings configuration."""

    def test_default_settings(self) -> None:
        settings = Settings(identity="Test Person")
        assert settings.identity == "Test Person"
        assert settings.log_level == "INFO"
        assert settings.device == "auto"

    def test_resolve_db_path(self) -> None:
        settings = Settings(identity="Matt Damon")
        db_path = settings.resolve_db_path()
        assert "Matt Damon" in str(db_path)
        assert db_path.name == "portrait.db"

    def test_resolve_data_dir(self) -> None:
        settings = Settings(identity="Matt Damon")
        data_dir = settings.resolve_data_dir()
        assert "Matt Damon" in str(data_dir)

    def test_resolve_images_dir(self) -> None:
        settings = Settings(identity="Matt Damon")
        images_dir = settings.resolve_images_dir()
        assert "images" in str(images_dir)
        assert "Matt Damon" in str(images_dir)

    def test_resolve_videos_dir(self) -> None:
        settings = Settings(identity="Matt Damon")
        videos_dir = settings.resolve_videos_dir()
        assert "videos" in str(videos_dir)

    def test_resolve_seeds_dir(self) -> None:
        settings = Settings(identity="Matt Damon")
        seeds_dir = settings.resolve_seeds_dir()
        assert "seeds" in str(seeds_dir)
        assert "Matt Damon" in str(seeds_dir)

    def test_load_settings(self) -> None:
        settings = load_settings("Test Identity")
        assert settings.identity == "Test Identity"

    def test_database_settings_defaults(self) -> None:
        settings = Settings(identity="test")
        assert settings.database.echo is False
        assert "portrait.db" in settings.database.path

    def test_search_settings_defaults(self) -> None:
        settings = Settings(identity="test")
        assert "duckduckgo" in settings.search.enabled_sources
        assert settings.search.max_results_per_source == 500

    def test_download_settings_defaults(self) -> None:
        settings = Settings(identity="test")
        assert settings.download.max_concurrent == 10
        assert settings.download.timeout == 30
        assert settings.download.retries == 3

    def test_face_detection_settings(self) -> None:
        settings = Settings(identity="test")
        assert settings.face_detection.detector == "insightface"
        assert settings.face_detection.min_confidence == 0.5

    def test_face_verification_settings(self) -> None:
        settings = Settings(identity="test")
        assert settings.face_verification.strict_threshold == 0.50

        assert settings.face_verification.normal_threshold == 0.45

        assert settings.face_verification.permissive_threshold == 0.35

    def test_quality_settings_weights_sum(self) -> None:
        settings = Settings(identity="test")
        total = sum(settings.quality.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_pipeline_settings_stages(self) -> None:
        settings = Settings(identity="test")
        assert "search" in settings.pipeline.stages
        assert "download" in settings.pipeline.stages
        assert "export" in settings.pipeline.stages
        assert len(settings.pipeline.stages) >= 8
