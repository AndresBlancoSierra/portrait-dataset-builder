"""Test plugin registry and source registration."""

from __future__ import annotations

import pytest

from portrait_dataset_builder.plugins import PluginRegistry, image_source, video_source


class TestPluginRegistry:
    def test_image_sources_registered(self) -> None:
        sources = PluginRegistry.list_image_sources()
        assert "google" in sources
        assert "bing" in sources
        assert "duckduckgo" in sources
        assert "wikimedia" in sources
        assert "wikipedia" in sources
        assert "imdb" in sources

    def test_video_sources_registered(self) -> None:
        sources = PluginRegistry.list_video_sources()
        assert "youtube" in sources

    def test_get_image_source(self) -> None:
        cls = PluginRegistry.get_image_source("google")
        assert cls is not None
        assert hasattr(cls, "search")

    def test_get_video_source(self) -> None:
        cls = PluginRegistry.get_video_source("youtube")
        assert cls is not None
        assert hasattr(cls, "search")
        assert hasattr(cls, "download")

    def test_unknown_source_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown image source"):
            PluginRegistry.get_image_source("nonexistent")

    def test_unknown_video_source_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown video source"):
            PluginRegistry.get_video_source("nonexistent")

    def test_decorator_registers(self) -> None:
        @image_source("test_custom")
        class TestSource:
            pass

        assert "test_custom" in PluginRegistry.list_image_sources()

    def test_video_decorator_registers(self) -> None:
        @video_source("test_custom_video")
        class TestVideoSource:
            pass

        assert "test_custom_video" in PluginRegistry.list_video_sources()
