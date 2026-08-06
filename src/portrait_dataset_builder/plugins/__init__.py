"""Plugin registry for extensible source providers."""

from __future__ import annotations

from typing import Any

from portrait_dataset_builder.logging import get_logger

logger = get_logger("plugins")


class PluginRegistry:
    """Central registry for image/video source plugins."""

    _image_sources: dict[str, type] = {}
    _video_sources: dict[str, type] = {}

    @classmethod
    def register_image_source(cls, name: str, source_class: type) -> None:
        cls._image_sources[name] = source_class
        logger.debug("Registered image source: {}", name)

    @classmethod
    def register_video_source(cls, name: str, source_class: type) -> None:
        cls._video_sources[name] = source_class
        logger.debug("Registered video source: {}", name)

    @classmethod
    def get_image_source(cls, name: str) -> type:
        if name not in cls._image_sources:
            raise KeyError(f"Unknown image source: {name}. Available: {list(cls._image_sources)}")
        return cls._image_sources[name]

    @classmethod
    def get_video_source(cls, name: str) -> type:
        if name not in cls._video_sources:
            raise KeyError(f"Unknown video source: {name}. Available: {list(cls._video_sources)}")
        return cls._video_sources[name]

    @classmethod
    def list_image_sources(cls) -> list[str]:
        return list(cls._image_sources)

    @classmethod
    def list_video_sources(cls) -> list[str]:
        return list(cls._video_sources)


def image_source(name: str) -> Any:
    """Decorator to register an image source plugin."""

    def decorator(cls: type) -> type:
        PluginRegistry.register_image_source(name, cls)
        return cls

    return decorator


def video_source(name: str) -> Any:
    """Decorator to register a video source plugin."""

    def decorator(cls: type) -> type:
        PluginRegistry.register_video_source(name, cls)
        return cls

    return decorator
