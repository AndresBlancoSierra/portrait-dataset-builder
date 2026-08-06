"""Video source providers — re-export base types and auto-register plugins."""

from portrait_dataset_builder.sources.video.base import VideoResult, VideoSource
from portrait_dataset_builder.sources.video.youtube import YouTubeVideoSource

__all__ = [
    "VideoResult",
    "VideoSource",
    "YouTubeVideoSource",
]
