"""Test configuration for portrait-dataset-builder."""

from __future__ import annotations

# Ensure all source plugins are registered before any test runs
from portrait_dataset_builder.sources import image as _image_sources  # noqa: F401
from portrait_dataset_builder.sources import video as _video_sources  # noqa: F401
