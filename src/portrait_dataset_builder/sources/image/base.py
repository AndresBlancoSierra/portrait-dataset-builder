"""Base interface for image source providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImageResult:
    """A single image result from a search provider."""

    url: str
    title: str = ""
    source_provider: str = ""
    source_url: str = ""
    thumbnail_url: str = ""
    width: int = 0
    height: int = 0
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageSource(ABC):
    """Abstract base class for image search providers."""

    provider_name: str = "base"

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[ImageResult]:
        """Search for images matching the query."""

    async def setup(self) -> None:  # noqa: B027
        """Optional initialization."""

    async def teardown(self) -> None:  # noqa: B027
        """Optional cleanup."""
