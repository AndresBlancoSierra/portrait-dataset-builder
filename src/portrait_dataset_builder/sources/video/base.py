"""Base interface for video source providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoResult:
    """A single video result from a search provider."""

    url: str
    title: str = ""
    source_provider: str = ""
    duration: float = 0.0
    thumbnail_url: str = ""
    description: str = ""
    upload_date: str = ""
    channel: str = ""
    view_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoSource(ABC):
    """Abstract base class for video search providers."""

    provider_name: str = "base"

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[VideoResult]:
        """Search for videos matching the query."""

    @abstractmethod
    async def download(
        self,
        url: str,
        output_path: str,
        quality: str = "best[height<=1080]",
    ) -> str | None:
        """Download video and return local file path."""

    async def setup(self) -> None:  # noqa: B027
        pass

    async def teardown(self) -> None:  # noqa: B027
        pass
