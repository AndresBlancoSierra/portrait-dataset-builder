"""YouTube video source using yt-dlp."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from portrait_dataset_builder.logging import get_logger
from portrait_dataset_builder.plugins import video_source
from portrait_dataset_builder.sources.video.base import VideoResult, VideoSource

logger = get_logger("source.youtube")


@video_source("youtube")
class YouTubeVideoSource(VideoSource):
    """Search and download YouTube videos using yt-dlp."""

    provider_name = "youtube"

    async def search(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[VideoResult]:
        search_query = f"{query} interview OR press conference OR appearance"
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-download",
            "--match-filter",
            "duration>10 & duration<3600",
            f"ytsearch{max_results}:{search_query}",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except TimeoutError:
            logger.error("YouTube search timed out for '{}'", query)
            return []
        except FileNotFoundError:
            logger.error("yt-dlp not found. Install it with: pip install yt-dlp")
            return []

        results: list[VideoResult] = []
        if stdout:
            for line in stdout.decode("utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    results.append(
                        VideoResult(
                            url=data.get("url", data.get("webpage_url", "")),
                            title=data.get("title", ""),
                            source_provider=self.provider_name,
                            duration=float(data.get("duration", 0)),
                            thumbnail_url=data.get("thumbnail", ""),
                            description=data.get("description", ""),
                            upload_date=data.get("upload_date", ""),
                            channel=data.get("channel", data.get("uploader", "")),
                            view_count=int(data.get("view_count", 0)),
                            metadata={
                                "id": data.get("id", ""),
                                "categories": data.get("categories", []),
                                "tags": data.get("tags", []),
                            },
                        )
                    )
                except json.JSONDecodeError:
                    continue

        logger.info("YouTube: found {} results for '{}'", len(results), query)
        return results

    async def download(
        self,
        url: str,
        output_path: str,
        quality: str = "best[height<=1080]",
    ) -> str | None:
        output_template = str(Path(output_path) / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f",
            quality,
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            "--no-playlist",
            "--socket-timeout",
            "30",
            "--retries",
            "3",
            url,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        except TimeoutError:
            logger.error("YouTube download timed out for {}", url)
            return None
        except FileNotFoundError:
            logger.error("yt-dlp not found")
            return None

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8") if stderr else "unknown error"
            logger.error("YouTube download failed: {}", err_msg[:200])
            return None

        output_dir = Path(output_path)
        for f in output_dir.iterdir():
            if f.suffix in [".mp4", ".webm", ".mkv"]:
                return str(f)

        return None
