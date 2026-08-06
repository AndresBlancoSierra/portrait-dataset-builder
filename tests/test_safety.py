"""Tests for the safety pipeline stages."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from portrait_dataset_builder.pipeline.safety_gate import (
    SafetyGateStage,
    _combined_ai_score,
    _detect_ai_metadata,
    _detect_ai_visual,
)
from portrait_dataset_builder.pipeline.url_safety_filter import (
    URLSafetyFilterStage,
    get_trust_score,
)
from portrait_dataset_builder.sources.image.base import ImageResult

# ── URL Safety Filter tests ─────────────────────────────────────────────────


class TestTrustScoring:
    def test_flickr_provider(self):
        trust_map = {"flickr": 0.90, "unknown": 0.30}
        assert get_trust_score("https://example.com/img.jpg", "flickr", trust_map) == 0.90

    def test_wikimedia_url(self):
        trust_map = {"wikimedia": 0.95, "unknown": 0.30}
        assert get_trust_score("https://upload.wikimedia.org/img.jpg", "unknown", trust_map) == 0.95

    def test_gov_url(self):
        trust_map = {"unknown": 0.30}
        assert get_trust_score("https://whitehouse.gov/photo.jpg", "unknown", trust_map) == 1.0

    def test_edu_url(self):
        trust_map = {"unknown": 0.30}
        assert get_trust_score("https://mit.edu/img.jpg", "unknown", trust_map) == 1.0

    def test_unknown_provider(self):
        trust_map = {"unknown": 0.30}
        assert get_trust_score("https://random-site.com/img.jpg", "unknown", trust_map) == 0.30

    def test_bbc_url(self):
        trust_map = {"unknown": 0.30}
        assert get_trust_score("https://bbc.com/news/photo.jpg", "unknown", trust_map) == 1.0


class TestURLSafetyFilterStage:
    @pytest.fixture
    def stage(self):
        return URLSafetyFilterStage()

    @pytest.mark.asyncio
    async def test_should_run_with_results(self, stage):
        ctx = MagicMock()
        ctx.metadata = {"image_results": [MagicMock()]}
        assert await stage.should_run(ctx) is True

    @pytest.mark.asyncio
    async def test_should_run_empty(self, stage):
        ctx = MagicMock()
        ctx.metadata = {"image_results": []}
        assert await stage.should_run(ctx) is False

    @pytest.mark.asyncio
    async def test_blocks_nsfw_urls(self, stage):
        ctx = MagicMock()
        ctx.metadata = {
            "image_results": [
                ImageResult(url="https://example.com/erotic-photo.jpg", source_provider="unknown"),
                ImageResult(url="https://example.com/portrait.jpg", source_provider="flickr"),
            ]
        }
        ctx.settings.safety.blocked_url_keywords = ["erotic", "nude", "porn"]
        ctx.settings.safety.source_trust_scores = {"flickr": 0.90, "unknown": 0.30}

        result = await stage.execute(ctx)

        assert result.items_rejected == 1
        assert result.items_succeeded == 1
        assert len(ctx.metadata["image_results"]) == 1

    @pytest.mark.asyncio
    async def test_blocks_low_trust(self, stage):
        ctx = MagicMock()
        ctx.metadata = {
            "image_results": [
                ImageResult(url="https://random-site.com/img.jpg", source_provider="unknown"),
            ]
        }
        ctx.settings.safety.blocked_url_keywords = ["porn"]
        ctx.settings.safety.source_trust_scores = {"unknown": 0.20}

        result = await stage.execute(ctx)

        assert result.items_rejected == 1
        assert result.items_succeeded == 0

    @pytest.mark.asyncio
    async def test_allows_safe_urls(self, stage):
        ctx = MagicMock()
        ctx.metadata = {
            "image_results": [
                ImageResult(
                    url="https://flickr.com/photo/123",
                    source_provider="flickr",
                ),
                ImageResult(
                    url="https://upload.wikimedia.org/img.jpg",
                    source_provider="wikimedia",
                ),
            ]
        }
        ctx.settings.safety.blocked_url_keywords = ["porn"]
        ctx.settings.safety.source_trust_scores = {
            "flickr": 0.90, "wikimedia": 0.95, "unknown": 0.30,
        }

        result = await stage.execute(ctx)

        assert result.items_succeeded == 2
        assert result.items_rejected == 0


# ── Safety Gate tests ────────────────────────────────────────────────────────


class TestAIDetection:
    def test_metadata_detects_stable_diffusion(self):
        img = PILImage.new("RGB", (100, 100), color="blue")
        img.info = {"exif": b"software=Stable Diffusion v2.1"}
        detected, score = _detect_ai_metadata(img)
        assert detected is True
        assert score >= 0.80

    def test_metadata_detects_midjourney(self):
        img = PILImage.new("RGB", (100, 100), color="blue")
        img.info = {"parameters": "--style raw --ar 16:9"}
        detected, score = _detect_ai_metadata(img)
        assert detected is True
        assert score >= 0.80

    def test_metadata_clean_photo(self):
        img = PILImage.new("RGB", (100, 100), color="blue")
        img.info = {}
        detected, score = _detect_ai_metadata(img)
        assert detected is False
        assert score == 0.0

    def test_visual_detects_flat_image(self):
        flat = np.ones((64, 64, 3), dtype=np.uint8) * 128
        detected, score = _detect_ai_visual(flat)
        assert detected is True
        assert score >= 0.60

    def test_visual_accepts_textured_image(self):
        rng = np.random.RandomState(42)
        textured = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        detected, score = _detect_ai_visual(textured)
        assert detected is False

    def test_combined_score_metadata_only(self):
        score = _combined_ai_score(0.90, 0.0, True, False)
        assert score >= 0.80

    def test_combined_score_visual_only(self):
        score = _combined_ai_score(0.0, 0.70, False, True)
        assert score >= 0.60

    def test_combined_score_clean(self):
        score = _combined_ai_score(0.0, 0.0, False, False)
        assert score == 0.0


class TestSafetyGateStage:
    @pytest.fixture
    def stage(self):
        return SafetyGateStage()

    @pytest.mark.asyncio
    async def test_should_run_with_downloaded(self, stage):
        ctx = MagicMock()
        ctx.db_path = Path("/tmp/test.db")
        mock_repo = MagicMock()

        async def fake_count(state, limit=100000):
            return 5

        mock_repo.count_by_state = fake_count
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        sg = "portrait_dataset_builder.pipeline.safety_gate"
        with patch(f"{sg}.get_engine"), \
             patch(f"{sg}.get_session", return_value=mock_session), \
             patch(f"{sg}.ImageRepository", return_value=mock_repo):
            assert await stage.should_run(ctx) is True

    @pytest.mark.asyncio
    async def test_fail_closed_without_model(self, stage):
        ctx = MagicMock()
        ctx.db_path = Path("/tmp/test.db")
        ctx.settings.safety.fail_closed = True
        ctx.settings.safety.nsfw_threshold = 0.20
        ctx.settings.safety.clip_safety_threshold = 0.50
        ctx.settings.safety.ai_mode = "strict"

        mock_repo = MagicMock()

        async def fake_count(state, limit=100000):
            return 5

        async def fake_get(state, limit=100000):
            return []

        mock_repo.count_by_state = fake_count
        mock_repo.get_by_state = fake_get
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        sg = "portrait_dataset_builder.pipeline.safety_gate"
        with patch(f"{sg}._load_nsfw_model", return_value=None), \
             patch(f"{sg}._load_clip", return_value=(None, None, None, None)), \
             patch(f"{sg}.get_engine"), \
             patch(f"{sg}.get_session", return_value=mock_session), \
             patch(f"{sg}.ImageRepository", return_value=mock_repo):
            from portrait_dataset_builder.core.pipeline import StageStatus
            result = await stage.execute(ctx)
            assert result.status == StageStatus.FAILED
            assert "NSFW model unavailable" in result.errors[0]
