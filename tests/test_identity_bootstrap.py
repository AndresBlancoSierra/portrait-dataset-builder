"""Test identity bootstrap pipeline stage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from portrait_dataset_builder.config.settings import Settings
from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.pipeline.identity_bootstrap import (
    FaceCandidate,
    IdentityBootstrapStage,
    IdentityProfile,
    SeedCandidate,
)


class TestFaceCandidate:
    def test_creation(self) -> None:
        emb = np.random.randn(512).astype(np.float32)
        c = FaceCandidate(
            image_id=1,
            face_id=2,
            embedding=emb,
            confidence=0.95,
            bbox=(10.0, 20.0, 100.0, 120.0),
            yaw=-5.0,
            pitch=2.0,
            quality_score=0.8,
        )
        assert c.image_id == 1
        assert c.face_id == 2
        assert len(c.embedding) == 512
        assert c.confidence == 0.95
        assert c.yaw == -5.0
        assert c.quality_score == 0.8


class TestSeedCandidate:
    def test_creation(self) -> None:
        emb = np.random.randn(512).astype(np.float32)
        face = FaceCandidate(
            image_id=1,
            face_id=1,
            embedding=emb,
            confidence=0.9,
            bbox=(0.0, 0.0, 50.0, 50.0),
            yaw=0.0,
            pitch=0.0,
            quality_score=0.7,
        )
        seed = SeedCandidate(face=face)
        assert seed.identity_consistency == 0.0
        assert seed.seed_confidence == 0.0


class TestIdentityProfile:
    def test_creation(self) -> None:
        mean_emb = np.random.randn(512).astype(np.float32)
        median_emb = np.random.randn(512).astype(np.float32)
        profile = IdentityProfile(
            mean_prototype=mean_emb,
            geometric_median=median_emb,
            identity_confidence=0.85,
            seed_count=10,
            outlier_count=2,
            total_candidates=20,
        )
        assert profile.identity_confidence == 0.85
        assert profile.seed_count == 10
        assert profile.outlier_count == 2
        assert profile.total_candidates == 20


class TestIdentityBootstrapStage:
    def setup_method(self) -> None:
        self.stage = IdentityBootstrapStage()
        self.settings = Settings(
            identity="Test Person",
            identity_bootstrap={"enabled": True, "candidate_count": 5, "min_candidates": 2, "min_cluster_size": 2},
        )

    def test_stage_name(self) -> None:
        assert self.stage.name == "identity_bootstrap"

    def test_guess_extension_jpeg(self) -> None:
        ext = self.stage._guess_extension("image/jpeg", b"")
        assert ext == ".jpg"

    def test_guess_extension_png(self) -> None:
        ext = self.stage._guess_extension("image/png", b"")
        assert ext == ".png"

    def test_guess_extension_from_content_jpeg(self) -> None:
        ext = self.stage._guess_extension(None, b"\xff\xd8\xff\xe0")
        assert ext == ".jpg"

    def test_guess_extension_from_content_png(self) -> None:
        ext = self.stage._guess_extension(None, b"\x89PNG\r\n\x1a\n")
        assert ext == ".png"

    def test_guess_extension_unknown(self) -> None:
        ext = self.stage._guess_extension("image/webp", b"")
        assert ext == ".webp"

    def test_select_diverse_sample(self) -> None:
        results = [
            MagicMock(source_provider="a"),
            MagicMock(source_provider="a"),
            MagicMock(source_provider="b"),
            MagicMock(source_provider="b"),
            MagicMock(source_provider="c"),
        ]
        sample = self.stage._select_diverse_sample(results, 3)
        sources = [r.source_provider for r in sample]
        assert sources == ["a", "b", "c"]

    def test_select_diverse_sample_fewer_than_count(self) -> None:
        results = [
            MagicMock(source_provider="a"),
            MagicMock(source_provider="a"),
        ]
        sample = self.stage._select_diverse_sample(results, 5)
        assert len(sample) == 2

    def test_compute_pose_diversity_no_seeds(self) -> None:
        emb = np.random.randn(512).astype(np.float32)
        face = FaceCandidate(
            image_id=1, face_id=1, embedding=emb,
            confidence=0.9, bbox=(0, 0, 50, 50), yaw=0.0, pitch=0.0,
        )
        seed = SeedCandidate(face=face)
        diversity = self.stage._compute_pose_diversity(seed, [])
        assert diversity == 1.0

    def test_compute_pose_diversity_same_pose(self) -> None:
        emb = np.random.randn(512).astype(np.float32)
        face1 = FaceCandidate(
            image_id=1, face_id=1, embedding=emb,
            confidence=0.9, bbox=(0, 0, 50, 50), yaw=10.0, pitch=5.0,
        )
        face2 = FaceCandidate(
            image_id=2, face_id=2, embedding=emb,
            confidence=0.9, bbox=(0, 0, 50, 50), yaw=10.0, pitch=5.0,
        )
        seed1 = SeedCandidate(face=face1)
        seed2 = SeedCandidate(face=face2)
        diversity = self.stage._compute_pose_diversity(seed2, [seed1])
        assert diversity == 0.0

    def test_compute_pose_diversity_different_pose(self) -> None:
        emb = np.random.randn(512).astype(np.float32)
        face1 = FaceCandidate(
            image_id=1, face_id=1, embedding=emb,
            confidence=0.9, bbox=(0, 0, 50, 50), yaw=-40.0, pitch=0.0,
        )
        face2 = FaceCandidate(
            image_id=2, face_id=2, embedding=emb,
            confidence=0.9, bbox=(0, 0, 50, 50), yaw=40.0, pitch=0.0,
        )
        seed1 = SeedCandidate(face=face1)
        seed2 = SeedCandidate(face=face2)
        diversity = self.stage._compute_pose_diversity(seed2, [seed1])
        assert diversity > 0.5

    def test_geometric_median_single_vector(self) -> None:
        vec = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = IdentityBootstrapStage._geometric_median(vec)
        np.testing.assert_allclose(result, vec[0], atol=1e-4)

    def test_geometric_median_two_same_vectors(self) -> None:
        vec = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        result = IdentityBootstrapStage._geometric_median(vec)
        np.testing.assert_allclose(result, vec[0], atol=1e-4)

    def test_geometric_median_two_opposite_vectors(self) -> None:
        vec = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
        result = IdentityBootstrapStage._geometric_median(vec)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-4)

    def test_cluster_identical(self) -> None:
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [
            FaceCandidate(i, i, emb, 0.9, (0, 0, 50, 50), 0.0, 0.0, 0.8)
            for i in range(4)
        ]
        result = self.stage._cluster_identities(candidates, threshold=0.5)
        assert len(result["main_cluster"]) == 4
        assert len(result["outliers"]) == 0

    def test_cluster_two_groups(self) -> None:
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        candidates = [
            FaceCandidate(i, i, emb_a, 0.9, (0, 0, 50, 50), 0.0, 0.0, 0.8)
            for i in range(3)
        ] + [
            FaceCandidate(i + 3, i + 3, emb_b, 0.9, (0, 0, 50, 50), 0.0, 0.0, 0.8)
            for i in range(3)
        ]
        result = self.stage._cluster_identities(candidates, threshold=0.5)
        assert len(result["main_cluster"]) == 3
        assert len(result["outliers"]) == 3

    def test_rank_and_select_seeds_empty(self) -> None:
        seeds = self.stage._rank_and_select_seeds([], np.zeros(512), 10, 20)
        assert seeds == []

    def test_rank_and_select_seeds_limits_count(self) -> None:
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        center = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [
            FaceCandidate(i, i, emb, 0.9, (0, 0, 50, 50), 0.0, 0.0, 0.8)
            for i in range(10)
        ]
        seeds = self.stage._rank_and_select_seeds(candidates, center, target_count=5, max_count=20)
        assert len(seeds) == 5

    def test_rank_and_select_seeds_max_limit(self) -> None:
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        center = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [
            FaceCandidate(i, i, emb, 0.9, (0, 0, 50, 50), 0.0, 0.0, 0.8)
            for i in range(5)
        ]
        seeds = self.stage._rank_and_select_seeds(candidates, center, target_count=20, max_count=3)
        assert len(seeds) == 3

    def test_build_profile_empty(self) -> None:
        profile = self.stage._build_profile([], {"main_cluster": [], "outliers": []}, 0)
        assert profile.identity_confidence == 0.0
        assert profile.seed_count == 0

    def test_build_profile_confidence(self) -> None:
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        seeds = [
            SeedCandidate(
                face=FaceCandidate(i, i, emb, 0.9, (0, 0, 50, 50), 0.0, 0.0, 0.8),
                identity_consistency=0.9,
                quality_score=0.8,
                seed_confidence=0.85,
            )
            for i in range(5)
        ]
        cluster = [FaceCandidate(i, i, emb, 0.9, (0, 0, 50, 50), 0.0, 0.0, 0.8) for i in range(5)]
        profile = self.stage._build_profile(
            seeds,
            {"main_cluster": cluster, "outliers": []},
            total_candidates=5,
        )
        assert profile.identity_confidence > 0.7
        assert profile.seed_count == 5
        assert profile.outlier_count == 0

    def test_build_profile_with_outliers(self) -> None:
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        seeds = [
            SeedCandidate(
                face=FaceCandidate(i, i, emb, 0.9, (0, 0, 50, 50), 0.0, 0.0, 0.8),
                identity_consistency=0.9,
                quality_score=0.8,
                seed_confidence=0.85,
            )
            for i in range(5)
        ]
        cluster = [FaceCandidate(i, i, emb, 0.9, (0, 0, 50, 50), 0.0, 0.0, 0.8) for i in range(5)]
        outliers = [FaceCandidate(i, i, emb, 0.9, (0, 0, 50, 50), 0.0, 0.0, 0.8) for i in range(3)]
        profile = self.stage._build_profile(
            seeds,
            {"main_cluster": cluster, "outliers": outliers},
            total_candidates=8,
        )
        assert profile.outlier_count == 3
        assert profile.identity_confidence > 0.3
