"""Benchmark test suite for validating pipeline improvements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from portrait_dataset_builder.coverage import (
    DatasetScore,
    ExpressionDiversity,
    LightingDiversity,
    PoseCoverage,
    TemporalDiversity,
)

BENCHMARK_DIR = Path(__file__).parent
SAMPLE_DATA_DIR = BENCHMARK_DIR / "sample_data"


@dataclass
class BenchmarkResult:
    metric: str
    value: float
    threshold: float
    passed: bool
    details: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.metric}: {self.value:.3f} "
            f"(threshold: {self.threshold:.3f}) {self.details}"
        )


class BenchmarkSuite:
    def __init__(self) -> None:
        self.results: list[BenchmarkResult] = []

    def add(self, metric: str, value: float, threshold: float, details: str = "") -> None:
        passed = value >= threshold
        result = BenchmarkResult(metric, value, threshold, passed, details)
        self.results.append(result)

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("  Benchmark Results")
        lines.append("=" * 60)
        for r in self.results:
            lines.append(f"  {r}")
        lines.append("-" * 60)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        lines.append(f"  {passed}/{total} benchmarks passed")
        lines.append("=" * 60)
        return "\n".join(lines)


class TestCoverageMetrics:
    """Test coverage analysis metrics."""

    def test_pose_coverage_grid(self) -> None:
        pc = PoseCoverage()
        poses = [
            (0.0, 0.0),
            (45.0, 10.0),
            (-45.0, -10.0),
            (30.0, 5.0),
            (-30.0, -5.0),
            (60.0, 15.0),
            (-60.0, -15.0),
        ]
        result = pc.compute(poses)

        assert result["total_images"] == 7
        assert result["populated_bins"] > 0
        assert result["coverage_pct"] > 0

    def test_pose_coverage_empty(self) -> None:
        pc = PoseCoverage()
        result = pc.compute([])
        assert result["coverage_pct"] == 0.0
        assert result["total_images"] == 0

    def test_pose_coverage_ascii(self) -> None:
        pc = PoseCoverage()
        poses = [(0.0, 0.0), (45.0, 10.0), (-45.0, -10.0)]
        ascii_art = pc.render_ascii(poses)
        assert "│" in ascii_art
        assert "Coverage:" in ascii_art

    def test_expression_diversity(self) -> None:
        ed = ExpressionDiversity()
        expressions = ["neutral", "smile", "neutral", "smile", "laugh"]
        result = ed.compute(expressions)

        assert result["unique_expressions"] == 3
        assert result["total_images"] == 5
        assert "neutral" in result["distribution"]
        assert result["entropy"] > 0

    def test_expression_diversity_single(self) -> None:
        ed = ExpressionDiversity()
        result = ed.compute(["neutral", "neutral", "neutral"])
        assert result["unique_expressions"] == 1
        assert result["entropy"] == 0.0

    def test_lighting_diversity(self) -> None:
        ld = LightingDiversity()
        data = [
            {"brightness": 0.3, "contrast": 0.5, "shadow_ratio": 0.1},
            {"brightness": 0.7, "contrast": 0.8, "shadow_ratio": 0.3},
            {"brightness": 0.5, "contrast": 0.6, "shadow_ratio": 0.2},
        ]
        result = ld.compute(data)

        assert result["brightness_range"][0] < result["brightness_range"][1]
        assert result["diversity_score"] > 0

    def test_temporal_diversity(self) -> None:
        td = TemporalDiversity()
        frames = [
            {"yaw": 0, "pitch": 0},
            {"yaw": 45, "pitch": 10},
            {"yaw": -45, "pitch": -10},
            {"yaw": 0, "pitch": 0},
        ]
        result = td.compute_video(frames)

        assert result["new_poses"] == 3
        assert result["total_frames"] == 4
        assert result["diversity_ratio"] > 0

    def test_dataset_score(self) -> None:
        ds = DatasetScore.compute(
            {
                "purity": 0.95,
                "pose_div": 0.80,
                "expr_div": 0.70,
                "light_div": 0.60,
                "dup_rate": 0.05,
                "avg_quality": 0.85,
            }
        )

        assert ds.overall > 0
        assert ds.overall <= 10
        assert ds.identity_purity == 0.95

    def test_dataset_score_render(self) -> None:
        ds = DatasetScore.compute(
            {
                "purity": 0.95,
                "pose_div": 0.80,
                "expr_div": 0.70,
                "light_div": 0.60,
                "dup_rate": 0.05,
                "avg_quality": 0.85,
            }
        )
        rendered = ds.render()
        assert "OVERALL" in rendered
        assert "█" in rendered


class TestVectorBackend:
    """Test vector backend implementations."""

    def test_usearch_backend_add_search(self) -> None:
        try:
            from portrait_dataset_builder.vector_backend import USearchBackend

            backend = USearchBackend(dim=4, metric="cosine")
            backend.add(0, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            backend.add(1, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            backend.add(2, np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32))

            results = backend.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=2)
            assert len(results) == 2
            assert results[0][0] == 0
        except ImportError:
            pytest.skip("usearch not installed")

    def test_usearch_backend_remove(self) -> None:
        try:
            from portrait_dataset_builder.vector_backend import USearchBackend

            backend = USearchBackend(dim=4, metric="cosine")
            backend.add(0, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            backend.add(1, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

            backend.remove(0)
            assert 0 not in backend
            assert 1 in backend
        except ImportError:
            pytest.skip("usearch not installed")

    def test_union_find(self) -> None:
        from portrait_dataset_builder.core.clustering import UnionFind

        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(1, 2)

        assert uf.find(0) == uf.find(1)
        assert uf.find(0) == uf.find(2)
        assert uf.find(0) == uf.find(3)
        assert uf.find(0) != uf.find(4)


class TestDuplicateDetection:
    """Test duplicate detection logic."""

    def test_hash_cluster(self) -> None:
        import imagehash
        from PIL import Image as PILImage

        from portrait_dataset_builder.pipeline.duplicates import DuplicateDetectionStage

        stage = DuplicateDetectionStage()

        img1 = PILImage.new("RGB", (100, 100), (255, 0, 0))
        img2 = PILImage.new("RGB", (100, 100), (255, 0, 0))
        img3 = PILImage.new("RGB", (100, 100), (0, 0, 255))

        hashes = {
            1: imagehash.phash(img1),
            2: imagehash.phash(img2),
            3: imagehash.phash(img3),
        }

        groups = stage._cluster_hashes(hashes, threshold=5)
        assert len(groups) >= 1


class TestQualityMetrics:
    """Test quality scoring algorithms."""

    def test_jpeg_blockiness(self) -> None:
        import numpy as np

        from portrait_dataset_builder.pipeline.quality import QualityStage

        stage = QualityStage()

        clean = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        score_clean = stage._estimate_jpeg_quality(clean)
        assert 0.0 <= score_clean <= 1.0

    def test_occlusion_landmark_based(self) -> None:
        from portrait_dataset_builder.pipeline.quality import QualityStage

        stage = QualityStage()

        class MockFace:
            landmark_left_eye_x = 100.0
            landmark_left_eye_y = 100.0
            landmark_right_eye_x = 200.0
            landmark_right_eye_y = 100.0
            landmark_nose_x = 150.0
            landmark_nose_y = 150.0
            landmark_left_mouth_x = 120.0
            landmark_left_mouth_y = 180.0
            landmark_right_mouth_x = 180.0
            landmark_right_mouth_y = 180.0
            bbox_x = 50.0
            bbox_y = 50.0
            bbox_w = 200.0
            bbox_h = 200.0

        score = stage._estimate_occlusion(MockFace())
        assert 0.0 <= score <= 1.0
