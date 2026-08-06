"""Test quality assessment logic."""

from __future__ import annotations

import numpy as np

from portrait_dataset_builder.pipeline.quality import QualityStage


class TestQualityAssessment:
    def setup_method(self) -> None:
        self.stage = QualityStage()

    def test_assess_lighting_bright(self) -> None:
        gray = np.full((100, 100), 200, dtype=np.uint8)
        score = self.stage._assess_lighting(gray)
        assert 0.0 <= score <= 1.0

    def test_assess_lighting_dark(self) -> None:
        gray = np.full((100, 100), 30, dtype=np.uint8)
        score = self.stage._assess_lighting(gray)
        assert 0.0 <= score <= 1.0

    def test_assess_lighting_optimal(self) -> None:
        gray = np.full((100, 100), 128, dtype=np.uint8)
        score = self.stage._assess_lighting(gray)
        assert score >= 0.5

    def test_estimate_noise_clean(self) -> None:
        gray = np.full((100, 100), 128, dtype=np.uint8)
        score = self.stage._estimate_noise(gray)
        assert 0.0 <= score <= 1.0

    def test_estimate_noise_small_image(self) -> None:
        gray = np.full((2, 2), 128, dtype=np.uint8)
        score = self.stage._estimate_noise(gray)
        assert score == 0.5

    def test_compute_scores_structure(self) -> None:
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        class FakeFace:
            bbox_x = 20.0
            bbox_y = 20.0
            bbox_w = 100.0
            bbox_h = 120.0
            yaw = 5.0
            pitch = -2.0
            confidence = 0.95
            face_width = 100
            face_height = 120
            landmark_left_eye_x = 80.0
            landmark_left_eye_y = 60.0
            landmark_right_eye_x = 140.0
            landmark_right_eye_y = 60.0
            landmark_nose_x = 110.0
            landmark_nose_y = 90.0
            landmark_left_mouth_x = 90.0
            landmark_left_mouth_y = 110.0
            landmark_right_mouth_x = 130.0
            landmark_right_mouth_y = 110.0

        class FakeImage:
            width = 200
            height = 200

        scores = self.stage._compute_scores(img, FakeFace(), FakeImage())
        assert "resolution" in scores
        assert "sharpness" in scores
        assert "blur" in scores
        assert "noise" in scores
        assert "lighting" in scores
        assert "face_size" in scores
        assert "frontal" in scores

        for key, value in scores.items():
            assert 0.0 <= value <= 1.0, f"Score {key} = {value} out of range"
