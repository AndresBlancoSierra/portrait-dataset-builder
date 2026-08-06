"""Test classification logic."""

from __future__ import annotations

from portrait_dataset_builder.pipeline.classification import ClassificationStage
from portrait_dataset_builder.taxonomy import (
    ExpressionLabel,
    HorizontalPose,
    LightingLabel,
    VerticalPose,
)


class TestClassification:
    def setup_method(self) -> None:
        self.stage = ClassificationStage()

    def _make_face(self, yaw: float = 0.0, pitch: float = 0.0) -> object:
        class FakeFace:
            pass

        face = FakeFace()
        face.yaw = yaw
        face.pitch = pitch
        face.confidence = 0.95
        face.landmark_left_eye_x = 100.0
        face.landmark_left_eye_y = 150.0
        face.landmark_right_eye_x = 200.0
        face.landmark_right_eye_y = 150.0
        face.landmark_nose_x = 150.0
        face.landmark_nose_y = 200.0
        face.landmark_left_mouth_x = 130.0
        face.landmark_left_mouth_y = 250.0
        face.landmark_right_mouth_x = 170.0
        face.landmark_right_mouth_y = 250.0
        return face

    def _make_settings(self):
        class S:
            yaw_frontal_threshold = 15.0
            yaw_profile_threshold = 60.0
            pitch_up_threshold = -15.0
            pitch_down_threshold = 15.0

        return S()

    # ── Horizontal pose tests ────────────────────────────────────────────────

    def test_horizontal_pose_frontal(self) -> None:
        face = self._make_face(yaw=5.0, pitch=0.0)
        settings = self._make_settings()
        assert self.stage._classify_horizontal_pose(face, settings) == HorizontalPose.FRONTAL

    def test_horizontal_pose_profile_left(self) -> None:
        face = self._make_face(yaw=70.0)
        settings = self._make_settings()
        assert self.stage._classify_horizontal_pose(face, settings) == HorizontalPose.PROFILE_LEFT

    def test_horizontal_pose_profile_right(self) -> None:
        face = self._make_face(yaw=-70.0)
        settings = self._make_settings()
        assert self.stage._classify_horizontal_pose(face, settings) == HorizontalPose.PROFILE_RIGHT

    def test_horizontal_pose_three_quarter_left(self) -> None:
        face = self._make_face(yaw=35.0)
        settings = self._make_settings()
        assert self.stage._classify_horizontal_pose(face, settings) == HorizontalPose.QUARTER_LEFT

    def test_horizontal_pose_three_quarter_right(self) -> None:
        face = self._make_face(yaw=-35.0)
        settings = self._make_settings()
        assert self.stage._classify_horizontal_pose(face, settings) == HorizontalPose.QUARTER_RIGHT

    def test_horizontal_pose_boundary_frontal_quarter(self) -> None:
        face = self._make_face(yaw=15.0)
        settings = self._make_settings()
        assert self.stage._classify_horizontal_pose(face, settings) == HorizontalPose.FRONTAL

    def test_horizontal_pose_boundary_quarter_profile(self) -> None:
        face = self._make_face(yaw=60.0)
        settings = self._make_settings()
        assert self.stage._classify_horizontal_pose(face, settings) == HorizontalPose.PROFILE_LEFT

    # ── Vertical pose tests ──────────────────────────────────────────────────

    def test_vertical_pose_neutral(self) -> None:
        face = self._make_face(pitch=0.0)
        settings = self._make_settings()
        assert self.stage._classify_vertical_pose(face, settings) == VerticalPose.NEUTRAL

    def test_vertical_pose_looking_up(self) -> None:
        face = self._make_face(pitch=-20.0)
        settings = self._make_settings()
        assert self.stage._classify_vertical_pose(face, settings) == VerticalPose.LOOKING_UP

    def test_vertical_pose_looking_down(self) -> None:
        face = self._make_face(pitch=20.0)
        settings = self._make_settings()
        assert self.stage._classify_vertical_pose(face, settings) == VerticalPose.LOOKING_DOWN

    def test_vertical_pose_boundary_up(self) -> None:
        face = self._make_face(pitch=-15.0)
        settings = self._make_settings()
        assert self.stage._classify_vertical_pose(face, settings) == VerticalPose.NEUTRAL

    def test_vertical_pose_boundary_down(self) -> None:
        face = self._make_face(pitch=15.0)
        settings = self._make_settings()
        assert self.stage._classify_vertical_pose(face, settings) == VerticalPose.NEUTRAL

    # ── Combined horizontal + vertical (independent axes) ────────────────────

    def test_combined_three_quarter_left_looking_up(self) -> None:
        face = self._make_face(yaw=35.0, pitch=-20.0)
        settings = self._make_settings()
        h = self.stage._classify_horizontal_pose(face, settings)
        v = self.stage._classify_vertical_pose(face, settings)
        assert h == HorizontalPose.QUARTER_LEFT
        assert v == VerticalPose.LOOKING_UP

    def test_combined_profile_right_looking_down(self) -> None:
        face = self._make_face(yaw=-70.0, pitch=20.0)
        settings = self._make_settings()
        h = self.stage._classify_horizontal_pose(face, settings)
        v = self.stage._classify_vertical_pose(face, settings)
        assert h == HorizontalPose.PROFILE_RIGHT
        assert v == VerticalPose.LOOKING_DOWN

    def test_combined_frontal_looking_up(self) -> None:
        face = self._make_face(yaw=5.0, pitch=-20.0)
        settings = self._make_settings()
        h = self.stage._classify_horizontal_pose(face, settings)
        v = self.stage._classify_vertical_pose(face, settings)
        assert h == HorizontalPose.FRONTAL
        assert v == VerticalPose.LOOKING_UP

    # ── Legacy angle derivation ──────────────────────────────────────────────

    def test_derive_angle_neutral_vertical(self) -> None:
        assert self.stage._derive_angle("frontal", "neutral") == "frontal"
        assert self.stage._derive_angle("three_quarter_left", "neutral") == "three_quarter_left"

    def test_derive_angle_non_neutral_vertical(self) -> None:
        assert self.stage._derive_angle("frontal", "looking_up") == "looking_up"
        assert self.stage._derive_angle("three_quarter_left", "looking_down") == "looking_down"

    # ── Expression tests ─────────────────────────────────────────────────────

    def test_expression_neutral(self) -> None:
        face = self._make_face()
        assert self.stage._classify_expression(face) == ExpressionLabel.NEUTRAL

    def test_expression_smile(self) -> None:
        face = self._make_face()
        # Wider mouth relative to eye distance -> smile
        face.landmark_left_mouth_x = 110.0
        face.landmark_right_mouth_x = 190.0
        assert self.stage._classify_expression(face) == ExpressionLabel.SMILE

    def test_expression_laugh(self) -> None:
        face = self._make_face()
        # Very wide mouth -> laugh
        face.landmark_left_mouth_x = 90.0
        face.landmark_right_mouth_x = 210.0
        assert self.stage._classify_expression(face) == ExpressionLabel.LAUGH

    def test_expression_speaking(self) -> None:
        face = self._make_face()
        # Large nose-to-mouth distance -> speaking
        face.landmark_nose_y = 180.0
        face.landmark_left_mouth_y = 270.0
        face.landmark_right_mouth_y = 270.0
        assert self.stage._classify_expression(face) == ExpressionLabel.SPEAKING

    def test_expression_zero_eye_distance_returns_neutral(self) -> None:
        face = self._make_face()
        face.landmark_left_eye_x = 150.0
        face.landmark_left_eye_y = 150.0
        face.landmark_right_eye_x = 150.0
        face.landmark_right_eye_y = 150.0
        assert self.stage._classify_expression(face) == ExpressionLabel.NEUTRAL

    # ── Lighting tests ───────────────────────────────────────────────────────

    def test_lighting_returns_balanced_for_no_local_path(self) -> None:
        class FakeImage:
            local_path = None
        face = self._make_face()
        assert self.stage._classify_lighting(FakeImage(), face) == LightingLabel.BALANCED
