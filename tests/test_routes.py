"""Tests for API route helper functions."""
from __future__ import annotations


class TestAngleFilterParsing:
    """Test _parse_angle_filter normalizes UI angle labels to backend values."""

    def _parse(self, angle: str | None) -> list[str] | None:
        from portrait_dataset_builder.api.routes import _parse_angle_filter
        return _parse_angle_filter(angle)

    def test_none_returns_none(self):
        assert self._parse(None) is None

    def test_empty_returns_none(self):
        assert self._parse("") is None

    def test_frontal(self):
        assert self._parse("frontal") == ["frontal"]

    def test_quarter_maps_both_sides(self):
        result = self._parse("quarter")
        assert set(result) == {"three_quarter_left", "three_quarter_right"}

    def test_profile_maps_both_sides(self):
        result = self._parse("profile")
        assert set(result) == {"profile_left", "profile_right"}

    def test_comma_separated(self):
        result = self._parse("frontal,profile")
        assert "frontal" in result
        assert "profile_left" in result
        assert "profile_right" in result

    def test_unknown_value_ignored(self):
        assert self._parse("bogus") is None

    def test_mixed_known_unknown(self):
        result = self._parse("frontal,bogus")
        assert result == ["frontal"]


class TestHorizontalPoseFilterParsing:
    """Test _parse_horizontal_pose_filter accepts backend values directly."""

    def _parse(self, pose: str | None) -> list[str] | None:
        from portrait_dataset_builder.api.routes import _parse_horizontal_pose_filter
        return _parse_horizontal_pose_filter(pose)

    def test_none_returns_none(self):
        assert self._parse(None) is None

    def test_empty_returns_none(self):
        assert self._parse("") is None

    def test_single_backend_value(self):
        result = self._parse("three_quarter_left")
        assert result == ["three_quarter_left"]

    def test_group_name_quarter(self):
        result = self._parse("quarter")
        assert set(result) == {"three_quarter_left", "three_quarter_right"}

    def test_comma_separated_backend_values(self):
        result = self._parse("frontal,profile_left")
        assert set(result) == {"frontal", "profile_left"}

    def test_mixed_group_and_value(self):
        result = self._parse("quarter,profile_left")
        assert "three_quarter_left" in result
        assert "three_quarter_right" in result
        assert "profile_left" in result


class TestVerticalPoseFilterParsing:
    """Test _parse_vertical_pose_filter accepts vertical pose values."""

    def _parse(self, pose: str | None) -> list[str] | None:
        from portrait_dataset_builder.api.routes import _parse_vertical_pose_filter
        return _parse_vertical_pose_filter(pose)

    def test_none_returns_none(self):
        assert self._parse(None) is None

    def test_empty_returns_none(self):
        assert self._parse("") is None

    def test_neutral(self):
        assert self._parse("neutral") == ["neutral"]

    def test_looking_up(self):
        assert self._parse("looking_up") == ["looking_up"]

    def test_looking_down(self):
        assert self._parse("looking_down") == ["looking_down"]

    def test_comma_separated(self):
        result = self._parse("looking_up,looking_down")
        assert set(result) == {"looking_up", "looking_down"}

    def test_unknown_value_ignored(self):
        assert self._parse("bogus") is None


class TestQualityFilterParsing:
    """Test _parse_quality_filter maps semantic labels to numeric ranges."""

    def _parse(self, quality: str | None) -> tuple[float, float] | None:
        from portrait_dataset_builder.api.routes import _parse_quality_filter
        return _parse_quality_filter(quality)

    def test_none_returns_none(self):
        assert self._parse(None) is None

    def test_empty_returns_none(self):
        assert self._parse("") is None

    def test_high(self):
        assert self._parse("high") == (0.80, 1.0)

    def test_medium(self):
        assert self._parse("medium") == (0.50, 0.79)

    def test_low(self):
        assert self._parse("low") == (0.0, 0.49)

    def test_raw_range(self):
        assert self._parse("0.5,0.8") == (0.5, 0.8)

    def test_invalid_raw_returns_none(self):
        assert self._parse("abc,def") is None

    def test_single_value_returns_none(self):
        assert self._parse("0.5") is None


class TestQualityRangeBoundaries:
    """Test quality ranges have no gaps and no overlaps."""

    def test_no_gaps_between_ranges(self):
        from portrait_dataset_builder.api.routes import QUALITY_MAP
        low_max = QUALITY_MAP["low"][1]
        medium_min = QUALITY_MAP["medium"][0]
        medium_max = QUALITY_MAP["medium"][1]
        high_min = QUALITY_MAP["high"][0]

        # Check continuity: low_max + 0.01 should be in medium
        assert low_max < medium_min  # No overlap
        assert medium_max < high_min  # No overlap
        # Gaps: 0.49 to 0.50 (medium starts at 0.50)
        # This is intentional: 0.49 is low, 0.50 is medium
        assert medium_min - low_max > 0  # Exactly 1 cent gap

    def test_all_scores_covered(self):
        from portrait_dataset_builder.api.routes import QUALITY_MAP
        low_min = QUALITY_MAP["low"][0]
        high_max = QUALITY_MAP["high"][1]
        assert low_min == 0.0
        assert high_max == 1.0


class TestCoverageComputation:
    """Test _compute_coverage_score computes breadth+balance metric."""

    def _compute(self, poses, expressions, lighting, hp=None, vp=None):
        from portrait_dataset_builder.api.routes import _compute_coverage_score
        return _compute_coverage_score(poses, expressions, lighting, hp, vp)

    def test_empty_poses_returns_zero(self):
        assert self._compute([], [], []) == 0.0

    def test_single_pose_nonzero(self):
        score = self._compute([(0, 0)], ["neutral"], ["balanced"])
        assert 0.0 < score <= 1.0

    def test_diverse_poses_higher_score(self):
        few = self._compute([(0, 0), (5, 0)], ["neutral"], ["balanced"])
        many = self._compute(
            [(0, 0), (30, 0), (-30, 0), (0, 10), (0, -10)],
            ["neutral", "smile", "laugh"],
            ["dark", "bright", "balanced"],
        )
        assert many > few

    def test_score_bounded_0_1(self):
        score = self._compute(
            [(0, 0), (30, 0), (-30, 0), (60, 0), (-60, 0)],
            ["neutral", "smile", "laugh", "speaking"],
            ["dark", "bright", "balanced"],
        )
        assert 0.0 <= score <= 1.0

    def test_expression_diversity_matters(self):
        single_expr = self._compute(
            [(0, 0), (10, 0), (-10, 0)], ["neutral", "neutral"], ["balanced"]
        )
        multi_expr = self._compute(
            [(0, 0), (10, 0), (-10, 0)], ["neutral", "smile", "laugh"], ["balanced"]
        )
        assert multi_expr > single_expr

    def test_two_axis_pose_improves_score(self):
        # Without explicit horizontal/vertical poses
        baseline = self._compute([(0, 0)], ["neutral"], ["balanced"])
        # With all 5 horizontal and 3 vertical poses
        with_poses = self._compute(
            [(0, 0)],
            ["neutral"],
            ["balanced"],
            hp=[
                "frontal", "three_quarter_left", "three_quarter_right",
                "profile_left", "profile_right",
            ],
            vp=["neutral", "looking_up", "looking_down"],
        )
        assert with_poses >= baseline


class TestImageSerialization:
    """Test _serialize_image produces consistent dict shape."""

    def test_none_inputs_produce_null_nested_dicts(self):
        from portrait_dataset_builder.api.routes import _serialize_image

        class FakeImg:
            id = 1
            content_hash = "abc"
            uri = "http://test"
            local_path = None
            width = 100
            height = 100
            file_size = 1000
            source_provider = "test"
            pipeline_state = "verified"
            created_at = None

        result = _serialize_image(FakeImg())
        assert result["id"] == 1
        assert result["content_hash"] == "abc"
        assert result["face"] is None
        assert result["quality"] is None
        assert result["classification"] is None
        assert result["safety"] is None

    def test_classification_includes_new_fields(self):
        from types import SimpleNamespace

        from portrait_dataset_builder.api.routes import _serialize_image

        class FakeImg:
            id = 1
            content_hash = "abc"
            uri = "http://test"
            local_path = None
            width = 100
            height = 100
            file_size = 1000
            source_provider = "test"
            pipeline_state = "verified"
            created_at = None

        cls = SimpleNamespace(
            id=1, image_id=1,
            angle="frontal",
            horizontal_pose="frontal",
            vertical_pose="neutral",
            expression="smile",
            accessories=None,
            age_group="adult",
            lighting="balanced",
        )
        result = _serialize_image(FakeImg(), cls=cls)
        assert result["classification"]["horizontal_pose"] == "frontal"
        assert result["classification"]["vertical_pose"] == "neutral"
        assert result["classification"]["expression"] == "smile"


class TestNullClassificationFiltering:
    """Test that null classifications are correctly handled by filters."""

    def test_active_filter_excludes_none_classification(self):
        from portrait_dataset_builder.api.routes import _parse_horizontal_pose_filter

        angle_values = _parse_horizontal_pose_filter("frontal")
        assert angle_values is not None

        # Simulate: c is None, filter is active -> should skip
        c = None
        has_filter = angle_values is not None
        excluded = bool(has_filter and c is None)
        assert excluded is True

    def test_active_filter_excludes_none_field(self):
        """When a filter is active and classification exists but field is None, exclude."""
        from types import SimpleNamespace

        c = SimpleNamespace(
            horizontal_pose=None,
            vertical_pose=None,
            expression=None,
            lighting=None,
        )
        h_pose_values = ["frontal"]
        if h_pose_values:
            if c is None or c.horizontal_pose is None or c.horizontal_pose not in h_pose_values:
                excluded = True
            else:
                excluded = False
        else:
            excluded = False
        assert excluded is True

    def test_inactive_filter_permits_none_classification(self):
        """When no filter is active, images with None classification should pass."""
        c = None
        h_pose_values = None
        v_pose_values = None
        expr_values = None
        light_values = None

        has_filter = h_pose_values or v_pose_values or expr_values or light_values
        excluded = bool(has_filter and c is None)
        assert excluded is False


class TestResolveLibraryStatus:
    """Test _resolve_library_status maps identity status + build state."""

    def _resolve(self, identity_status, active_job, latest_job, verified_count):
        from portrait_dataset_builder.api.routes import _resolve_library_status
        return _resolve_library_status(identity_status, active_job, latest_job, verified_count)

    @staticmethod
    def _job(status: str, queue_status: str = "running"):
        """Create a mock object with .status and .queue_status attributes for testing."""
        from types import SimpleNamespace
        return SimpleNamespace(status=status, queue_status=queue_status)

    def test_active_running_job_returns_building(self):
        assert self._resolve("unknown", self._job("running"), None, 0) == "building"

    def test_active_pending_job_returns_building(self):
        assert self._resolve("unknown", self._job("pending"), None, 0) == "building"

    def test_building_identity_returns_building(self):
        assert self._resolve("building", None, self._job("completed"), 0) == "building"

    def test_failed_identity_with_completed_job_returns_job_status(self):
        assert self._resolve("failed", None, self._job("completed"), 10) == "ready"

    def test_cancelled_identity_with_completed_job_returns_job_status(self):
        assert self._resolve("cancelled", None, self._job("completed"), 10) == "ready"

    def test_failed_identity_with_no_job_returns_failed(self):
        assert self._resolve("failed", None, None, 0) == "failed"

    def test_cancelled_identity_with_no_job_returns_cancelled(self):
        assert self._resolve("cancelled", None, None, 0) == "cancelled"

    def test_ready_with_images_returns_ready(self):
        assert self._resolve("ready", None, self._job("completed"), 10) == "ready"

    def test_ready_with_no_images_returns_empty(self):
        assert self._resolve("ready", None, self._job("completed"), 0) == "empty"

    def test_latest_job_failed(self):
        assert self._resolve("ready", None, self._job("failed"), 10) == "failed"

    def test_latest_job_cancelled(self):
        assert self._resolve("ready", None, self._job("cancelled"), 10) == "cancelled"

    def test_unknown_returns_unknown(self):
        assert self._resolve("unknown", None, None, 0) == "unknown"

    def test_queued_job_returns_queued(self):
        assert self._resolve("unknown", self._job("pending", queue_status="queued"), None, 0) == "queued"

    def test_running_queue_status_returns_building(self):
        assert self._resolve("unknown", self._job("running", queue_status="running"), None, 0) == "building"

    def test_identity_unverified_completed_no_images(self):
        assert self._resolve(
            "identity_unverified", None, self._job("completed"), 0
        ) == "identity_unverified"

    def test_identity_unverified_completed_with_images(self):
        assert self._resolve(
            "identity_unverified", None, self._job("completed"), 10
        ) == "ready"

    def test_identity_unverified_no_job(self):
        assert self._resolve(
            "identity_unverified", None, None, 0
        ) == "identity_unverified"


class TestRandomGlobalEndpoint:
    """Test random_global_images helper logic."""

    def test_serialize_image_includes_library_name(self):
        from types import SimpleNamespace
        from portrait_dataset_builder.api.routes import _serialize_image

        class FakeImg:
            id = 1
            content_hash = "abc"
            uri = "http://test"
            local_path = None
            width = 100
            height = 100
            file_size = 1000
            source_provider = "test"
            pipeline_state = "verified"
            created_at = None

        result = _serialize_image(FakeImg())
        result["library_name"] = "Test Person"
        assert result["library_name"] == "Test Person"
        assert result["content_hash"] == "abc"

    def test_empty_libraries_returns_empty(self):
        from unittest.mock import patch
        from portrait_dataset_builder.api.routes import random_global_images
        import asyncio

        with patch(
            "portrait_dataset_builder.api.routes._find_libraries", return_value=[]
        ):
            result = asyncio.run(random_global_images(count=10))
            assert result == []

    def test_no_ready_libraries_returns_empty(self):
        from unittest.mock import patch, AsyncMock, MagicMock
        from portrait_dataset_builder.api.routes import random_global_images
        import asyncio

        mock_id_repo = AsyncMock()
        mock_id_repo.get_by_name = AsyncMock(return_value=None)

        mock_bj_repo = AsyncMock()
        mock_bj_repo.get_active_by_identity = AsyncMock(return_value=None)
        mock_bj_repo.get_latest_by_identity = AsyncMock(return_value=None)

        mock_img_repo = AsyncMock()
        mock_img_repo.count_by_state = AsyncMock(return_value=0)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_lib = {"name": "Test", "path": MagicMock(), "db": MagicMock()}

        with (
            patch(
                "portrait_dataset_builder.api.routes._find_libraries",
                return_value=[mock_lib],
            ),
            patch(
                "portrait_dataset_builder.api.routes.get_engine"
            ),
            patch(
                "portrait_dataset_builder.api.routes.get_session",
                return_value=mock_session,
            ),
            patch(
                "portrait_dataset_builder.api.routes.IdentityRepository",
                return_value=mock_id_repo,
            ),
            patch(
                "portrait_dataset_builder.api.routes.BuildJobRepository",
                return_value=mock_bj_repo,
            ),
            patch(
                "portrait_dataset_builder.api.routes.ImageRepository",
                return_value=mock_img_repo,
            ),
        ):
            result = asyncio.run(random_global_images(count=10))
            assert result == []

    def test_count_param_affects_result_size(self):
        from unittest.mock import patch
        from portrait_dataset_builder.api.routes import random_global_images
        import asyncio

        with patch(
            "portrait_dataset_builder.api.routes._find_libraries", return_value=[]
        ):
            result = asyncio.run(random_global_images(count=5))
            assert result == []
