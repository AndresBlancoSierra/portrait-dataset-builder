"""Test database models and repositories."""

from __future__ import annotations

from pathlib import Path

import pytest

from portrait_dataset_builder.database import (
    Classification,
    Face,
    Image,
    QualityScore,
    Video,
)
from portrait_dataset_builder.database.engine import (
    _engine_cache,
    get_engine,
    get_session,
    init_db,
)
from portrait_dataset_builder.database.repository import (
    ClassificationRepository,
    FaceRepository,
    IdentityRepository,
    ImageRepository,
    QualityRepository,
    VideoRepository,
)

TEST_DB = Path("/tmp/test_portrait_builder.db")


@pytest.fixture(autouse=True)
async def setup_db() -> None:
    """Create a fresh test database for each test."""
    TEST_DB.unlink(missing_ok=True)
    _engine_cache.pop(str(TEST_DB.resolve()), None)
    engine = get_engine(TEST_DB)
    await init_db(engine)
    yield
    TEST_DB.unlink(missing_ok=True)
    _engine_cache.pop(str(TEST_DB.resolve()), None)


@pytest.fixture
def engine():
    return get_engine(TEST_DB)


class TestImageRepository:
    @pytest.mark.asyncio
    async def test_add_image(self, engine) -> None:
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            img = Image(
                uri="test://example.com/img.jpg",
                local_path="/tmp/test.jpg",
                source_type="image_search",
                source_provider="test",
                content_hash="abc123",
                width=1920,
                height=1080,
                file_size=1024,
                mime_type="image/jpeg",
                pipeline_state="downloaded",
            )
            result = await repo.add(img)
            assert result.id is not None

    @pytest.mark.asyncio
    async def test_count(self, engine) -> None:
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            assert await repo.count() == 0

            img = Image(
                uri="test://1",
                content_hash="h1",
                pipeline_state="downloaded",
            )
            await repo.add(img)
            assert await repo.count() == 1

    @pytest.mark.asyncio
    async def test_exists(self, engine) -> None:
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            assert await repo.exists("nonexistent") is False

            img = Image(
                uri="test://1",
                content_hash="exists",
                pipeline_state="downloaded",
            )
            await repo.add(img)
            assert await repo.exists("exists") is True

    @pytest.mark.asyncio
    async def test_get_by_hash(self, engine) -> None:
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            img = Image(
                uri="test://1",
                content_hash="findme",
                pipeline_state="downloaded",
            )
            await repo.add(img)
            found = await repo.get_by_hash("findme")
            assert found is not None
            assert found.uri == "test://1"

    @pytest.mark.asyncio
    async def test_update_state(self, engine) -> None:
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            img = Image(
                uri="test://1",
                content_hash="state_test",
                pipeline_state="downloaded",
            )
            await repo.add(img)
            await repo.update_state(img.id, "verified")

            found = await repo.get_by_id(img.id)
            assert found is not None
            assert found.pipeline_state == "verified"

    @pytest.mark.asyncio
    async def test_get_by_state(self, engine) -> None:
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            for i in range(5):
                state = "downloaded" if i < 3 else "verified"
                img = Image(
                    uri=f"test://{i}",
                    content_hash=f"state_{i}",
                    pipeline_state=state,
                )
                await repo.add(img)

            downloaded = await repo.get_by_state("downloaded")
            assert len(downloaded) == 3

            verified = await repo.get_by_state("verified")
            assert len(verified) == 2

    @pytest.mark.asyncio
    async def test_count_by_state(self, engine) -> None:
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            for i in range(3):
                img = Image(
                    uri=f"test://{i}",
                    content_hash=f"cs_{i}",
                    pipeline_state="downloaded",
                )
                await repo.add(img)
            assert await repo.count_by_state("downloaded") == 3
            assert await repo.count_by_state("verified") == 0


class TestFaceRepository:
    @pytest.mark.asyncio
    async def test_add_face(self, engine) -> None:
        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            img = Image(
                uri="test://face",
                content_hash="face_img",
                pipeline_state="downloaded",
            )
            await img_repo.add(img)

            face_repo = FaceRepository(session)
            face = Face(
                image_id=img.id,
                bbox_x=10.0,
                bbox_y=20.0,
                bbox_w=100.0,
                bbox_h=120.0,
                yaw=5.0,
                pitch=-3.0,
                roll=1.0,
                confidence=0.95,
                face_width=100,
                face_height=120,
            )
            result = await face_repo.add(face)
            assert result.id is not None

    @pytest.mark.asyncio
    async def test_get_by_image_id(self, engine) -> None:
        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            img = Image(
                uri="test://face2",
                content_hash="face_img2",
                pipeline_state="downloaded",
            )
            await img_repo.add(img)

            face_repo = FaceRepository(session)
            for _ in range(3):
                face = Face(
                    image_id=img.id,
                    confidence=0.9,
                    face_width=100,
                    face_height=100,
                )
                await face_repo.add(face)

            faces = await face_repo.get_by_image_id(img.id)
            assert len(faces) == 3


class TestVideoRepository:
    @pytest.mark.asyncio
    async def test_add_and_get_video(self, engine) -> None:
        async with get_session(engine) as session:
            repo = VideoRepository(session)
            video = Video(
                url="https://youtube.com/watch?v=test",
                title="Test Video",
                source="youtube",
                duration=120.0,
                pipeline_state="pending",
            )
            await repo.add(video)

            found = await repo.get_by_url("https://youtube.com/watch?v=test")
            assert found is not None
            assert found.title == "Test Video"

    @pytest.mark.asyncio
    async def test_update_local_path(self, engine) -> None:
        async with get_session(engine) as session:
            repo = VideoRepository(session)
            video = Video(
                url="https://youtube.com/watch?v=test2",
                source="youtube",
                pipeline_state="pending",
            )
            await repo.add(video)
            await repo.update_local_path(video.id, "/tmp/video.mp4")

            found = await repo.get_by_id(video.id)
            assert found is not None
            assert found.local_path == "/tmp/video.mp4"
            assert found.pipeline_state == "downloaded"


class TestIdentityRepository:
    @pytest.mark.asyncio
    async def test_get_or_create(self, engine) -> None:
        async with get_session(engine) as session:
            repo = IdentityRepository(session)
            identity = await repo.get_or_create("Matt Damon")
            assert identity.id is not None
            assert identity.name == "Matt Damon"

            identity2 = await repo.get_or_create("Matt Damon")
            assert identity.id == identity2.id


class TestQualityRepository:
    @pytest.mark.asyncio
    async def test_add_quality_score(self, engine) -> None:
        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            img = Image(
                uri="test://quality",
                content_hash="quality_img",
                pipeline_state="downloaded",
            )
            await img_repo.add(img)

            q_repo = QualityRepository(session)
            quality = QualityScore(
                image_id=img.id,
                resolution_score=0.9,
                sharpness_score=0.85,
                blur_score=0.95,
                noise_score=0.9,
                lighting_score=0.8,
                occlusion_score=1.0,
                face_size_score=0.7,
                frontal_score=0.95,
                jpeg_score=0.85,
                final_score=0.87,
            )
            await q_repo.add(quality)

            found = await q_repo.get_by_image_id(img.id)
            assert found is not None
            assert found.final_score == 0.87


class TestClassificationRepository:
    @pytest.mark.asyncio
    async def test_add_classification(self, engine) -> None:
        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            img = Image(
                uri="test://cls",
                content_hash="cls_img",
                pipeline_state="downloaded",
            )
            await img_repo.add(img)

            cls_repo = ClassificationRepository(session)
            cls = Classification(
                image_id=img.id,
                angle="frontal",
                expression="smile",
                age_group="adult",
                lighting="studio",
            )
            await cls_repo.add(cls)

            found = await cls_repo.get_by_image_id(img.id)
            assert found is not None
            assert found.angle == "frontal"
            assert found.expression == "smile"
