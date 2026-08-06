"""Test duplicate detection logic."""

from __future__ import annotations

from PIL import Image as PILImage

from portrait_dataset_builder.pipeline.duplicates import DuplicateDetectionStage


class TestDuplicateDetection:
    def setup_method(self) -> None:
        self.stage = DuplicateDetectionStage()

    def test_cluster_exact_duplicates(self) -> None:
        import imagehash

        img1 = PILImage.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = PILImage.new("RGB", (100, 100), color=(255, 0, 0))

        h1 = imagehash.phash(img1, hash_size=16)
        h2 = imagehash.phash(img2, hash_size=16)

        hashes = {1: h1, 2: h2}
        groups = self.stage._cluster_hashes(hashes, threshold=5)

        assert len(groups) == 1
        assert set(groups[0]) == {1, 2}

    def test_cluster_different_images(self) -> None:
        import imagehash

        img1 = PILImage.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = PILImage.new("RGB", (100, 100), color=(0, 0, 255))
        # Draw distinct patterns to make hashes truly different
        from PIL import ImageDraw

        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([10, 10, 50, 50], fill=(0, 255, 0))
        draw2 = ImageDraw.Draw(img2)
        draw2.ellipse([10, 10, 90, 90], fill=(255, 255, 0))

        h1 = imagehash.phash(img1, hash_size=16)
        h2 = imagehash.phash(img2, hash_size=16)

        hashes = {1: h1, 2: h2}
        groups = self.stage._cluster_hashes(hashes, threshold=5)

        assert len(groups) == 0

    def test_cluster_mixed(self) -> None:
        import imagehash
        from PIL import ImageDraw

        img1 = PILImage.new("RGB", (100, 100), color=(255, 0, 0))
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([10, 10, 50, 50], fill=(0, 255, 0))

        img2 = img1.copy()

        img3 = PILImage.new("RGB", (100, 100), color=(0, 0, 255))
        draw3 = ImageDraw.Draw(img3)
        draw3.ellipse([10, 10, 90, 90], fill=(255, 255, 0))

        h1 = imagehash.phash(img1, hash_size=16)
        h2 = imagehash.phash(img2, hash_size=16)
        h3 = imagehash.phash(img3, hash_size=16)

        hashes = {1: h1, 2: h2, 3: h3}
        groups = self.stage._cluster_hashes(hashes, threshold=5)

        assert len(groups) == 1
        assert set(groups[0]) == {1, 2}
