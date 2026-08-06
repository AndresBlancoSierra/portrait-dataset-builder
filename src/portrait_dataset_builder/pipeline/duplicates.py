"""Duplicate detection pipeline stage with multi-level dedup and Union-Find clustering."""

from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
from PIL import Image as PILImage

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session
from portrait_dataset_builder.database.repository import (
    ImageRepository,
    QualityRepository,
)
from portrait_dataset_builder.core.clustering import UnionFind
from portrait_dataset_builder.logging import get_logger

logger = get_logger("stage.duplicates")


class DuplicateDetectionStage(PipelineStage):
    """Detect and remove duplicate/near-duplicate images using multi-level detection."""

    def __init__(self) -> None:
        super().__init__("duplicates")
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

    async def should_run(self, context: PipelineContext) -> bool:
        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            count = await repo.count_by_state("verified")
        return count > 10

    def _init_clip(self) -> None:
        if self._clip_model is not None:
            return
        try:
            import open_clip

            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            logger.info("OpenCLIP ViT-B-32 loaded for duplicate detection")
        except Exception as e:
            logger.warning("OpenCLIP not available, skipping CLIP dedup: {}", e)

    async def execute(self, context: PipelineContext) -> StageResult:
        engine = get_engine(context.db_path)

        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            images = await img_repo.get_by_state("verified", limit=100000)

        settings = context.settings.duplicates
        id_to_idx: dict[int, int] = {img.id: i for i, img in enumerate(images)}
        n = len(images)
        uf = UnionFind(n)

        total_original = n

        if "phash" in settings.enabled_methods:
            phash_groups = self._find_phash_duplicates(images, settings.phash_threshold)
            self._merge_groups(uf, phash_groups, id_to_idx)
            logger.info("pHash: {} groups after level 1", len(phash_groups))

        surviving = self._get_surviving_groups(uf, n)
        if "dhash" in settings.enabled_methods:
            surviving_images = [images[i] for group in surviving for i in group]
            dhash_groups = self._find_dhash_duplicates(
                surviving_images, settings.dhash_threshold
            )
            self._merge_groups(uf, dhash_groups, id_to_idx)
            logger.info("dHash: {} groups after level 2", len(dhash_groups))

        surviving = self._get_surviving_groups(uf, n)
        if "clip" in settings.enabled_methods:
            self._init_clip()
            if self._clip_model is not None:
                surviving_images = [images[i] for group in surviving for i in group]
                clip_groups = self._find_clip_duplicates(
                    surviving_images, settings.embedding_similarity_threshold
                )
                self._merge_groups(uf, clip_groups, id_to_idx)
                logger.info("CLIP: {} groups after level 3", len(clip_groups))

        final_groups = self._get_surviving_groups(uf, n)
        marked_for_removal: set[int] = set()

        for group in final_groups:
            if len(group) < 2:
                continue
            group_ids = [images[i].id for i in group]
            best_id = await self._pick_best_image(group_ids, engine)
            for img_id in group_ids:
                if img_id != best_id:
                    marked_for_removal.add(img_id)

        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            for img_id in marked_for_removal:
                await img_repo.update_state(img_id, "duplicate")

        logger.info(
            "Duplicates: {} total images, {} groups, {} marked for removal",
            total_original,
            len(final_groups),
            len(marked_for_removal),
        )

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=total_original,
            items_succeeded=total_original - len(marked_for_removal),
            items_skipped=len(marked_for_removal),
            metadata={
                "duplicate_groups": len([g for g in final_groups if len(g) > 1]),
                "removed": len(marked_for_removal),
            },
        )

    def _find_phash_duplicates(
        self, images: list, threshold: int
    ) -> list[list[int]]:
        hashes: dict[int, imagehash.ImageHash] = {}
        for img in images:
            if not img.local_path or not Path(img.local_path).exists():
                continue
            try:
                pil_img = PILImage.open(img.local_path)
                h = imagehash.phash(pil_img, hash_size=16)
                hashes[img.id] = h
            except Exception:
                continue
        return self._cluster_hashes(hashes, threshold)

    def _find_dhash_duplicates(
        self, images: list, threshold: int
    ) -> list[list[int]]:
        hashes: dict[int, imagehash.ImageHash] = {}
        for img in images:
            if not img.local_path or not Path(img.local_path).exists():
                continue
            try:
                pil_img = PILImage.open(img.local_path)
                h = imagehash.dhash(pil_img, hash_size=16)
                hashes[img.id] = h
            except Exception:
                continue
        return self._cluster_hashes(hashes, threshold)

    def _find_clip_duplicates(
        self, images: list, threshold: float
    ) -> list[list[int]]:
        if self._clip_model is None:
            return []

        import torch

        embeddings: dict[int, np.ndarray] = {}
        for img in images:
            if not img.local_path or not Path(img.local_path).exists():
                continue
            try:
                pil_img = PILImage.open(img.local_path).convert("RGB")
                tensor = self._clip_preprocess(pil_img).unsqueeze(0)
                with torch.no_grad():
                    emb = self._clip_model.encode_image(tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings[img.id] = emb.squeeze().numpy()
            except Exception:
                continue

        ids = list(embeddings.keys())
        n = len(ids)
        {img_id: i for i, img_id in enumerate(ids)}
        uf_local = UnionFind(n)

        emb_matrix = np.array([embeddings[img_id] for img_id in ids])
        sim_matrix = emb_matrix @ emb_matrix.T

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    uf_local.union(i, j)

        groups: list[list[int]] = []
        component_map: dict[int, list[int]] = {}
        for i in range(n):
            root = uf_local.find(i)
            if root not in component_map:
                component_map[root] = []
            component_map[root].append(ids[i])

        for component in component_map.values():
            if len(component) > 1:
                groups.append(component)

        return groups

    def _cluster_hashes(
        self, hashes: dict[int, imagehash.ImageHash], threshold: int
    ) -> list[list[int]]:
        ids = list(hashes.keys())
        n = len(ids)
        {img_id: i for i, img_id in enumerate(ids)}
        uf = UnionFind(n)

        for i in range(n):
            for j in range(i + 1, n):
                if hashes[ids[i]] - hashes[ids[j]] <= threshold:
                    uf.union(i, j)

        groups: list[list[int]] = []
        component_map: dict[int, list[int]] = {}
        for i in range(n):
            root = uf.find(i)
            if root not in component_map:
                component_map[root] = []
            component_map[root].append(ids[i])

        for component in component_map.values():
            if len(component) > 1:
                groups.append(component)

        return groups

    def _merge_groups(
        self,
        uf: UnionFind,
        groups: list[list[int]],
        id_to_idx: dict[int, int],
    ) -> None:
        for group in groups:
            indices = [id_to_idx[img_id] for img_id in group if img_id in id_to_idx]
            if len(indices) < 2:
                continue
            for i in range(1, len(indices)):
                uf.union(indices[0], indices[i])

    def _get_surviving_groups(self, uf: UnionFind, n: int) -> list[list[int]]:
        component_map: dict[int, list[int]] = {}
        for i in range(n):
            root = uf.find(i)
            if root not in component_map:
                component_map[root] = []
            component_map[root].append(i)
        return list(component_map.values())

    async def _pick_best_image(self, group: list[int], engine) -> int:
        best_id = group[0]
        best_score = -1.0

        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            q_repo = QualityRepository(session)

            for img_id in group:
                img = await img_repo.get_by_id(img_id)
                if not img:
                    continue

                quality = await q_repo.get_by_image_id(img_id)
                quality_score = quality.final_score if quality else 0.0

                resolution_score = min(1.0, (img.width or 0) / 1024)
                file_size_score = min(1.0, (img.file_size or 0) / (500 * 1024))

                total = (
                    quality_score * 0.4
                    + resolution_score * 0.3
                    + file_size_score * 0.1
                    + 0.2
                )

                if total > best_score:
                    best_score = total
                    best_id = img_id

        return best_id

    async def teardown(self, context: PipelineContext) -> None:
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
