"""Face verification pipeline stage with dual prototypes."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.core.similarity import cosine_similarity
from portrait_dataset_builder.database import get_engine, get_session
from portrait_dataset_builder.database.repository import (
    FaceRepository,
    IdentityImageRepository,
    IdentityRepository,
    ImageRepository,
)
from portrait_dataset_builder.logging import get_logger

logger = get_logger("stage.face_verification")


class FaceVerificationStage(PipelineStage):
    """Verify detected faces against identity seed embeddings using dual prototypes."""

    def __init__(self) -> None:
        super().__init__("face_verification")

    async def should_run(self, context: PipelineContext) -> bool:
        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            count = await repo.count_by_state("face_detected")
        return count > 0

    async def execute(self, context: PipelineContext) -> StageResult:
        engine = get_engine(context.db_path)

        seed_embeddings = await self._load_seed_embeddings(context, engine)

        if len(seed_embeddings) == 0:
            logger.warning(
                "No seeds found (no manual seeds, no identity_bootstrap profile). "
                "All images will be REJECTED — identity cannot be established."
            )
            threshold = 1.0
            prototypes = None
        else:
            threshold = self._get_threshold(context.settings.face_verification)
            prototypes = self._build_prototypes(seed_embeddings)

        logger.info(
            "Verification: {} seed embeddings, {} prototypes, threshold={:.3f}",
            len(seed_embeddings),
            2 if prototypes else 0,
            threshold,
        )

        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            images = await img_repo.get_by_state("face_detected", limit=100000)

        verified = 0
        rejected = 0
        failed = 0
        errors: list[str] = []

        _progress = context.metadata.get("_progress_task")
        total = len(images)

        for i, image_record in enumerate(images):
            if i % 10 == 0:
                await asyncio.sleep(0)
            if _progress:
                _progress["items_processed"] = i
                _progress["items_total"] = total

            if not image_record.local_path or not Path(image_record.local_path).exists():
                continue

            async with get_session(engine) as session:
                face_repo = FaceRepository(session)
                faces = await face_repo.get_by_image_id(image_record.id)

            best_similarity = 0.0

            for face_record in faces:
                if face_record.embedding is None:
                    continue

                face_embedding = np.frombuffer(face_record.embedding, dtype=np.float32)

                if prototypes is not None:
                    sim_mean = cosine_similarity(face_embedding, prototypes["mean"])
                    sim_median = cosine_similarity(
                        face_embedding, prototypes["geometric_median"]
                    )
                    similarity = max(sim_mean, sim_median)
                elif len(seed_embeddings) > 0:
                    similarities = [
                        cosine_similarity(face_embedding, seed_emb)
                        for seed_emb in seed_embeddings
                    ]
                    similarity = max(similarities)
                else:
                    similarity = 0.0

                if similarity > best_similarity:
                    best_similarity = similarity

            is_verified = best_similarity >= threshold

            async with get_session(engine) as session:
                img_repo = ImageRepository(session)
                if is_verified:
                    await img_repo.update_state(image_record.id, "verified")
                    verified += 1
                else:
                    await img_repo.update_state(image_record.id, "rejected")
                    rejected += 1

        logger.info(
            "Verification: {} verified, {} rejected, {} failed",
            verified,
            rejected,
            failed,
        )

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(images),
            items_succeeded=verified,
            items_rejected=rejected,
            items_failed=failed,
            errors=errors,
            metadata={
                "verified": verified,
                "rejected": rejected,
                "threshold": threshold,
            },
        )

    def _build_prototypes(self, seed_embeddings: list[np.ndarray]) -> dict[str, np.ndarray]:
        seed_matrix = np.array(seed_embeddings)

        mean_prototype = np.mean(seed_matrix, axis=0)
        norm = np.linalg.norm(mean_prototype)
        if norm > 0:
            mean_prototype = mean_prototype / norm

        median_prototype = self._geometric_median(seed_matrix)
        norm = np.linalg.norm(median_prototype)
        if norm > 0:
            median_prototype = median_prototype / norm

        return {
            "mean": mean_prototype.astype(np.float32),
            "geometric_median": median_prototype.astype(np.float32),
        }

    @staticmethod
    def _geometric_median(
        vectors: np.ndarray, eps: float = 1e-5, max_iter: int = 100
    ) -> np.ndarray:
        y = np.mean(vectors, axis=0)
        for _ in range(max_iter):
            distances = np.linalg.norm(vectors - y, axis=1)
            nonzero = distances > eps
            if not nonzero.all():
                return y
            weights = 1.0 / distances[nonzero]
            weights /= weights.sum()
            y_new = np.average(vectors[nonzero], axis=0, weights=weights)
            if np.linalg.norm(y_new - y) < eps:
                return y_new
            y = y_new
        return y

    async def _load_seed_embeddings(
        self, context: PipelineContext, engine
    ) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []

        # 1. Check identity bootstrap profile from context metadata (in-memory)
        profile = context.metadata.get("identity_profile")
        if profile is not None and len(profile.seed_embeddings) > 0:
            logger.info(
                "Using {} seed embeddings from identity bootstrap profile",
                len(profile.seed_embeddings),
            )
            return list(profile.seed_embeddings)

        # 2. Check identity.seed_embedding (persisted by bootstrap, survives restart)
        async with get_session(engine) as session:
            identity_repo = IdentityRepository(session)
            identity = await identity_repo.get_or_create(context.identity)

            if identity.seed_embedding is not None:
                seed_emb = np.frombuffer(identity.seed_embedding, dtype=np.float32)
                logger.info(
                    "Using geometric median prototype from identity.seed_embedding"
                )
                return [seed_emb]

        # 3. Check IdentityImage table (manual seeds linked via DB)
        async with get_session(engine) as session:
            id_img_repo = IdentityImageRepository(session)
            seed_images = await id_img_repo.get_by_identity(identity.id)

            for seed_img in seed_images:
                face_repo = FaceRepository(session)
                faces = await face_repo.get_by_image_id(seed_img.image_id)
                for face in faces:
                    if face.embedding is not None:
                        emb = np.frombuffer(face.embedding, dtype=np.float32)
                        embeddings.append(emb)

        if len(embeddings) > 0:
            logger.info(
                "Using {} seed embeddings from IdentityImage table",
                len(embeddings),
            )
            return embeddings

        # 4. Fallback: embed seed image files from disk
        seed_dir = context.resolve_seeds_dir()
        if seed_dir.exists():
            embeddings = await self._embed_seeds_from_files(context, seed_dir)
            if len(embeddings) > 0:
                logger.info(
                    "Using {} seed embeddings from {} seed files",
                    len(embeddings),
                    seed_dir,
                )

        return embeddings

    async def _embed_seeds_from_files(
        self, context: PipelineContext, seed_dir: Path
    ) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        try:
            from insightface.app import FaceAnalysis

            from portrait_dataset_builder.compute import (
                get_ctx_id,
                get_onnx_providers,
                resolve_device,
            )

            resolved = resolve_device(context.settings.effective_device)
            providers = get_onnx_providers(resolved)
            ctx_id = get_ctx_id(resolved)

            app = FaceAnalysis(
                name=context.settings.face_detection.model_name,
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            import cv2

            for img_path in seed_dir.iterdir():
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                faces = app.get(img)
                if faces:
                    best = max(faces, key=lambda f: f.det_score)
                    embeddings.append(best.normed_embedding)

        except ImportError:
            logger.warning("InsightFace not available for seed embedding")

        return embeddings

    def _get_threshold(self, settings) -> float:
        mode = settings.mode
        if mode == "strict":
            return settings.strict_threshold
        elif mode == "permissive":
            return settings.permissive_threshold
        return settings.normal_threshold
