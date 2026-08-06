"""Identity Bootstrap pipeline stage — auto-discover seeds from search results.

Downloads a sample of search results, validates identity through face clustering,
and builds an identity profile (mean + geometric median prototypes) for use
by the face_verification stage.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from PIL import Image as PILImage

from portrait_dataset_builder.core.clustering import (
    UnionFind,
    build_cosine_similarity_matrix,
)
from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.core.similarity import cosine_similarity
from portrait_dataset_builder.database import get_engine, get_session, init_db
from portrait_dataset_builder.database.models import Image
from portrait_dataset_builder.database.repository import (
    IdentityRepository,
    ImageRepository,
)
from portrait_dataset_builder.logging import get_logger

logger = get_logger("stage.identity_bootstrap")


@dataclass
class FaceCandidate:
    """A face candidate discovered during bootstrap."""

    image_id: int
    face_id: int
    embedding: np.ndarray
    confidence: float
    bbox: tuple[float, float, float, float]
    yaw: float
    pitch: float
    quality_score: float = 0.0


@dataclass
class SeedCandidate:
    """A ranked seed candidate."""

    face: FaceCandidate
    identity_consistency: float = 0.0
    quality_score: float = 0.0
    pose_diversity_score: float = 0.0
    seed_confidence: float = 0.0


@dataclass
class IdentityProfile:
    """The computed identity profile."""

    mean_prototype: np.ndarray
    geometric_median: np.ndarray
    identity_confidence: float
    seed_count: int
    outlier_count: int
    total_candidates: int
    selected_seeds: list[SeedCandidate] = field(default_factory=list)
    seed_embeddings: list[np.ndarray] = field(default_factory=list)


class IdentityBootstrapStage(PipelineStage):
    """Auto-discover identity seeds from search results via face clustering.

    This stage runs after url_safety_filter and before download. It:
    1. Samples search results and downloads a small batch
    2. Validates via safety (NSFW + AI-generated)
    3. Detects faces and computes ArcFace embeddings
    4. Builds pairwise similarity matrix
    5. Clusters faces with Union-Find
    6. Identifies main cluster and removes outliers
    7. Ranks and selects diverse seeds
    8. Builds identity profile (mean + geometric median)
    9. Persists to database and stores in context.metadata
    """

    def __init__(self) -> None:
        super().__init__("identity_bootstrap")

    async def should_run(self, context: PipelineContext) -> bool:
        if not context.settings.identity_bootstrap.enabled:
            logger.info("Identity bootstrap disabled, skipping")
            return False

        results = context.metadata.get("image_results", [])
        if len(results) == 0:
            logger.info("No search results, skipping identity bootstrap")
            return False

        if len(results) < context.settings.identity_bootstrap.min_candidates:
            logger.info(
                "Only {} search results, need {} minimum, skipping",
                len(results),
                context.settings.identity_bootstrap.min_candidates,
            )
            return False

        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            id_repo = IdentityRepository(session)
            identity = await id_repo.get_or_create(context.identity)

            has_bootstrap_profile = identity.seed_embedding is not None
            has_established = identity.status == "identity_established"

        if has_established:
            logger.info("Identity already established, skipping bootstrap")
            return False

        if has_bootstrap_profile:
            logger.info(
                "Bootstrap profile exists but identity not established, re-running bootstrap"
            )
            return True

        return True

    async def execute(self, context: PipelineContext) -> StageResult:
        settings = context.settings.identity_bootstrap
        image_results = context.metadata.get("image_results", [])

        _progress = context.metadata.get("_progress_task")

        if _progress:
            _progress["items_processed"] = 0
            _progress["items_total"] = settings.candidate_count

        # 1. Download sample batch
        logger.info(
            "Downloading {} candidate images from {} search results",
            settings.candidate_count,
            len(image_results),
        )
        downloaded_images = await self._download_sample(context, image_results)
        if _progress:
            _progress["items_processed"] = 20

        # 2. Safety filtering
        logger.info("Running safety checks on {} candidates", len(downloaded_images))
        safe_images = await self._filter_safety(context, downloaded_images)
        if _progress:
            _progress["items_processed"] = 35

        # 3. Face detection + ArcFace embeddings
        logger.info("Detecting faces in {} safe candidates", len(safe_images))
        candidates = await self._detect_faces(context, safe_images)
        if _progress:
            _progress["items_processed"] = 55

        if len(candidates) < settings.min_cluster_size:
            logger.warning(
                "Only {} face candidates found (need {}), cannot establish identity",
                len(candidates),
                settings.min_cluster_size,
            )
            await self._persist_result(
                context, IdentityProfile(
                    mean_prototype=np.zeros(512, dtype=np.float32),
                    geometric_median=np.zeros(512, dtype=np.float32),
                    identity_confidence=0.0,
                    seed_count=0,
                    outlier_count=0,
                    total_candidates=len(candidates),
                )
            )
            return StageResult(
                stage_name=self.name,
                status=StageStatus.COMPLETED,
                items_processed=len(downloaded_images),
                items_succeeded=0,
                items_rejected=len(downloaded_images),
                metadata={"identity_confidence": 0.0, "reason": "insufficient_candidates"},
            )

        # 4. Pairwise similarity + clustering
        logger.info(
            "Building similarity matrix for {} face candidates",
            len(candidates),
        )
        cluster_result = self._cluster_identities(candidates, settings.similarity_threshold)
        if _progress:
            _progress["items_processed"] = 70

        # 5. Rank and select seeds
        logger.info(
            "Main cluster: {} faces, outliers: {}",
            len(cluster_result["main_cluster"]),
            len(cluster_result["outliers"]),
        )
        seeds = self._rank_and_select_seeds(
            cluster_result["main_cluster"],
            cluster_result["cluster_center"],
            settings.target_seeds,
            settings.max_seeds,
        )
        if _progress:
            _progress["items_processed"] = 85

        # 6. Build identity profile
        profile = self._build_profile(seeds, cluster_result, len(candidates))

        # 7. Persist to database
        await self._persist_result(context, profile)
        if _progress:
            _progress["items_processed"] = 100

        # 8. Store in context metadata for face_verification
        context.metadata["identity_profile"] = profile

        logger.info(
            "Identity bootstrap complete: {} seeds, {} outliers, confidence={:.2f}",
            profile.seed_count,
            profile.outlier_count,
            profile.identity_confidence,
        )

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(downloaded_images),
            items_succeeded=profile.seed_count,
            items_rejected=len(downloaded_images) - profile.seed_count,
            metadata={
                "identity_confidence": profile.identity_confidence,
                "seed_count": profile.seed_count,
                "outlier_count": profile.outlier_count,
                "total_candidates": len(candidates),
                "cluster_size": len(cluster_result["main_cluster"]),
            },
        )

    async def _download_sample(
        self,
        context: PipelineContext,
        image_results: list,
    ) -> list[Path]:
        """Download a sample of search results for identity bootstrap."""
        settings = context.settings.identity_bootstrap
        images_dir = context.resolve_images_dir()
        images_dir.mkdir(parents=True, exist_ok=True)

        # Take a diverse sample: spread across different sources
        sample = self._select_diverse_sample(image_results, settings.candidate_count)

        engine = get_engine(context.db_path)
        await init_db(engine)

        downloaded: list[Path] = []
        sem = asyncio.Semaphore(10)

        async with httpx.AsyncClient(
            timeout=context.settings.download.timeout,
            follow_redirects=True,
            headers={"User-Agent": context.settings.search.user_agent},
        ) as client:

            async def _download_one(result: Any) -> None:
                nonlocal downloaded
                async with sem:
                    try:
                        resp = await client.get(result.url)
                        resp.raise_for_status()
                        content = resp.content
                        if len(content) < 1000:
                            return

                        content_hash = hashlib.sha256(content).hexdigest()

                        async with get_session(engine) as session:
                            repo = ImageRepository(session)
                            existing = await repo.get_by_hash(content_hash)
                            if existing is not None:
                                if existing.local_path and Path(existing.local_path).exists():
                                    downloaded.append(Path(existing.local_path))
                                logger.debug("Reused existing image: {}", result.url[:80])
                                return

                        ext = self._guess_extension(result.mime_type, content)
                        local_path = images_dir / f"bootstrap_{content_hash}{ext}"
                        local_path.write_bytes(content)

                        try:
                            with PILImage.open(local_path) as img:
                                w, h = img.size
                        except Exception:
                            w, h = 0, 0

                        async with get_session(engine) as session:
                            repo = ImageRepository(session)
                            image = Image(
                                uri=result.url,
                                local_path=str(local_path),
                                source_type="image_search",
                                source_provider=result.source_provider,
                                content_hash=content_hash,
                                width=w,
                                height=h,
                                file_size=len(content),
                                mime_type=result.mime_type or ext.lstrip("."),
                                pipeline_state="downloaded",
                            )
                            await repo.add(image)
                            downloaded.append(local_path)

                    except Exception:
                        pass

            tasks = [_download_one(r) for r in sample]
            await asyncio.gather(*tasks)

        logger.info("Downloaded {}/{} bootstrap candidates", len(downloaded), len(sample))
        return downloaded

    def _select_diverse_sample(self, results: list, count: int) -> list:
        """Select a diverse sample spread across different sources."""
        by_source: dict[str, list] = {}
        for r in results:
            key = r.source_provider or "unknown"
            by_source.setdefault(key, []).append(r)

        sample = []
        sources = list(by_source.keys())
        idx = dict.fromkeys(sources, 0)

        while len(sample) < count:
            added = False
            for source in sources:
                if idx[source] < len(by_source[source]) and len(sample) < count:
                    sample.append(by_source[source][idx[source]])
                    idx[source] += 1
                    added = True
            if not added:
                break

        return sample

    def _guess_extension(self, mime_type: str | None, content: bytes) -> str:
        mime_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }
        if mime_type and mime_type in mime_map:
            return mime_map[mime_type]
        if content[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if content[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
            return ".webp"
        return ".jpg"

    async def _filter_safety(
        self, context: PipelineContext, image_paths: list[Path]
    ) -> list[Path]:
        """Run lenient safety filtering on bootstrap images.

        Uses a higher NSFW threshold and skips AI visual detection
        since bootstrap images are small search thumbnails that may
        trigger false positives.
        """
        from portrait_dataset_builder.compute import resolve_device
        from portrait_dataset_builder.pipeline.safety_gate import (
            _detect_ai_metadata,
            _detect_nsfw_model,
            _load_nsfw_model,
        )

        resolved_device = resolve_device(context.settings.effective_device)
        _load_nsfw_model(resolved_device.actual)

        nsfw_threshold = context.settings.safety.nsfw_threshold
        bootstrap_nsfw_threshold = max(nsfw_threshold, 0.50)
        safe: list[Path] = []

        for img_path in image_paths:
            try:
                pil_img = PILImage.open(img_path).convert("RGB")

                nsfw_reject, nsfw_score = _detect_nsfw_model(pil_img)
                if nsfw_reject or nsfw_score >= bootstrap_nsfw_threshold:
                    continue

                metadata_detected, _ = _detect_ai_metadata(pil_img)
                if metadata_detected:
                    continue

                safe.append(img_path)
            except Exception:
                continue

        logger.info("Safety: {}/{} candidates passed", len(safe), len(image_paths))
        return safe

    async def _detect_faces(
        self, context: PipelineContext, image_paths: list[Path]
    ) -> list[FaceCandidate]:
        """Detect faces and compute ArcFace embeddings."""
        from portrait_dataset_builder.compute import (
            get_ctx_id,
            get_onnx_providers,
            resolve_device,
        )
        from portrait_dataset_builder.compute.resize import resize_for_inference

        resolved = resolve_device(context.settings.effective_device)
        providers = get_onnx_providers(resolved)
        ctx_id = get_ctx_id(resolved)

        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=context.settings.face_detection.model_name,
            providers=providers,
        )
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        max_dim = context.settings.compute.inference_max_dimension
        min_size = context.settings.identity_bootstrap.min_face_size

        candidates: list[FaceCandidate] = []

        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                resized, scale = resize_for_inference(img, max_dim=max_dim)
                faces = app.get(resized)

                if not faces:
                    continue

                best_face = max(faces, key=lambda f: f.det_score)
                if best_face.det_score < context.settings.face_detection.min_confidence:
                    continue

                bbox = best_face.bbox.astype(float) * scale
                face_w = int(bbox[2] - bbox[0])
                face_h = int(bbox[3] - bbox[1])

                if face_w < min_size or face_h < min_size:
                    continue

                quality = self._compute_face_quality(
                    img, best_face, face_w, face_h
                )

                candidate = FaceCandidate(
                    image_id=0,
                    face_id=0,
                    embedding=best_face.normed_embedding.copy(),
                    confidence=float(best_face.det_score),
                    bbox=(
                        float(bbox[0]), float(bbox[1]),
                        float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]),
                    ),
                    yaw=float(best_face["pose"][0]),
                    pitch=float(best_face["pose"][1]),
                    quality_score=quality,
                )
                candidates.append(candidate)

            except Exception:
                continue

        logger.info("Detected {} faces from {} images", len(candidates), len(image_paths))
        return candidates

    def _compute_face_quality(
        self, img: np.ndarray, face: Any, face_w: int, face_h: int
    ) -> float:
        """Compute a quick quality score for a face candidate."""
        h, w = img.shape[:2]
        face_area_ratio = (face_w * face_h) / (w * h + 1)
        face_size_score = min(1.0, face_area_ratio * 10)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 500.0)

        yaw = abs(float(face["pose"][0]))
        pitch = abs(float(face["pose"][1]))
        frontal_score = max(0.0, 1.0 - (yaw / 90 + pitch / 90))

        return (
            face_size_score * 0.3
            + sharpness * 0.3
            + frontal_score * 0.2
            + float(face.det_score) * 0.2
        )

    def _cluster_identities(
        self, candidates: list[FaceCandidate], threshold: float
    ) -> dict[str, Any]:
        """Cluster face candidates using Union-Find on pairwise similarity."""
        if len(candidates) == 0:
            return {
                "main_cluster": [],
                "outliers": [],
                "low_confidence": [],
                "cluster_center": np.zeros(512, dtype=np.float32),
                "similarity_matrix": np.zeros((0, 0)),
            }

        embeddings = np.array([c.embedding for c in candidates])
        sim_matrix = build_cosine_similarity_matrix(embeddings)

        uf = UnionFind(len(candidates))
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if sim_matrix[i, j] >= threshold:
                    uf.union(i, j)

        components = uf.components()

        if not components:
            return {
                "main_cluster": candidates,
                "outliers": [],
                "low_confidence": [],
                "cluster_center": np.mean(embeddings, axis=0),
                "similarity_matrix": sim_matrix,
            }

        main_cluster_indices = max(components, key=len)
        main_cluster = [candidates[i] for i in main_cluster_indices]
        outlier_indices = set(range(len(candidates))) - set(main_cluster_indices)
        outliers = [candidates[i] for i in outlier_indices]

        main_embeddings = np.array([c.embedding for c in main_cluster])
        cluster_center = np.mean(main_embeddings, axis=0)
        norm = np.linalg.norm(cluster_center)
        if norm > 0:
            cluster_center = cluster_center / norm

        return {
            "main_cluster": main_cluster,
            "outliers": outliers,
            "low_confidence": [],
            "cluster_center": cluster_center,
            "similarity_matrix": sim_matrix,
        }

    def _rank_and_select_seeds(
        self,
        main_cluster: list[FaceCandidate],
        cluster_center: np.ndarray,
        target_count: int,
        max_count: int,
    ) -> list[SeedCandidate]:
        """Rank candidates and select diverse seeds."""
        if len(main_cluster) == 0:
            return []

        for face in main_cluster:
            consistency = cosine_similarity(face.embedding, cluster_center)

            seed = SeedCandidate(
                face=face,
                identity_consistency=consistency,
                quality_score=face.quality_score,
                seed_confidence=consistency * 0.6 + face.quality_score * 0.4,
            )
            main_cluster[main_cluster.index(face)] = seed

        seeds: list[SeedCandidate] = []
        remaining = list(main_cluster)

        remaining.sort(key=lambda s: s.seed_confidence, reverse=True)
        seeds.append(remaining.pop(0))

        while len(seeds) < min(target_count, max_count) and remaining:
            for candidate in remaining:
                candidate.pose_diversity_score = self._compute_pose_diversity(
                    candidate, seeds
                )

            remaining.sort(
                key=lambda s: s.seed_confidence * 0.7 + s.pose_diversity_score * 0.3,
                reverse=True,
            )
            seeds.append(remaining.pop(0))

        return seeds

    def _compute_pose_diversity(
        self, candidate: SeedCandidate, selected: list[SeedCandidate]
    ) -> float:
        """Compute how different this candidate's pose is from already-selected seeds."""
        if not selected:
            return 1.0

        min_distance = float("inf")
        for s in selected:
            dyaw = abs(candidate.face.yaw - s.face.yaw)
            dpitch = abs(candidate.face.pitch - s.face.pitch)
            distance = (dyaw ** 2 + dpitch ** 2) ** 0.5
            min_distance = min(min_distance, distance)

        return min(1.0, min_distance / 60.0)

    def _build_profile(
        self,
        seeds: list[SeedCandidate],
        cluster_result: dict[str, Any],
        total_candidates: int,
    ) -> IdentityProfile:
        """Build the identity profile from selected seeds."""
        if not seeds:
            return IdentityProfile(
                mean_prototype=np.zeros(512, dtype=np.float32),
                geometric_median=np.zeros(512, dtype=np.float32),
                identity_confidence=0.0,
                seed_count=0,
                outlier_count=len(cluster_result.get("outliers", [])),
                total_candidates=total_candidates,
            )

        embeddings = np.array([s.face.embedding for s in seeds])

        mean_prototype = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_prototype)
        if norm > 0:
            mean_prototype = mean_prototype / norm

        geometric_median = self._geometric_median(embeddings)
        norm = np.linalg.norm(geometric_median)
        if norm > 0:
            geometric_median = geometric_median / norm

        avg_consistency = np.mean([s.identity_consistency for s in seeds])
        main_cluster_size = len(cluster_result["main_cluster"])
        outlier_count = len(cluster_result["outliers"])

        cluster_dominance = main_cluster_size / max(1, total_candidates)
        outlier_penalty = max(0, 1.0 - (outlier_count / max(1, total_candidates)))
        size_bonus = min(1.0, main_cluster_size / 10.0)

        confidence = (
            cluster_dominance * 0.30
            + float(avg_consistency) * 0.30
            + outlier_penalty * 0.15
            + size_bonus * 0.15
            + (float(avg_consistency) * 0.10)
        )
        confidence = max(0.0, min(1.0, confidence))

        return IdentityProfile(
            mean_prototype=mean_prototype.astype(np.float32),
            geometric_median=geometric_median.astype(np.float32),
            identity_confidence=confidence,
            seed_count=len(seeds),
            outlier_count=outlier_count,
            total_candidates=total_candidates,
            selected_seeds=seeds,
            seed_embeddings=[s.face.embedding for s in seeds],
        )

    @staticmethod
    def _geometric_median(
        vectors: np.ndarray, eps: float = 1e-5, max_iter: int = 100
    ) -> np.ndarray:
        """Compute geometric median using Weiszfeld's algorithm."""
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

    async def _persist_result(
        self, context: PipelineContext, profile: IdentityProfile
    ) -> None:
        """Persist identity bootstrap results to database.

        Saves seed embeddings to identity.seed_embedding for use by
        face_verification on resume. Does NOT create Image/Face/IdentityImage
        records — the bootstrap profile is passed via context.metadata
        and face_verification reads from there directly.
        """
        engine = get_engine(context.db_path)

        async with get_session(engine) as session:
            id_repo = IdentityRepository(session)
            identity = await id_repo.get_or_create(context.identity)

            identity.seed_embedding = profile.geometric_median.tobytes()
            min_conf = context.settings.identity_bootstrap.min_identity_confidence
            if profile.identity_confidence >= min_conf:
                identity.status = "identity_established"
            else:
                identity.status = "identity_unverified"
            await id_repo.update(identity)
