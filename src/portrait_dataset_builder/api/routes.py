"""API routes for WHO? frontend."""

from __future__ import annotations

import asyncio
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from portrait_dataset_builder.config.settings import load_settings
from portrait_dataset_builder.database.engine import (
    _configure_sqlite,
    _migrate_db,
    dispose_engine,
    get_engine,
    get_session,
)
from portrait_dataset_builder.database.models import Base, BuildJob
from portrait_dataset_builder.database.repository import (
    BuildJobRepository,
    ClassificationRepository,
    FaceRepository,
    IdentityRepository,
    ImageRepository,
    QualityRepository,
    ReviewQueueRepository,
    SafetyScoreRepository,
)
from portrait_dataset_builder.taxonomy import (
    ExpressionLabel,
    HorizontalPose,
    LightingLabel,
    QualityLevel,
    VerticalPose,
)

router = APIRouter()

# ── Build concurrency control ──────────────────────────────────────────────────

_build_semaphore: asyncio.Semaphore | None = None
_build_queue: list[str] = []


def _get_build_semaphore() -> asyncio.Semaphore:
    """Return or initialize the build semaphore with max_concurrent_builds."""
    global _build_semaphore
    if _build_semaphore is None:
        settings = load_settings("default")
        max_concurrent = settings.pipeline.max_concurrent_builds
        _build_semaphore = asyncio.Semaphore(max_concurrent)
    return _build_semaphore


# ── In-memory build tasks (tracks running asyncio tasks only) ─────────────────

_build_tasks: dict[str, asyncio.Task[None]] = {}

# ── Pipeline stage labels ─────────────────────────────────────────────────────

STAGE_LABELS: dict[str, str] = {
    "search": "Searching",
    "url_safety_filter": "Filtering unsafe URLs",
    "identity_bootstrap": "Discovering identity",
    "download": "Downloading",
    "safety_gate": "Checking content safety",
    "face_detection": "Detecting faces",
    "face_verification": "Verifying identity",
    "semantic_filter": "Filtering non-portraits",
    "quality": "Scoring quality",
    "duplicates": "Removing duplicates",
    "classification": "Classifying",
    "export": "Exporting",
    "cleanup": "Cleaning up rejected images",
}

ALL_STAGES = list(STAGE_LABELS.keys())

# ── Stage progress mapping ────────────────────────────────────────────────────
# Maps each stage to the DB states used to compute real-time progress.
# None = in-memory stage (no DB progress available).

STAGE_PROGRESS_MAP: dict[str, dict[str, list[str]] | None] = {
    "search": None,
    "url_safety_filter": None,
    "identity_bootstrap": None,
    "download": None,
    "safety_gate": {"input_states": ["downloaded"], "done_states": ["rejected"]},
    "face_detection": {"input_states": ["downloaded"], "done_states": ["face_detected", "no_face"]},
    "face_verification": {
        "input_states": ["face_detected"],
        "done_states": ["verified", "rejected"],
    },
    "semantic_filter": {"input_states": ["verified"], "done_states": ["rejected"]},
    "quality": {"input_states": ["verified"], "done_states": []},
    "duplicates": {"input_states": ["verified"], "done_states": ["duplicate"]},
    "classification": {"input_states": ["verified"], "done_states": []},
    "cleanup": None,
}

# ── Filter normalization maps (using taxonomy constants) ─────────────────────

ANGLE_MAP: dict[str, list[str]] = HorizontalPose.GROUPS

QUALITY_MAP: dict[str, tuple[float, float]] = QualityLevel.RANGES

EXPRESSION_VALID = set(ExpressionLabel.ALL)

LIGHTING_VALID = set(LightingLabel.ALL)

HORIZONTAL_POSE_VALID = set(HorizontalPose.ALL)

VERTICAL_POSE_VALID = set(VerticalPose.ALL)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _data_root() -> Path:
    return Path("data")


def _find_libraries() -> list[dict[str, Any]]:
    root = _data_root()
    if not root.exists():
        return []
    libs = []
    for d in sorted(root.iterdir()):
        db = d / "portrait.db"
        if db.exists():
            _migrate_db(db)
            libs.append({"name": d.name, "path": d, "db": db})
    return libs


def _serialize_image(
    img: Any,
    face: Any = None,
    quality: Any = None,
    cls: Any = None,
    safety: Any = None,
) -> dict:
    return {
        "id": img.id,
        "content_hash": img.content_hash,
        "uri": img.uri,
        "local_path": img.local_path or "",
        "width": img.width or 0,
        "height": img.height or 0,
        "file_size": img.file_size or 0,
        "source_provider": img.source_provider,
        "pipeline_state": img.pipeline_state,
        "created_at": img.created_at.isoformat() if img.created_at else "",
        "face": {
            "id": face.id,
            "image_id": face.image_id,
            "bbox_x": face.bbox_x,
            "bbox_y": face.bbox_y,
            "bbox_w": face.bbox_w,
            "bbox_h": face.bbox_h,
            "yaw": face.yaw,
            "pitch": face.pitch,
            "roll": face.roll,
            "confidence": face.confidence,
            "face_width": face.face_width or 0,
            "face_height": face.face_height or 0,
        }
        if face
        else None,
        "quality": {
            "id": quality.id,
            "image_id": quality.image_id,
            "resolution_score": quality.resolution_score,
            "sharpness_score": quality.sharpness_score,
            "blur_score": quality.blur_score,
            "noise_score": quality.noise_score,
            "lighting_score": quality.lighting_score,
            "occlusion_score": quality.occlusion_score,
            "face_size_score": quality.face_size_score,
            "frontal_score": quality.frontal_score,
            "jpeg_score": quality.jpeg_score,
            "final_score": quality.final_score,
        }
        if quality
        else None,
        "classification": {
            "id": cls.id,
            "image_id": cls.image_id,
            "angle": cls.angle,
            "horizontal_pose": cls.horizontal_pose,
            "vertical_pose": cls.vertical_pose,
            "expression": cls.expression,
            "accessories": cls.accessories,
            "age_group": cls.age_group,
            "lighting": cls.lighting,
        }
        if cls
        else None,
        "safety": {
            "is_nsfw": safety.is_nsfw,
            "nsfw_score": safety.nsfw_score,
            "is_ai_generated": safety.is_ai_generated,
            "ai_probability": safety.ai_probability,
            "real_photo_score": safety.real_photo_score,
            "source_trust_score": safety.source_trust_score,
            "rejection_reason": safety.rejection_reason,
        }
        if safety
        else None,
    }


async def _get_library_data(name: str) -> dict[str, Any]:
    root = _data_root() / name
    db = root / "portrait.db"
    if not db.exists():
        raise HTTPException(404, f"Library '{name}' not found")
    _migrate_db(db)
    return {"name": name, "path": root, "db": db}


def _serialize_build_job(job: BuildJob | None, identity: str) -> dict[str, Any]:
    if not job:
        return {
            "id": 0,
            "status": "unknown",
            "current_stage": None,
            "stage_label": None,
            "items_processed": 0,
            "items_total": 0,
            "error": None,
            "started_at": None,
            "completed_at": None,
            "created_at": None,
        }
    return {
        "id": job.id,
        "status": job.status,
        "queue_status": job.queue_status,
        "queue_position": job.queue_position,
        "current_stage": job.current_stage,
        "stage_label": job.stage_label,
        "items_processed": job.items_processed,
        "items_total": job.items_total,
        "error": job.error,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


def _resolve_library_status(
    identity_status: str,
    active_job: BuildJob | None,
    latest_job: BuildJob | None,
    verified_count: int,
) -> str:
    """Map BuildJob/Identity status to a frontend Library status.

    Returns one of: queued, building, ready, empty, failed, cancelled,
    identity_unverified, unknown.
    """
    if active_job:
        if active_job.queue_status == "queued":
            return "queued"
        if active_job.status in ("pending", "running"):
            return "building"
    if identity_status == "building":
        return "building"

    if latest_job:
        if latest_job.status == "completed":
            if verified_count > 0:
                return "ready"
            if identity_status == "identity_unverified":
                return "identity_unverified"
            return "empty"
        if latest_job.status == "failed":
            return "failed"
        if latest_job.status == "cancelled":
            return "cancelled"

    if identity_status == "failed":
        return "failed"
    if identity_status == "cancelled":
        return "cancelled"
    if identity_status == "identity_unverified":
        return "identity_unverified"
    if identity_status == "ready":
        return "ready" if verified_count > 0 else "empty"

    return "unknown"


def _compute_coverage_score(
    poses: list[tuple[float, float]],
    expressions: list[str],
    lighting_data: list[str],
    horizontal_poses: list[str] | None = None,
    vertical_poses: list[str] | None = None,
) -> float:
    """Compute a reference coverage score using existing infrastructure.

    Combines breadth (how many different categories are populated)
    with balance (how evenly distributed images are across categories).

    Now considers both horizontal and vertical pose dimensions.
    """
    if not poses:
        return 0.0

    from portrait_dataset_builder.coverage import (
        ExpressionDiversity,
        PoseCoverage,
    )

    # Pose coverage breadth: populated bins / total bins
    pc = PoseCoverage()
    pose_result = pc.compute(poses)
    pose_breadth = pose_result["coverage_pct"]

    # Horizontal pose diversity (5 categories)
    if horizontal_poses:
        hp_counts = Counter(horizontal_poses)
        unique_hp = len(hp_counts)
        hp_breadth = min(1.0, unique_hp / 5.0)
    else:
        hp_breadth = 0.0

    # Vertical pose diversity (3 categories)
    if vertical_poses:
        vp_counts = Counter(vertical_poses)
        unique_vp = len(vp_counts)
        vp_breadth = min(1.0, unique_vp / 3.0)
    else:
        vp_breadth = 0.0

    # Combined two-axis pose score
    two_axis_pose = 0.5 * hp_breadth + 0.5 * vp_breadth

    # Expression coverage breadth + balance (entropy)
    ed = ExpressionDiversity()
    if expressions:
        expr_result = ed.compute(expressions)
        expr_breadth = min(1.0, expr_result["unique_expressions"] / 4.0)
        expr_balance = expr_result["entropy"]
    else:
        expr_breadth = 0.0
        expr_balance = 0.0

    # Lighting coverage breadth + balance
    if lighting_data:
        light_counts = Counter(lighting_data)
        unique_lighting = len(light_counts)
        light_breadth = min(1.0, unique_lighting / 3.0)
        total_light = len(lighting_data)
        probs = [c / total_light for c in light_counts.values()]
        light_entropy = -sum(p * math.log2(p + 1e-10) for p in probs)
        max_light_entropy = math.log2(unique_lighting) if unique_lighting > 1 else 1.0
        light_balance = light_entropy / max_light_entropy if max_light_entropy > 0 else 0.0
    else:
        light_breadth = 0.0
        light_balance = 0.0

    # Combine: weighted mix of all dimensions
    pose_score = 0.4 * pose_breadth + 0.6 * two_axis_pose
    expr_score = 0.5 * expr_breadth + 0.5 * expr_balance
    light_score = 0.5 * light_breadth + 0.5 * light_balance

    coverage = pose_score * 0.45 + expr_score * 0.30 + light_score * 0.25
    return max(0.0, min(1.0, coverage))


def _parse_quality_filter(quality: str | None) -> tuple[float, float] | None:
    """Parse quality filter from UI value or raw range string.

    Accepts: "high", "medium", "low", or "min,max" range.
    Returns (min, max) tuple or None if no filter.
    """
    if not quality:
        return None
    q = quality.strip().lower()
    if q in QUALITY_MAP:
        return QUALITY_MAP[q]
    parts = q.split(",")
    if len(parts) == 2:
        try:
            return (float(parts[0]), float(parts[1]))
        except ValueError:
            return None
    return None


def _parse_angle_filter(angle: str | None) -> list[str] | None:
    """Parse angle filter, mapping UI labels to backend classification values.

    Returns list of valid angle values or None if no filter.
    """
    if not angle:
        return None
    result: list[str] = []
    for a in angle.split(","):
        a = a.strip().lower()
        if a in ANGLE_MAP:
            result.extend(ANGLE_MAP[a])
    return result if result else None


def _parse_horizontal_pose_filter(pose: str | None) -> list[str] | None:
    """Parse horizontal pose filter.

    Accepts comma-separated values: 'frontal', 'three_quarter_left', etc.
    Also accepts UI group names: 'quarter', 'profile'.
    Returns list of valid horizontal pose values or None if no filter.
    """
    if not pose:
        return None
    result: list[str] = []
    for p in pose.split(","):
        p = p.strip().lower()
        if p in ANGLE_MAP:
            result.extend(ANGLE_MAP[p])
        elif p in HORIZONTAL_POSE_VALID:
            result.append(p)
    return result if result else None


def _parse_vertical_pose_filter(pose: str | None) -> list[str] | None:
    """Parse vertical pose filter.

    Accepts comma-separated values: 'neutral', 'looking_up', 'looking_down'.
    Returns list of valid vertical pose values or None if no filter.
    """
    if not pose:
        return None
    result: list[str] = []
    for p in pose.split(","):
        p = p.strip().lower()
        if p in VERTICAL_POSE_VALID:
            result.append(p)
    return result if result else None


# ── Library endpoints ────────────────────────────────────────────────────────


class CreateLibraryRequest(BaseModel):
    name: str


@router.get("/libraries")
async def list_libraries() -> list[dict]:
    libs = _find_libraries()
    result = []
    for lib in libs:
        lib_data = await _build_library_response(lib["name"])
        result.append(lib_data)
    return result


async def _build_library_response(name: str) -> dict[str, Any]:
    """Build a consistent library response dict used by list and get endpoints."""
    root = _data_root() / name
    db = root / "portrait.db"
    if not db.exists():
        raise HTTPException(404, f"Library '{name}' not found")

    engine = get_engine(db)
    async with get_session(engine) as session:
        img_repo = ImageRepository(session)
        face_repo = FaceRepository(session)
        q_repo = QualityRepository(session)
        cls_repo = ClassificationRepository(session)
        id_repo = IdentityRepository(session)
        bj_repo = BuildJobRepository(session)

        total = await img_repo.count()
        verified_imgs = await img_repo.get_by_state("verified", limit=100000)
        verified_count = len(verified_imgs)

        # Batch fetch quality scores (eliminates N+1)
        image_ids = [img.id for img in verified_imgs]
        quality_map = await q_repo.get_by_image_ids(image_ids)

        avg_quality = 0.0
        best_hash = None
        scores = []
        best_score = -1
        for img in verified_imgs:
            q = quality_map.get(img.id)
            if q:
                scores.append(q.final_score)
                if q.final_score > best_score:
                    best_score = q.final_score
                    best_hash = img.content_hash
        if scores:
            avg_quality = sum(scores) / len(scores)

        # Build status mapping
        identity = await id_repo.get_by_name(name)
        identity_status = identity.status if identity else "unknown"
        active_job = await bj_repo.get_active_by_identity(name)
        latest_job = await bj_repo.get_latest_by_identity(name)

        build_progress = _serialize_build_job(active_job or latest_job, name)

        # Coverage: batch fetch faces and classifications (eliminates N+1)
        best_faces = await face_repo.get_best_by_image_ids(image_ids)
        cls_map = await cls_repo.get_by_image_ids(image_ids)

        poses = []
        expressions: list[str] = []
        lighting_data: list[str] = []
        horizontal_poses: list[str] = []
        vertical_poses: list[str] = []
        for img in verified_imgs:
            best = best_faces.get(img.id)
            if best:
                poses.append((best.yaw or 0, best.pitch or 0))
            c = cls_map.get(img.id)
            if c:
                if c.expression:
                    expressions.append(c.expression)
                if c.lighting:
                    lighting_data.append(c.lighting)
                if c.horizontal_pose:
                    horizontal_poses.append(c.horizontal_pose)
                if c.vertical_pose:
                    vertical_poses.append(c.vertical_pose)

        coverage = _compute_coverage_score(
            poses, expressions, lighting_data, horizontal_poses, vertical_poses
        )

        library_status = _resolve_library_status(
            identity_status, active_job, latest_job, verified_count
        )

    return {
        "name": name,
        "image_count": total,
        "quality_score": round(avg_quality, 3),
        "coverage_score": round(coverage, 3),
        "updated_at": "",
        "thumbnail_hash": best_hash,
        "status": library_status,
        "build": build_progress,
    }


@router.post("/libraries")
async def create_library(req: CreateLibraryRequest) -> dict:
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Library name cannot be empty")

    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()
    job = await worker.enqueue(name)

    return {"name": name, "status": "queued", "build_job_id": job.id}


@router.get("/libraries/{name}")
async def get_library(name: str) -> dict:
    return await _build_library_response(name)


@router.delete("/libraries/{name}")
async def delete_library(name: str) -> None:
    lib = await _get_library_data(name)
    dispose_engine(lib["db"])
    if lib["path"].exists():
        shutil.rmtree(lib["path"])


@router.delete("/libraries/{name}/images/{content_hash}")
async def delete_image(name: str, content_hash: str) -> None:
    lib = await _get_library_data(name)
    engine = get_engine(lib["db"])
    async with get_session(engine) as session:
        repo = ImageRepository(session)
        img = await repo.get_by_hash(content_hash)
        if not img:
            raise HTTPException(404, "Image not found")
        if img.local_path:
            p = Path(img.local_path)
            if p.exists():
                p.unlink()
        await repo.delete(img.id)


# ── Image endpoints ──────────────────────────────────────────────────────────


@router.get("/libraries/{name}/images")
async def list_images(
    name: str,
    sort: str = "quality",
    angle: str | None = None,
    horizontal_pose: str | None = None,
    vertical_pose: str | None = None,
    expression: str | None = None,
    lighting: str | None = None,
    quality: str | None = None,
) -> list[dict]:
    lib = await _get_library_data(name)
    engine = get_engine(lib["db"])
    async with get_session(engine) as session:
        img_repo = ImageRepository(session)
        face_repo = FaceRepository(session)
        q_repo = QualityRepository(session)
        cls_repo = ClassificationRepository(session)

        imgs = await img_repo.get_by_state("verified", limit=100000)

        safety_repo = SafetyScoreRepository(session)
        has_safety = await safety_repo.table_exists()

        # Parse filters using taxonomy constants
        # horizontal_pose takes precedence over legacy angle param
        h_pose_values = _parse_horizontal_pose_filter(horizontal_pose) or _parse_angle_filter(angle)
        v_pose_values = _parse_vertical_pose_filter(vertical_pose)
        quality_range = _parse_quality_filter(quality)
        expr_values = [e.strip().lower() for e in expression.split(",")] if expression else None
        light_values = [lv.strip().lower() for lv in lighting.split(",")] if lighting else None

        has_classification_filter = h_pose_values or v_pose_values or expr_values or light_values

        # Batch fetch all related data (eliminates N+1)
        image_ids = [img.id for img in imgs]
        best_faces = await face_repo.get_best_by_image_ids(image_ids)
        quality_map = await q_repo.get_by_image_ids(image_ids)
        cls_map = await cls_repo.get_by_image_ids(image_ids)
        safety_map = await safety_repo.get_by_image_ids(image_ids) if has_safety else {}

        results = []
        for img in imgs:
            face = best_faces.get(img.id)
            q = quality_map.get(img.id)
            c = cls_map.get(img.id)
            s = safety_map.get(img.id)

            # If classification filters are active but image has no classification, skip it
            if has_classification_filter and c is None:
                continue

            # Horizontal pose filter: exact set membership
            if h_pose_values and (c is None or c.horizontal_pose is None
                    or c.horizontal_pose not in h_pose_values):
                continue

            # Vertical pose filter: exact match
            if v_pose_values and (c is None or c.vertical_pose is None
                    or c.vertical_pose not in v_pose_values):
                continue

            # Expression filter: exact match
            if expr_values and (c is None or c.expression is None
                    or c.expression not in expr_values):
                continue

            # Lighting filter: exact match
            if light_values and (c is None or c.lighting is None
                    or c.lighting not in light_values):
                continue

            # Quality filter: semantic mapping or range
            if quality_range and q:
                lo, hi = quality_range
                if not (lo <= q.final_score <= hi):
                    continue

            results.append(_serialize_image(img, face, q, c, s))

        if sort == "quality":
            results.sort(
                key=lambda r: r["quality"]["final_score"] if r["quality"] else 0,
                reverse=True,
            )
        elif sort == "random":
            import random

            random.shuffle(results)

    return results


@router.get("/libraries/{name}/images/{content_hash}")
async def get_image(name: str, content_hash: str) -> dict:
    lib = await _get_library_data(name)
    engine = get_engine(lib["db"])
    async with get_session(engine) as session:
        img_repo = ImageRepository(session)
        face_repo = FaceRepository(session)
        q_repo = QualityRepository(session)
        cls_repo = ClassificationRepository(session)
        safety_repo = SafetyScoreRepository(session)

        img = await img_repo.get_by_hash(content_hash)
        if not img:
            raise HTTPException(404, "Image not found")

        faces = await face_repo.get_by_image_id(img.id)
        face = max(faces, key=lambda f: f.confidence or 0) if faces else None
        q = await q_repo.get_by_image_id(img.id)
        c = await cls_repo.get_by_image_id(img.id)

        # Safe safety query for old DBs without safety_scores table
        s = None
        has_safety = await safety_repo.table_exists()
        if has_safety:
            s = await safety_repo.get_by_image_id(img.id)

        return _serialize_image(img, face, q, c, s)


@router.get("/libraries/{name}/images/{content_hash}/file")
async def get_image_file(name: str, content_hash: str) -> FileResponse:
    lib = await _get_library_data(name)
    engine = get_engine(lib["db"])
    async with get_session(engine) as session:
        img_repo = ImageRepository(session)
        img = await img_repo.get_by_hash(content_hash)
        if not img or not img.local_path:
            raise HTTPException(404, "Image not found")

    file_path = Path(img.local_path)
    if not file_path.exists():
        file_path = lib["path"] / "images" / f"{content_hash}.jpg"
        if not file_path.exists():
            raise HTTPException(404, "Image file not found on disk")

    return FileResponse(file_path, media_type="image/jpeg")


# ── Practice global random endpoint ─────────────────────────────────────────


@router.get("/practice/random-global")
async def random_global_images(count: int = 100) -> list[dict]:
    """Return a balanced random sample of images across all ready libraries.

    Samples proportionally from each ready library to ensure fair representation,
    then shuffles the result so images from different libraries are interleaved.
    """
    import random as _random

    libs = _find_libraries()
    ready_libs: list[dict[str, Any]] = []

    for lib in libs:
        engine = get_engine(lib["db"])
        async with get_session(engine) as session:
            id_repo = IdentityRepository(session)
            bj_repo = BuildJobRepository(session)
            img_repo = ImageRepository(session)

            identity = await id_repo.get_by_name(lib["name"])
            identity_status = identity.status if identity else "unknown"
            active_job = await bj_repo.get_active_by_identity(lib["name"])
            latest_job = await bj_repo.get_latest_by_identity(lib["name"])
            verified_count = await img_repo.count_by_state("verified")

            status = _resolve_library_status(
                identity_status, active_job, latest_job, verified_count
            )
            if status == "ready" and verified_count > 0:
                ready_libs.append({**lib, "verified_count": verified_count})

    if not ready_libs:
        return []

    total_images = sum(lib["verified_count"] for lib in ready_libs)
    all_images: list[dict] = []

    for lib in ready_libs:
        proportion = lib["verified_count"] / total_images
        sample_size = max(1, round(count * proportion))

        engine = get_engine(lib["db"])
        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            face_repo = FaceRepository(session)
            q_repo = QualityRepository(session)
            cls_repo = ClassificationRepository(session)
            safety_repo = SafetyScoreRepository(session)

            sampled = await img_repo.get_random_by_state("verified", limit=sample_size)

            image_ids = [img.id for img in sampled]
            best_faces = await face_repo.get_best_by_image_ids(image_ids)
            quality_map = await q_repo.get_by_image_ids(image_ids)
            cls_map = await cls_repo.get_by_image_ids(image_ids)
            has_safety = await safety_repo.table_exists()
            safety_map = (
                await safety_repo.get_by_image_ids(image_ids) if has_safety else {}
            )

            for img in sampled:
                serialized = _serialize_image(
                    img,
                    best_faces.get(img.id),
                    quality_map.get(img.id),
                    cls_map.get(img.id),
                    safety_map.get(img.id),
                )
                serialized["library_name"] = lib["name"]
                all_images.append(serialized)

    _random.shuffle(all_images)
    return all_images[:count]


# ── Build endpoints ──────────────────────────────────────────────────────────


@router.post("/libraries/{name}/build")
async def start_build(name: str) -> dict:
    lib = await _get_library_data(name)

    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()
    job = await worker.enqueue(name)

    return {"name": name, "status": "queued", "build_job_id": job.id}


@router.post("/libraries/{name}/cancel")
async def cancel_build(name: str) -> None:
    lib = await _get_library_data(name)
    engine = get_engine(lib["db"])

    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()

    async with get_session(engine) as session:
        bj_repo = BuildJobRepository(session)
        id_repo = IdentityRepository(session)

        active = await bj_repo.get_active_by_identity(name)
        if active:
            await worker.cancel_job(active.id)

        identity = await id_repo.get_by_name(name)
        if identity:
            identity.status = "cancelled"
            await id_repo.update(identity)


@router.get("/builds/queue")
async def get_build_queue() -> dict:
    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()
    return await worker.get_status()


@router.get("/libraries/{name}/build/progress")
async def get_build_progress(name: str) -> dict:
    lib = await _get_library_data(name)
    engine = get_engine(lib["db"])
    async with get_session(engine) as session:
        bj_repo = BuildJobRepository(session)
        id_repo = IdentityRepository(session)
        img_repo = ImageRepository(session)

        job = await bj_repo.get_active_by_identity(name)
        if not job:
            job = await bj_repo.get_latest_by_identity(name)

        # Compute library status for frontend routing
        identity = await id_repo.get_by_name(name)
        identity_status = identity.status if identity else "unknown"

        if not job:
            return {
                "id": 0,
                "status": "pending",
                "library_status": identity_status,
                "current_stage": None,
                "stage_label": "Queued",
                "items_processed": 0,
                "items_total": 0,
                "error": None,
                "elapsed_ms": 0,
                "started_at": None,
                "completed_at": None,
                "stages_completed": [],
            }

        verified_count = (await img_repo.get_by_state("verified", limit=1)).__len__()
        active_job = await bj_repo.get_active_by_identity(name)
        library_status = _resolve_library_status(identity_status, active_job, job, verified_count)

        # Compute real-time progress from DB based on current stage
        items_processed = job.items_processed or 0
        items_total = job.items_total or 0

        if job.current_stage and job.status == "running":
            progress_info = STAGE_PROGRESS_MAP.get(job.current_stage)
            if progress_info is not None:
                input_count = 0
                for state in progress_info["input_states"]:
                    input_count += await img_repo.count_by_state(state)
                done_count = 0
                for state in progress_info["done_states"]:
                    done_count += await img_repo.count_by_state(state)
                items_total = input_count + done_count
                items_processed = done_count

    elapsed_ms = 0
    if job.started_at:
        from datetime import datetime

        now = datetime.utcnow()
        elapsed_ms = int((now - job.started_at).total_seconds() * 1000)

    return {
        "id": job.id,
        "status": job.status,
        "library_status": library_status,
        "current_stage": job.current_stage,
        "stage_label": job.stage_label,
        "items_processed": items_processed,
        "items_total": items_total,
        "error": job.error,
        "elapsed_ms": elapsed_ms,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "stages_completed": [],
    }


# ── Queue endpoints ───────────────────────────────────────────────────────────


class BatchEnqueueRequest(BaseModel):
    names: list[str]


@router.post("/queue/batch")
async def batch_enqueue(req: BatchEnqueueRequest) -> dict:
    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()
    result = await worker.enqueue_batch(req.names)
    return result


@router.post("/queue/{job_id}/cancel")
async def queue_cancel_job(job_id: int) -> dict:
    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()
    ok = await worker.cancel_job(job_id)
    if not ok:
        raise HTTPException(404, "Job not found or not cancellable")
    return {"ok": True}


@router.post("/queue/{job_id}/retry")
async def queue_retry_job(job_id: int) -> dict:
    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()
    ok = await worker.retry_job(job_id)
    if not ok:
        raise HTTPException(404, "Job not found or not retryable (max retries reached)")
    return {"ok": True}


@router.delete("/queue/{job_id}")
async def queue_remove_job(job_id: int) -> dict:
    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()
    ok = await worker.remove_job(job_id)
    if not ok:
        raise HTTPException(404, "Job not found or not removable")
    return {"ok": True}


@router.post("/queue/pause")
async def queue_pause() -> dict:
    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()
    await worker.pause()
    return {"ok": True, "paused": True}


@router.post("/queue/resume")
async def queue_resume() -> dict:
    from portrait_dataset_builder.core.queue_worker import get_queue_worker

    worker = get_queue_worker()
    await worker.resume()
    return {"ok": True, "paused": False}


# ── Build execution ──────────────────────────────────────────────────────────


async def _run_build(task_id: str, identity: str, job_id: int) -> None:
    import uuid

    lib_root = _data_root() / identity
    db_path = lib_root / "portrait.db"

    engine = get_engine(db_path)
    build_id = str(uuid.uuid4())[:8]

    _migrate_db(db_path)
    await _configure_sqlite(db_path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with get_session(engine) as session:
        bj_repo = BuildJobRepository(session)
        await bj_repo.update_status(job_id, "running", stage_label="Starting")

    try:
        settings = load_settings(identity)
        from portrait_dataset_builder.core.orchestrator import PipelineOrchestrator
        from portrait_dataset_builder.core.pipeline import PipelineContext
        from portrait_dataset_builder.logging import setup_logging
        from portrait_dataset_builder.pipeline.classification import ClassificationStage
        from portrait_dataset_builder.pipeline.download import DownloadStage
        from portrait_dataset_builder.pipeline.duplicates import DuplicateDetectionStage
        from portrait_dataset_builder.pipeline.export import ExportStage
        from portrait_dataset_builder.pipeline.face_detection import FaceDetectionStage
        from portrait_dataset_builder.pipeline.face_verification import FaceVerificationStage
        from portrait_dataset_builder.pipeline.identity_bootstrap import IdentityBootstrapStage
        from portrait_dataset_builder.pipeline.quality import QualityStage
        from portrait_dataset_builder.pipeline.safety_gate import SafetyGateStage
        from portrait_dataset_builder.pipeline.search import SearchStage
        from portrait_dataset_builder.pipeline.semantic_filter import SemanticFilterStage
        from portrait_dataset_builder.pipeline.url_safety_filter import URLSafetyFilterStage

        setup_logging(settings.log_level)

        context = PipelineContext(
            identity=identity,
            output_dir=settings.resolve_data_dir(),
            db_path=settings.resolve_db_path(),
            settings=settings,
        )

        stage_map = {
            "search": SearchStage,
            "url_safety_filter": URLSafetyFilterStage,
            "identity_bootstrap": IdentityBootstrapStage,
            "download": DownloadStage,
            "safety_gate": SafetyGateStage,
            "face_detection": FaceDetectionStage,
            "face_verification": FaceVerificationStage,
            "semantic_filter": SemanticFilterStage,
            "quality": QualityStage,
            "duplicates": DuplicateDetectionStage,
            "classification": ClassificationStage,
            "export": ExportStage,
        }

        stages = []
        for stage_name in settings.pipeline.stages:
            if stage_name in stage_map:
                stages.append(stage_map[stage_name]())

        async def _update_job(**kwargs: Any) -> None:
            eng = get_engine(db_path)
            async with get_session(eng) as sess:
                r = BuildJobRepository(sess)
                await r.update_status(job_id, **kwargs)

        for stage in stages:
            orig_execute = stage.execute

            async def _patched_execute(
                ctx: PipelineContext,
                _orig: Any = orig_execute,
                _sname: str = stage.name,
            ) -> Any:
                await _update_job(
                    status="running",
                    current_stage=_sname,
                    stage_label=STAGE_LABELS.get(_sname, _sname),
                )
                result = await _orig(ctx)
                await _update_job(status="running")
                return result

            stage.execute = _patched_execute  # type: ignore

        orchestrator = PipelineOrchestrator(stages, context, build_id=build_id)
        await orchestrator.run()

        verified_count = 0
        async with get_session(engine) as sess:
            from portrait_dataset_builder.database.repository import ImageRepository

            img_repo = ImageRepository(sess)
            verified = await img_repo.get_by_state("verified", limit=1)
            verified_count = len(verified)

        if verified_count == 0:
            async with get_session(engine) as sess:
                id_repo = IdentityRepository(sess)
                ident = await id_repo.get_by_name(identity)
                if ident and ident.status not in ("identity_established", "identity_unverified"):
                    ident.status = "empty"
                    await id_repo.update(ident)
                bj_repo = BuildJobRepository(sess)
                await bj_repo.update_status(
                    job_id,
                    "completed",
                    error=None,
                    stage_label="No valid images",
                )
        else:
            async with get_session(engine) as sess:
                id_repo = IdentityRepository(sess)
                ident = await id_repo.get_by_name(identity)
                if ident:
                    ident.status = "ready"
                    await id_repo.update(ident)
                bj_repo = BuildJobRepository(sess)
                await bj_repo.update_status(job_id, "completed", stage_label="Complete")

    except asyncio.CancelledError:
        async with get_session(engine) as sess:
            id_repo = IdentityRepository(sess)
            ident = await id_repo.get_by_name(identity)
            if ident:
                ident.status = "cancelled"
                await id_repo.update(ident)
            bj_repo = BuildJobRepository(sess)
            await bj_repo.update_status(job_id, "cancelled", stage_label="Cancelled")
    except Exception as e:
        async with get_session(engine) as sess:
            id_repo = IdentityRepository(sess)
            ident = await id_repo.get_by_name(identity)
            if ident:
                ident.status = "failed"
                await id_repo.update(ident)
            bj_repo = BuildJobRepository(sess)
            await bj_repo.update_status(job_id, "failed", error=str(e), stage_label=f"Failed: {e}")
    finally:
        _build_tasks.pop(task_id, None)


# ── Coverage endpoint ────────────────────────────────────────────────────────


@router.get("/libraries/{name}/coverage")
async def get_coverage(name: str) -> dict:
    lib = await _get_library_data(name)
    engine = get_engine(lib["db"])
    async with get_session(engine) as session:
        img_repo = ImageRepository(session)
        face_repo = FaceRepository(session)
        cls_repo = ClassificationRepository(session)

        verified = await img_repo.get_by_state("verified", limit=100000)

        # Batch fetch (eliminates N+1)
        image_ids = [img.id for img in verified]
        best_faces = await face_repo.get_best_by_image_ids(image_ids)
        cls_map = await cls_repo.get_by_image_ids(image_ids)

        poses = []
        expressions = []
        lighting_data = []
        age_groups = []

        for img in verified:
            best = best_faces.get(img.id)
            if best:
                poses.append((best.yaw or 0, best.pitch or 0))

            c = cls_map.get(img.id)
            if c:
                if c.expression:
                    expressions.append(c.expression)
                if c.lighting:
                    lighting_data.append(c.lighting)
                if c.age_group:
                    age_groups.append(c.age_group)

    from portrait_dataset_builder.coverage import PoseCoverage

    pc = PoseCoverage()
    result = pc.compute(poses)

    expr_counter = Counter(expressions)
    light_counter = Counter(lighting_data)
    age_counter = Counter(age_groups)

    yaw_step = 10.0
    pitch_step = 10.0
    yaw_bins = [round(-90 + (i + 0.5) * yaw_step, 1) for i in range(18)]
    pitch_bins = [round(-30 + (i + 0.5) * pitch_step, 1) for i in range(6)]

    return {
        "yaw_bins": yaw_bins,
        "pitch_bins": pitch_bins,
        "heatmap": [list(row) for row in zip(*result["grid"], strict=False)],
        "expressions": dict(expr_counter),
        "lighting": dict(light_counter),
        "age_groups": dict(age_counter),
        "horizontal_poses": dict(Counter(
            cls_map.get(img.id).horizontal_pose
            for img in verified
            if cls_map.get(img.id) and cls_map.get(img.id).horizontal_pose
        )),
        "vertical_poses": dict(Counter(
            cls_map.get(img.id).vertical_pose
            for img in verified
            if cls_map.get(img.id) and cls_map.get(img.id).vertical_pose
        )),
    }


# ── Stats endpoint ───────────────────────────────────────────────────────────


@router.get("/libraries/{name}/stats")
async def get_stats(name: str) -> dict:
    lib = await _get_library_data(name)
    engine = get_engine(lib["db"])
    async with get_session(engine) as session:
        img_repo = ImageRepository(session)
        face_repo = FaceRepository(session)
        q_repo = QualityRepository(session)
        cls_repo = ClassificationRepository(session)

        total = await img_repo.count()
        verified_imgs = await img_repo.get_by_state("verified", limit=100000)
        verified_count = len(verified_imgs)

        # Batch fetch (eliminates N+1)
        image_ids = [img.id for img in verified_imgs]
        quality_map = await q_repo.get_by_image_ids(image_ids)
        best_faces = await face_repo.get_best_by_image_ids(image_ids)
        cls_map = await cls_repo.get_by_image_ids(image_ids)

        scores = []
        yaws = []
        expressions = Counter()
        angles = Counter()

        for img in verified_imgs:
            q = quality_map.get(img.id)
            if q:
                scores.append(q.final_score)

            best = best_faces.get(img.id)
            if best:
                yaws.append(best.yaw or 0)

            c = cls_map.get(img.id)
            if c:
                if c.expression:
                    expressions[c.expression] += 1
                if c.angle:
                    angles[c.angle] += 1

    avg_quality = sum(scores) / len(scores) if scores else 0
    avg_yaw = sum(yaws) / len(yaws) if yaws else 0

    return {
        "total_images": total,
        "verified_images": verified_count,
        "avg_quality": round(avg_quality, 3),
        "avg_yaw": round(avg_yaw, 1),
        "expressions": dict(expressions),
        "angles": dict(angles),
        "horizontal_poses": dict(Counter(
            cls_map.get(img.id).horizontal_pose
            for img in verified_imgs
            if cls_map.get(img.id) and cls_map.get(img.id).horizontal_pose
        )),
        "vertical_poses": dict(Counter(
            cls_map.get(img.id).vertical_pose
            for img in verified_imgs
            if cls_map.get(img.id) and cls_map.get(img.id).vertical_pose
        )),
    }


# ── Review endpoints ─────────────────────────────────────────────────────────


@router.get("/libraries/{name}/review")
async def get_review_queue(name: str) -> list[dict]:
    lib = await _get_library_data(name)
    engine = get_engine(lib["db"])
    async with get_session(engine) as session:
        rq_repo = ReviewQueueRepository(session)
        img_repo = ImageRepository(session)
        face_repo = FaceRepository(session)
        q_repo = QualityRepository(session)
        cls_repo = ClassificationRepository(session)
        safety_repo = SafetyScoreRepository(session)

        has_safety = await safety_repo.table_exists()
        pending = await rq_repo.get_pending(limit=100)

        # Collect image IDs from review entries
        image_ids = []
        entry_by_image: dict[int, object] = {}
        imgs_by_id: dict[int, Any] = {}
        for entry in pending:
            img = await img_repo.get_by_id(entry.image_id)
            if img:
                image_ids.append(img.id)
                entry_by_image[img.id] = entry
                imgs_by_id[img.id] = img

        # Batch fetch related data (eliminates N+1)
        best_faces = await face_repo.get_best_by_image_ids(image_ids)
        quality_map = await q_repo.get_by_image_ids(image_ids)
        cls_map = await cls_repo.get_by_image_ids(image_ids)
        safety_map = await safety_repo.get_by_image_ids(image_ids) if has_safety else {}

        results = []
        for img_id in image_ids:
            img = imgs_by_id[img_id]
            face = best_faces.get(img_id)
            q = quality_map.get(img_id)
            c = cls_map.get(img_id)
            s = safety_map.get(img_id)
            results.append(_serialize_image(img, face, q, c, s))

    return results


@router.post("/libraries/{name}/review/{content_hash}")
async def review_image(name: str, content_hash: str, body: dict) -> None:
    lib = await _get_library_data(name)
    engine = get_engine(lib["db"])
    async with get_session(engine) as session:
        img_repo = ImageRepository(session)
        rq_repo = ReviewQueueRepository(session)

        img = await img_repo.get_by_hash(content_hash)
        if not img:
            raise HTTPException(404, "Image not found")

        pending = await rq_repo.get_pending(limit=1000)
        for entry in pending:
            if entry.image_id == img.id:
                answer = "yes" if body.get("accepted") else "no"
                await rq_repo.mark_reviewed(entry.id, answer)
                break


_BOOKS_ROOT = Path("data/books")


def _find_books() -> list[dict[str, Any]]:
    if not _BOOKS_ROOT.exists():
        return []
    books = []
    for d in sorted(_BOOKS_ROOT.iterdir()):
        meta = d / "metadata.json"
        pages_dir = d / "pages"
        if meta.exists() and pages_dir.exists():
            try:
                data = json.loads(meta.read_text())
                data["slug"] = d.name
                books.append(data)
            except (json.JSONDecodeError, OSError):
                continue
    return books


@router.get("/books")
async def list_books() -> list[dict[str, Any]]:
    return _find_books()


@router.get("/books/pages/random")
async def random_book_pages(count: int = 100, book: str | None = None) -> list[dict[str, Any]]:
    all_books = _find_books()
    if not all_books:
        return []

    if book:
        all_books = [b for b in all_books if b["slug"] == book]
        if not all_books:
            return []

    all_pages: list[dict[str, Any]] = []
    for b in all_books:
        book_dir = _BOOKS_ROOT / b["slug"]
        pages_dir = book_dir / "pages"
        for f in sorted(pages_dir.iterdir(), key=lambda p: _page_sort_key(p)):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                page_num = _parse_page_number(f)
                if page_num is not None:
                    all_pages.append({
                        "slug": b["slug"],
                        "title": b["title"],
                        "page_number": page_num,
                    })

    random.shuffle(all_pages)
    return all_pages[:count]


def _parse_page_number(path: Path) -> int | None:
    import re
    m = re.search(r'(\d+)', path.stem)
    return int(m.group(1)) if m else None


def _page_sort_key(path: Path) -> tuple[int, str]:
    n = _parse_page_number(path)
    return (n if n is not None else 0, path.name)


@router.get("/books/pages/{slug}/{page_num}/file")
async def get_book_page_file(slug: str, page_num: int) -> FileResponse:
    pages_dir = _BOOKS_ROOT / slug / "pages"
    if not pages_dir.exists():
        raise HTTPException(404, "Book not found")
    for f in pages_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            n = _parse_page_number(f)
            if n == page_num:
                return FileResponse(f)
    raise HTTPException(404, "Page not found")
