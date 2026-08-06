"""Safety Gate pipeline stage — NSFW detection + AI-generated image detection."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from portrait_dataset_builder.compute import (
    ResolvedDevice,
    get_torch_device,
    log_device_info,
    resolve_device,
)
from portrait_dataset_builder.compute.models import get_clip_model, get_nsfw_model
from portrait_dataset_builder.compute.resize import resize_for_inference
from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session
from portrait_dataset_builder.database.models import SafetyScore
from portrait_dataset_builder.database.repository import ImageRepository, SafetyScoreRepository
from portrait_dataset_builder.logging import get_logger

logger = get_logger("stage.safety_gate")

NSFW_PIPELINE: Any = None
CLIP_TEXT_FEATURES: Any = None
CLIP_RESOLVED: ResolvedDevice | None = None


def _load_nsfw_model(device: str = "cpu") -> Any:
    global NSFW_PIPELINE
    if NSFW_PIPELINE is not None:
        return NSFW_PIPELINE
    try:
        NSFW_PIPELINE = get_nsfw_model(device)
        logger.info("NSFW model loaded successfully (device={})", device)
    except Exception as e:
        logger.error("Failed to load NSFW model: {}", e)
        NSFW_PIPELINE = None
    return NSFW_PIPELINE


def _load_clip(device: str = "cpu") -> tuple[Any, Any, Any, Any]:
    try:
        model, preprocess = get_clip_model(device)
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return model, preprocess, tokenizer, None
    except Exception as e:
        logger.error("Failed to load CLIP for safety gate: {}", e)
        return None, None, None, None


def _detect_nsfw_model(image: PILImage.Image) -> tuple[bool, float]:
    """Primary NSFW detection using Falconsai ViT-B/32."""
    model = NSFW_PIPELINE
    if model is None:
        return True, 0.0
    try:
        results = model(image, top_k=None)
        nsfw_score = 0.0
        for r in results:
            if r["label"] == "nsfw":
                nsfw_score = r["score"]
                break
        return nsfw_score >= 0.20, nsfw_score
    except Exception as e:
        logger.warning("NSFW model inference failed: {}", e)
        return True, 0.0


def _detect_nsfw_clip(image: PILImage.Image, device: str = "cpu") -> tuple[bool, float]:
    """Secondary NSFW detection using CLIP zero-shot safety prompts."""
    model, preprocess, tokenizer, _ = _load_clip(device)
    if model is None:
        return False, 0.5
    try:
        import torch
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        prompts = ["a safe portrait photograph", "nsfw explicit inappropriate content"]
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            text_features = model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            nsfw_clip_score = similarity[0][1].item()
        is_nsfw = nsfw_clip_score >= 0.50
        return is_nsfw, nsfw_clip_score
    except Exception as e:
        logger.warning("CLIP safety check failed: {}", e)
        return False, 0.5


def _detect_ai_metadata(image: PILImage.Image) -> tuple[bool, float]:
    """Metadata-based AI detection: check EXIF, PNG chunks, file structure."""
    try:
        info = image.info or {}
        if "exif" in info:
            exif_str = str(info["exif"]).lower()
            ai_markers = ["stable diffusion", "midjourney", "dalle", "dall-e",
                          "firefly", "craiyon", "nightcafe", "novelai", "stablediffusion"]
            if any(marker in exif_str for marker in ai_markers):
                return True, 0.90
        if "prompt" in info or "parameters" in info:
            return True, 0.85
        fmt = getattr(image, "format", "") or ""
        fmt_lower = fmt.lower()
        if fmt_lower == "webp":
            try:
                exif_data = image.getexif()
                if exif_data and any(
                    k in str(exif_data).lower()
                    for k in ("generation", "ai", "stable")
                ):
                    return True, 0.80
            except Exception:
                pass
        return False, 0.0
    except Exception:
        return False, 0.0


def _detect_ai_visual(pixels: np.ndarray) -> tuple[bool, float]:
    """Visual heuristics for AI detection: noise patterns, color distribution."""
    try:
        if len(pixels.shape) != 3:
            return False, 0.0
        gray = np.mean(pixels, axis=2) if len(pixels.shape) == 3 else pixels
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        from scipy.ndimage import convolve
        edge_map = convolve(gray.astype(np.float64), laplacian)
        edge_std = np.std(edge_map)
        if edge_std < 3.0:
            return True, 0.70
        h, w = gray.shape[:2]
        block_size = 8
        blocks_h = h // block_size
        blocks_w = w // block_size
        if blocks_h < 2 or blocks_w < 2:
            return False, 0.0
        block_means = np.zeros((blocks_h, blocks_w))
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_means[i, j] = np.mean(block)
        spatial_variance = np.var(block_means)
        if spatial_variance < 10.0:
            return True, 0.65
        return False, 0.0
    except Exception:
        return False, 0.0


def _combined_ai_score(
    metadata_score: float, visual_score: float, metadata_detected: bool, visual_detected: bool
) -> float:
    """Combine metadata and visual AI scores."""
    if metadata_detected and metadata_score >= 0.80:
        return min(1.0, metadata_score * 1.1)
    if visual_detected and visual_score >= 0.65:
        combined = (
            (visual_score * 0.6 + metadata_score * 0.4)
            if metadata_detected
            else visual_score
        )
        return combined
    if metadata_detected and visual_detected:
        return (metadata_score + visual_score) / 2.0
    if metadata_detected:
        return metadata_score
    if visual_detected:
        return visual_score
    return max(metadata_score, visual_score)


class SafetyGateStage(PipelineStage):
    """Comprehensive content safety: NSFW detection + AI-generated image filtering."""

    def __init__(self) -> None:
        super().__init__("safety_gate")
        self._resolved: ResolvedDevice | None = None

    async def should_run(self, context: PipelineContext) -> bool:
        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            count = await repo.count_by_state("downloaded")
        return count > 0

    async def execute(self, context: PipelineContext) -> StageResult:
        global CLIP_RESOLVED

        self._resolved = resolve_device(context.settings.effective_device)
        torch_device = get_torch_device(self._resolved)
        device_str = str(torch_device)
        log_device_info(self._resolved)

        max_dim = context.settings.compute.inference_max_dimension
        nsfw_threshold = context.settings.safety.nsfw_threshold
        ai_mode = context.settings.safety.ai_mode
        fail_closed = context.settings.safety.fail_closed

        _ = _load_nsfw_model(device_str)
        CLIP_RESOLVED = self._resolved

        if fail_closed and NSFW_PIPELINE is None:
            logger.error("NSFW model unavailable and fail_closed=true — aborting")
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                errors=["NSFW model unavailable (fail_closed=true)"],
            )

        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            images = await img_repo.get_by_state("downloaded", limit=100000)

        safe_count = 0
        nsfw_count = 0
        ai_count = 0
        error_count = 0
        errors: list[str] = []

        _progress = context.metadata.get("_progress_task")
        total = len(images)

        for i, image_record in enumerate(images):
            if i % 5 == 0:
                await asyncio.sleep(0)
            if _progress:
                _progress["items_processed"] = i
                _progress["items_total"] = total

            if not image_record.local_path or not Path(image_record.local_path).exists():
                error_count += 1
                continue

            try:
                image = PILImage.open(image_record.local_path).convert("RGB")
                w, h = image.size
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = image.resize((new_w, new_h), PILImage.LANCZOS)

                nsfw_reject, nsfw_score = _detect_nsfw_model(image)
                clip_reject, clip_score = _detect_nsfw_clip(image, device=device_str)

                is_nsfw = nsfw_reject or (nsfw_score < nsfw_threshold and clip_reject)
                combined_nsfw_score = max(nsfw_score, clip_score)

                metadata_detected, metadata_score = _detect_ai_metadata(image)
                pixels = np.array(image)
                visual_detected, visual_score = _detect_ai_visual(pixels)
                ai_score = _combined_ai_score(
                    metadata_score, visual_score,
                    metadata_detected, visual_detected,
                )

                is_ai = False
                rejection_reason = None

                if is_nsfw:
                    is_ai = False
                    rejection_reason = f"nsfw_score={combined_nsfw_score:.3f}"
                elif ai_mode == "strict":
                    if ai_score >= 0.65 or metadata_detected:
                        is_ai = True
                        rejection_reason = f"ai_score={ai_score:.3f}"
                elif ai_mode == "balanced":
                    if ai_score >= 0.80 and metadata_detected:
                        is_ai = True
                        rejection_reason = f"ai_score={ai_score:.3f}"

                source_trust = image_record.source_provider or "unknown"
                trust_scores = context.settings.safety.source_trust_scores
                default_trust = trust_scores.get("unknown", 0.3)
                source_trust_score = trust_scores.get(source_trust, default_trust)
                for key in trust_scores:
                    if key in source_trust.lower():
                        source_trust_score = trust_scores[key]
                        break

                async with get_session(engine) as session:
                    s_repo = SafetyScoreRepository(session)
                    safety = SafetyScore(
                        image_id=image_record.id,
                        is_nsfw=is_nsfw,
                        nsfw_score=combined_nsfw_score,
                        is_ai_generated=is_ai,
                        ai_probability=ai_score,
                        real_photo_score=1.0 - ai_score,
                        source=source_trust,
                        source_trust_score=source_trust_score,
                        rejection_reason=rejection_reason,
                        pipeline_state="rejected" if (is_nsfw or is_ai) else "passed",
                    )
                    await s_repo.add(safety)

                if is_nsfw:
                    nsfw_count += 1
                elif is_ai:
                    ai_count += 1
                else:
                    safe_count += 1

                async with get_session(engine) as session:
                    img_repo = ImageRepository(session)
                    if is_nsfw or is_ai:
                        await img_repo.update_state(image_record.id, "rejected")
                    else:
                        await img_repo.update_state(image_record.id, "downloaded")

            except Exception as e:
                error_count += 1
                errors.append(f"Image {image_record.id}: {e}")

        logger.info(
            "Safety Gate: {} safe, {} nsfw, {} ai_generated, {} errors | device={}",
            safe_count, nsfw_count, ai_count, error_count, device_str,
        )

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(images),
            items_succeeded=safe_count,
            items_rejected=nsfw_count + ai_count,
            items_failed=error_count,
            errors=errors[:50],
            metadata={
                "nsfw_rejected": nsfw_count,
                "ai_rejected": ai_count,
                "safe": safe_count,
                "errors": error_count,
            },
        )
