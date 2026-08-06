"""Semantic filter pipeline stage using CLIP to reject non-portrait images."""

from __future__ import annotations

import asyncio
from pathlib import Path

from portrait_dataset_builder.compute import (
    ResolvedDevice,
    get_torch_device,
    log_device_info,
    resolve_device,
)
from portrait_dataset_builder.compute.models import get_clip_model
from portrait_dataset_builder.compute.resize import resize_for_inference
from portrait_dataset_builder.core.pipeline import (
    PipelineContext,
    PipelineStage,
    StageResult,
    StageStatus,
)
from portrait_dataset_builder.database import get_engine, get_session
from portrait_dataset_builder.database.repository import ImageRepository
from portrait_dataset_builder.logging import get_logger

logger = get_logger("stage.semantic_filter")

POSITIVE_PROMPTS = [
    "a photo of a single person's face",
    "a headshot portrait of one person",
    "a close-up photo of a man",
    "a close-up photo of a woman",
    "a celebrity photo at an event",
    "a studio portrait photograph",
    "a red carpet photo of an actor",
]

NEGATIVE_PROMPTS = [
    "a text document or screenshot",
    "a meme with text overlay",
    "a movie poster or graphic design",
    "a group photo with many people",
    "a landscape or scenery photo",
    "a product or object photo",
    "a cartoon or illustration",
    "a screenshot of a website",
    "a math problem or equation",
    "a logo or brand image",
    "a black image or solid color",
    "a movie scene with actors",
    "a drawing or sketch",
]


class SemanticFilterStage(PipelineStage):
    """Reject non-portrait images using CLIP semantic matching."""

    def __init__(self) -> None:
        super().__init__("semantic_filter")
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._pos_tokenized = None
        self._neg_tokenized = None
        self._resolved: ResolvedDevice | None = None

    async def should_run(self, context: PipelineContext) -> bool:
        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            count = await repo.count_by_state("verified")
        return count > 0

    def _init_model(self, context: PipelineContext) -> None:
        if self._model is not None:
            return

        import open_clip
        import torch

        self._resolved = resolve_device(context.settings.effective_device)
        torch_device = get_torch_device(self._resolved)
        log_device_info(self._resolved)

        model_name = "ViT-B-32"

        self._model, _, self._preprocess = get_clip_model(str(torch_device))
        self._tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            self._pos_tokenized = self._tokenizer(POSITIVE_PROMPTS).to(torch_device)
            self._neg_tokenized = self._tokenizer(NEGATIVE_PROMPTS).to(torch_device)

        self._pos_features = self._model.encode_text(self._pos_tokenized)
        self._neg_features = self._model.encode_text(self._neg_tokenized)
        self._pos_features = self._pos_features / self._pos_features.norm(dim=-1, keepdim=True)
        self._neg_features = self._neg_features / self._neg_features.norm(dim=-1, keepdim=True)

        logger.info(
            "CLIP model loaded: {} | {} pos prompts, {} neg prompts | device={}",
            model_name,
            len(POSITIVE_PROMPTS),
            len(NEGATIVE_PROMPTS),
            torch_device,
        )

    async def execute(self, context: PipelineContext) -> StageResult:
        self._init_model(context)

        import cv2
        import torch

        max_dim = context.settings.compute.inference_max_dimension
        torch_device = get_torch_device(self._resolved)

        engine = get_engine(context.db_path)
        async with get_session(engine) as session:
            repo = ImageRepository(session)
            images = await repo.get_by_state("verified", limit=100000)

        pos_threshold = context.settings.semantic_filter.pos_threshold
        neg_threshold = context.settings.semantic_filter.neg_threshold

        passed = 0
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

            try:
                img = cv2.imread(image_record.local_path)
                if img is None:
                    failed += 1
                    continue

                resized_img, _ = resize_for_inference(img, max_dim=max_dim)
                img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                img_tensor = self._preprocess(
                    __import__("PIL").Image.fromarray(img_rgb)
                ).unsqueeze(0).to(torch_device)

                with torch.no_grad():
                    img_features = self._model.encode_image(img_tensor)
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

                    pos_sims = (img_features @ self._pos_features.T).squeeze(0)
                    neg_sims = (img_features @ self._neg_features.T).squeeze(0)

                    best_pos = pos_sims.max().item()
                    best_neg = neg_sims.max().item()

                is_portrait = best_pos >= pos_threshold and best_neg < neg_threshold

                async with get_session(engine) as session:
                    img_repo = ImageRepository(session)
                    if is_portrait:
                        passed += 1
                    else:
                        await img_repo.update_state(image_record.id, "rejected")
                        rejected += 1
                        logger.debug(
                            "Rejected: {} (pos={:.3f} neg={:.3f})",
                            image_record.content_hash[:16],
                            best_pos,
                            best_neg,
                        )

            except Exception as e:
                failed += 1
                errors.append(f"Image {image_record.id}: {type(e).__name__}: {e}")
                if len(errors) <= 5:
                    logger.error("Semantic filter error on image {}: {}", image_record.id, e)

        logger.info(
            "Semantic filter: {} passed, {} rejected, {} failed "
            "(pos_thresh={:.3f}, neg_thresh={:.3f})",
            passed,
            rejected,
            failed,
            pos_threshold,
            neg_threshold,
        )

        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            items_processed=len(images),
            items_succeeded=passed,
            items_rejected=rejected,
            items_failed=failed,
            errors=errors[:50],
            metadata={
                "passed": passed,
                "rejected": rejected,
                "pos_threshold": pos_threshold,
                "neg_threshold": neg_threshold,
            },
        )

    async def teardown(self, context: PipelineContext) -> None:
        self._model = None
