"""Shared model cache to avoid duplicate model loading across pipeline stages."""

from __future__ import annotations

import threading
from typing import Any

from portrait_dataset_builder.logging import get_logger

logger = get_logger("compute.models")

_clip_lock = threading.Lock()
_clip_cache: dict[str, tuple[Any, Any]] = {}


def get_clip_model(device: str = "cpu") -> tuple[Any, Any]:
    """Get or create a shared CLIP model instance.

    Returns (model, preprocess) tuple. The model is cached per device.
    """
    cache_key = f"clip_{device}"

    if cache_key in _clip_cache:
        return _clip_cache[cache_key]

    with _clip_lock:
        if cache_key in _clip_cache:
            return _clip_cache[cache_key]

        import open_clip
        import torch

        model_name = "ViT-B-32"
        pretrained = "openai"

        logger.info("Loading CLIP model: {} (device={})", model_name, device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device)
        model.eval()

        _clip_cache[cache_key] = (model, preprocess)
        logger.info("CLIP model loaded and cached on {}", device)

        return model, preprocess


def get_nsfw_model(device: str = "cpu") -> Any:
    """Get or create a shared NSFW classifier.

    Returns the HuggingFace pipeline object.
    """
    cache_key = f"nsfw_{device}"

    if cache_key in _clip_cache:
        return _clip_cache[cache_key]

    with _clip_lock:
        if cache_key in _clip_cache:
            return _clip_cache[cache_key]

        from portrait_dataset_builder.config.settings import get_settings

        settings = get_settings()
        import torch
        from transformers import pipeline as hf_pipeline

        torch_device = 0 if device == "cuda" else -1
        logger.info("Loading NSFW model: {} (device={})", settings.safety.nsfw_model, device)

        nsfw_pipeline = hf_pipeline(
            "image-classification",
            model=settings.safety.nsfw_model,
            device=torch_device,
        )

        _clip_cache[cache_key] = nsfw_pipeline
        logger.info("NSFW model loaded on {}", device)

        return nsfw_pipeline


def release_clip_model(device: str = "cpu") -> None:
    """Release cached CLIP model for the given device."""
    cache_key = f"clip_{device}"
    with _clip_lock:
        _clip_cache.pop(cache_key, None)
    logger.info("CLIP model released for device={}", device)


def release_all_models() -> None:
    """Release all cached models."""
    with _clip_lock:
        _clip_cache.clear()
    logger.info("All cached models released")
