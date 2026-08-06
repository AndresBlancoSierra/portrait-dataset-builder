"""Image resize utilities for inference optimization."""

from __future__ import annotations

import cv2
import numpy as np


def resize_for_inference(
    img: np.ndarray,
    max_dim: int = 1600,
) -> tuple[np.ndarray, float]:
    """Resize image for inference while preserving aspect ratio.

    Returns (resized_image, scale_factor) where scale_factor converts
    coordinates from resized space back to original space.

    If the image is already smaller than max_dim, returns it unchanged
    with scale_factor=1.0.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0

    if w >= h:
        new_w = max_dim
        new_h = int(h * max_dim / w)
    else:
        new_h = max_dim
        new_w = int(w * max_dim / h)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale = h / new_h
    return resized, scale


def scale_bbox(
    bbox: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    """Scale bounding box from resized coordinates back to original.

    Args:
        bbox: (x, y, w, h) in resized image coordinates
        scale: resize scale factor (original_size / resized_size)

    Returns:
        (x, y, w, h) in original image coordinates
    """
    x, y, w, h = bbox
    return (x * scale, y * scale, w * scale, h * scale)


def scale_landmarks(
    landmarks: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Scale face landmarks from resized coordinates back to original.

    Args:
        landmarks: (N, 2) array of (x, y) coordinates
        scale: resize scale factor

    Returns:
        Scaled landmarks array
    """
    return landmarks * scale
