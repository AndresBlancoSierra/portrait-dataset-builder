"""Tests for compute module: device resolution, providers, resize."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from portrait_dataset_builder.compute import (
    ResolvedDevice,
    _check_cuda_onnx,
    _check_cuda_torch,
    get_ctx_id,
    get_gpu_semaphore,
    get_onnx_providers,
    get_torch_device,
    resolve_device,
)
from portrait_dataset_builder.compute.resize import resize_for_inference, scale_bbox, scale_landmarks


class TestResolveDevice:
    """Tests for device resolution logic."""

    def test_auto_with_cuda(self) -> None:
        with (
            patch("portrait_dataset_builder.compute._check_cuda_onnx", return_value=True),
            patch("portrait_dataset_builder.compute._check_cuda_torch", return_value=(True, "RTX 3050", 4000)),
        ):
            result = resolve_device("auto")
            assert result.actual == "cuda"
            assert result.ctx_id == 0
            assert result.torch_device == "cuda"
            assert result.onnx_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
            assert result.gpu_name == "RTX 3050"
            assert result.vram_mb == 4000

    def test_auto_without_cuda(self) -> None:
        with (
            patch("portrait_dataset_builder.compute._check_cuda_onnx", return_value=False),
            patch("portrait_dataset_builder.compute._check_cuda_torch", return_value=(False, None, None)),
        ):
            result = resolve_device("auto")
            assert result.actual == "cpu"
            assert result.ctx_id == -1
            assert result.torch_device == "cpu"
            assert result.onnx_providers == ["CPUExecutionProvider"]
            assert result.gpu_name is None

    def test_cpu_always_cpu(self) -> None:
        result = resolve_device("cpu")
        assert result.actual == "cpu"
        assert result.ctx_id == -1
        assert result.torch_device == "cpu"

    def test_cuda_fallback_when_unavailable(self) -> None:
        with (
            patch("portrait_dataset_builder.compute._check_cuda_onnx", return_value=False),
            patch("portrait_dataset_builder.compute._check_cuda_torch", return_value=(False, None, None)),
        ):
            result = resolve_device("cuda")
            assert result.actual == "cpu"
            assert result.ctx_id == -1

    def test_cuda_with_only_torch(self) -> None:
        with (
            patch("portrait_dataset_builder.compute._check_cuda_onnx", return_value=False),
            patch("portrait_dataset_builder.compute._check_cuda_torch", return_value=(True, "RTX 3050", 4000)),
        ):
            result = resolve_device("auto")
            assert result.actual == "cuda"
            assert result.onnx_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_unknown_device_falls_back_to_cpu(self) -> None:
        result = resolve_device("unknown_gpu")
        assert result.actual == "cpu"

    def test_resolved_device_is_frozen(self) -> None:
        result = resolve_device("cpu")
        with pytest.raises(AttributeError):
            result.actual = "cuda"  # type: ignore[misc]


class TestComputeHelpers:
    """Tests for helper functions."""

    def test_get_onnx_providers(self) -> None:
        resolved = ResolvedDevice(
            requested="auto", actual="cuda",
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ctx_id=0, torch_device="cuda",
            gpu_name="RTX 3050", vram_mb=4000, cuda_available=True,
        )
        providers = get_onnx_providers(resolved)
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert providers is not resolved.onnx_providers  # should be a copy

    def test_get_ctx_id(self) -> None:
        resolved = ResolvedDevice(
            requested="auto", actual="cuda",
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ctx_id=0, torch_device="cuda",
            gpu_name="RTX 3050", vram_mb=4000, cuda_available=True,
        )
        assert get_ctx_id(resolved) == 0

    def test_get_torch_device(self) -> None:
        resolved = ResolvedDevice(
            requested="auto", actual="cpu",
            onnx_providers=["CPUExecutionProvider"],
            ctx_id=-1, torch_device="cpu",
            gpu_name=None, vram_mb=None, cuda_available=False,
        )
        assert get_torch_device(resolved) == "cpu"

    def test_gpu_semaphore_is_singleton(self) -> None:
        s1 = get_gpu_semaphore()
        s2 = get_gpu_semaphore()
        assert s1 is s2

    def test_gpu_semaphore_is_asyncio(self) -> None:
        import asyncio
        sem = get_gpu_semaphore()
        assert isinstance(sem, asyncio.Semaphore)


class TestResizeForInference:
    """Tests for image resize utility."""

    def test_small_image_not_resized(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        resized, scale = resize_for_inference(img, max_dim=1600)
        assert resized.shape == img.shape
        assert scale == 1.0

    def test_large_landscape_resized(self) -> None:
        img = np.zeros((1000, 3000, 3), dtype=np.uint8)
        resized, scale = resize_for_inference(img, max_dim=1600)
        assert resized.shape[1] == 1600
        assert resized.shape[0] == 533
        assert abs(scale - 3000 / 1600) < 0.01

    def test_large_portrait_resized(self) -> None:
        img = np.zeros((3000, 1000, 3), dtype=np.uint8)
        resized, scale = resize_for_inference(img, max_dim=1600)
        assert resized.shape[0] == 1600
        assert resized.shape[1] == 533
        assert abs(scale - 3000 / 1600) < 0.01

    def test_exact_max_dim_not_resized(self) -> None:
        img = np.zeros((1600, 1200, 3), dtype=np.uint8)
        resized, scale = resize_for_inference(img, max_dim=1600)
        assert resized.shape == img.shape
        assert scale == 1.0

    def test_scale_bbox_round_trip(self) -> None:
        bbox = (100.0, 200.0, 50.0, 60.0)
        scale = 2.0
        scaled = scale_bbox(bbox, scale)
        assert scaled == (200.0, 400.0, 100.0, 120.0)

    def test_scale_bbox_identity(self) -> None:
        bbox = (10.0, 20.0, 30.0, 40.0)
        scaled = scale_bbox(bbox, 1.0)
        assert scaled == bbox

    def test_scale_landmarks(self) -> None:
        landmarks = np.array([[100.0, 200.0], [300.0, 400.0]])
        scaled = scale_landmarks(landmarks, 2.0)
        np.testing.assert_array_equal(scaled, np.array([[200.0, 400.0], [600.0, 800.0]]))

    def test_scale_landmarks_identity(self) -> None:
        landmarks = np.array([[10.0, 20.0]])
        scaled = scale_landmarks(landmarks, 1.0)
        np.testing.assert_array_equal(scaled, landmarks)
