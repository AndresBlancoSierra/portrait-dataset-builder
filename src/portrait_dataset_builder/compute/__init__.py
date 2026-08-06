"""GPU detection, device resolution, and compute utilities."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from portrait_dataset_builder.logging import get_logger

logger = get_logger("compute")

_gpu_semaphore: asyncio.Semaphore | None = None
_gpu_semaphore_lock = threading.Lock()


@dataclass(frozen=True)
class ResolvedDevice:
    """Resolved compute device with all details."""

    requested: str
    actual: str  # "cuda" or "cpu"
    onnx_providers: list[str]
    ctx_id: int
    torch_device: str
    gpu_name: str | None
    vram_mb: int | None
    cuda_available: bool


def _check_cuda_onnx() -> bool:
    """Check if CUDAExecutionProvider is available in ONNX Runtime."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        return "CUDAExecutionProvider" in providers
    except Exception:
        return False


def _check_cuda_torch() -> tuple[bool, str | None, int | None]:
    """Check if CUDA is available via PyTorch. Returns (available, gpu_name, vram_mb)."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, "total_memory", getattr(props, "total_mem", 0)) // (1024 * 1024)
            return True, name, vram
    except Exception:
        pass
    return False, None, None


def _get_gpu_name_nvidia_smi() -> str | None:
    """Fallback GPU name detection via nvidia-smi."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def resolve_device(requested: str) -> ResolvedDevice:
    """Resolve device string to actual compute configuration.

    Resolution logic:
      "auto" -> check CUDA availability -> cuda if available, else cpu
      "cuda" -> try CUDA, fallback to cpu with warning
      "cpu"  -> always CPU
    """
    cuda_onnx = _check_cuda_onnx()
    cuda_torch, gpu_name, vram = _check_cuda_torch()

    if not gpu_name:
        gpu_name = _get_gpu_name_nvidia_smi()

    if requested == "auto":
        if cuda_onnx and cuda_torch:
            actual = "cuda"
        elif cuda_torch:
            actual = "cuda"
            logger.warning(
                "torch CUDA available but ONNX Runtime lacks CUDA provider. "
                "InsightFace will use CPU. Install onnxruntime-gpu for full GPU support."
            )
        else:
            actual = "cpu"
    elif requested == "cuda":
        if cuda_onnx or cuda_torch:
            actual = "cuda"
        else:
            logger.error("CUDA requested but not available. Falling back to CPU.")
            actual = "cpu"
    elif requested == "cpu":
        actual = "cpu"
    else:
        logger.warning("Unknown device '{}', falling back to CPU", requested)
        actual = "cpu"

    if actual == "cuda":
        onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
        torch_device = "cuda"
    else:
        onnx_providers = ["CPUExecutionProvider"]
        ctx_id = -1
        torch_device = "cpu"

    resolved = ResolvedDevice(
        requested=requested,
        actual=actual,
        onnx_providers=onnx_providers,
        ctx_id=ctx_id,
        torch_device=torch_device,
        gpu_name=gpu_name if actual == "cuda" else None,
        vram_mb=vram,
        cuda_available=cuda_onnx or cuda_torch,
    )

    return resolved


def log_device_info(resolved: ResolvedDevice) -> None:
    """Log device configuration in a readable format."""
    if resolved.actual == "cuda":
        logger.info(
            "Device: {} | Provider: {} | Fallback: {} | VRAM: {}MB",
            resolved.gpu_name or "GPU",
            resolved.onnx_providers[0],
            resolved.onnx_providers[-1],
            resolved.vram_mb or "unknown",
        )
    else:
        reason = ""
        if resolved.cuda_available:
            reason = " (CUDA hardware detected but ONNX provider unavailable)"
        logger.info(
            "Device: CPU | Provider: CPUExecutionProvider{}",
            reason,
        )


def get_onnx_providers(resolved: ResolvedDevice) -> list[str]:
    """Return ONNX Runtime providers for the resolved device."""
    return list(resolved.onnx_providers)


def get_ctx_id(resolved: ResolvedDevice) -> int:
    """Return InsightFace context ID (0 for GPU, -1 for CPU)."""
    return resolved.ctx_id


def get_torch_device(resolved: ResolvedDevice) -> str:
    """Return torch device string."""
    return resolved.torch_device


def get_gpu_semaphore() -> asyncio.Semaphore:
    """Return or create the global GPU semaphore (max 1 concurrent GPU build)."""
    global _gpu_semaphore
    if _gpu_semaphore is None:
        with _gpu_semaphore_lock:
            if _gpu_semaphore is None:
                _gpu_semaphore = asyncio.Semaphore(1)
    return _gpu_semaphore
