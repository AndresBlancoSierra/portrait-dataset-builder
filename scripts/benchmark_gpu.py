"""GPU vs CPU Benchmark for portrait-dataset-builder pipeline stages.

Usage:
    uv run python scripts/benchmark_gpu.py [--images N] [--stages STAGE,...] [--device DEVICE]

Examples:
    uv run python scripts/benchmark_gpu.py                    # benchmark all stages, 10 images
    uv run python scripts/benchmark_gpu.py --images 20        # benchmark all stages, 20 images
    uv run python scripts/benchmark_gpu.py --stages face_detection,semantic_filter
    uv run python scripts/benchmark_gpu.py --device cpu        # force CPU only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from portrait_dataset_builder.compute import resolve_device, log_device_info


def find_test_images(data_dir: Path, n: int) -> list[Path]:
    """Find up to N test images from the data directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    for ext in extensions:
        images.extend(data_dir.rglob(f"*{ext}"))
    return sorted(images)[:n]


def benchmark_face_detection(images: list[Path], device: str) -> dict:
    """Benchmark InsightFace face detection on CPU vs GPU."""
    from portrait_dataset_builder.compute import get_ctx_id, get_onnx_providers

    resolved = resolve_device(device)
    providers = get_onnx_providers(resolved)
    ctx_id = get_ctx_id(resolved)

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    times = []
    face_counts = []
    max_dim = 1600

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            if w >= h:
                new_w = max_dim
                new_h = int(h * max_dim / w)
            else:
                new_h = max_dim
                new_w = int(w * max_dim / h)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale = h / new_h

        start = time.perf_counter()
        faces = app.get(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        face_counts.append(len(faces))

    return {
        "stage": "face_detection",
        "device": resolved.actual,
        "gpu_name": resolved.gpu_name,
        "images": len(times),
        "total_time_s": sum(times),
        "avg_time_ms": (sum(times) / len(times)) * 1000 if times else 0,
        "min_time_ms": min(times) * 1000 if times else 0,
        "max_time_ms": max(times) * 1000 if times else 0,
        "images_per_sec": len(times) / sum(times) if sum(times) > 0 else 0,
        "total_faces": sum(face_counts),
    }


def benchmark_clip_inference(images: list[Path], device: str) -> dict:
    """Benchmark CLIP image encoding (semantic filter core)."""
    import torch
    import open_clip
    from PIL import Image as PILImage

    resolved = resolve_device(device)
    torch_device = torch.device(resolved.torch_device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(torch_device)
    model.eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompts = ["a photo of a single person's face", "a landscape photo"]
    text_tokens = tokenizer(prompts).to(torch_device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    times = []
    for img_path in images:
        try:
            pil_img = PILImage.open(img_path).convert("RGB")
            w, h = pil_img.size
            if max(w, h) > 1600:
                scale = 1600 / max(w, h)
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)), PILImage.LANCZOS
                )

            img_tensor = preprocess(pil_img).unsqueeze(0).to(torch_device)

            start = time.perf_counter()
            with torch.no_grad():
                img_features = model.encode_image(img_tensor)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                sims = (img_features @ text_features.T).softmax(dim=-1)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception:
            continue

    return {
        "stage": "clip_inference",
        "device": resolved.actual,
        "gpu_name": resolved.gpu_name,
        "images": len(times),
        "total_time_s": sum(times),
        "avg_time_ms": (sum(times) / len(times)) * 1000 if times else 0,
        "min_time_ms": min(times) * 1000 if times else 0,
        "max_time_ms": max(times) * 1000 if times else 0,
        "images_per_sec": len(times) / sum(times) if sum(times) > 0 else 0,
    }


def benchmark_nsfw_inference(images: list[Path], device: str) -> dict:
    """Benchmark NSFW classification."""
    import torch
    from transformers import pipeline as hf_pipeline
    from PIL import Image as PILImage

    resolved = resolve_device(device)
    torch_device = 0 if resolved.actual == "cuda" else -1

    nsfw_pipeline = hf_pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=torch_device,
    )

    times = []
    for img_path in images:
        try:
            pil_img = PILImage.open(img_path).convert("RGB")
            w, h = pil_img.size
            if max(w, h) > 1600:
                scale = 1600 / max(w, h)
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)), PILImage.LANCZOS
                )

            start = time.perf_counter()
            _ = nsfw_pipeline(pil_img, top_k=None)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception:
            continue

    return {
        "stage": "nsfw_classification",
        "device": resolved.actual,
        "gpu_name": resolved.gpu_name,
        "images": len(times),
        "total_time_s": sum(times),
        "avg_time_ms": (sum(times) / len(times)) * 1000 if times else 0,
        "min_time_ms": min(times) * 1000 if times else 0,
        "max_time_ms": max(times) * 1000 if times else 0,
        "images_per_sec": len(times) / sum(times) if sum(times) > 0 else 0,
    }


STAGE_BENCHMARKS = {
    "face_detection": benchmark_face_detection,
    "semantic_filter": benchmark_clip_inference,
    "safety_gate": benchmark_nsfw_inference,
}


def print_results(results: list[dict]) -> None:
    """Print benchmark results in a table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\n--- {r['stage']} ({r['device']}) ---")
        if r.get("gpu_name"):
            print(f"  GPU: {r['gpu_name']}")
        print(f"  Images: {r['images']}")
        print(f"  Total time: {r['total_time_s']:.2f}s")
        print(f"  Avg per image: {r['avg_time_ms']:.1f}ms")
        print(f"  Min: {r['min_time_ms']:.1f}ms | Max: {r['max_time_ms']:.1f}ms")
        print(f"  Throughput: {r['images_per_sec']:.2f} images/sec")
        if "total_faces" in r:
            print(f"  Faces detected: {r['total_faces']}")

    if len(results) >= 2:
        cpu_results = {r["stage"]: r for r in results if r["device"] == "cpu"}
        gpu_results = {r["stage"]: r for r in results if r["device"] == "cuda"}

        print("\n" + "=" * 70)
        print("COMPARISON (CPU vs GPU)")
        print("=" * 70)

        for stage in cpu_results:
            if stage in gpu_results:
                cpu_r = cpu_results[stage]
                gpu_r = gpu_results[stage]
                speedup = cpu_r["avg_time_ms"] / gpu_r["avg_time_ms"] if gpu_r["avg_time_ms"] > 0 else 0
                print(f"\n  {stage}:")
                print(f"    CPU: {cpu_r['avg_time_ms']:.1f}ms/img ({cpu_r['images_per_sec']:.2f} img/s)")
                print(f"    GPU: {gpu_r['avg_time_ms']:.1f}ms/img ({gpu_r['images_per_sec']:.2f} img/s)")
                print(f"    Speedup: {speedup:.1f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU pipeline stages")
    parser.add_argument("--images", type=int, default=10, help="Number of test images")
    parser.add_argument("--stages", type=str, default="all", help="Comma-separated stages or 'all'")
    parser.add_argument("--device", type=str, default="all", help="'all', 'cpu', or 'cuda'")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    images = find_test_images(data_dir, args.images)

    if not images:
        print("No test images found in data/")
        return

    print(f"Found {len(images)} test images")
    print(f"First image: {images[0].name}")

    stages = list(STAGE_BENCHMARKS.keys()) if args.stages == "all" else args.stages.split(",")
    devices = ["cpu", "cuda"] if args.device == "all" else [args.device]

    results = []
    for stage in stages:
        if stage not in STAGE_BENCHMARKS:
            print(f"Unknown stage: {stage}")
            continue
        for device in devices:
            print(f"\nBenchmarking {stage} on {device}...")
            try:
                result = STAGE_BENCHMARKS[stage](images, device)
                results.append(result)
                print(f"  Done: {result['avg_time_ms']:.1f}ms/img")
            except Exception as e:
                print(f"  Failed: {e}")

    print_results(results)


if __name__ == "__main__":
    main()
