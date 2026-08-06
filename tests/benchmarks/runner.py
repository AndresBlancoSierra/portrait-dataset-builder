"""Benchmark runner for comparing pipeline versions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
BASELINES_DIR = BENCHMARK_DIR / "baselines"
BASELINES_DIR.mkdir(exist_ok=True)


def save_baseline(results: dict[str, float], name: str) -> Path:
    """Save benchmark results as a named baseline."""
    baseline = {
        "name": name,
        "timestamp": datetime.now(UTC).isoformat(),
        "results": results,
    }
    path = BASELINES_DIR / f"{name}.json"
    path.write_text(json.dumps(baseline, indent=2))
    return path


def load_baseline(name: str) -> dict:
    """Load a baseline by name."""
    path = BASELINES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {name}")
    return json.loads(path.read_text())


def compare_baselines(baseline_a: str, baseline_b: str) -> dict:
    """Compare two baselines and return delta report."""
    a = load_baseline(baseline_a)
    b = load_baseline(baseline_b)

    comparison = {
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "timestamp_a": a["timestamp"],
        "timestamp_b": b["timestamp"],
        "metrics": {},
    }

    all_metrics = set(list(a["results"].keys()) + list(b["results"].keys()))

    for metric in all_metrics:
        val_a = a["results"].get(metric, 0.0)
        val_b = b["results"].get(metric, 0.0)
        delta = val_b - val_a

        is_error_metric = "error" in metric.lower() or "rate" in metric.lower()
        improved = delta < 0 if is_error_metric else delta > 0

        comparison["metrics"][metric] = {
            "value_a": val_a,
            "value_b": val_b,
            "delta": delta,
            "improved": improved,
        }

    return comparison


def format_comparison(comparison: dict) -> str:
    """Format comparison results for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("  Baseline Comparison")
    lines.append("=" * 60)
    lines.append(f"  A: {comparison['baseline_a']} ({comparison['timestamp_a']})")
    lines.append(f"  B: {comparison['baseline_b']} ({comparison['timestamp_b']})")
    lines.append("-" * 60)

    for metric, data in comparison["metrics"].items():
        arrow = "↑" if data["improved"] else "↓" if data["delta"] != 0 else "="
        lines.append(
            f"  {metric:30s} {data['value_a']:.3f} → {data['value_b']:.3f} "
            f"({data['delta']:+.3f}) {arrow}"
        )

    lines.append("-" * 60)
    improved = sum(1 for m in comparison["metrics"].values() if m["improved"])
    total = len(comparison["metrics"])
    lines.append(f"  {improved}/{total} metrics improved")
    lines.append("=" * 60)

    return "\n".join(lines)
