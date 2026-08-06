"""Coverage analysis and dataset scoring module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portrait_dataset_builder.logging import get_logger

logger = get_logger("coverage")


@dataclass
class PoseCoverage:
    yaw_bins: int = 18
    pitch_bins: int = 6
    yaw_range: tuple[float, float] = (-90.0, 90.0)
    pitch_range: tuple[float, float] = (-30.0, 30.0)

    def compute(self, poses: list[tuple[float, float]]) -> dict:
        grid = np.zeros((self.pitch_bins, self.yaw_bins))

        yaw_step = (self.yaw_range[1] - self.yaw_range[0]) / self.yaw_bins
        pitch_step = (self.pitch_range[1] - self.pitch_range[0]) / self.pitch_bins

        for yaw, pitch in poses:
            col = int((yaw - self.yaw_range[0]) / yaw_step)
            row = int((pitch - self.pitch_range[0]) / pitch_step)
            col = max(0, min(self.yaw_bins - 1, col))
            row = max(0, min(self.pitch_bins - 1, row))
            grid[row, col] += 1

        total_bins = self.yaw_bins * self.pitch_bins
        populated_bins = int(np.sum(grid > 0))
        coverage_pct = populated_bins / total_bins

        gaps = []
        for r in range(self.pitch_bins):
            for c in range(self.yaw_bins):
                if grid[r, c] == 0:
                    yaw_center = self.yaw_range[0] + (c + 0.5) * yaw_step
                    pitch_center = self.pitch_range[0] + (r + 0.5) * pitch_step
                    gaps.append(
                        {
                            "yaw": round(yaw_center, 1),
                            "pitch": round(pitch_center, 1),
                        }
                    )

        return {
            "grid": grid.tolist(),
            "populated_bins": populated_bins,
            "total_bins": total_bins,
            "coverage_pct": round(coverage_pct, 3),
            "gaps": gaps,
            "total_images": len(poses),
        }

    def render_ascii(self, poses: list[tuple[float, float]]) -> str:
        result = self.compute(poses)
        grid = np.array(result["grid"])

        chars = " ░▒▓█"
        lines = []

        yaw_step = (self.yaw_range[1] - self.yaw_range[0]) / self.yaw_bins
        pitch_step = (self.pitch_range[1] - self.pitch_range[0]) / self.pitch_bins

        max_val = grid.max() if grid.max() > 0 else 1

        for r in range(self.pitch_bins - 1, -1, -1):
            pitch_center = self.pitch_range[0] + (r + 0.5) * pitch_step
            line = f"  {pitch_center:+6.1f}° │"
            for c in range(self.yaw_bins):
                val = grid[r, c]
                char_idx = min(4, int(val / max_val * 5)) if max_val > 0 else 0
                line += chars[char_idx]
            lines.append(line)

        lines.append("         └" + "─" * self.yaw_bins)
        labels = ""
        for c in range(0, self.yaw_bins + 1, 3):
            yaw_val = self.yaw_range[0] + c * yaw_step
            labels += f"{yaw_val:+.0f}"
            if c + 3 <= self.yaw_bins:
                padding = 3 - len(f"{yaw_val:+.0f}")
                labels += " " * max(0, padding)
        lines.append(f"          {labels}")

        lines.append("")
        lines.append(
            f"  Coverage: {result['populated_bins']}/{result['total_bins']} bins populated "
            f"({result['coverage_pct']:.0%})"
        )

        if result["gaps"]:
            lines.append(f"  Gaps: {len(result['gaps'])} empty bins")

        return "\n".join(lines)


@dataclass
class ExpressionDiversity:
    def compute(self, expressions: list[str]) -> dict:
        from collections import Counter

        counts = Counter(expressions)
        total = len(expressions)
        unique = len(counts)

        distribution = {}
        for expr, count in counts.most_common():
            distribution[expr] = {
                "count": count,
                "pct": round(count / total, 3) if total > 0 else 0,
            }

        if total > 0:
            probs = np.array([c / total for c in counts.values()])
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(unique) if unique > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0

        return {
            "unique_expressions": unique,
            "total_images": total,
            "distribution": distribution,
            "entropy": round(normalized_entropy, 3),
        }


@dataclass
class LightingDiversity:
    def compute(self, lighting_data: list[dict]) -> dict:
        if not lighting_data:
            return {
                "brightness_range": [0, 0],
                "contrast_range": [0, 0],
                "shadow_range": [0, 0],
                "diversity_score": 0.0,
            }

        brightness_vals = [d.get("brightness", 0) for d in lighting_data]
        contrast_vals = [d.get("contrast", 0) for d in lighting_data]
        shadow_vals = [d.get("shadow_ratio", 0) for d in lighting_data]

        brightness_range = [min(brightness_vals), max(brightness_vals)]
        contrast_range = [min(contrast_vals), max(contrast_vals)]
        shadow_range = [min(shadow_vals), max(shadow_vals)]

        brightness_spread = brightness_range[1] - brightness_range[0]
        contrast_spread = contrast_range[1] - contrast_range[0]
        shadow_spread = shadow_range[1] - shadow_range[0]

        diversity_score = (brightness_spread + contrast_spread + shadow_spread) / 3.0

        return {
            "brightness_range": [round(v, 3) for v in brightness_range],
            "contrast_range": [round(v, 3) for v in contrast_range],
            "shadow_range": [round(v, 3) for v in shadow_range],
            "diversity_score": round(min(1.0, diversity_score), 3),
        }


@dataclass
class TemporalDiversity:
    def compute_video(self, frames: list[dict]) -> dict:
        if not frames:
            return {
                "new_poses": 0,
                "diversity_ratio": 0.0,
                "total_frames": 0,
            }

        unique_poses = set()
        for frame in frames:
            yaw_bin = round(frame.get("yaw", 0) / 10) * 10
            pitch_bin = round(frame.get("pitch", 0) / 10) * 10
            unique_poses.add((yaw_bin, pitch_bin))

        return {
            "new_poses": len(unique_poses),
            "diversity_ratio": round(
                len(unique_poses) / max(1, len(frames)), 3
            ),
            "total_frames": len(frames),
        }


@dataclass
class DatasetScore:
    identity_purity: float = 0.0
    pose_diversity: float = 0.0
    expression_diversity: float = 0.0
    lighting_diversity: float = 0.0
    duplicate_rate: float = 0.0
    average_quality: float = 0.0
    overall: float = 0.0

    @classmethod
    def compute(cls, stats: dict) -> DatasetScore:
        overall = (
            stats.get("purity", 0.0) * 0.25
            + stats.get("pose_div", 0.0) * 0.20
            + stats.get("expr_div", 0.0) * 0.15
            + stats.get("light_div", 0.0) * 0.10
            + (1 - stats.get("dup_rate", 0.0)) * 0.10
            + stats.get("avg_quality", 0.0) * 0.20
        ) * 10

        return cls(
            identity_purity=stats.get("purity", 0.0),
            pose_diversity=stats.get("pose_div", 0.0),
            expression_diversity=stats.get("expr_div", 0.0),
            lighting_diversity=stats.get("light_div", 0.0),
            duplicate_rate=stats.get("dup_rate", 0.0),
            average_quality=stats.get("avg_quality", 0.0),
            overall=round(overall, 1),
        )

    def render(self) -> str:
        def bar(value: float, width: int = 20) -> str:
            filled = int(value * width)
            return "█" * filled + "░" * (width - filled)

        lines = [
            f"  Identity purity    {bar(self.identity_purity)}  {self.identity_purity:.0%}",
            f"  Pose diversity     {bar(self.pose_diversity)}  {self.pose_diversity:.0%}",
            f"  Expression div.    "
            f"{bar(self.expression_diversity)}  {self.expression_diversity:.0%}",
            f"  Lighting div.      {bar(self.lighting_diversity)}  {self.lighting_diversity:.0%}",
            f"  Duplicate rate     {bar(self.duplicate_rate)}  {self.duplicate_rate:.0%}",
            f"  Average quality    {bar(self.average_quality)}  {self.average_quality:.2f}",
            "",
            f"  {'━' * 42}",
            f"  OVERALL                          {self.overall:.1f} / 10",
        ]
        return "\n".join(lines)
