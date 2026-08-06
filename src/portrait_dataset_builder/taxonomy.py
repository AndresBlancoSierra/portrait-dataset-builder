"""Shared taxonomy for WHO? Portrait Dataset Builder.

Single source of truth for all classification labels used across
backend, API, frontend, and tests.
"""
from __future__ import annotations

# ── Horizontal pose (yaw-based) ──────────────────────────────────────────────

class HorizontalPose:
    FRONTAL = "frontal"
    QUARTER_LEFT = "three_quarter_left"
    QUARTER_RIGHT = "three_quarter_right"
    PROFILE_LEFT = "profile_left"
    PROFILE_RIGHT = "profile_right"

    ALL = [FRONTAL, QUARTER_LEFT, QUARTER_RIGHT, PROFILE_LEFT, PROFILE_RIGHT]

    # UI grouping: user-facing label → list of backend values
    GROUPS: dict[str, list[str]] = {
        "frontal": [FRONTAL],
        "quarter": [QUARTER_LEFT, QUARTER_RIGHT],
        "profile": [PROFILE_LEFT, PROFILE_RIGHT],
    }

    # User-friendly display labels
    LABELS: dict[str, str] = {
        FRONTAL: "Frontal",
        QUARTER_LEFT: "\u00be Left",
        QUARTER_RIGHT: "\u00be Right",
        PROFILE_LEFT: "Profile Left",
        PROFILE_RIGHT: "Profile Right",
    }


# ── Vertical pose (pitch-based) ──────────────────────────────────────────────

class VerticalPose:
    NEUTRAL = "neutral"
    LOOKING_UP = "looking_up"
    LOOKING_DOWN = "looking_down"

    ALL = [NEUTRAL, LOOKING_UP, LOOKING_DOWN]

    LABELS: dict[str, str] = {
        NEUTRAL: "Neutral",
        LOOKING_UP: "Looking Up",
        LOOKING_DOWN: "Looking Down",
    }


# ── Expression ───────────────────────────────────────────────────────────────

class ExpressionLabel:
    NEUTRAL = "neutral"
    SMILE = "smile"
    LAUGH = "laugh"
    SPEAKING = "speaking"

    ALL = [NEUTRAL, SMILE, LAUGH, SPEAKING]

    LABELS: dict[str, str] = {
        NEUTRAL: "Neutral",
        SMILE: "Smile",
        LAUGH: "Laugh",
        SPEAKING: "Speaking",
    }


# ── Lighting ─────────────────────────────────────────────────────────────────

class LightingLabel:
    DARK = "dark"
    BRIGHT = "bright"
    BALANCED = "balanced"

    ALL = [DARK, BRIGHT, BALANCED]

    LABELS: dict[str, str] = {
        DARK: "Dark",
        BRIGHT: "Bright",
        BALANCED: "Balanced",
    }


# ── Quality ──────────────────────────────────────────────────────────────────

class QualityLevel:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    ALL = [HIGH, MEDIUM, LOW]

    RANGES: dict[str, tuple[float, float]] = {
        HIGH: (0.80, 1.0),
        MEDIUM: (0.50, 0.79),
        LOW: (0.0, 0.49),
    }

    LABELS: dict[str, str] = {
        HIGH: "High (80+)",
        MEDIUM: "Medium (50-80)",
        LOW: "Low (<50)",
    }
