/**
 * Shared taxonomy for WHO? Portrait Dataset Builder.
 *
 * Single source of truth for all classification labels used across
 * frontend, API, and backend.
 */

export type HorizontalPose =
  | 'frontal'
  | 'three_quarter_left'
  | 'three_quarter_right'
  | 'profile_left'
  | 'profile_right';

export type VerticalPose = 'neutral' | 'looking_up' | 'looking_down';

export type ExpressionLabel = 'neutral' | 'smile' | 'laugh' | 'speaking';

export type LightingLabel = 'dark' | 'bright' | 'balanced';

export type QualityLevel = 'high' | 'medium' | 'low';

// UI grouping: user-facing filter key → backend values
export const HORIZONTAL_POSE_GROUPS: Record<string, HorizontalPose[]> = {
  frontal: ['frontal'],
  quarter: ['three_quarter_left', 'three_quarter_right'],
  profile: ['profile_left', 'profile_right'],
};

// User-friendly display labels
export const HORIZONTAL_POSE_LABELS: Record<HorizontalPose, string> = {
  frontal: 'Frontal',
  three_quarter_left: '\u00be Left',
  three_quarter_right: '\u00be Right',
  profile_left: 'Profile Left',
  profile_right: 'Profile Right',
};

export const VERTICAL_POSE_LABELS: Record<VerticalPose, string> = {
  neutral: 'Neutral',
  looking_up: 'Looking Up',
  looking_down: 'Looking Down',
};

export const EXPRESSION_LABELS: Record<ExpressionLabel, string> = {
  neutral: 'Neutral',
  smile: 'Smile',
  laugh: 'Laugh',
  speaking: 'Speaking',
};

export const LIGHTING_LABELS: Record<LightingLabel, string> = {
  dark: 'Dark',
  bright: 'Bright',
  balanced: 'Balanced',
};

export const QUALITY_LABELS: Record<QualityLevel, string> = {
  high: 'High (80+)',
  medium: 'Medium (50-80)',
  low: 'Low (<50)',
};

export const QUALITY_RANGES: Record<QualityLevel, [number, number]> = {
  high: [0.80, 1.0],
  medium: [0.50, 0.79],
  low: [0.0, 0.49],
};
