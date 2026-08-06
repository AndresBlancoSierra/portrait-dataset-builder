import { describe, it, expect } from 'vitest';
import type { Library, LibraryStatus, PipelineProgress, Classification } from '../types';
import type { HorizontalPose, VerticalPose, ExpressionLabel, LightingLabel } from '../types/taxonomy';
import {
  HORIZONTAL_POSE_LABELS,
  VERTICAL_POSE_LABELS,
  EXPRESSION_LABELS,
  LIGHTING_LABELS,
  QUALITY_LABELS,
} from '../types/taxonomy';

describe('Type definitions', () => {
  it('LibraryStatus includes all expected values', () => {
    const statuses: LibraryStatus[] = ['building', 'ready', 'empty', 'failed', 'cancelled', 'unknown', 'identity_unverified'];
    expect(statuses).toHaveLength(7);
    expect(statuses).toContain('building');
    expect(statuses).toContain('ready');
    expect(statuses).toContain('empty');
    expect(statuses).toContain('failed');
    expect(statuses).toContain('cancelled');
    expect(statuses).toContain('unknown');
    expect(statuses).toContain('identity_unverified');
    // Ensure old 'ready_empty' is gone
    expect(statuses).not.toContain('ready_empty');
  });

  it('PipelineProgress has library_status field', () => {
    const progress: PipelineProgress = {
      id: 1,
      status: 'completed',
      library_status: 'ready',
      current_stage: null,
      stage_label: null,
      items_processed: 100,
      items_total: 100,
      error: null,
      elapsed_ms: 5000,
      started_at: null,
      completed_at: null,
      stages_completed: [],
    };
    expect(progress.library_status).toBe('ready');
  });

  it('Library has required fields', () => {
    const lib: Library = {
      name: 'test',
      image_count: 10,
      quality_score: 0.8,
      coverage_score: 0.6,
      updated_at: '',
      thumbnail_hash: null,
      status: 'ready',
      build: { id: 1, status: 'completed', current_stage: null, stage_label: null, items_processed: 0, items_total: 0, error: null, started_at: null, completed_at: null, created_at: null },
    };
    expect(lib.status).toBe('ready');
    expect(lib.quality_score).toBe(0.8);
    expect(lib.coverage_score).toBe(0.6);
  });

  it('Classification includes horizontal_pose and vertical_pose', () => {
    const cls: Classification = {
      id: 1,
      image_id: 1,
      angle: 'frontal',
      horizontal_pose: 'frontal',
      vertical_pose: 'neutral',
      expression: 'smile',
      accessories: null,
      age_group: 'adult',
      lighting: 'balanced',
    };
    expect(cls.horizontal_pose).toBe('frontal');
    expect(cls.vertical_pose).toBe('neutral');
  });
});

describe('Taxonomy constants', () => {
  it('HORIZONTAL_POSE_LABELS has 5 entries', () => {
    expect(Object.keys(HORIZONTAL_POSE_LABELS)).toHaveLength(5);
  });

  it('VERTICAL_POSE_LABELS has 3 entries', () => {
    expect(Object.keys(VERTICAL_POSE_LABELS)).toHaveLength(3);
  });

  it('EXPRESSION_LABELS has 4 entries', () => {
    expect(Object.keys(EXPRESSION_LABELS)).toHaveLength(4);
  });

  it('LIGHTING_LABELS has 3 entries', () => {
    expect(Object.keys(LIGHTING_LABELS)).toHaveLength(3);
  });

  it('QUALITY_LABELS has 3 entries', () => {
    expect(Object.keys(QUALITY_LABELS)).toHaveLength(3);
  });

  it('HorizontalPose labels are user-friendly', () => {
    expect(HORIZONTAL_POSE_LABELS.frontal).toBe('Frontal');
    expect(HORIZONTAL_POSE_LABELS.three_quarter_left).toBe('\u00be Left');
    expect(HORIZONTAL_POSE_LABELS.profile_right).toBe('Profile Right');
  });

  it('VerticalPose labels are user-friendly', () => {
    expect(VERTICAL_POSE_LABELS.looking_up).toBe('Looking Up');
    expect(VERTICAL_POSE_LABELS.looking_down).toBe('Looking Down');
  });
});
