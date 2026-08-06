export interface ImageRecord {
  id: number;
  content_hash: string;
  uri: string;
  local_path: string;
  width: number;
  height: number;
  file_size: number;
  source_provider: string;
  pipeline_state: string;
  created_at: string;
}

export interface FaceRecord {
  id: number;
  image_id: number;
  bbox_x: number;
  bbox_y: number;
  bbox_w: number;
  bbox_h: number;
  yaw: number;
  pitch: number;
  roll: number;
  confidence: number;
  face_width: number;
  face_height: number;
}

export interface QualityScore {
  id: number;
  image_id: number;
  resolution_score: number;
  sharpness_score: number;
  blur_score: number;
  noise_score: number;
  lighting_score: number;
  occlusion_score: number;
  face_size_score: number;
  frontal_score: number;
  jpeg_score: number;
  final_score: number;
}

export interface Classification {
  id: number;
  image_id: number;
  angle: string | null;
  horizontal_pose: string | null;
  vertical_pose: string | null;
  expression: string | null;
  accessories: Record<string, boolean> | null;
  age_group: string | null;
  lighting: string | null;
}

export interface SafetyScore {
  is_nsfw: boolean;
  nsfw_score: number;
  is_ai_generated: boolean;
  ai_probability: number;
  real_photo_score: number;
  source_trust_score: number;
  rejection_reason: string | null;
}

export type LibraryStatus = 'queued' | 'building' | 'ready' | 'empty' | 'failed' | 'cancelled' | 'unknown' | 'identity_unverified';

export interface BuildJobInfo {
  id?: number;
  status: string;
  queue_status?: string;
  queue_position?: number | null;
  current_stage: string | null;
  stage_label: string | null;
  items_processed: number;
  items_total: number;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string | null;
}

export interface Library {
  name: string;
  image_count: number;
  quality_score: number;
  coverage_score: number;
  updated_at: string;
  thumbnail_hash: string | null;
  status: LibraryStatus;
  build: BuildJobInfo;
}

export interface PipelineProgress {
  id: number;
  status: string;
  library_status: LibraryStatus;
  current_stage: string | null;
  stage_label: string | null;
  items_processed: number;
  items_total: number;
  error: string | null;
  elapsed_ms: number;
  started_at: string | null;
  completed_at: string | null;
  stages_completed: string[];
}

export interface ImageWithMetadata extends ImageRecord {
  face: FaceRecord | null;
  quality: QualityScore | null;
  classification: Classification | null;
  safety: SafetyScore | null;
}

export interface CoverageData {
  yaw_bins: number[];
  pitch_bins: number[];
  heatmap: number[][];
  expressions: Record<string, number>;
  lighting: Record<string, number>;
  age_groups: Record<string, number>;
  horizontal_poses: Record<string, number>;
  vertical_poses: Record<string, number>;
}

export interface Stats {
  total_images: number;
  verified_images: number;
  avg_quality: number;
  avg_yaw: number;
  expressions: Record<string, number>;
  angles: Record<string, number>;
  horizontal_poses: Record<string, number>;
  vertical_poses: Record<string, number>;
}

export interface BuildQueueStatus {
  jobs: QueueJob[];
  active_job: number | null;
  queue_paused: boolean;
  max_concurrent: number;
}

export interface QueueJob {
  id: number;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  position: number | null;
  stage?: string;
  stage_label?: string;
  processed?: number;
  total?: number;
  error?: string;
}

export interface BatchEnqueueResult {
  added: string[];
  already_exists: Array<{ name: string; status: string }>;
  queued: number;
}

export interface GlobalImage extends ImageWithMetadata {
  library_name: string;
}

export interface Book {
  title: string;
  slug: string;
  category: string;
  author?: string;
  page_count: number;
}

export interface BookPage {
  slug: string;
  title: string;
  page_number: number;
}
