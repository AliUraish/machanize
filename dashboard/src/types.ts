export type Health = {
  status: string;
  phase: number;
  mode: string;
  robot_movement_enabled: false;
  classes: string[];
};

export type Episode = {
  episode_id: string;
  dataset_root: string;
  dataset_episode_index: number;
  project_name: string;
  robot_type: string;
  task: string;
  outcome: string;
  review_status: string;
  processing_status: string;
  frame_count: number | null;
  camera_keys: string[];
};

export type Frame = {
  episode_id: string;
  camera_key: string;
  frame_id: string;
  frame_index: number;
  timestamp: number;
  image_path: string;
  image_url: string;
  width: number;
  height: number;
};

export type SynchronizedFrames = {
  camera_keys: string[];
  synchronized_frame_count: number;
  frames: Record<string, Frame[]>;
};

export type Box = {
  class_name: string;
  x_center: number;
  y_center: number;
  width: number;
  height: number;
  confidence?: number | null;
};

export type Annotation = {
  episode_id: string;
  camera_key: string;
  frame_id: string;
  image_path?: string;
  boxes: Box[];
  approved: boolean;
  source: "manual" | "prediction";
  model_id?: string | null;
};

export type YoloModel = {
  model_id: string;
  model_path: string;
  base_model: string;
  created_at: string;
  metrics: Record<string, unknown>;
};

export type TrainingJob = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;
  completed_at?: string | null;
  model?: YoloModel | null;
  error?: string | null;
};
