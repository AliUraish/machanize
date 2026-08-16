export type Health = {
  status: string;
  phase: number;
  mode: string;
  robot_movement_enabled: false;
  classes: string[];
  task_analysis_model: string;
  task_analysis_fps: number;
  task_template_approval: "manual";
};

export type TimestampEvidence = {
  timestamp_seconds: number;
  description: string;
};

export type TaskStage = {
  name: string;
  description: string;
  start_time_seconds: number;
  end_time_seconds: number;
  expected_object_relationships: string[];
  expected_robot_behavior: string;
  expected_gripper_behavior: string;
  evidence: TimestampEvidence[];
  confidence: number;
  uncertainty: string[];
};

export type PossibleFailure = {
  failure_type: string;
  description: string;
  related_stage_names: string[];
  detectable_evidence: string[];
};

export type TaskTemplateDraft = {
  task_description: string;
  ordered_task_stages: TaskStage[];
  success_conditions: string[];
  possible_failure_types: PossibleFailure[];
  important_timestamps_and_evidence: TimestampEvidence[];
  confidence: number;
  uncertainty: string[];
};

export type TaskTemplate = TaskTemplateDraft & {
  template_version: number;
  source_episode: {
    episode_id: string;
    dataset_episode_index: number;
    project_name: string;
  };
  model_version: string;
  video_fps: number;
  approval_status: "draft" | "approved";
  created_at: string;
  updated_at: string;
  approved_at?: string | null;
};

export type TaskAnalysisJob = {
  job_id: string;
  episode_id: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;
  completed_at?: string | null;
  result?: TaskTemplate | null;
  error?: string | null;
};

export type RuntimeMode = "off" | "monitor" | "active";
export type ACTState = "ready" | "running" | "stopped" | "error";
export type RuntimeConnectionState =
  | "off"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "error";

export type RuntimeHealth = {
  status: string;
  service_role: "pi_runtime";
  training_backend_controls_robot: false;
  model_id: string;
  api_key_configured: boolean;
  active_enabled: boolean;
  default_mode: "off";
  monitor_sample_fps: number;
  control_fps: number;
  cloud_timeout_seconds: number;
  stop_threshold: number;
  consecutive_stop_predictions: number;
  robot_configured: boolean;
  robot_connected: boolean;
  control_loop_status: string;
  act_state: ACTState;
  reset_auth_configured: boolean;
  stop_latched: boolean;
};

export type RuntimeSession = {
  session_id: string;
  template_episode_id: string;
  template_version: number;
  template_revision: string;
  model_id: string;
  mode: RuntimeMode;
  connection_state: RuntimeConnectionState;
  created_at: string;
  started_at?: string | null;
  stopped_at?: string | null;
  last_sample_at?: string | null;
  last_decision_at?: string | null;
  last_latency_ms?: number | null;
  stop_latched: boolean;
  stop_reason?: string | null;
  thresholds: Record<string, unknown>;
};

export type RuntimeTelemetry = {
  sequence: number;
  timestamp?: string | null;
  joints: Record<string, number>;
  proposed_actions: Record<string, number>;
  executed: boolean;
  block_reason?: string | null;
  robot_connected: boolean;
  control_loop_status: string;
  act_state: ACTState;
  control_loop_error?: string | null;
  current_stage: string;
  progress: number;
  risk: number;
  confidence: number;
  evidence: Array<{ timestamp: string; description: string }>;
  decision: "CONTINUE" | "STOP";
  recommend_stop: boolean;
  monitor_result: string;
  stop_latched: boolean;
  stop_reason?: string | null;
  provider_connection: RuntimeConnectionState;
  mode: RuntimeMode;
  latency_ms?: number | null;
};

export type RuntimeControlStatus = {
  act_state: ACTState;
  robot_connected: boolean;
  stop_latched: boolean;
  block_reason?: string | null;
};

export type RuntimeDecision = {
  decision_id: string;
  session_id: string;
  sample_id: number;
  sample_timestamp: string;
  received_at: string;
  model_id: string;
  mode: RuntimeMode;
  report: {
    current_stage: string;
    progress: number;
    correct: boolean;
    failure_type?: string | null;
    confidence: number;
    evidence: Array<{ timestamp: string; description: string }>;
    recommend_stop: boolean;
  };
  latency_ms: number;
  validation_status: "valid";
  stop_streak: number;
  local_result: "continue" | "alert" | "stop_requested";
  safety_reason?: string | null;
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
  suggested?: boolean;
  accepted?: boolean;
};

export type Annotation = {
  episode_id: string;
  camera_key: string;
  frame_id: string;
  image_path?: string;
  boxes: Box[];
  approved: boolean;
  source: "manual" | "prediction" | "grounding_dino";
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

export type LabelingJob = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;
  episode_ids: string[];
  confidence: number;
  total_frames: number;
  processed_frames: number;
  labeled_frames: number;
  total_boxes: number;
  current_episode_id?: string | null;
  current_camera_key?: string | null;
  completed_at?: string | null;
  errors: string[];
};
