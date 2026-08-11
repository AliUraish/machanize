import type {
  Annotation,
  Box,
  Episode,
  Frame,
  Health,
  LabelingJob,
  TaskAnalysisJob,
  TaskTemplate,
  TaskTemplateDraft,
  SynchronizedFrames,
  TrainingJob,
  YoloModel
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers }
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  episodes: () => request<Episode[]>("/api/episodes"),
  models: () => request<YoloModel[]>("/api/models"),
  startTaskAnalysis: (episodeId: string, confirmUnknownAsSuccess: boolean) =>
    request<TaskAnalysisJob>("/api/analysis/start", {
      method: "POST",
      body: JSON.stringify({
        episode_id: episodeId,
        confirm_unknown_as_success: confirmUnknownAsSuccess
      })
    }),
  taskAnalysisJob: (jobId: string) =>
    request<TaskAnalysisJob>(`/api/analysis/jobs/${jobId}`),
  taskTemplate: (episodeId: string) =>
    request<TaskTemplate>(`/api/analysis/templates/${episodeId}`),
  saveTaskTemplate: (episodeId: string, draft: TaskTemplateDraft) =>
    request<TaskTemplate>(`/api/analysis/templates/${episodeId}`, {
      method: "PUT",
      body: JSON.stringify(draft)
    }),
  approveTaskTemplate: (episodeId: string) =>
    request<TaskTemplate>(`/api/analysis/templates/${episodeId}/approve`, {
      method: "POST",
      body: JSON.stringify({ confirm: true })
    }),
  extract: (episodeId: string, cameraKey: string, stride: number) =>
    request<Frame[]>(`/api/episodes/${episodeId}/extract`, {
      method: "POST",
      body: JSON.stringify({ camera_key: cameraKey, stride })
    }),
  extractSynchronized: (episodeId: string, cameraKeys: string[], stride: number) =>
    request<SynchronizedFrames>(`/api/episodes/${episodeId}/extract-synchronized`, {
      method: "POST",
      body: JSON.stringify({ camera_keys: cameraKeys, stride })
    }),
  frames: (episodeId: string, cameraKey: string) =>
    request<Frame[]>(
      `/api/episodes/${episodeId}/frames?camera_key=${encodeURIComponent(cameraKey)}`
    ),
  annotation: (episodeId: string, frameId: string, cameraKey: string) =>
    request<Annotation>(
      `/api/episodes/${episodeId}/frames/${frameId}/annotation?camera_key=${encodeURIComponent(cameraKey)}`
    ),
  autoLabel: (episodeId: string, frame: Frame, confidence: number) =>
    request<Annotation>(
      `/api/episodes/${episodeId}/frames/${frame.frame_id}/auto-label`,
      {
        method: "POST",
        body: JSON.stringify({ camera_key: frame.camera_key, confidence })
      }
    ),
  saveAnnotation: (
    episodeId: string,
    frame: Frame,
    boxes: Box[],
    approved: boolean,
    source: Annotation["source"],
    modelId?: string
  ) =>
    request<{ status: string; approved: boolean }>(
      `/api/episodes/${episodeId}/frames/${frame.frame_id}/annotation`,
      {
        method: "PUT",
        body: JSON.stringify({
          camera_key: frame.camera_key,
          boxes: boxes.map(({ class_name, x_center, y_center, width, height, confidence }) => ({
            class_name,
            x_center,
            y_center,
            width,
            height,
            confidence
          })),
          approved,
          source,
          model_id: modelId
        })
      }
    ),
  startBatchLabeling: (episodeIds: string[], confidence: number) =>
    request<LabelingJob>("/api/labeling/start", {
      method: "POST",
      body: JSON.stringify({ episode_ids: episodeIds, confidence })
    }),
  labelingJob: (jobId: string) => request<LabelingJob>(`/api/labeling/${jobId}`),
  startTraining: (epochs: number, device: string | null, episodeIds: string[]) =>
    request<{
      job_id: string;
      status: string;
      training_images: number;
      validation_images: number;
      camera_images: Record<string, number>;
    }>(
      "/api/training/start",
      {
        method: "POST",
        body: JSON.stringify({
          base_model: "yolo26n.pt",
          epochs,
          image_size: 640,
          device,
          episode_ids: episodeIds
        })
      }
    ),
  trainingJob: (jobId: string) => request<TrainingJob>(`/api/training/${jobId}`),
  predict: (modelId: string, episodeId: string, cameraKey: string) =>
    request<{ status: string; frames: number }>(`/api/models/${modelId}/predict/${episodeId}`, {
      method: "POST",
      body: JSON.stringify({ camera_key: cameraKey, confidence: 0.25 })
    }),
  prediction: (modelId: string, episodeId: string, frameId: string, cameraKey: string) =>
    request<Annotation>(
      `/api/models/${modelId}/predictions/${episodeId}/${frameId}?camera_key=${encodeURIComponent(cameraKey)}`
    )
};
