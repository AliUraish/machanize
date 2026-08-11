import type {
  Annotation,
  Box,
  Episode,
  Frame,
  Health,
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
          boxes,
          approved,
          source,
          model_id: modelId
        })
      }
    ),
  startTraining: (epochs: number, device: string | null) =>
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
        body: JSON.stringify({ base_model: "yolo26n.pt", epochs, image_size: 640, device })
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
