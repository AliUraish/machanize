import type {
  RuntimeControlStatus,
  RuntimeDecision,
  RuntimeHealth,
  RuntimeMode,
  RuntimeSession,
  TaskTemplate
} from "./types";

const runtimeBaseUrl = (import.meta.env.VITE_RUNTIME_BASE_URL || "http://127.0.0.1:8001")
  .replace(/\/$/, "");

async function runtimeRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${runtimeBaseUrl}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers }
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const runtimeApi = {
  baseUrl: runtimeBaseUrl,
  health: () => runtimeRequest<RuntimeHealth>("/health"),
  createSession: (taskTemplate: TaskTemplate) =>
    runtimeRequest<RuntimeSession>("/api/runtime/sessions", {
      method: "POST",
      body: JSON.stringify({
        template_episode_id: taskTemplate.source_episode.episode_id,
        task_template: taskTemplate
      })
    }),
  session: (sessionId: string) =>
    runtimeRequest<RuntimeSession>(`/api/runtime/sessions/${sessionId}`),
  setMode: (sessionId: string, mode: RuntimeMode, confirmActive: boolean) =>
    runtimeRequest<RuntimeSession>(`/api/runtime/sessions/${sessionId}/mode`, {
      method: "PUT",
      body: JSON.stringify({ mode, confirm_active: confirmActive })
    }),
  stop: (sessionId: string) =>
    runtimeRequest<RuntimeSession>(`/api/runtime/sessions/${sessionId}/stop`, {
      method: "POST"
    }),
  startAct: () =>
    runtimeRequest<RuntimeControlStatus>("/api/runtime/control/start", {
      method: "POST",
      body: JSON.stringify({ confirm: true })
    }),
  stopAct: () =>
    runtimeRequest<RuntimeControlStatus>("/api/runtime/control/stop", {
      method: "POST",
      body: JSON.stringify({ confirm: true })
    }),
  decisions: (sessionId: string) =>
    runtimeRequest<RuntimeDecision[]>(`/api/runtime/sessions/${sessionId}/decisions`),
  resetStopLatch: (resetToken: string) =>
    runtimeRequest<RuntimeSession | { status: string; stop_latched: boolean }>(
      "/stop-latch/reset",
      {
        method: "POST",
        headers: { "X-Machanize-Reset-Token": resetToken },
        body: JSON.stringify({ confirm: true })
      }
    ),
  combinedStreamUrl: () => `${runtimeBaseUrl}/stream/combined.mjpeg`,
  runtimeWebSocketUrl: () => {
    const url = new URL(runtimeBaseUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = "/ws/runtime";
    return url.toString();
  }
};
