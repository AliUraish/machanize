import { expect, test } from "@playwright/test";

const episode = {
  episode_id: "episode-1",
  dataset_root: "/mock",
  dataset_episode_index: 0,
  project_name: "demo",
  robot_type: "so101",
  task: "Place the blue object in the glass.",
  outcome: "success",
  review_status: "approved",
  processing_status: "completed",
  frame_count: 120,
  camera_keys: ["observation.images.front", "observation.images.wrist"]
};

const draft = {
  template_version: 1,
  task_description: episode.task,
  ordered_task_stages: [{
    name: "carrying",
    description: "Carry the object toward the glass.",
    start_time_seconds: 0,
    end_time_seconds: 1,
    expected_object_relationships: ["object in gripper"],
    expected_robot_behavior: "Move toward glass.",
    expected_gripper_behavior: "Remain closed.",
    evidence: [{ timestamp_seconds: 0.5, description: "Object remains in gripper." }],
    confidence: 0.92,
    uncertainty: []
  }],
  success_conditions: ["Object is inside glass."],
  possible_failure_types: [{
    failure_type: "object dropped",
    description: "Object leaves the gripper.",
    related_stage_names: ["carrying"],
    detectable_evidence: ["Object moves away from gripper."]
  }],
  important_timestamps_and_evidence: [{ timestamp_seconds: 0.5, description: "Object carried." }],
  confidence: 0.92,
  uncertainty: [],
  source_episode: { episode_id: episode.episode_id, dataset_episode_index: 0, project_name: "demo" },
  model_version: "gemini-robotics-er-1.6-preview",
  video_fps: 5,
  approval_status: "draft",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  approved_at: null
};

test("analyzes, requires approval, and starts monitoring in OFF", async ({ page }) => {
  let approved = false;
  let runtimeMode = "off";
  let actState = "ready";
  await page.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const json = (body: unknown, status = 200) => route.fulfill({
      body: JSON.stringify(body),
      contentType: "application/json",
      status
    });

    if (path === "/api/health") return json({
      status: "ok", phase: 3, mode: "training", robot_movement_enabled: false,
      classes: ["blue_object", "glass", "gripper"],
      task_analysis_model: "gemini-robotics-er-1.6-preview", task_analysis_fps: 5,
      task_template_approval: "manual"
    });
    if (path === "/api/episodes") return json([episode]);
    if (path === "/api/models") return json([]);
    if (path.includes("/frames")) return json([]);
    if (path === `/api/analysis/templates/${episode.episode_id}` && request.method() === "GET") {
      return approved ? json({ ...draft, approval_status: "approved", approved_at: "2026-01-01T00:01:00Z" }) : json({ detail: "Task analysis not found." }, 404);
    }
    if (path === "/api/analysis/start") return json({
      job_id: "job-1", episode_id: episode.episode_id, status: "queued",
      created_at: "2026-01-01T00:00:00Z"
    });
    if (path === "/api/analysis/jobs/job-1") return json({
      job_id: "job-1", episode_id: episode.episode_id, status: "completed",
      created_at: "2026-01-01T00:00:00Z", completed_at: "2026-01-01T00:00:05Z", result: draft
    });
    if (path === `/api/analysis/templates/${episode.episode_id}` && request.method() === "PUT") {
      return json(draft);
    }
    if (path.endsWith("/approve")) {
      approved = true;
      return json({ ...draft, approval_status: "approved", approved_at: "2026-01-01T00:01:00Z" });
    }
    if (path === "/health") return json({
      status: "ok", service_role: "pi_runtime", training_backend_controls_robot: false,
      model_id: "gemini-3.1-flash-live-preview", api_key_configured: true,
      active_enabled: false, default_mode: "off", monitor_sample_fps: 5, control_fps: 30,
      cloud_timeout_seconds: 4, stop_threshold: 0.9, consecutive_stop_predictions: 3,
      robot_configured: true, robot_connected: true, control_loop_status: actState,
      act_state: actState,
      reset_auth_configured: true, stop_latched: false
    });
    const session = {
      session_id: "session-1", template_episode_id: episode.episode_id,
      template_version: 1,
      template_revision: "revision", model_id: "gemini-3.1-flash-live-preview",
      mode: runtimeMode, connection_state: runtimeMode === "monitor" ? "connected" : "off",
      created_at: "2026-01-01T00:01:00Z", stop_latched: false, thresholds: {}
    };
    if (path === "/api/runtime/sessions" && request.method() === "POST") return json(session);
    if (path.endsWith("/mode")) {
      runtimeMode = ((await request.postDataJSON()) as { mode: string }).mode;
      return json({ ...session, mode: runtimeMode, connection_state: "connected" });
    }
    if (path === "/api/runtime/control/start") {
      actState = "running";
      return json({ act_state: actState, robot_connected: true, stop_latched: false });
    }
    if (path === "/api/runtime/control/stop") {
      actState = "stopped";
      return json({
        act_state: actState,
        robot_connected: true,
        stop_latched: false,
        block_reason: "ACT is stopped"
      });
    }
    if (path === "/api/runtime/sessions/session-1") return json({
      ...session, mode: runtimeMode, connection_state: runtimeMode === "monitor" ? "connected" : "off"
    });
    if (path.endsWith("/decisions")) return json(runtimeMode === "monitor" ? [{
      decision_id: "decision-1", session_id: "session-1", sample_id: 1,
      sample_timestamp: "2026-01-01T00:01:01Z", received_at: "2026-01-01T00:01:01Z",
      model_id: "gemini-3.1-flash-live-preview", mode: "monitor",
      report: { current_stage: "carrying", progress: 0.5, correct: true,
        failure_type: null, confidence: 0.91,
        evidence: [{ timestamp: "2026-01-01T00:01:01Z", description: "Object remains in gripper." }],
        recommend_stop: false },
      latency_ms: 84, validation_status: "valid", stop_streak: 0, local_result: "continue"
    }] : []);
    return route.continue();
  });

  await page.goto("/");
  await page.getByRole("button", { name: new RegExp(episode.task) }).click();
  await page.getByRole("button", { name: "Analyze selected success" }).click();
  await expect(page.getByText("Analysis: completed")).toBeVisible({ timeout: 4_000 });
  await expect(page.getByRole("button", { name: /Start Live monitoring/ })).toBeDisabled();

  await page.getByRole("button", { name: "Approve template" }).click();
  await expect(page.getByText("approved", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: /Start Live monitoring/ }).click();
  await expect(page.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: "ACTIVE" })).toBeDisabled();

  await page.getByRole("button", { name: "MONITOR", exact: true }).click();
  await expect(page.locator(".runtime-readout").getByText("carrying", { exact: true })).toBeVisible({
    timeout: 4_000
  });
  await expect(page.getByText("84 ms")).toBeVisible();
  await expect(page.locator(".runtime-evidence").getByText(/Object remains in gripper/)).toBeVisible();
  await page.getByRole("button", { name: "Start ACT" }).click();
  await expect(page.getByText("RUNNING", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Stop ACT" }).click();
  await expect(page.getByText("STOPPED", { exact: true })).toBeVisible();
});
