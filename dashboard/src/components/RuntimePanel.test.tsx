import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { RuntimePanel } from "./RuntimePanel";
import type { RuntimeHealth, RuntimeSession, TaskTemplate } from "../types";

const template: TaskTemplate = {
  template_version: 1,
  task_description: "Place the blue object in the glass.",
  ordered_task_stages: [
    {
      name: "carrying",
      description: "Carry the object.",
      start_time_seconds: 0,
      end_time_seconds: 1,
      expected_object_relationships: ["object in gripper"],
      expected_robot_behavior: "Move toward glass.",
      expected_gripper_behavior: "Remain closed.",
      evidence: [],
      confidence: 0.9,
      uncertainty: []
    }
  ],
  success_conditions: ["Object is in glass."],
  possible_failure_types: [],
  important_timestamps_and_evidence: [],
  confidence: 0.9,
  uncertainty: [],
  source_episode: { episode_id: "episode-1", dataset_episode_index: 0, project_name: "demo" },
  model_version: "gemini-robotics-er-1.6-preview",
  video_fps: 5,
  approval_status: "approved",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  approved_at: "2026-01-01T00:00:00Z"
};

const health: RuntimeHealth = {
  status: "ok",
  service_role: "pi_runtime",
  training_backend_controls_robot: false,
  model_id: "gemini-3.1-flash-live-preview",
  api_key_configured: true,
  active_enabled: false,
  default_mode: "off",
  monitor_sample_fps: 5,
  control_fps: 30,
  cloud_timeout_seconds: 4,
  stop_threshold: 0.9,
  consecutive_stop_predictions: 3,
  robot_configured: true,
  robot_connected: true,
  control_loop_status: "ready",
  act_state: "ready",
  reset_auth_configured: true,
  stop_latched: false
};

const session: RuntimeSession = {
  session_id: "session-1",
  template_episode_id: "episode-1",
  template_version: 1,
  template_revision: "revision",
  model_id: health.model_id,
  mode: "monitor",
  connection_state: "connected",
  created_at: "2026-01-01T00:00:00Z",
  stop_latched: false,
  thresholds: {}
};

describe("RuntimePanel", () => {
  it("keeps ACTIVE disabled and exposes monitoring state", () => {
    const onModeChange = vi.fn().mockResolvedValue(undefined);
    const onStartAct = vi.fn().mockResolvedValue(undefined);
    render(
      <RuntimePanel
        decisions={[]}
        error=""
        health={health}
        onModeChange={onModeChange}
        onReset={vi.fn().mockResolvedValue(undefined)}
        onStartAct={onStartAct}
        onStart={vi.fn().mockResolvedValue(undefined)}
        onStopAct={vi.fn().mockResolvedValue(undefined)}
        onStop={vi.fn().mockResolvedValue(undefined)}
        session={session}
        socketState="connected"
        streamUrl="http://pi:8001/stream/combined.mjpeg"
        template={template}
        telemetry={null}
      />
    );

    expect(screen.getByRole("button", { name: "ACTIVE" })).toBeDisabled();
    expect(screen.getAllByText("connected").length).toBeGreaterThan(0);
    expect(screen.getByText(/ACTIVE is disabled/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "OFF" }));
    expect(onModeChange).toHaveBeenCalledWith("off");
    fireEvent.click(screen.getByRole("button", { name: "Start ACT" }));
    expect(onStartAct).toHaveBeenCalledOnce();
    expect(screen.getByText("READY")).toBeInTheDocument();
  });

  it("cannot start without an approved template", () => {
    render(
      <RuntimePanel
        decisions={[]}
        error=""
        health={health}
        onModeChange={vi.fn()}
        onReset={vi.fn()}
        onStartAct={vi.fn()}
        onStart={vi.fn()}
        onStopAct={vi.fn()}
        onStop={vi.fn()}
        session={null}
        socketState="disconnected"
        streamUrl="http://pi:8001/stream/combined.mjpeg"
        template={{ ...template, approval_status: "draft", approved_at: null }}
        telemetry={null}
      />
    );

    expect(screen.getByRole("button", { name: /Start Live/ })).toBeDisabled();
    expect(screen.getByText(/Approve the selected task template/)).toBeInTheDocument();
  });
});
