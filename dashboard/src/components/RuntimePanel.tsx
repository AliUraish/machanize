import { StageTimeline } from "./StageTimeline";
import type {
  RuntimeDecision,
  RuntimeHealth,
  RuntimeMode,
  RuntimeSession,
  RuntimeTelemetry,
  TaskTemplate
} from "../types";

type Props = {
  health: RuntimeHealth | null;
  template: TaskTemplate | null;
  session: RuntimeSession | null;
  decisions: RuntimeDecision[];
  error: string;
  telemetry: RuntimeTelemetry | null;
  socketState: string;
  streamUrl: string;
  onStart: () => Promise<void>;
  onStop: () => Promise<void>;
  onReset: () => Promise<void>;
  onStartAct: () => Promise<void>;
  onStopAct: () => Promise<void>;
  onModeChange: (mode: RuntimeMode) => Promise<void>;
};

export function RuntimePanel({
  health,
  template,
  session,
  decisions,
  error,
  telemetry,
  socketState,
  streamUrl,
  onStart,
  onStop,
  onReset,
  onStartAct,
  onStopAct,
  onModeChange
}: Props) {
  const latest = decisions.at(-1);
  const ready = template?.approval_status === "approved";
  const sessionRunning = session != null && session.stopped_at == null;
  const stopLatched = telemetry?.stop_latched ?? session?.stop_latched ?? health?.stop_latched;
  const actState = telemetry?.act_state ?? health?.act_state ?? "stopped";
  const monitorReady = sessionRunning
    && session.mode === "monitor"
    && session.connection_state === "connected";
  const canStartAct = ready
    && monitorReady
    && health?.robot_connected === true
    && !stopLatched
    && (actState === "ready" || actState === "stopped");
  const currentStage = telemetry?.current_stage ?? latest?.report.current_stage;
  const confidence = telemetry?.confidence ?? latest?.report.confidence;
  const latency = telemetry?.latency_ms ?? latest?.latency_ms;
  const decision = telemetry?.decision ?? (stopLatched ? "STOP" : "CONTINUE");

  return (
    <section className="runtime-panel">
      <div className="panel-heading">
        Live monitoring
        <span className={`connection-state ${session?.connection_state ?? "off"}`}>
          {session?.connection_state ?? "off"}
        </span>
      </div>
      <p className="analysis-meta">
        Pi {runtimeBaseLabel(health)} · control {health?.control_fps ?? "—"} FPS · AI sample {health?.monitor_sample_fps ?? 5} FPS
      </p>
      <div className="runtime-video">
        {health?.robot_configured ? (
          <img alt="Live combined front and wrist cameras" src={streamUrl} />
        ) : (
          <span>Pi robot factory is not configured.</span>
        )}
      </div>
      <div className="runtime-health-grid">
        <span>Pi WebSocket</span><strong>{socketState}</strong>
        <span>Robot</span><strong>{telemetry?.robot_connected ?? health?.robot_connected ? "connected" : "offline"}</strong>
        <span>Control loop</span><strong>{telemetry?.control_loop_status ?? health?.control_loop_status ?? "unknown"}</strong>
        <span>Provider</span><strong>{telemetry?.provider_connection ?? session?.connection_state ?? "off"}</strong>
      </div>
      {!sessionRunning ? (
        <button disabled={!ready || !health?.api_key_configured} onClick={() => void onStart()} type="button">
          Start Live monitoring session
        </button>
      ) : (
        <button className="runtime-stop" onClick={() => void onStop()} type="button">
          Stop Live monitoring session
        </button>
      )}
      {!ready ? <small>Approve the selected task template before starting runtime monitoring.</small> : null}
      {health && !health.api_key_configured ? <small>GEMINI_API_KEY is not configured in the runtime backend.</small> : null}

      <div className="runtime-modes" aria-label="Runtime mode">
        {(["off", "monitor", "active"] as RuntimeMode[]).map((mode) => (
          <button
            aria-pressed={session?.mode === mode}
            disabled={!sessionRunning || (mode === "active" && !health?.active_enabled)}
            key={mode}
            onClick={() => void onModeChange(mode)}
            type="button"
          >
            {mode.toUpperCase()}
          </button>
        ))}
      </div>
      {health && !health.active_enabled ? (
        <small>ACTIVE is disabled by backend configuration.</small>
      ) : null}

      <div className="act-controls">
        <div>
          <span>ACT state</span>
          <strong className={`act-state ${actState}`}>{actState.toUpperCase()}</strong>
        </div>
        <button disabled={!canStartAct} onClick={() => void onStartAct()} type="button">
          Start ACT
        </button>
        <button
          className="runtime-stop"
          disabled={actState !== "running"}
          onClick={() => void onStopAct()}
          type="button"
        >
          Stop ACT
        </button>
      </div>
      {!monitorReady && actState !== "running" ? (
        <small>Start ACT requires an approved template and a connected MONITOR session.</small>
      ) : null}

      {stopLatched ? (
        <div className="stop-alert" role="alert">
          STOP LATCHED · {telemetry?.stop_reason ?? session?.stop_reason}
          <button
            disabled={!health?.reset_auth_configured}
            onClick={() => void onReset()}
            type="button"
          >
            Authenticated manual reset
          </button>
        </div>
      ) : latest?.report.recommend_stop ? (
        <div className="monitor-alert" role="alert">
          Stop recommended · streak {latest.stop_streak}
        </div>
      ) : null}

      {template ? (
        <StageTimeline
          currentStage={currentStage}
          stages={template.ordered_task_stages}
        />
      ) : null}

      <div className="runtime-readout">
        <span>Stage</span><strong>{currentStage ?? "—"}</strong>
        <span>Risk</span><strong>{telemetry ? `${Math.round(telemetry.risk * 100)}%` : "—"}</strong>
        <span>Confidence</span><strong>{confidence != null ? `${Math.round(confidence * 100)}%` : "—"}</strong>
        <span>Latency</span><strong>{latency != null ? `${Math.round(latency)} ms` : "—"}</strong>
        <span>Decision</span><strong className={decision === "STOP" ? "decision-stop" : "decision-continue"}>{decision}</strong>
      </div>

      {(telemetry?.evidence.length || latest?.report.evidence.length) ? (
        <div className="runtime-evidence">
          <strong>Timestamp evidence</strong>
          {(telemetry?.evidence ?? latest?.report.evidence ?? []).map((item) => (
            <p key={`${item.timestamp}-${item.description}`}>
              <time>{item.timestamp}</time> {item.description}
            </p>
          ))}
        </div>
      ) : null}

      <div className="decision-history">
        <strong>Saved decisions ({decisions.length})</strong>
        {decisions.slice(-5).reverse().map((decision) => (
          <p key={decision.decision_id}>
            {decision.report.current_stage} · {Math.round(decision.report.confidence * 100)}% · {decision.local_result}
          </p>
        ))}
      </div>
      {error ? <p className="error-copy">{error}</p> : null}
    </section>
  );
}

function runtimeBaseLabel(health: RuntimeHealth | null): string {
  if (!health) return "offline";
  return health.robot_connected ? "connected" : "reachable";
}
