import type { Episode, TaskAnalysisJob, TaskTemplate } from "../types";

type Props = {
  episode: Episode | null;
  healthModel?: string;
  job: TaskAnalysisJob | null;
  selectedEpisodeCount: number;
  template: TaskTemplate | null;
  templateText: string;
  onApprove: () => Promise<void>;
  onStart: () => Promise<void>;
  onSave: () => Promise<void>;
  onTemplateTextChange: (value: string) => void;
};

export function TaskAnalysisPanel({
  episode,
  healthModel,
  job,
  selectedEpisodeCount,
  template,
  templateText,
  onApprove,
  onStart,
  onSave,
  onTemplateTextChange
}: Props) {
  const busy = job?.status === "queued" || job?.status === "running";
  const hasBothCameras = episode?.camera_keys.some((key) => key.endsWith(".front"))
    && episode.camera_keys.some((key) => key.endsWith(".wrist"));
  const outcome = episode?.outcome.toLowerCase();
  const validSelection = (outcome === "success" || outcome === "unknown")
    && selectedEpisodeCount === 1
    && hasBothCameras;

  return (
    <section className="task-analysis-panel">
      <div className="panel-heading">
        Demonstration analysis
        <span className={`approval-state ${template?.approval_status ?? "none"}`}>
          {template?.approval_status ?? "not analyzed"}
        </span>
      </div>
      <p className="analysis-meta">
        {healthModel ?? "gemini-robotics-er-1.6-preview"} · Front + Wrist · 5 FPS
      </p>
      <button disabled={!validSelection || busy} onClick={() => void onStart()} type="button">
        {busy
          ? "Analyzing demonstration…"
          : outcome === "unknown"
            ? "Confirm success and analyze"
            : "Analyze selected success"}
      </button>
      {!validSelection ? (
        <small>Select exactly one successful or unknown-outcome episode containing Front and Wrist video.</small>
      ) : outcome === "unknown" ? (
        <small>Starting analysis explicitly confirms this unknown-outcome episode as a successful demonstration.</small>
      ) : null}
      {job ? <p className={`job-status ${job.status}`}>Analysis: {job.status}</p> : null}
      {job?.error ? <p className="error-copy">{job.error}</p> : null}

      {template ? (
        <>
          <div className="analysis-summary">
            <strong>{template.task_description}</strong>
            <ol>
              {template.ordered_task_stages.map((stage) => (
                <li key={`${stage.name}-${stage.start_time_seconds}`}>
                  <span>{stage.name}</span>
                  <small>{stage.start_time_seconds.toFixed(2)}–{stage.end_time_seconds.toFixed(2)}s · {Math.round(stage.confidence * 100)}%</small>
                </li>
              ))}
            </ol>
            <small>{template.success_conditions.length} success conditions · {template.possible_failure_types.length} possible failures · {template.uncertainty.length} uncertainties</small>
          </div>
          <label>
            Editable structured task template
            <textarea
              aria-label="Editable structured task template"
              onChange={(event) => onTemplateTextChange(event.target.value)}
              spellCheck={false}
              value={templateText}
            />
          </label>
          <p className="analysis-meta">
            Confidence {Math.round(template.confidence * 100)}% · source {template.source_episode.episode_id}
          </p>
          <div className="approval-actions">
            <button onClick={() => void onSave()} type="button">Save draft</button>
            <button className="primary" onClick={() => void onApprove()} type="button">
              {template.approval_status === "approved" ? "Re-approve current draft" : "Approve template"}
            </button>
          </div>
          <small>Gemini output always starts as a draft. Only this approval action can approve it.</small>
        </>
      ) : null}
    </section>
  );
}
