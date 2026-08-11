import type { LabelingJob } from "../types";

type Props = {
  confidence: number;
  job: LabelingJob | null;
  selectedEpisodeCount: number;
  onConfidenceChange: (confidence: number) => void;
  onStart: () => Promise<void>;
};

export function BatchLabelPanel({
  confidence,
  job,
  selectedEpisodeCount,
  onConfidenceChange,
  onStart
}: Props) {
  const busy = job?.status === "queued" || job?.status === "running";
  const progress = job && job.total_frames > 0
    ? Math.round((job.processed_frames / job.total_frames) * 100)
    : 0;

  return (
    <section className="batch-label-panel">
      <div className="panel-heading">Grounding DINO</div>
      <label>
        Auto-label confidence
        <div className="threshold-control">
          <input
            aria-label="Auto-label confidence"
            disabled={busy}
            max={0.9}
            min={0.1}
            onInput={(event) => onConfidenceChange(Number(event.currentTarget.value))}
            step={0.05}
            type="range"
            value={confidence}
          />
          <output>{Math.round(confidence * 100)}%</output>
        </div>
      </label>
      <button
        disabled={busy || selectedEpisodeCount === 0}
        onClick={() => void onStart()}
        type="button"
      >
        {busy ? "Auto-labeling…" : `Auto-label ${selectedEpisodeCount} selected`}
      </button>
      <small>Every frame from every available camera is saved automatically. No approval required.</small>
      {job ? (
        <div className="labeling-progress" aria-live="polite">
          <progress max={100} value={progress} />
          <span>{job.processed_frames} / {job.total_frames} frames · {job.total_boxes} boxes</span>
          <span className={`job-status ${job.status}`}>{job.status}</span>
          {job.errors.length > 0 ? <span className="error-copy">{job.errors.length} errors</span> : null}
        </div>
      ) : null}
    </section>
  );
}
