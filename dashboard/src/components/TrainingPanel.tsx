import { useState } from "react";
import type { TrainingJob, YoloModel } from "../types";

type Props = {
  models: YoloModel[];
  selectedModelId: string;
  job: TrainingJob | null;
  labelingReady: boolean;
  onModelChange: (modelId: string) => void;
  onPredict: () => Promise<void>;
  onTrain: (epochs: number, device: string | null) => Promise<void>;
  selectedEpisodeCount: number;
};

export function TrainingPanel({
  models,
  selectedModelId,
  job,
  labelingReady,
  onModelChange,
  onPredict,
  onTrain,
  selectedEpisodeCount
}: Props) {
  const [epochs, setEpochs] = useState(50);
  const [device, setDevice] = useState("mps");
  const busy = job?.status === "queued" || job?.status === "running";

  return (
    <section className="training-panel">
      <div className="panel-heading">YOLO Nano</div>
      <label>
        Epochs
        <input
          max={1000}
          min={1}
          onChange={(event) => setEpochs(Number(event.target.value))}
          type="number"
          value={epochs}
        />
      </label>
      <label>
        Training device
        <select value={device} onChange={(event) => setDevice(event.target.value)}>
          <option value="mps">Mac GPU (MPS)</option>
          <option value="cpu">CPU / Raspberry Pi</option>
          <option value="">Automatic</option>
        </select>
      </label>
      <button
        disabled={busy || selectedEpisodeCount < 2 || !labelingReady}
        onClick={() => void onTrain(epochs, device || null)}
        type="button"
      >
        {busy ? "Training…" : "Train yolo26n"}
      </button>
      {selectedEpisodeCount < 2 ? (
        <small>Select at least two auto-labeled episodes for train/validation.</small>
      ) : !labelingReady ? (
        <small>Finish DINO auto-labeling for this selection before training.</small>
      ) : null}
      {job ? <p className={`job-status ${job.status}`}>Job: {job.status}</p> : null}
      {job?.error ? <p className="error-copy">{job.error}</p> : null}

      <label>
        Review model
        <select value={selectedModelId} onChange={(event) => onModelChange(event.target.value)}>
          <option value="">Select trained model</option>
          {models.map((model) => (
            <option key={model.model_id} value={model.model_id}>
              {model.model_id}
            </option>
          ))}
        </select>
      </label>
      <button disabled={!selectedModelId} onClick={() => void onPredict()} type="button">
        Run predictions
      </button>
    </section>
  );
}
