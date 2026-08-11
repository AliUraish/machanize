import { useEffect, useState } from "react";
import { api } from "./api";
import { EpisodeSidebar } from "./components/EpisodeSidebar";
import { FrameAnnotator } from "./components/FrameAnnotator";
import { StatusHeader } from "./components/StatusHeader";
import { TrainingPanel } from "./components/TrainingPanel";
import type { Annotation, Box, Episode, Frame, Health, TrainingJob, YoloModel } from "./types";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [models, setModels] = useState<YoloModel[]>([]);
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null);
  const [selectedCameraKey, setSelectedCameraKey] = useState("");
  const [frames, setFrames] = useState<Frame[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [annotation, setAnnotation] = useState<Annotation | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [selectedClass, setSelectedClass] = useState("");
  const [selectedModelId, setSelectedModelId] = useState("");
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [message, setMessage] = useState("Ready");
  const [error, setError] = useState("");
  const currentFrame = frames[frameIndex] ?? null;
  const cameraKey = selectedCameraKey;

  useEffect(() => {
    void Promise.all([api.health(), api.episodes(), api.models()])
      .then(([nextHealth, nextEpisodes, nextModels]) => {
        setHealth(nextHealth);
        setEpisodes(nextEpisodes);
        setModels(nextModels);
        setSelectedClass(nextHealth.classes[0] ?? "");
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!selectedEpisode || !cameraKey) return;
    let cancelled = false;
    void api
      .frames(selectedEpisode.episode_id, cameraKey)
      .then((nextFrames) => {
        if (!cancelled) {
          setFrames(nextFrames);
          setFrameIndex((index) => Math.min(index, Math.max(nextFrames.length - 1, 0)));
        }
      })
      .catch((reason: Error) => {
        if (!cancelled) setError(reason.message);
      });
    return () => {
      cancelled = true;
    };
  }, [cameraKey, selectedEpisode]);

  useEffect(() => {
    if (!selectedEpisode || !currentFrame) {
      setAnnotation(null);
      setBoxes([]);
      return;
    }
    let cancelled = false;
    const manualAnnotation = () =>
      api.annotation(selectedEpisode.episode_id, currentFrame.frame_id, currentFrame.camera_key);
    const request = selectedModelId
      ? api
          .prediction(
            selectedModelId,
            selectedEpisode.episode_id,
            currentFrame.frame_id,
            currentFrame.camera_key
          )
          .catch(manualAnnotation)
      : manualAnnotation();
    void request
      .then((nextAnnotation) => {
        if (!cancelled) {
          setAnnotation(nextAnnotation);
          setBoxes(nextAnnotation.boxes);
        }
      })
      .catch((reason: Error) => {
        if (!cancelled) setError(reason.message);
      });
    return () => {
      cancelled = true;
    };
  }, [currentFrame, selectedEpisode, selectedModelId]);

  useEffect(() => {
    if (!job || (job.status !== "queued" && job.status !== "running")) return;
    const timer = window.setInterval(() => {
      void api.trainingJob(job.job_id).then((nextJob) => {
        setJob(nextJob);
        if (nextJob.status === "completed") {
          void api.models().then(setModels);
        }
      }).catch((reason: Error) => setError(reason.message));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  const approved = annotation?.approved === true;
  const progress = frames.length === 0 ? 0 : Math.round(((frameIndex + 1) / frames.length) * 100);

  const extract = async () => {
    if (!selectedEpisode || !cameraKey) return;
    setError("");
    setMessage("Extracting frames from the LeRobot episode…");
    try {
      const synchronized = await api.extractSynchronized(
        selectedEpisode.episode_id,
        selectedEpisode.camera_keys,
        5
      );
      const nextFrames = synchronized.frames[cameraKey] ?? [];
      setFrames(nextFrames);
      setFrameIndex(0);
      setMessage(
        `${synchronized.synchronized_frame_count} synchronized frame pairs ready for review`
      );
    } catch (reason) {
      setError((reason as Error).message);
    }
  };

  const save = async (approve: boolean) => {
    if (!selectedEpisode || !currentFrame) return;
    setError("");
    try {
      await api.saveAnnotation(
        selectedEpisode.episode_id,
        currentFrame,
        boxes,
        approve,
        annotation?.source ?? "manual",
        annotation?.model_id ?? undefined
      );
      setAnnotation((previous) => ({
        episode_id: selectedEpisode.episode_id,
        camera_key: currentFrame.camera_key,
        frame_id: currentFrame.frame_id,
        boxes,
        approved: approve,
        source: previous?.source ?? "manual",
        model_id: previous?.model_id
      }));
      setMessage(approve ? "Frame approved" : "Draft saved");
      if (approve && frameIndex < frames.length - 1) setFrameIndex((index) => index + 1);
    } catch (reason) {
      setError((reason as Error).message);
    }
  };

  const startTraining = async (epochs: number, device: string | null) => {
    setError("");
    try {
      const started = await api.startTraining(epochs, device);
      setJob(await api.trainingJob(started.job_id));
      const cameraSummary = Object.entries(started.camera_images)
        .map(([key, count]) => `${cameraLabel(key)} ${count}`)
        .join(" · ");
      setMessage(
        `Training queued: ${started.training_images} train · ${started.validation_images} validation · ${cameraSummary}`
      );
    } catch (reason) {
      setError((reason as Error).message);
    }
  };

  const runPredictions = async () => {
    if (!selectedEpisode || !currentFrame || !selectedModelId) return;
    setError("");
    try {
      await api.predict(selectedModelId, selectedEpisode.episode_id, currentFrame.camera_key);
      const prediction = await api.prediction(
        selectedModelId,
        selectedEpisode.episode_id,
        currentFrame.frame_id,
        currentFrame.camera_key
      );
      setAnnotation(prediction);
      setBoxes(prediction.boxes);
      setMessage("Predictions loaded. Correct boxes, then approve.");
    } catch (reason) {
      setError((reason as Error).message);
    }
  };

  const removeBox = (indexToRemove: number) => {
    setBoxes((currentBoxes) => currentBoxes.filter((_, index) => index !== indexToRemove));
  };

  const selectEpisode = (episode: Episode) => {
    setSelectedEpisode(episode);
    setSelectedCameraKey(episode.camera_keys[0] ?? "");
    setFrames([]);
    setFrameIndex(0);
    setAnnotation(null);
    setBoxes([]);
  };

  const selectCamera = (camera: string) => {
    setSelectedCameraKey(camera);
    setFrames([]);
    setAnnotation(null);
    setBoxes([]);
  };

  return (
    <div className="app-shell">
      <StatusHeader health={health} />
      <main className="workspace">
        <EpisodeSidebar
          episodes={episodes}
          onSelect={selectEpisode}
          selectedId={selectedEpisode?.episode_id}
        />

        <section className="review-workspace">
          <div className="review-toolbar">
            <div>
              <span className="eyebrow">{selectedEpisode?.project_name ?? "NO EPISODE"}</span>
              <h2>{selectedEpisode?.task || "Select an episode"}</h2>
            </div>
            <div className="camera-controls">
              <label>
                Camera stream
                <select
                  disabled={!selectedEpisode}
                  onChange={(event) => selectCamera(event.target.value)}
                  value={cameraKey}
                >
                  {selectedEpisode?.camera_keys.map((key) => (
                    <option key={key} value={key}>
                      {cameraLabel(key)}
                    </option>
                  ))}
                </select>
              </label>
              <button disabled={!selectedEpisode} onClick={() => void extract()} type="button">
                {selectedEpisode && selectedEpisode.camera_keys.length > 1
                  ? "Extract both cameras"
                  : "Extract camera"}
              </button>
            </div>
          </div>

          <FrameAnnotator
            boxes={boxes}
            frame={currentFrame}
            onBoxesChange={setBoxes}
            selectedClass={selectedClass}
          />

          <div className="timeline">
            <button
              aria-label="Previous frame"
              disabled={frameIndex === 0}
              onClick={() => setFrameIndex((index) => Math.max(0, index - 1))}
              type="button"
            >
              ←
            </button>
            <div className="timeline-track">
              <span style={{ width: `${progress}%` }} />
            </div>
            <strong>
              {frames.length === 0 ? "0 / 0" : `${frameIndex + 1} / ${frames.length}`}
            </strong>
            <button
              aria-label="Next frame"
              disabled={frameIndex >= frames.length - 1}
              onClick={() => setFrameIndex((index) => Math.min(frames.length - 1, index + 1))}
              type="button"
            >
              →
            </button>
          </div>
        </section>

        <aside className="inspector">
          <section>
            <div className="panel-heading">Frame label</div>
            <label>
              Draw class
              <select value={selectedClass} onChange={(event) => setSelectedClass(event.target.value)}>
                {health?.classes.map((className) => (
                  <option key={className}>{className}</option>
                ))}
              </select>
            </label>
            <div className="box-list">
              {boxes.map((box, index) => (
                <div className="box-row" key={`${box.class_name}-${index}`}>
                  <span>{box.class_name}</span>
                  {box.confidence != null ? <small>{Math.round(box.confidence * 100)}%</small> : null}
                  <button
                    aria-label={`Remove ${box.class_name} box`}
                    onClick={() => removeBox(index)}
                    type="button"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
            <div className="approval-actions">
              <button disabled={!currentFrame} onClick={() => void save(false)} type="button">
                Save draft
              </button>
              <button
                className="primary"
                disabled={!currentFrame}
                onClick={() => void save(true)}
                type="button"
              >
                {approved ? "Approved" : "Approve frame"}
              </button>
            </div>
          </section>

          <TrainingPanel
            job={job}
            models={models}
            onModelChange={setSelectedModelId}
            onPredict={runPredictions}
            onTrain={startTraining}
            selectedModelId={selectedModelId}
          />
        </aside>
      </main>
      <footer className="status-bar">
        <span>{message}</span>
        {error ? <strong>{error}</strong> : null}
        <span>
          {currentFrame
            ? `${cameraLabel(currentFrame.camera_key).toUpperCase()} · FRAME ${currentFrame.frame_id} · ${currentFrame.timestamp.toFixed(2)}s`
            : ""}
        </span>
      </footer>
    </div>
  );
}

function cameraLabel(cameraKey: string) {
  return cameraKey.split(".").at(-1) ?? cameraKey;
}
