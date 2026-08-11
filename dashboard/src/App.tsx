import { useEffect, useState } from "react";
import { api } from "./api";
import { BatchLabelPanel } from "./components/BatchLabelPanel";
import { BoxReviewList } from "./components/BoxReviewList";
import { EpisodeSidebar } from "./components/EpisodeSidebar";
import { FrameAnnotator } from "./components/FrameAnnotator";
import { StatusHeader } from "./components/StatusHeader";
import { TaskAnalysisPanel } from "./components/TaskAnalysisPanel";
import { TrainingPanel } from "./components/TrainingPanel";
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
  TrainingJob,
  YoloModel
} from "./types";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [models, setModels] = useState<YoloModel[]>([]);
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null);
  const [selectedEpisodeIds, setSelectedEpisodeIds] = useState<string[]>([]);
  const [selectedCameraKey, setSelectedCameraKey] = useState("");
  const [frames, setFrames] = useState<Frame[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [annotation, setAnnotation] = useState<Annotation | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [selectedClass, setSelectedClass] = useState("");
  const [selectedModelId, setSelectedModelId] = useState("");
  const [autoLabelConfidence, setAutoLabelConfidence] = useState(0.5);
  const [labelingJob, setLabelingJob] = useState<LabelingJob | null>(null);
  const [labelingRevision, setLabelingRevision] = useState(0);
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [analysisJob, setAnalysisJob] = useState<TaskAnalysisJob | null>(null);
  const [taskTemplate, setTaskTemplate] = useState<TaskTemplate | null>(null);
  const [taskTemplateText, setTaskTemplateText] = useState("");
  const [message, setMessage] = useState("Ready");
  const [error, setError] = useState("");
  const currentFrame = frames[frameIndex] ?? null;
  const cameraKey = selectedCameraKey;
  const labelingReady = labelingJob?.status === "completed"
    && labelingJob.episode_ids.length === selectedEpisodeIds.length
    && selectedEpisodeIds.every((episodeId) => labelingJob.episode_ids.includes(episodeId));
  const loadTaskTemplate = (record: TaskTemplate) => {
    setTaskTemplate(record);
    setTaskTemplateText(JSON.stringify(toTaskTemplateDraft(record), null, 2));
  };

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
          setBoxes(
            nextAnnotation.boxes.map((box) => ({
              ...box,
              suggested: nextAnnotation.source === "grounding_dino",
              accepted: nextAnnotation.source === "grounding_dino" ? true : undefined
            }))
          );
        }
      })
      .catch((reason: Error) => {
        if (!cancelled) setError(reason.message);
      });
    return () => {
      cancelled = true;
    };
  }, [currentFrame, labelingRevision, selectedEpisode, selectedModelId]);

  useEffect(() => {
    if (!labelingJob || (labelingJob.status !== "queued" && labelingJob.status !== "running")) {
      return;
    }
    const timer = window.setInterval(() => {
      void api.labelingJob(labelingJob.job_id).then((nextJob) => {
        setLabelingJob(nextJob);
        setMessage(
          `DINO: ${nextJob.processed_frames}/${nextJob.total_frames} frames · ${nextJob.total_boxes} boxes`
        );
        if (nextJob.status === "completed" || nextJob.status === "failed") {
          setLabelingRevision((revision) => revision + 1);
          if (selectedEpisode && cameraKey) {
            void api.frames(selectedEpisode.episode_id, cameraKey).then(setFrames);
          }
          setMessage(
            nextJob.status === "completed"
              ? `Auto-labeling complete: ${nextJob.labeled_frames} frames · ${nextJob.total_boxes} boxes`
              : "Auto-labeling failed"
          );
        }
      }).catch((reason: Error) => setError(reason.message));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [cameraKey, labelingJob, selectedEpisode]);

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

  useEffect(() => {
    if (!selectedEpisode) {
      setTaskTemplate(null);
      setTaskTemplateText("");
      return;
    }
    let cancelled = false;
    void api.taskTemplate(selectedEpisode.episode_id)
      .then((record) => {
        if (!cancelled) loadTaskTemplate(record);
      })
      .catch((reason: Error) => {
        if (!cancelled && reason.message !== "Task analysis not found.") setError(reason.message);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedEpisode]);

  useEffect(() => {
    if (!analysisJob || (analysisJob.status !== "queued" && analysisJob.status !== "running")) {
      return;
    }
    const timer = window.setInterval(() => {
      void api.taskAnalysisJob(analysisJob.job_id).then((nextJob) => {
        setAnalysisJob(nextJob);
        setMessage(`Gemini demonstration analysis: ${nextJob.status}`);
        if (nextJob.status === "completed" && nextJob.result) {
          loadTaskTemplate(nextJob.result);
          setMessage("Gemini analysis ready as a draft. Review, edit, and explicitly approve it.");
        }
      }).catch((reason: Error) => {
        if (reason.message.startsWith("Analysis job is no longer available")) {
          setAnalysisJob(null);
          setMessage("Analysis stopped because the API restarted.");
        }
        setError(reason.message);
      });
    }, 1500);
    return () => window.clearInterval(timer);
  }, [analysisJob]);

  const approved = annotation?.approved === true;
  const unresolvedSuggestions = boxes.filter(
    (box) => box.suggested && box.accepted !== true
  ).length;
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
    if (approve && unresolvedSuggestions > 0) {
      setError("Accept or delete every suggested box before approving this frame.");
      return;
    }
    setError("");
    const acceptedBoxes = boxes.filter((box) => !box.suggested || box.accepted === true);
    try {
      await api.saveAnnotation(
        selectedEpisode.episode_id,
        currentFrame,
        acceptedBoxes,
        approve,
        annotation?.source ?? "manual",
        annotation?.model_id ?? undefined
      );
      setAnnotation((previous) => ({
        episode_id: selectedEpisode.episode_id,
        camera_key: currentFrame.camera_key,
        frame_id: currentFrame.frame_id,
        boxes: acceptedBoxes,
        approved: approve,
        source: previous?.source ?? "manual",
        model_id: previous?.model_id
      }));
      setBoxes(acceptedBoxes);
      setMessage(approve ? "Frame approved" : "Draft saved");
      if (approve && frameIndex < frames.length - 1) setFrameIndex((index) => index + 1);
    } catch (reason) {
      setError((reason as Error).message);
    }
  };

  const startTraining = async (epochs: number, device: string | null) => {
    setError("");
    try {
      if (selectedEpisodeIds.length < 2) {
        throw new Error("Select at least two auto-labeled episodes before training.");
      }
      if (!labelingReady) {
        throw new Error("Finish automatic DINO labeling for the selected episodes first.");
      }
      const started = await api.startTraining(epochs, device, selectedEpisodeIds);
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

  const startBatchLabeling = async () => {
    if (selectedEpisodeIds.length === 0) return;
    setError("");
    setMessage(`Queuing ${selectedEpisodeIds.length} episodes for automatic labeling…`);
    try {
      const started = await api.startBatchLabeling(
        selectedEpisodeIds,
        autoLabelConfidence
      );
      setLabelingJob(started);
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

  const startTaskAnalysis = async () => {
    if (!selectedEpisode) return;
    setError("");
    setMessage("Building synchronized Front + Wrist evidence and sending it to Gemini…");
    try {
      setAnalysisJob(await api.startTaskAnalysis(
        selectedEpisode.episode_id,
        selectedEpisode.outcome.toLowerCase() === "unknown"
      ));
    } catch (reason) {
      setError((reason as Error).message);
    }
  };

  const saveTaskTemplate = async () => {
    if (!selectedEpisode) return;
    setError("");
    try {
      const draft = JSON.parse(taskTemplateText) as TaskTemplateDraft;
      const saved = await api.saveTaskTemplate(selectedEpisode.episode_id, draft);
      loadTaskTemplate(saved);
      setMessage("Task template saved as a draft. It is not approved.");
    } catch (reason) {
      setError(reason instanceof SyntaxError ? `Invalid JSON: ${reason.message}` : (reason as Error).message);
    }
  };

  const approveTaskTemplate = async () => {
    if (!selectedEpisode) return;
    setError("");
    try {
      const draft = JSON.parse(taskTemplateText) as TaskTemplateDraft;
      await api.saveTaskTemplate(selectedEpisode.episode_id, draft);
      const approvedTemplate = await api.approveTaskTemplate(selectedEpisode.episode_id);
      loadTaskTemplate(approvedTemplate);
      setMessage("Task template explicitly approved by the user.");
    } catch (reason) {
      setError(reason instanceof SyntaxError ? `Invalid JSON: ${reason.message}` : (reason as Error).message);
    }
  };

  const selectEpisode = (episode: Episode) => {
    setSelectedEpisode(episode);
    setSelectedEpisodeIds((ids) =>
      ids.includes(episode.episode_id) ? ids : [...ids, episode.episode_id]
    );
    setSelectedCameraKey(episode.camera_keys[0] ?? "");
    setFrames([]);
    setFrameIndex(0);
    setAnnotation(null);
    setBoxes([]);
    setAnalysisJob(null);
    setTaskTemplate(null);
    setTaskTemplateText("");
  };

  const toggleEpisode = (episodeId: string) => {
    setSelectedEpisodeIds((ids) =>
      ids.includes(episodeId) ? ids.filter((id) => id !== episodeId) : [...ids, episodeId]
    );
  };

  const toggleAllEpisodes = () => {
    setSelectedEpisodeIds((ids) =>
      ids.length === episodes.length ? [] : episodes.map((episode) => episode.episode_id)
    );
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
          activeId={selectedEpisode?.episode_id}
          episodes={episodes}
          onOpen={selectEpisode}
          onToggle={toggleEpisode}
          onToggleAll={toggleAllEpisodes}
          selectedIds={selectedEpisodeIds}
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
          <TaskAnalysisPanel
            episode={selectedEpisode}
            healthModel={health?.task_analysis_model}
            job={analysisJob}
            onApprove={approveTaskTemplate}
            onSave={saveTaskTemplate}
            onStart={startTaskAnalysis}
            onTemplateTextChange={setTaskTemplateText}
            selectedEpisodeCount={selectedEpisodeIds.length}
            template={taskTemplate}
            templateText={taskTemplateText}
          />
          <BatchLabelPanel
            confidence={autoLabelConfidence}
            job={labelingJob}
            onConfidenceChange={setAutoLabelConfidence}
            onStart={startBatchLabeling}
            selectedEpisodeCount={selectedEpisodeIds.length}
          />
          <section>
            <div className="panel-heading">Label review (optional)</div>
            <label>
              Draw class
              <select value={selectedClass} onChange={(event) => setSelectedClass(event.target.value)}>
                {health?.classes.map((className) => (
                  <option key={className}>{className}</option>
                ))}
              </select>
            </label>
            <BoxReviewList
              boxes={boxes}
              classNames={health?.classes ?? []}
              onChange={setBoxes}
            />
            <div className="approval-actions">
              <button disabled={!currentFrame} onClick={() => void save(false)} type="button">
                Save draft
              </button>
              <button
                className="primary"
                disabled={!currentFrame || unresolvedSuggestions > 0}
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
            selectedEpisodeCount={selectedEpisodeIds.length}
            labelingReady={labelingReady}
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

function toTaskTemplateDraft(record: TaskTemplate): TaskTemplateDraft {
  return {
    task_description: record.task_description,
    ordered_task_stages: record.ordered_task_stages,
    success_conditions: record.success_conditions,
    possible_failure_types: record.possible_failure_types,
    important_timestamps_and_evidence: record.important_timestamps_and_evidence,
    confidence: record.confidence,
    uncertainty: record.uncertainty
  };
}
