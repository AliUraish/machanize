"""Background Grounding DINO labeling and YOLO training jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from machanize.perception.annotations import AnnotationStore
from machanize.perception.episodes import EpisodeRepository, FrameExtractor
from machanize.perception.grounding_dino import MODEL_ID, GroundingDinoDetector
from machanize.perception.yolo import YoloExport, YoloTrainer


@dataclass
class TrainingJob:
    job_id: str
    status: str
    created_at: str
    completed_at: str | None = None
    model: dict[str, Any] | None = None
    error: str | None = None


class TrainingJobManager:
    def __init__(self, trainer: YoloTrainer) -> None:
        self.trainer = trainer
        self._jobs: dict[str, TrainingJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="machanize-yolo")
        self._lock = RLock()

    def submit(
        self,
        export: YoloExport,
        *,
        base_model: str,
        epochs: int,
        image_size: int,
        device: str | None,
    ) -> TrainingJob:
        job = TrainingJob(
            job_id=str(uuid4()),
            status="queued",
            created_at=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._executor.submit(
            self._run,
            job.job_id,
            export,
            base_model,
            epochs,
            image_size,
            device,
        )
        return job

    def get(self, job_id: str) -> TrainingJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as error:
                raise KeyError(f"Unknown training job: {job_id}") from error

    def to_api(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return asdict(self.get(job_id))

    def _run(
        self,
        job_id: str,
        export: YoloExport,
        base_model: str,
        epochs: int,
        image_size: int,
        device: str | None,
    ) -> None:
        with self._lock:
            self._jobs[job_id].status = "running"
        try:
            model = self.trainer.train(
                export,
                base_model=base_model,
                epochs=epochs,
                image_size=image_size,
                device=device,
            )
        except Exception as error:  # noqa: BLE001 - surface training failures in the dashboard
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(error)
                job.completed_at = datetime.now(UTC).isoformat()
            return

        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.model = model.to_api()
            job.completed_at = datetime.now(UTC).isoformat()


@dataclass
class LabelingJob:
    job_id: str
    status: str
    created_at: str
    episode_ids: list[str]
    confidence: float
    total_frames: int = 0
    processed_frames: int = 0
    labeled_frames: int = 0
    total_boxes: int = 0
    current_episode_id: str | None = None
    current_camera_key: str | None = None
    completed_at: str | None = None
    errors: list[str] = field(default_factory=list)


class LabelingJobManager:
    """Auto-label every frame from selected episodes without robot access."""

    def __init__(
        self,
        episodes: EpisodeRepository,
        frames: FrameExtractor,
        annotations: AnnotationStore,
        detector: GroundingDinoDetector,
    ) -> None:
        self.episodes = episodes
        self.frames = frames
        self.annotations = annotations
        self.detector = detector
        self._jobs: dict[str, LabelingJob] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="machanize-dino",
        )
        self._lock = RLock()

    def submit(self, episode_ids: list[str], *, confidence: float) -> LabelingJob:
        if not episode_ids:
            raise ValueError("Select at least one episode to auto-label.")
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("Each episode may only be selected once.")
        for episode_id in episode_ids:
            self.episodes.get(episode_id)
        job = LabelingJob(
            job_id=str(uuid4()),
            status="queued",
            created_at=datetime.now(UTC).isoformat(),
            episode_ids=list(episode_ids),
            confidence=confidence,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._executor.submit(self._run, job.job_id)
        return job

    def get(self, job_id: str) -> LabelingJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as error:
                raise KeyError(f"Unknown labeling job: {job_id}") from error

    def to_api(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return asdict(self.get(job_id))

    def _run(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id].status = "running"

        for episode_id in self.get(job_id).episode_ids:
            with self._lock:
                self._jobs[job_id].current_episode_id = episode_id
            try:
                episode = self.episodes.get(episode_id)
                camera_keys = list(episode.camera_keys)
                if not camera_keys:
                    raise ValueError("Episode has no camera streams.")
                frames_by_camera = self.frames.extract_synchronized(
                    episode,
                    camera_keys=camera_keys,
                    stride=1,
                    overwrite=True,
                )
            except Exception as error:  # noqa: BLE001 - continue the remaining episodes
                self._record_error(job_id, episode_id, str(error))
                continue

            episode_frames = [
                frame for camera_frames in frames_by_camera.values() for frame in camera_frames
            ]
            with self._lock:
                self._jobs[job_id].total_frames += len(episode_frames)

            for frame in episode_frames:
                with self._lock:
                    self._jobs[job_id].current_camera_key = frame.camera_key
                try:
                    boxes = self.detector.detect(
                        frame.image_path,
                        threshold=self.get(job_id).confidence,
                    )
                    self.annotations.save(
                        episode_id=frame.episode_id,
                        camera_key=frame.camera_key,
                        frame_id=frame.frame_id,
                        image_path=frame.image_path,
                        boxes=boxes,
                        approved=True,
                        source="grounding_dino",
                        model_id=MODEL_ID,
                    )
                except Exception as error:  # noqa: BLE001 - report a frame error and continue
                    self._record_error(
                        job_id,
                        f"{frame.episode_id}/{frame.camera_key}/{frame.frame_id}",
                        str(error),
                    )
                else:
                    with self._lock:
                        job = self._jobs[job_id]
                        job.labeled_frames += 1
                        job.total_boxes += len(boxes)
                finally:
                    with self._lock:
                        self._jobs[job_id].processed_frames += 1

        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed" if job.labeled_frames > 0 else "failed"
            if job.status == "failed" and not job.errors:
                job.errors.append("No frames were labeled.")
            job.current_episode_id = None
            job.current_camera_key = None
            job.completed_at = datetime.now(UTC).isoformat()

    def _record_error(self, job_id: str, location: str, message: str) -> None:
        with self._lock:
            self._jobs[job_id].errors.append(f"{location}: {message}")
