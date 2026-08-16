"""Build multimodal episode evidence and obtain a human-reviewed task template."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Protocol
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field, model_validator

from machanize.perception.episodes import EpisodeRecord

GEMINI_ROBOTICS_MODEL = "gemini-robotics-er-1.6-preview"
ANALYSIS_FPS = 5.0
logger = logging.getLogger(__name__)


class TimestampEvidence(BaseModel):
    timestamp_seconds: float = Field(ge=0)
    description: str = Field(min_length=1)


class TaskStage(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    start_time_seconds: float = Field(ge=0)
    end_time_seconds: float = Field(ge=0)
    expected_object_relationships: list[str]
    expected_robot_behavior: str = Field(min_length=1)
    expected_gripper_behavior: str = Field(min_length=1)
    evidence: list[TimestampEvidence]
    confidence: float = Field(ge=0, le=1)
    uncertainty: list[str]

    @model_validator(mode="after")
    def validate_time_range(self) -> TaskStage:
        if self.end_time_seconds < self.start_time_seconds:
            raise ValueError("Stage end time must not be before its start time.")
        return self


class PossibleFailure(BaseModel):
    failure_type: str = Field(min_length=1)
    description: str = Field(min_length=1)
    related_stage_names: list[str]
    detectable_evidence: list[str]


class TaskTemplateDraft(BaseModel):
    task_description: str = Field(min_length=1)
    ordered_task_stages: list[TaskStage] = Field(min_length=1)
    success_conditions: list[str] = Field(min_length=1)
    possible_failure_types: list[PossibleFailure] = Field(min_length=1)
    important_timestamps_and_evidence: list[TimestampEvidence]
    confidence: float = Field(ge=0, le=1)
    uncertainty: list[str]


class SourceEpisode(BaseModel):
    episode_id: str
    dataset_episode_index: int
    project_name: str


class TaskTemplateRecord(TaskTemplateDraft):
    template_version: int = Field(default=1, ge=1)
    source_episode: SourceEpisode
    model_version: str
    video_fps: float
    approval_status: Literal["draft", "approved"]
    created_at: str
    updated_at: str
    approved_at: str | None = None


@dataclass(frozen=True)
class EpisodeEvidence:
    video_path: Path
    telemetry_path: Path
    telemetry: list[dict[str, Any]]
    fps: float


class TaskAnalysisProvider(Protocol):
    """Replaceable provider boundary for demonstration analysis."""

    model: str

    def analyze(
        self,
        episode: EpisodeRecord,
        evidence: EpisodeEvidence,
    ) -> TaskTemplateDraft: ...


class EpisodeEvidenceBuilder:
    """Render synchronized front/wrist frames and telemetry into one evidence package."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        dataset_factory: Callable[[EpisodeRecord], Any] | None = None,
        video_encoder: Callable[[Path, Path, float], None] | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self._dataset_factory = dataset_factory
        self._video_encoder = video_encoder or _encode_mp4

    def build(self, episode: EpisodeRecord, *, fps: float = ANALYSIS_FPS) -> EpisodeEvidence:
        if episode.outcome.lower() != "success":
            raise ValueError("Gemini task analysis requires one successful episode.")
        front_key = _camera_key(episode.camera_keys, "front")
        wrist_key = _camera_key(episode.camera_keys, "wrist")
        if fps <= 0 or fps > 24:
            raise ValueError("Analysis FPS must be greater than 0 and no more than 24.")

        info = json.loads((episode.dataset_root / "meta" / "info.json").read_text())
        source_fps = float(info.get("fps", 30))
        stride = max(1, round(source_fps / fps))
        actual_fps = source_fps / stride
        feature_names = {
            key: list(feature.get("names") or [])
            for key, feature in info.get("features", {}).items()
            if isinstance(feature, dict)
        }
        dataset = self._load_dataset(episode)
        output_directory = self.output_root / _safe_part(episode.episode_id)
        output_directory.mkdir(parents=True, exist_ok=True)
        video_path = output_directory / "front-wrist-evidence.mp4"
        telemetry_path = output_directory / "telemetry.json"
        telemetry: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="machanize-analysis-") as temporary:
            frame_directory = Path(temporary)
            for output_index, local_index in enumerate(range(0, len(dataset), stride)):
                sample = dataset[local_index]
                row = _telemetry_row(sample, feature_names)
                telemetry.append(row)
                frame = _compose_frame(
                    _to_image(sample[front_key]),
                    _to_image(sample[wrist_key]),
                    task=episode.task,
                    telemetry=row,
                )
                frame.save(frame_directory / f"frame-{output_index:06d}.png")
            if not telemetry:
                raise ValueError("The selected episode contains no frames.")
            self._video_encoder(frame_directory, video_path, actual_fps)

        telemetry_path.write_text(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "task_description": episode.task,
                    "source_fps": source_fps,
                    "analysis_fps": actual_fps,
                    "samples": telemetry,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return EpisodeEvidence(video_path, telemetry_path, telemetry, actual_fps)

    def _load_dataset(self, episode: EpisodeRecord) -> Any:
        if self._dataset_factory is not None:
            return self._dataset_factory(episode)
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as error:
            raise RuntimeError(
                "LeRobot dataset support is required for demonstration analysis."
            ) from error
        return LeRobotDataset(
            f"machanize/analysis-{_safe_part(episode.episode_id)}",
            root=episode.dataset_root,
            episodes=[episode.dataset_episode_index],
        )


class GeminiRoboticsAnalyzer:
    """Use the normal Gemini generateContent API; this never trains or modifies Gemini."""

    def __init__(
        self,
        *,
        model: str = GEMINI_ROBOTICS_MODEL,
        client_factory: Callable[[], Any] | None = None,
        processing_timeout_seconds: float = 300,
    ) -> None:
        self.model = model
        self._client_factory = client_factory
        self.processing_timeout_seconds = processing_timeout_seconds

    def analyze(self, episode: EpisodeRecord, evidence: EpisodeEvidence) -> TaskTemplateDraft:
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise RuntimeError(
                "Google GenAI support is not installed. Install the 'vision' extra."
            ) from error

        client = self._client_factory() if self._client_factory else genai.Client()
        uploaded = client.files.upload(file=str(evidence.video_path))
        try:
            uploaded = self._wait_until_ready(client, uploaded)
            prompt = _analysis_prompt(episode, evidence)
            response = client.models.generate_content(
                model=self.model,
                contents=types.Content(
                    parts=[
                        types.Part(
                            file_data=types.FileData(
                                file_uri=uploaded.uri,
                                mime_type=uploaded.mime_type or "video/mp4",
                            ),
                            video_metadata=types.VideoMetadata(fps=evidence.fps),
                        ),
                        types.Part(text=prompt),
                    ]
                ),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TaskTemplateDraft,
                    temperature=0.2,
                ),
            )
            if getattr(response, "parsed", None) is not None:
                return TaskTemplateDraft.model_validate(response.parsed)
            if not response.text:
                raise RuntimeError("Gemini returned an empty task analysis.")
            return TaskTemplateDraft.model_validate_json(response.text)
        finally:
            name = getattr(uploaded, "name", None)
            if name:
                try:
                    client.files.delete(name=name)
                except Exception:
                    logger.warning("Could not delete uploaded Gemini evidence file.", exc_info=True)

    def _wait_until_ready(self, client: Any, uploaded: Any) -> Any:
        deadline = time.monotonic() + self.processing_timeout_seconds
        while _file_state(uploaded) == "processing":
            if time.monotonic() >= deadline:
                raise TimeoutError("Gemini video processing timed out.")
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if _file_state(uploaded) in {"failed", "error"}:
            raise RuntimeError("Gemini could not process the evidence video.")
        return uploaded


class TaskTemplateStore:
    """Persist model drafts and require a separate user action for approval."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def get(self, episode_id: str) -> TaskTemplateRecord | None:
        path = self._path(episode_id)
        if not path.is_file():
            return None
        return TaskTemplateRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_records(self, *, approved_only: bool = False) -> list[TaskTemplateRecord]:
        if not self.root.is_dir():
            return []
        records = [
            TaskTemplateRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.root.glob("*.json"))
            if path.name != "approval-history.json"
        ]
        if approved_only:
            records = [record for record in records if record.approval_status == "approved"]
        return records

    def create_model_draft(
        self,
        episode: EpisodeRecord,
        draft: TaskTemplateDraft,
        *,
        model_version: str,
        video_fps: float,
    ) -> TaskTemplateRecord:
        now = _now()
        record = TaskTemplateRecord(
            **draft.model_dump(),
            template_version=1,
            source_episode=SourceEpisode(
                episode_id=episode.episode_id,
                dataset_episode_index=episode.dataset_episode_index,
                project_name=episode.project_name,
            ),
            model_version=model_version,
            video_fps=video_fps,
            approval_status="draft",
            created_at=now,
            updated_at=now,
        )
        self._write(record)
        self._append_history(record, "model_draft_created")
        return record

    def save_user_draft(self, episode_id: str, draft: TaskTemplateDraft) -> TaskTemplateRecord:
        existing = self._required(episode_id)
        record = TaskTemplateRecord(
            **draft.model_dump(),
            template_version=existing.template_version + 1,
            source_episode=existing.source_episode,
            model_version=existing.model_version,
            video_fps=existing.video_fps,
            approval_status="draft",
            created_at=existing.created_at,
            updated_at=_now(),
            approved_at=None,
        )
        self._write(record)
        self._append_history(record, "user_draft_saved")
        return record

    def approve(self, episode_id: str) -> TaskTemplateRecord:
        existing = self._required(episode_id)
        now = _now()
        record = existing.model_copy(
            update={"approval_status": "approved", "updated_at": now, "approved_at": now}
        )
        self._write(record)
        self._append_history(record, "user_approved")
        return record

    def _required(self, episode_id: str) -> TaskTemplateRecord:
        record = self.get(episode_id)
        if record is None:
            raise KeyError(f"No task analysis exists for episode: {episode_id}")
        return record

    def _write(self, record: TaskTemplateRecord) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(record.source_episode.episode_id).write_text(
            record.model_dump_json(indent=2), encoding="utf-8"
        )

    def _path(self, episode_id: str) -> Path:
        return self.root / f"{_safe_part(episode_id)}.json"

    def _append_history(self, record: TaskTemplateRecord, event: str) -> None:
        import hashlib

        self.root.mkdir(parents=True, exist_ok=True)
        content = record.model_dump_json()
        history = {
            "event_id": str(uuid4()),
            "event": event,
            "episode_id": record.source_episode.episode_id,
            "approval_status": record.approval_status,
            "model_version": record.model_version,
            "timestamp": record.updated_at,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
        with (self.root / "approval-history.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(history, sort_keys=True) + "\n")


class TaskAnalysisService:
    def __init__(
        self,
        evidence_builder: EpisodeEvidenceBuilder,
        analyzer: TaskAnalysisProvider,
        store: TaskTemplateStore,
    ) -> None:
        self.evidence_builder = evidence_builder
        self.analyzer = analyzer
        self.store = store

    def analyze(self, episode: EpisodeRecord) -> TaskTemplateRecord:
        evidence = self.evidence_builder.build(episode, fps=ANALYSIS_FPS)
        draft = self.analyzer.analyze(episode, evidence)
        return self.store.create_model_draft(
            episode,
            draft,
            model_version=self.analyzer.model,
            video_fps=evidence.fps,
        )


@dataclass
class TaskAnalysisJob:
    job_id: str
    episode_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class TaskAnalysisJobManager:
    def __init__(self, service: TaskAnalysisService) -> None:
        self.service = service
        self._jobs: dict[str, TaskAnalysisJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="machanize-gemini")
        self._lock = RLock()

    def submit(
        self,
        episode: EpisodeRecord,
        *,
        confirm_unknown_as_success: bool = False,
    ) -> TaskAnalysisJob:
        outcome = episode.outcome.lower()
        if outcome == "failure":
            raise ValueError("A recorded failure cannot be used as the successful demonstration.")
        if outcome != "success" and not confirm_unknown_as_success:
            raise ValueError(
                "Confirm that this unknown-outcome episode is a successful demonstration."
            )
        analysis_episode = replace(episode, outcome="success") if outcome != "success" else episode
        job = TaskAnalysisJob(str(uuid4()), episode.episode_id, "queued", _now())
        with self._lock:
            self._jobs[job.job_id] = job
        self._executor.submit(self._run, job.job_id, analysis_episode)
        return job

    def to_api(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return asdict(self._jobs[job_id])
            except KeyError as error:
                raise KeyError(f"Unknown task-analysis job: {job_id}") from error

    def _run(self, job_id: str, episode: EpisodeRecord) -> None:
        with self._lock:
            self._jobs[job_id].status = "running"
        try:
            result = self.service.analyze(episode)
        except Exception as error:  # noqa: BLE001 - surface background failures in the GUI
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(error)
                job.completed_at = _now()
            return
        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.result = result.model_dump(mode="json")
            job.completed_at = _now()


def _analysis_prompt(episode: EpisodeRecord, evidence: EpisodeEvidence) -> str:
    telemetry = json.dumps(evidence.telemetry, separators=(",", ":"))
    return f"""
Analyze this successful robot demonstration as evidence for a reusable task template.
This is analysis only: do not issue robot commands and do not claim that any model was trained.

The side-by-side video shows FRONT on the left and WRIST on the right. It is sampled at
{evidence.fps:.3f} FPS. Each frame includes the task, timestamp, joint observations, recorded
actions, and proposed actions when available.

Task description: {episode.task}
Source episode: {episode.episode_id}

Return a detailed structured template. Infer ordered stages such as approaching, grasping,
grasped, carrying, approaching destination, placing, and success when supported by evidence.
Include expected object relationships, robot and gripper behavior, success conditions, possible
failures (including missed grasp, object dropped, wrong direction, stuck, incorrect placement,
and success condition not reached where relevant), timestamped evidence, confidence, and explicit
uncertainty. Use seconds from the video. Do not invent evidence that is not visible or present in
the telemetry. This output is always a draft requiring human approval.

Sampled telemetry JSON:
{telemetry}
""".strip()


def _telemetry_row(sample: Any, names: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "timestamp_seconds": float(_scalar(sample.get("timestamp", 0))),
        "frame_index": int(_scalar(sample.get("frame_index", 0))),
        "joint_observations": _named_values(
            sample.get("observation.state"), names.get("observation.state", [])
        ),
        "recorded_actions": _named_values(sample.get("action"), names.get("action", [])),
        "proposed_actions": _named_values(
            sample.get("machanize.proposed_action"),
            names.get("machanize.proposed_action", []),
        ),
    }


def _named_values(value: Any, names: list[str]) -> dict[str, float] | None:
    if value is None:
        return None
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    values = np.asarray(array).reshape(-1).tolist()
    labels = (
        names if len(names) == len(values) else [f"value_{index}" for index in range(len(values))]
    )
    return {name: round(float(item), 5) for name, item in zip(labels, values, strict=True)}


def _compose_frame(
    front: Image.Image,
    wrist: Image.Image,
    *,
    task: str,
    telemetry: dict[str, Any],
) -> Image.Image:
    view_size = (640, 480)
    front = ImageOps.fit(front.convert("RGB"), view_size)
    wrist = ImageOps.fit(wrist.convert("RGB"), view_size)
    canvas = Image.new("RGB", (1280, 620), "#080a0c")
    canvas.paste(front, (0, 28))
    canvas.paste(wrist, (640, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "FRONT", fill="#72f1b8")
    draw.text((652, 8), "WRIST", fill="#72f1b8")
    draw.text((12, 518), f"Task: {task}", fill="white")
    draw.text(
        (12, 540),
        f"t={telemetry['timestamp_seconds']:.3f}s  frame={telemetry['frame_index']}",
        fill="#ffcf66",
    )
    draw.text(
        (12, 562), f"Joints: {_compact_values(telemetry['joint_observations'])}", fill="#cdd3d0"
    )
    draw.text(
        (12, 584), f"Action: {_compact_values(telemetry['recorded_actions'])}", fill="#cdd3d0"
    )
    return canvas


def _compact_values(values: dict[str, float] | None) -> str:
    if not values:
        return "unavailable"
    return "  ".join(f"{key}={value:.2f}" for key, value in values.items())


def _encode_mp4(frame_directory: Path, destination: Path, fps: float) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required to create the side-by-side evidence video.")
    subprocess.run(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(frame_directory / "frame-%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


def _camera_key(camera_keys: tuple[str, ...], name: str) -> str:
    matches = [key for key in camera_keys if key == name or key.endswith(f".{name}")]
    if len(matches) != 1:
        raise ValueError(f"The selected episode must contain exactly one {name} camera stream.")
    return matches[0]


def _to_image(value: Any) -> Image.Image:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(array.max()) <= 1:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _file_state(file: Any) -> str:
    state = getattr(file, "state", "active")
    name = getattr(state, "name", None)
    return str(name or state).rsplit(".", 1)[-1].lower()


def _safe_part(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "_.-" else "_" for character in value
    )
    return safe.strip("._") or "episode"


def _now() -> str:
    return datetime.now(UTC).isoformat()
