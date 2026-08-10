"""Stateful episode recorder and Machanize review manifest writer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from machanize.recording.sink import EpisodeSink


class EpisodeOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


@dataclass
class ActiveEpisode:
    episode_id: str
    started_at: str
    task: str
    metadata: dict[str, Any] = field(default_factory=dict)
    frame_count: int = 0


class EpisodeRecorder:
    """Record one episode at a time and queue it for later review."""

    def __init__(
        self,
        sink: EpisodeSink,
        *,
        manifest_directory: str | Path,
        project_name: str,
        robot_type: str,
    ) -> None:
        self.sink = sink
        self.manifest_directory = Path(manifest_directory)
        self.project_name = project_name
        self.robot_type = robot_type
        self._active: ActiveEpisode | None = None
        self._closed = False
        self._lock = RLock()

    @property
    def is_recording(self) -> bool:
        return self._active is not None

    def start_episode(self, *, task: str, metadata: Mapping[str, Any] | None = None) -> str:
        with self._lock:
            self._require_open()
            if self._active is not None:
                raise RuntimeError("An episode is already being recorded.")
            self._active = ActiveEpisode(
                episode_id=str(uuid4()),
                started_at=_now(),
                task=task,
                metadata=dict(metadata or {}),
            )
            return self._active.episode_id

    def record_step(
        self,
        *,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
        executed_action: Mapping[str, Any],
    ) -> None:
        with self._lock:
            active = self._require_active()
            self.sink.add_frame(observation, proposed_action, executed_action, active.task)
            active.frame_count += 1

    def finish_episode(
        self,
        *,
        outcome: EpisodeOutcome,
        failure_type: str | None = None,
        notes: str | None = None,
    ) -> Path:
        with self._lock:
            active = self._require_active()
            if active.frame_count == 0:
                raise RuntimeError("Cannot finish an episode with no recorded frames.")
            dataset_episode_index = self.sink.save_episode()
            manifest = {
                **asdict(active),
                "finished_at": _now(),
                "dataset_episode_index": dataset_episode_index,
                "project_name": self.project_name,
                "robot_type": self.robot_type,
                "outcome": outcome.value,
                "failure_type": failure_type,
                "notes": notes,
                "review_status": "pending",
                "processing_status": "pending",
            }
            path = self._write_manifest(active.episode_id, manifest)
            self._active = None
            return path

    def abort_episode(self) -> None:
        with self._lock:
            self._require_active()
            self.sink.clear_episode()
            self._active = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._active is not None:
                raise RuntimeError(
                    "Finish or abort the active episode before closing the recorder."
                )
            self.sink.finalize()
            self._closed = True

    def _write_manifest(self, episode_id: str, manifest: dict[str, Any]) -> Path:
        self.manifest_directory.mkdir(parents=True, exist_ok=True)
        path = self.manifest_directory / f"{episode_id}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _require_active(self) -> ActiveEpisode:
        self._require_open()
        if self._active is None:
            raise RuntimeError("No episode is currently being recorded.")
        return self._active

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The episode recorder is closed.")


def _now() -> str:
    return datetime.now(UTC).isoformat()
