"""Bridge that automatically records actions sent through LeRobot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from machanize.adapters.lerobot import LeRobotAdapter
from machanize.recording.recorder import EpisodeOutcome, EpisodeRecorder


@dataclass(frozen=True)
class RecordedStep:
    observation: Mapping[str, Any]
    proposed_action: Mapping[str, Any]
    executed_action: Mapping[str, Any]


class MachanizeLeRobotBridge:
    """Connect, control, and record a LeRobot robot through Machanize."""

    def __init__(
        self,
        adapter: LeRobotAdapter,
        recorder: EpisodeRecorder,
        *,
        task: str,
    ) -> None:
        self.adapter = adapter
        self.recorder = recorder
        self.task = task

    def connect(self) -> None:
        self.adapter.connect()

    def start_episode(self, metadata: Mapping[str, Any] | None = None) -> str:
        return self.recorder.start_episode(task=self.task, metadata=metadata)

    def step(
        self,
        proposed_action: Mapping[str, Any],
        *,
        observation: Mapping[str, Any] | None = None,
    ) -> RecordedStep:
        current_observation = self.adapter.observe() if observation is None else observation
        executed_action = self.adapter.execute(proposed_action)
        self.recorder.record_step(
            observation=current_observation,
            proposed_action=proposed_action,
            executed_action=executed_action,
        )
        return RecordedStep(current_observation, proposed_action, executed_action)

    def finish_episode(
        self,
        *,
        outcome: EpisodeOutcome,
        failure_type: str | None = None,
        notes: str | None = None,
    ) -> Path:
        return self.recorder.finish_episode(
            outcome=outcome,
            failure_type=failure_type,
            notes=notes,
        )

    def abort_episode(self) -> None:
        self.recorder.abort_episode()

    def close(self) -> None:
        try:
            self.recorder.close()
        finally:
            self.adapter.disconnect()
