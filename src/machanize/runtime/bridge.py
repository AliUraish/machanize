"""Bridge that automatically records actions sent through LeRobot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from machanize.adapters.lerobot import LeRobotAdapter
from machanize.recording.recorder import EpisodeOutcome, EpisodeRecorder


class RuntimeBridgeHookLike(Protocol):
    def before_action(
        self,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
    ) -> None: ...

    def after_action(
        self,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
        executed_action: Mapping[str, Any],
    ) -> None: ...


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
        runtime_hook: RuntimeBridgeHookLike | None = None,
    ) -> None:
        self.adapter = adapter
        self.recorder = recorder
        self.task = task
        self.runtime_hook = runtime_hook

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
        if self.runtime_hook is not None:
            observe_proposal = getattr(self.runtime_hook, "observe_proposal", None)
            if callable(observe_proposal):
                observe_proposal(current_observation, proposed_action)
            self.runtime_hook.before_action(current_observation, proposed_action)
        executed_action = self.adapter.execute(proposed_action)
        self.recorder.record_step(
            observation=current_observation,
            proposed_action=proposed_action,
            executed_action=executed_action,
        )
        if self.runtime_hook is not None:
            self.runtime_hook.after_action(
                current_observation,
                proposed_action,
                executed_action,
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
            try:
                close_runtime = getattr(self.runtime_hook, "close", None)
                if callable(close_runtime):
                    close_runtime()
            finally:
                self.adapter.disconnect()
