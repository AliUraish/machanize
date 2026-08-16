"""Thread-safe live state shared by the Pi control loop and FastAPI."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any


class RuntimeStateHub:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sequence = 0
        self._frame_sequence = 0
        self._combined_jpeg: bytes | None = None
        self._state: dict[str, Any] = {
            "timestamp": None,
            "joints": {},
            "proposed_actions": {},
            "executed": False,
            "block_reason": None,
            "robot_connected": False,
            "control_loop_status": "stopped",
            "act_state": "stopped",
            "control_loop_error": None,
        }

    def update_observation(
        self,
        *,
        timestamp: str,
        joints: dict[str, float],
        proposed_actions: dict[str, float],
        combined_jpeg: bytes,
    ) -> None:
        self.update_telemetry(
            timestamp=timestamp,
            joints=joints,
            proposed_actions=proposed_actions,
        )
        self.update_frame(combined_jpeg)

    def update_telemetry(
        self,
        *,
        timestamp: str,
        joints: dict[str, float],
        proposed_actions: dict[str, float],
    ) -> None:
        with self._lock:
            self._sequence += 1
            self._state.update(
                timestamp=timestamp,
                joints=dict(joints),
                proposed_actions=dict(proposed_actions),
                executed=False,
                block_reason=None,
            )

    def update_frame(self, combined_jpeg: bytes) -> None:
        with self._lock:
            self._sequence += 1
            self._frame_sequence += 1
            self._combined_jpeg = combined_jpeg

    def update_gate(self, *, executed: bool, block_reason: str | None) -> None:
        with self._lock:
            self._sequence += 1
            self._state.update(executed=executed, block_reason=block_reason)

    def update_robot(self, *, connected: bool, status: str, error: str | None = None) -> None:
        with self._lock:
            self._sequence += 1
            self._state.update(
                robot_connected=connected,
                control_loop_status=status,
                act_state=status,
                control_loop_error=error,
            )

    def snapshot(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            return self._sequence, deepcopy(self._state)

    def latest_frame(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._frame_sequence, self._combined_jpeg
