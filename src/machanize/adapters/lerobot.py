"""Small compatibility layer around LeRobot's robot interface."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Any, Protocol, runtime_checkable

RobotData = Mapping[str, Any]


@runtime_checkable
class LeRobotLike(Protocol):
    """The subset of LeRobot used by Machanize."""

    @property
    def is_connected(self) -> bool: ...

    @property
    def observation_features(self) -> Mapping[str, Any]: ...

    @property
    def action_features(self) -> Mapping[str, Any]: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_observation(self) -> RobotData: ...

    def send_action(self, action: RobotData) -> RobotData | None: ...


class LeRobotAdapter:
    """Thread-safe access to a LeRobot-compatible robot instance."""

    def __init__(self, robot: LeRobotLike) -> None:
        self.robot = robot
        self._lock = RLock()

    @property
    def is_connected(self) -> bool:
        return bool(self.robot.is_connected)

    @property
    def observation_features(self) -> Mapping[str, Any]:
        return _resolve_features(self.robot.observation_features)

    @property
    def action_features(self) -> Mapping[str, Any]:
        return _resolve_features(self.robot.action_features)

    def connect(self) -> None:
        with self._lock:
            if not self.is_connected:
                self.robot.connect()

    def disconnect(self) -> None:
        with self._lock:
            if self.is_connected:
                self.robot.disconnect()

    def observe(self) -> RobotData:
        with self._lock:
            self._require_connection()
            return self.robot.get_observation()

    def execute(self, action: RobotData) -> RobotData:
        with self._lock:
            self._require_connection()
            executed_action = self.robot.send_action(action)
            return action if executed_action is None else executed_action

    def _require_connection(self) -> None:
        if not self.is_connected:
            raise ConnectionError("LeRobot is not connected.")


def _resolve_features(features: Any) -> Mapping[str, Any]:
    """Support LeRobot releases exposing features as properties or callables."""

    resolved = features() if callable(features) else features
    if not isinstance(resolved, Mapping):
        raise TypeError("LeRobot features must be a mapping.")
    return resolved
