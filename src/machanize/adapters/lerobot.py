"""Small compatibility layer around LeRobot's robot interface."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from threading import RLock
from typing import Any, Protocol, runtime_checkable

RobotData = Mapping[str, Any]
logger = logging.getLogger(__name__)


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
                try:
                    self.robot.connect()
                except BaseException as error:
                    cleanup_errors = self._cleanup_partial_connection()
                    for cleanup_error in cleanup_errors:
                        error.add_note(f"Machanize connection cleanup: {cleanup_error!r}")
                    raise

    def disconnect(self) -> None:
        with self._lock:
            if self.is_connected:
                self.robot.disconnect()
                return

            cleanup_errors = self._cleanup_partial_connection()
            if cleanup_errors:
                raise ExceptionGroup(
                    "Machanize could not completely clean up a partial robot connection.",
                    cleanup_errors,
                )

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

    def _cleanup_partial_connection(self) -> list[Exception]:
        """Best-effort cleanup when LeRobot failed partway through connect()."""

        cleanup_errors: list[Exception] = []
        bus = getattr(self.robot, "bus", None)
        if bus is not None and _is_connected(bus):
            cleanup_errors.extend(_disable_torque_and_disconnect_bus(bus))
        elif bus is None:
            try:
                self.robot.disconnect()
            except Exception as error:  # noqa: BLE001 - cleanup must tolerate device errors
                cleanup_errors.append(error)

        cameras = getattr(self.robot, "cameras", {})
        if isinstance(cameras, Mapping):
            for name, camera in cameras.items():
                if not _is_connected(camera):
                    continue
                try:
                    camera.disconnect()
                except Exception as error:
                    cleanup_errors.append(error)
                    logger.exception("Failed to disconnect partially connected camera %s.", name)

        return cleanup_errors


def _resolve_features(features: Any) -> Mapping[str, Any]:
    """Support LeRobot releases exposing features as properties or callables."""

    resolved = features() if callable(features) else features
    if not isinstance(resolved, Mapping):
        raise TypeError("LeRobot features must be a mapping.")
    return resolved


def _is_connected(component: Any) -> bool:
    try:
        return bool(component.is_connected)
    except Exception:  # noqa: BLE001 - broken state must still trigger cleanup
        # If connection state itself is broken, attempting cleanup is safer.
        return True


def _disable_torque_and_disconnect_bus(bus: Any) -> list[Exception]:
    """Disable motor torque, then close the bus even if torque disabling fails."""

    cleanup_errors: list[Exception] = []
    disable_torque = getattr(bus, "disable_torque", None)

    if callable(disable_torque):
        try:
            try:
                disable_torque(num_retry=5)
            except TypeError:
                disable_torque()
        except Exception as error:
            cleanup_errors.append(error)
            logger.exception("Failed to disable motor torque after robot connection failure.")

        try:
            bus.disconnect(disable_torque=False)
        except TypeError:
            try:
                bus.disconnect()
            except Exception as error:  # noqa: BLE001 - best-effort device cleanup
                cleanup_errors.append(error)
        except Exception as error:  # noqa: BLE001 - best-effort device cleanup
            cleanup_errors.append(error)
    else:
        try:
            bus.disconnect(disable_torque=True)
        except TypeError:
            try:
                bus.disconnect()
            except Exception as error:  # noqa: BLE001 - best-effort device cleanup
                cleanup_errors.append(error)
        except Exception as error:  # noqa: BLE001 - best-effort device cleanup
            cleanup_errors.append(error)
            try:
                bus.disconnect(disable_torque=False)
            except Exception as close_error:  # noqa: BLE001 - preserve all cleanup failures
                cleanup_errors.append(close_error)

    return cleanup_errors
