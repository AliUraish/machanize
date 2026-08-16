"""Synchronous robot-loop integration for the asynchronous runtime supervisor."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from threading import Event, Lock, Thread
from typing import Any, Protocol

import numpy as np

from machanize.media import compose_front_wrist_jpeg
from machanize.runtime.schemas import RuntimeMode, RuntimeSample
from machanize.runtime.state import RuntimeStateHub
from machanize.runtime.supervisor import RuntimeSupervisor
from machanize.safety import LocalSafetyMonitor, StopLatch, Watchdog


class SafeStopRequested(RuntimeError):
    """Raised before action execution when the local stop latch is set."""


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


class RuntimeBridgeHook:
    """Apply local safety synchronously and publish cloud samples without blocking ACT."""

    def __init__(
        self,
        supervisor: RuntimeSupervisor,
        loop: asyncio.AbstractEventLoop,
        *,
        stop_latch: StopLatch,
        local_safety: LocalSafetyMonitor | None = None,
        control_watchdog_seconds: float = 1.0,
        front_key: str = "front",
        wrist_key: str = "wrist",
        state_key: str = "observation.state",
        state_names: tuple[str, ...] = (),
        state_hub: RuntimeStateHub | None = None,
        monitor_sample_fps: float = 5.0,
    ) -> None:
        if not 0 < monitor_sample_fps <= 5:
            raise ValueError("Monitor sample FPS must be greater than zero and at most 5.")
        self.supervisor = supervisor
        self.loop = loop
        self.stop_latch = stop_latch
        self.local_safety = local_safety or LocalSafetyMonitor()
        self.front_key = front_key
        self.wrist_key = wrist_key
        self.state_key = state_key
        self.state_names = state_names
        self.state_hub = state_hub
        self.monitor_sample_fps = monitor_sample_fps
        self._sample_id = 0
        self._frame_id = 0
        self._last_monitor_sample = 0.0
        self._lock = Lock()
        self._latest_monitor_input: (
            tuple[
                Any,
                Any,
                str,
                dict[str, float],
                dict[str, float],
            ]
            | None
        ) = None
        self._monitor_event = Event()
        self._watchdog = Watchdog(control_watchdog_seconds)
        self._closed = Event()
        self._watchdog_thread = Thread(
            target=self._watchdog_loop,
            name="machanize-control-watchdog",
            daemon=True,
        )
        self._monitor_thread = Thread(
            target=self._monitor_loop,
            name="machanize-monitor-sampler",
            daemon=True,
        )
        self._watchdog_thread.start()
        self._monitor_thread.start()

    def before_action(
        self,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
    ) -> None:
        self._watchdog.heartbeat()
        violation = self.local_safety.violation(observation, proposed_action)
        if violation:
            self.stop_latch.request(violation)
        try:
            mode = self.supervisor.status().mode
        except RuntimeError:
            mode = RuntimeMode.OFF
        missing_cameras = [
            key for key in (self.front_key, self.wrist_key) if key not in observation
        ]
        if mode is RuntimeMode.ACTIVE and missing_cameras:
            self.stop_latch.request(
                f"ACTIVE input missing camera frame: {', '.join(missing_cameras)}"
            )
        if self.stop_latch.is_latched:
            raise SafeStopRequested(self.stop_latch.reason or "Safe stop requested.")
        if mode is RuntimeMode.ACTIVE and not self.supervisor.has_fresh_decision():
            self.stop_latch.request("ACTIVE fail-safe: no fresh validated monitor decision")
        if self.stop_latch.is_latched:
            raise SafeStopRequested(self.stop_latch.reason or "Safe stop requested.")

    def observe_proposal(
        self,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
    ) -> None:
        timestamp = datetime.now(UTC).isoformat()
        joints = _joint_values(
            observation,
            state_key=self.state_key,
            state_names=self.state_names,
            excluded={self.front_key, self.wrist_key},
        )
        actions = _numeric_mapping(proposed_action)
        if self.state_hub is not None:
            self.state_hub.update_telemetry(
                timestamp=timestamp,
                joints=joints,
                proposed_actions=actions,
            )
        if self.front_key not in observation or self.wrist_key not in observation:
            return
        with self._lock:
            self._latest_monitor_input = (
                _snapshot_frame(observation[self.front_key]),
                _snapshot_frame(observation[self.wrist_key]),
                timestamp,
                joints,
                actions,
            )
        self._monitor_event.set()

    def after_action(
        self,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
        executed_action: Mapping[str, Any],
    ) -> None:
        del observation, proposed_action, executed_action

    def close(self) -> None:
        self._closed.set()
        self._monitor_event.set()
        self._watchdog_thread.join(timeout=1)
        self._monitor_thread.join(timeout=1)

    def control_started(self) -> None:
        self._watchdog.heartbeat()

    def control_stopped(self) -> None:
        self._watchdog.disarm()

    def _submission_completed(self, future) -> None:
        try:
            future.result()
        except Exception:  # noqa: BLE001 - crossing thread/async boundary
            try:
                active = self.supervisor.status().mode is RuntimeMode.ACTIVE
            except RuntimeError:
                active = False
            if active:
                self.stop_latch.request("ACTIVE fail-safe: runtime sample submission failed")

    def _watchdog_loop(self) -> None:
        while not self._closed.wait(0.05):
            if self._watchdog.expired():
                self.stop_latch.request("Local control-loop watchdog expired")
                self._watchdog.disarm()

    def _monitor_loop(self) -> None:
        interval = 1 / self.monitor_sample_fps
        while not self._closed.is_set():
            self._monitor_event.wait(timeout=0.1)
            self._monitor_event.clear()
            remaining = interval - (time.monotonic() - self._last_monitor_sample)
            if remaining > 0 and self._closed.wait(remaining):
                return
            with self._lock:
                monitor_input, self._latest_monitor_input = self._latest_monitor_input, None
                frame_id = self._frame_id
                self._frame_id += 1
            if monitor_input is None:
                continue
            try:
                self._process_monitor_input(monitor_input, frame_id)
            except Exception as error:  # noqa: BLE001 - monitoring must fail closed in ACTIVE
                try:
                    active = self.supervisor.status().mode is RuntimeMode.ACTIVE
                except RuntimeError:
                    active = False
                if active:
                    self.stop_latch.request(
                        f"ACTIVE fail-safe: monitor sampler {type(error).__name__}"
                    )

    def _process_monitor_input(
        self,
        monitor_input: tuple[
            Any,
            Any,
            str,
            dict[str, float],
            dict[str, float],
        ],
        frame_id: int,
    ) -> None:
        front, wrist, timestamp, joints, actions = monitor_input
        combined_jpeg = compose_front_wrist_jpeg(
            front,
            wrist,
            sample_id=frame_id,
            timestamp=timestamp,
        )
        self._last_monitor_sample = time.monotonic()
        if self.state_hub is not None:
            self.state_hub.update_frame(combined_jpeg)
        try:
            record = self.supervisor.status()
        except RuntimeError:
            return
        if record.mode is RuntimeMode.OFF:
            return
        with self._lock:
            sample_id = self._sample_id
            self._sample_id += 1
        sample = RuntimeSample(
            session_id=record.session_id,
            sample_id=sample_id,
            timestamp=timestamp,
            monotonic_time=self._last_monitor_sample,
            joint_observations=joints,
            act_proposed_action=actions,
            combined_jpeg=combined_jpeg,
        )
        future = asyncio.run_coroutine_threadsafe(
            self.supervisor.submit_sample(sample),
            self.loop,
        )
        future.add_done_callback(self._submission_completed)


def _joint_values(
    observation: Mapping[str, Any],
    *,
    state_key: str,
    state_names: tuple[str, ...],
    excluded: set[str],
) -> dict[str, float]:
    if state_key in observation:
        values = _flatten(observation[state_key])
        names = (
            state_names
            if len(state_names) == len(values)
            else tuple(f"joint_{index}" for index in range(len(values)))
        )
        return {name: value for name, value in zip(names, values, strict=True)}
    return {
        key: float(value)
        for key, value in observation.items()
        if key not in excluded and _is_scalar(value)
    }


def _numeric_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items() if _is_scalar(value)}


def _flatten(value: Any) -> list[float]:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return [float(item) for item in array.reshape(-1).tolist()]


def _is_scalar(value: Any) -> bool:
    try:
        return np.asarray(value).ndim == 0
    except Exception:  # noqa: BLE001 - unknown observation types are ignored
        return False


def _snapshot_frame(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return array.copy()
