"""Native-rate LeRobot control loop owned by the Raspberry Pi runtime."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from threading import Event, RLock, Thread
from typing import Any, Protocol

from machanize.adapters import LeRobotAdapter
from machanize.runtime.integration import RuntimeBridgeHook, SafeStopRequested
from machanize.runtime.schemas import ACTState
from machanize.runtime.state import RuntimeStateHub
from machanize.safety import StopLatch


class ActionSource(Protocol):
    """Local ACT/teleoperation boundary. Cloud providers never implement this interface."""

    def start(self) -> None: ...

    def propose(self, observation: Mapping[str, Any]) -> Mapping[str, Any] | None: ...

    def stop(self) -> None: ...


class CallableActionSource:
    def __init__(
        self,
        propose: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
    ) -> None:
        self._propose = propose

    def start(self) -> None:
        return None

    def propose(self, observation: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return self._propose(observation)

    def stop(self) -> None:
        return None


class PiControlLoop:
    """Observe and gate every proposed action at native FPS before adapter.execute()."""

    def __init__(
        self,
        adapter: LeRobotAdapter,
        action_source: ActionSource,
        runtime_hook: RuntimeBridgeHook,
        state: RuntimeStateHub,
        stop_latch: StopLatch,
        *,
        control_fps: float = 30,
    ) -> None:
        if control_fps <= 0:
            raise ValueError("Control FPS must be positive.")
        self.adapter = adapter
        self.action_source = action_source
        self.runtime_hook = runtime_hook
        self.state = state
        self.stop_latch = stop_latch
        self.control_fps = control_fps
        self._closed = Event()
        self._act_enabled = Event()
        self._thread: Thread | None = None
        self._preview_started = False
        self._source_started = False
        self._lock = RLock()
        self._action_gate = RLock()

    @property
    def preview_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def act_running(self) -> bool:
        return self._act_enabled.is_set()

    def start_preview(self) -> None:
        with self._lock:
            if self.preview_running:
                return
            self._closed.clear()
            try:
                self.adapter.connect()
            except BaseException as error:
                self.state.update_robot(
                    connected=self.adapter.is_connected,
                    status=ACTState.ERROR.value,
                    error=f"Robot preview startup failed: {type(error).__name__}",
                )
                raise
            self.state.update_robot(connected=True, status=ACTState.READY.value)
            self._preview_started = True
            self._thread = Thread(
                target=self._run,
                name="machanize-pi-control-loop",
                daemon=True,
            )
            self._thread.start()

    def start_act(self) -> ACTState:
        with self._lock:
            if not self.preview_running or not self.adapter.is_connected:
                raise RuntimeError("Robot preview is not ready.")
            if self.stop_latch.is_latched:
                raise RuntimeError(
                    self.stop_latch.reason or "Clear the stop latch before starting ACT."
                )
            if self._act_enabled.is_set():
                return ACTState.RUNNING
            try:
                self.action_source.start()
            except BaseException as error:
                self.state.update_robot(
                    connected=self.adapter.is_connected,
                    status=ACTState.ERROR.value,
                    error=f"ACT startup failed: {type(error).__name__}",
                )
                raise
            self._source_started = True
            self.runtime_hook.control_started()
            self._act_enabled.set()
            self.state.update_robot(connected=True, status=ACTState.RUNNING.value)
            return ACTState.RUNNING

    def stop_act(self) -> ACTState:
        self._act_enabled.clear()
        with self._action_gate:
            self.runtime_hook.control_stopped()
        with self._lock:
            try:
                if self._source_started:
                    self.action_source.stop()
            except BaseException as error:
                self.state.update_robot(
                    connected=self.adapter.is_connected,
                    status=ACTState.ERROR.value,
                    error=f"ACT shutdown failed: {type(error).__name__}",
                )
                raise
            finally:
                self._source_started = False
            self.state.update_gate(executed=False, block_reason="ACT is stopped")
            self.state.update_robot(
                connected=self.adapter.is_connected,
                status=ACTState.STOPPED.value,
            )
            return ACTState.STOPPED

    def close(self) -> None:
        self._act_enabled.clear()
        with self._lock:
            self._closed.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=2)
        try:
            self.runtime_hook.close()
        finally:
            if self._source_started:
                try:
                    self.action_source.stop()
                finally:
                    self._source_started = False
            if self._preview_started:
                try:
                    self.adapter.disconnect()
                finally:
                    self._preview_started = False
            self.state.update_robot(connected=False, status=ACTState.STOPPED.value)

    def _run(self) -> None:
        interval = 1 / self.control_fps
        while not self._closed.is_set():
            started = time.monotonic()
            try:
                observation = self.adapter.observe()
                if not self._act_enabled.is_set():
                    self.runtime_hook.observe_proposal(observation, {})
                    self.state.update_gate(executed=False, block_reason="ACT is not running")
                else:
                    proposed_action = self.action_source.propose(observation)
                    self.runtime_hook.observe_proposal(
                        observation,
                        proposed_action if proposed_action is not None else {},
                    )
                    if proposed_action is None:
                        self.state.update_gate(executed=False, block_reason="No proposed action")
                    else:
                        self._execute_if_enabled(observation, proposed_action)
            except Exception as error:  # noqa: BLE001 - local failures must fail closed
                self._act_enabled.clear()
                self.runtime_hook.control_stopped()
                reason = f"Local runtime failure: {type(error).__name__}"
                self.stop_latch.request(reason)
                self.state.update_robot(
                    connected=self.adapter.is_connected,
                    status="error",
                    error=reason,
                )
                self.state.update_gate(executed=False, block_reason=reason)
                return
            self._closed.wait(max(interval - (time.monotonic() - started), 0))

    def _execute_if_enabled(
        self,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
    ) -> None:
        with self._action_gate:
            if not self._act_enabled.is_set():
                self.state.update_gate(executed=False, block_reason="ACT is stopped")
                return
            try:
                self.runtime_hook.before_action(observation, proposed_action)
            except SafeStopRequested as error:
                self.state.update_gate(executed=False, block_reason=str(error))
                return
            executed_action = self.adapter.execute(proposed_action)
            self.runtime_hook.after_action(
                observation,
                proposed_action,
                executed_action,
            )
            self.state.update_gate(executed=True, block_reason=None)
