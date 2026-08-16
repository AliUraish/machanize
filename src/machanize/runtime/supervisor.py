"""Rate-limited runtime monitoring orchestration and local fail-safe behavior."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from machanize.decision import DecisionGate, DecisionGateConfig, GateResult
from machanize.providers.base import MonitoringConnection, MonitoringProvider, ProviderCallbacks
from machanize.runtime.schemas import (
    ConnectionState,
    RobotStateReport,
    RuntimeDecision,
    RuntimeMode,
    RuntimeSample,
    RuntimeSessionRecord,
)
from machanize.runtime.store import RuntimeStore
from machanize.safety import StopLatch


@dataclass(frozen=True)
class RuntimeSupervisorConfig:
    model_id: str = "gemini-3.1-flash-live-preview"
    max_video_fps: float = 5.0
    cloud_timeout_seconds: float = 4.0
    active_enabled: bool = False
    stop_on_cloud_failure: bool = True
    decision_gate: DecisionGateConfig = field(default_factory=DecisionGateConfig)

    def __post_init__(self) -> None:
        if not 0 < self.max_video_fps <= 5:
            raise ValueError("Runtime monitoring rate must be greater than zero and at most 5 FPS.")
        if self.cloud_timeout_seconds <= 0:
            raise ValueError("Cloud timeout must be positive.")


class RuntimeSupervisor:
    def __init__(
        self,
        provider: MonitoringProvider,
        store: RuntimeStore,
        stop_latch: StopLatch,
        config: RuntimeSupervisorConfig,
        decision_sink: Callable[[RuntimeDecision], None] | None = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.stop_latch = stop_latch
        self.config = config
        self.decision_sink = decision_sink
        self.gate = DecisionGate(config.decision_gate)
        self.record: RuntimeSessionRecord | None = None
        self._template: dict[str, Any] | None = None
        self._stage_names: set[str] = set()
        self._failure_types: set[str] = set()
        self._connection: MonitoringConnection | None = None
        self._latest_sample: RuntimeSample | None = None
        self._awaiting_sample: RuntimeSample | None = None
        self._sent_monotonic: float | None = None
        self._last_dispatch = 0.0
        self._last_valid_decision_monotonic: float | None = None
        self._latest_decision: RuntimeDecision | None = None
        self._dispatch_event = asyncio.Event()
        self._closed = asyncio.Event()
        self._dispatch_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None

    async def create_session(self, approved_template: Mapping[str, Any]) -> RuntimeSessionRecord:
        if str(approved_template.get("approval_status")) != "approved":
            raise ValueError("Runtime monitoring requires an approved task template.")
        if self.record is not None and self.record.stopped_at is None:
            raise RuntimeError("A runtime session is already active.")
        template = json.loads(json.dumps(approved_template))
        source = template.get("source_episode", {})
        episode_id = str(source.get("episode_id", ""))
        if not episode_id:
            raise ValueError("Approved task template has no source episode.")
        revision = hashlib.sha256(
            json.dumps(template, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        now = _now()
        self.record = RuntimeSessionRecord(
            session_id=str(uuid4()),
            template_episode_id=episode_id,
            template_version=int(template.get("template_version", 1)),
            template_revision=revision,
            model_id=self.provider.model_id,
            created_at=now,
            stop_latched=self.stop_latch.is_latched,
            stop_reason=self.stop_latch.reason,
            thresholds={
                **asdict(self.config.decision_gate),
                "cloud_timeout_seconds": self.config.cloud_timeout_seconds,
                "max_video_fps": self.config.max_video_fps,
                "stop_on_cloud_failure": self.config.stop_on_cloud_failure,
            },
        )
        self._template = template
        self._stage_names = {
            str(stage.get("name", "")).strip().lower()
            for stage in template.get("ordered_task_stages", [])
        }
        self._failure_types = {
            str(failure.get("failure_type", "")).strip().lower()
            for failure in template.get("possible_failure_types", [])
        }
        self.gate.reset()
        self._latest_sample = None
        self._awaiting_sample = None
        self._sent_monotonic = None
        self._last_dispatch = 0.0
        self._last_valid_decision_monotonic = None
        self._latest_decision = None
        self._closed = asyncio.Event()
        self._dispatch_event = asyncio.Event()
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        self.store.save_session(self.record)
        return self.record

    async def set_mode(
        self,
        mode: RuntimeMode,
        *,
        confirm_active: bool = False,
    ) -> RuntimeSessionRecord:
        record = self._required_record()
        if record.stopped_at is not None:
            raise RuntimeError("Runtime session has already stopped.")
        if mode is RuntimeMode.ACTIVE:
            if not self.config.active_enabled:
                raise PermissionError("ACTIVE mode is disabled by backend configuration.")
            if not confirm_active:
                raise PermissionError("ACTIVE mode requires explicit operator confirmation.")
        if mode is RuntimeMode.OFF:
            await self._close_provider()
            self.gate.reset()
            self._update(mode=mode, connection_state=ConnectionState.OFF)
        else:
            if self._connection is None:
                await self._open_provider()
            self._update(
                mode=mode,
                started_at=record.started_at or _now(),
            )
            self._dispatch_event.set()
        self._event("mode_changed", {"mode": mode.value})
        return self._required_record()

    async def submit_sample(self, sample: RuntimeSample) -> None:
        record = self._required_record()
        if sample.session_id != record.session_id:
            raise ValueError("Runtime sample belongs to a different session.")
        if record.mode is RuntimeMode.OFF:
            return
        self._latest_sample = sample
        self._update(last_sample_at=sample.timestamp)
        self._dispatch_event.set()

    async def stop(self) -> RuntimeSessionRecord:
        self._required_record()
        await self._close_provider()
        self._closed.set()
        for task in (self._dispatch_task, self._watchdog_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._dispatch_task, self._watchdog_task) if task is not None),
            return_exceptions=True,
        )
        self._update(
            mode=RuntimeMode.OFF,
            connection_state=ConnectionState.OFF,
            stopped_at=_now(),
        )
        self._event("session_stopped", {})
        return self._required_record()

    def status(self) -> RuntimeSessionRecord:
        return self._required_record()

    @property
    def latest_decision(self) -> RuntimeDecision | None:
        return self._latest_decision

    def has_fresh_decision(self, *, max_age_seconds: float | None = None) -> bool:
        if self._last_valid_decision_monotonic is None:
            return False
        max_age = max_age_seconds or self.config.cloud_timeout_seconds
        return time.monotonic() - self._last_valid_decision_monotonic <= max_age

    def clear_stop_latch_by_operator(self) -> RuntimeSessionRecord:
        record = self._required_record()
        if record.mode is not RuntimeMode.OFF:
            raise RuntimeError("Switch to OFF before clearing stop.")
        self.stop_latch.clear_by_operator()
        self._update(stop_latched=False, stop_reason=None)
        self._event("stop_latch_cleared", {"actor": "local_operator"})
        return self._required_record()

    async def _open_provider(self) -> None:
        if self._template is None:
            raise RuntimeError("Runtime task template is unavailable.")
        callbacks = ProviderCallbacks(
            on_report=self._on_report,
            on_status=self._on_status,
            on_malformed=self._on_malformed,
        )
        try:
            self._connection = await self.provider.connect(
                approved_template=self._template,
                callbacks=callbacks,
            )
        except Exception as error:
            await self._on_status(ConnectionState.ERROR, type(error).__name__)
            raise

    async def _close_provider(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()
        self._awaiting_sample = None
        self._sent_monotonic = None

    async def _dispatch_loop(self) -> None:
        interval = 1 / self.config.max_video_fps
        while not self._closed.is_set():
            await self._dispatch_event.wait()
            self._dispatch_event.clear()
            if self._awaiting_sample is not None or self._latest_sample is None:
                continue
            record = self._required_record()
            if record.mode is RuntimeMode.OFF or self._connection is None:
                continue
            delay = interval - (time.monotonic() - self._last_dispatch)
            if delay > 0:
                await asyncio.sleep(delay)
            record = self._required_record()
            if record.mode is RuntimeMode.OFF or self._connection is None:
                continue
            sample, self._latest_sample = self._latest_sample, None
            self._awaiting_sample = sample
            self._sent_monotonic = time.monotonic()
            self._last_dispatch = self._sent_monotonic
            try:
                await self._connection.send(sample)
            except Exception as error:  # noqa: BLE001 - provider boundary reports failure locally
                await self._provider_failure(f"send_failed:{type(error).__name__}")

    async def _watchdog_loop(self) -> None:
        while not self._closed.is_set():
            await asyncio.sleep(0.1)
            if self._awaiting_sample is None or self._sent_monotonic is None:
                continue
            if time.monotonic() - self._sent_monotonic > self.config.cloud_timeout_seconds:
                await self._provider_failure("cloud_decision_timeout")

    async def _on_report(self, report: RobotStateReport) -> None:
        sample = self._awaiting_sample
        sent = self._sent_monotonic
        if sample is None or sent is None:
            await self._on_malformed("Received a report without an in-flight sample.")
            return
        if not self._semantic_report_is_valid(report):
            await self._on_malformed(
                "Report contains a stage or failure outside the approved template."
            )
            return
        now_monotonic = time.monotonic()
        gate_result = self.gate.evaluate(
            report,
            mode=self._required_record().mode,
            monotonic_time=now_monotonic,
        )
        safety_reason = None
        if gate_result is GateResult.STOP_REQUESTED:
            safety_reason = (
                f"Gemini repeated {report.failure_type or 'failure'} recommendation "
                f"at confidence {report.confidence:.3f}."
            )
            self.stop_latch.request(safety_reason)
        decision = RuntimeDecision(
            decision_id=str(uuid4()),
            session_id=sample.session_id,
            sample_id=sample.sample_id,
            sample_timestamp=sample.timestamp,
            received_at=_now(),
            model_id=self.provider.model_id,
            mode=self._required_record().mode,
            report=report,
            latency_ms=(now_monotonic - sent) * 1000,
            stop_streak=self.gate.stop_streak,
            local_result=gate_result.value,
            safety_reason=safety_reason,
        )
        self.store.append_decision(decision)
        self._latest_decision = decision
        self._last_valid_decision_monotonic = now_monotonic
        if self.decision_sink is not None:
            self.decision_sink(decision)
        self._awaiting_sample = None
        self._sent_monotonic = None
        self._update(
            last_decision_at=decision.received_at,
            last_latency_ms=decision.latency_ms,
            stop_latched=self.stop_latch.is_latched,
            stop_reason=self.stop_latch.reason,
        )
        self._dispatch_event.set()

    async def _on_status(self, state: ConnectionState, detail: str | None) -> None:
        if self.record is None:
            return
        self._update(connection_state=state)
        self._event("connection", {"state": state.value, "detail": detail})
        if state in {ConnectionState.DISCONNECTED, ConnectionState.ERROR}:
            await self._fail_safe_if_active(f"cloud_{state.value}")

    async def _on_malformed(self, reason: str) -> None:
        self._event("malformed_provider_output", {"reason": reason})
        self._awaiting_sample = None
        self._sent_monotonic = None
        await self._fail_safe_if_active("malformed_provider_output")
        self._dispatch_event.set()

    async def _provider_failure(self, reason: str) -> None:
        self._event("provider_failure", {"reason": reason})
        self._awaiting_sample = None
        self._sent_monotonic = None
        await self._fail_safe_if_active(reason)
        self._dispatch_event.set()

    async def _fail_safe_if_active(self, reason: str) -> None:
        record = self._required_record()
        if record.mode is RuntimeMode.ACTIVE and self.config.stop_on_cloud_failure:
            self.stop_latch.request(f"ACTIVE fail-safe: {reason}")
            self._update(
                stop_latched=self.stop_latch.is_latched,
                stop_reason=self.stop_latch.reason,
            )

    def _semantic_report_is_valid(self, report: RobotStateReport) -> bool:
        stage = report.current_stage.strip().lower()
        if stage != "unknown" and stage not in self._stage_names:
            return False
        if report.failure_type:
            failure = report.failure_type.strip().lower()
            if failure not in self._failure_types:
                return False
        return True

    def _event(self, event: str, details: dict[str, Any]) -> None:
        if self.record is None:
            return
        self.store.append_event(
            self.record.session_id,
            {"event": event, "timestamp": _now(), **details},
        )

    def _update(self, **changes: Any) -> None:
        record = self._required_record()
        self.record = record.model_copy(update=changes)
        self.store.save_session(self.record)

    def _required_record(self) -> RuntimeSessionRecord:
        if self.record is None:
            raise RuntimeError("No runtime session exists.")
        return self.record


def _now() -> str:
    return datetime.now(UTC).isoformat()
