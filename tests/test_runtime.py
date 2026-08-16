from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from machanize.analysis.task_template import (
    PossibleFailure,
    TaskStage,
    TaskTemplateDraft,
    TaskTemplateStore,
    TimestampEvidence,
)
from machanize.decision import DecisionGate, DecisionGateConfig, GateResult
from machanize.perception.episodes import EpisodeRecord
from machanize.providers.base import ProviderCallbacks
from machanize.providers.gemini_live import GeminiLiveConnection
from machanize.runtime.api import RuntimeSettings, create_runtime_app
from machanize.runtime.integration import RuntimeBridgeHook, SafeStopRequested
from machanize.runtime.schemas import (
    ConnectionState,
    DecisionEvidence,
    RobotStateReport,
    RuntimeMode,
    RuntimeSample,
)
from machanize.runtime.store import RuntimeStore
from machanize.runtime.supervisor import RuntimeSupervisor, RuntimeSupervisorConfig
from machanize.safety import StopLatch
from machanize.safety.watchdog import Watchdog


class FakeStopController:
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def request_safe_stop(self, reason: str) -> None:
        self.reasons.append(reason)


class FakeConnection:
    def __init__(self, callbacks: ProviderCallbacks) -> None:
        self.callbacks = callbacks
        self.samples: list[RuntimeSample] = []
        self.closed = False

    async def send(self, sample: RuntimeSample) -> None:
        self.samples.append(sample)

    async def close(self) -> None:
        self.closed = True
        await self.callbacks.on_status(ConnectionState.OFF, None)


class FakeProvider:
    model_id = "fake-live-model"

    def __init__(self) -> None:
        self.templates: list[dict] = []
        self.connection: FakeConnection | None = None

    async def connect(self, *, approved_template, callbacks) -> FakeConnection:
        self.templates.append(dict(approved_template))
        self.connection = FakeConnection(callbacks)
        await callbacks.on_status(ConnectionState.CONNECTED, None)
        return self.connection

    async def report(self, report: RobotStateReport) -> None:
        assert self.connection is not None
        await self.connection.callbacks.on_report(report)

    async def disconnect(self) -> None:
        assert self.connection is not None
        await self.connection.callbacks.on_status(ConnectionState.DISCONNECTED, "test")


def _template() -> dict:
    return {
        "task_description": "Place the blue object in the glass.",
        "ordered_task_stages": [{"name": "carrying"}, {"name": "success"}],
        "success_conditions": ["Object is inside glass."],
        "possible_failure_types": [{"failure_type": "object dropped"}],
        "important_timestamps_and_evidence": [],
        "confidence": 0.9,
        "uncertainty": [],
        "source_episode": {
            "episode_id": "episode-1",
            "dataset_episode_index": 0,
            "project_name": "demo",
        },
        "model_version": "gemini-robotics-er-1.6-preview",
        "video_fps": 5,
        "approval_status": "approved",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "approved_at": "2026-01-01T00:00:00Z",
    }


def _report(*, recommend_stop: bool = False, confidence: float = 0.9) -> RobotStateReport:
    return RobotStateReport(
        current_stage="carrying",
        progress=0.5,
        correct=not recommend_stop,
        failure_type="object dropped" if recommend_stop else None,
        confidence=confidence,
        evidence=[DecisionEvidence(timestamp="2026-01-01T00:00:01Z", description="Visible state")],
        recommend_stop=recommend_stop,
    )


def _sample(session_id: str, sample_id: int) -> RuntimeSample:
    return RuntimeSample(
        session_id=session_id,
        sample_id=sample_id,
        timestamp=f"2026-01-01T00:00:0{sample_id}Z",
        monotonic_time=time.monotonic(),
        joint_observations={"joint": 1.0},
        act_proposed_action={"joint": 1.1},
        combined_jpeg=b"jpeg",
    )


def test_decision_gate_requires_repeated_matching_stop_reports() -> None:
    gate = DecisionGate(DecisionGateConfig(stop_threshold=0.85, consecutive_stop_predictions=3))

    assert (
        gate.evaluate(_report(recommend_stop=True), mode=RuntimeMode.ACTIVE, monotonic_time=1)
        is GateResult.ALERT
    )
    assert (
        gate.evaluate(_report(recommend_stop=True), mode=RuntimeMode.ACTIVE, monotonic_time=2)
        is GateResult.ALERT
    )
    assert (
        gate.evaluate(_report(recommend_stop=True), mode=RuntimeMode.ACTIVE, monotonic_time=3)
        is GateResult.STOP_REQUESTED
    )
    assert gate.stop_streak == 3
    assert (
        gate.evaluate(_report(confidence=0.5), mode=RuntimeMode.ACTIVE, monotonic_time=4)
        is GateResult.CONTINUE
    )
    assert gate.stop_streak == 0


def test_correct_high_confidence_report_is_not_an_alert() -> None:
    gate = DecisionGate(DecisionGateConfig())

    assert (
        gate.evaluate(_report(confidence=0.99), mode=RuntimeMode.MONITOR, monotonic_time=1)
        is GateResult.CONTINUE
    )


def test_control_watchdog_is_disarmed_until_first_heartbeat() -> None:
    watchdog = Watchdog(1)

    assert not watchdog.expired(now=100)
    watchdog.heartbeat(now=100)
    assert not watchdog.expired(now=101)
    assert watchdog.expired(now=101.01)


def test_active_missing_camera_blocks_before_action() -> None:
    loop = asyncio.new_event_loop()
    controller = FakeStopController()
    latch = StopLatch(controller)
    supervisor = SimpleNamespace(
        status=lambda: SimpleNamespace(mode=RuntimeMode.ACTIVE),
    )
    hook = RuntimeBridgeHook(
        supervisor,
        loop,
        stop_latch=latch,
        front_key="front",
        wrist_key="wrist",
    )
    try:
        try:
            hook.before_action({"front": object()}, {"joint": 0.0})
        except SafeStopRequested as error:
            assert "wrist" in str(error)
        else:
            raise AssertionError("ACTIVE must block an action when a runtime camera is absent.")
    finally:
        hook.close()
        loop.close()

    assert controller.reasons == ["ACTIVE input missing camera frame: wrist"]


def test_runtime_supervisor_rate_limits_and_monitor_never_stops(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = FakeProvider()
        controller = FakeStopController()
        supervisor = RuntimeSupervisor(
            provider,
            RuntimeStore(tmp_path / "runtime"),
            StopLatch(controller),
            RuntimeSupervisorConfig(),
        )
        session = await supervisor.create_session(_template())
        await supervisor.set_mode(RuntimeMode.MONITOR)
        for sample_id in range(4):
            await supervisor.submit_sample(_sample(session.session_id, sample_id))
        await asyncio.sleep(0.03)
        assert provider.connection is not None
        assert len(provider.connection.samples) == 1
        await provider.report(_report(recommend_stop=True))
        await asyncio.sleep(0.03)
        assert controller.reasons == []
        decisions = supervisor.store.list_decisions(session.session_id)
        assert decisions[0].local_result == "alert"
        await supervisor.stop()

    asyncio.run(scenario())


def test_active_stops_after_three_reports_and_on_disconnect(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = FakeProvider()
        controller = FakeStopController()
        latch = StopLatch(controller)
        supervisor = RuntimeSupervisor(
            provider,
            RuntimeStore(tmp_path / "runtime"),
            latch,
            RuntimeSupervisorConfig(active_enabled=True),
        )
        session = await supervisor.create_session(_template())
        await supervisor.set_mode(RuntimeMode.ACTIVE, confirm_active=True)
        for sample_id in range(3):
            supervisor._last_dispatch = 0  # Avoid real-time delays in this unit test.
            await supervisor.submit_sample(_sample(session.session_id, sample_id))
            await asyncio.sleep(0.02)
            await provider.report(_report(recommend_stop=True))
        assert len(controller.reasons) == 1
        assert latch.is_latched
        assert supervisor.status().stop_latched
        await provider.disconnect()
        assert len(controller.reasons) == 1
        await supervisor.stop()
        assert supervisor.store.list_sessions()[0].session_id == session.session_id
        try:
            await supervisor.set_mode(RuntimeMode.MONITOR)
        except RuntimeError as error:
            assert "already stopped" in str(error)
        else:
            raise AssertionError("Stopped runtime sessions must not restart.")

    asyncio.run(scenario())


def test_runtime_api_defaults_active_off_and_never_returns_key(tmp_path: Path) -> None:
    repository_root = Path(__file__).parent.parent
    settings = RuntimeSettings(
        repository_root=repository_root,
        project_config=repository_root / "configs/projects/so101_blue_object_to_glass.yaml",
        task_templates_root=tmp_path / "templates",
        runtime_storage_root=tmp_path / "runtime",
    )
    store = TaskTemplateStore(settings.task_templates_root)
    episode = EpisodeRecord(
        episode_id="episode-1",
        dataset_root=tmp_path,
        dataset_episode_index=0,
        project_name="demo",
        robot_type="so101",
        task="Place the blue object in the glass.",
        outcome="success",
        review_status="approved",
        processing_status="completed",
        frame_count=1,
        camera_keys=("observation.images.front", "observation.images.wrist"),
    )
    evidence = TimestampEvidence(timestamp_seconds=0, description="Object visible")
    draft = TaskTemplateDraft(
        task_description=episode.task,
        ordered_task_stages=[
            TaskStage(
                name="carrying",
                description="Carry object.",
                start_time_seconds=0,
                end_time_seconds=1,
                expected_object_relationships=["object in gripper"],
                expected_robot_behavior="Move toward glass.",
                expected_gripper_behavior="Remain closed.",
                evidence=[evidence],
                confidence=0.9,
                uncertainty=[],
            )
        ],
        success_conditions=["Object in glass."],
        possible_failure_types=[
            PossibleFailure(
                failure_type="object dropped",
                description="Object leaves gripper.",
                related_stage_names=["carrying"],
                detectable_evidence=["Object moves away from gripper."],
            )
        ],
        important_timestamps_and_evidence=[evidence],
        confidence=0.9,
        uncertainty=[],
    )
    store.create_model_draft(
        episode,
        draft,
        model_version="gemini-robotics-er-1.6-preview",
        video_fps=5,
    )
    store.approve(episode.episode_id)
    client = TestClient(create_runtime_app(settings, provider=FakeProvider()))

    health = client.get("/api/runtime/health")
    session = client.post("/api/runtime/sessions", json={"template_episode_id": episode.episode_id})
    active = client.put(
        f"/api/runtime/sessions/{session.json()['session_id']}/mode",
        json={"mode": "active", "confirm_active": True},
    )

    assert health.status_code == 200
    assert health.json()["active_enabled"] is False
    assert "key" not in " ".join(health.json().keys()).replace("api_key_configured", "")
    assert session.json()["mode"] == "off"
    assert active.status_code == 403


def test_gemini_live_config_seeds_template_and_only_declares_report_tool() -> None:
    captured = {}

    class FakeLive:
        def connect(self, *, model, config):
            captured.update(model=model, config=config)
            return SimpleNamespace()

    client = SimpleNamespace(aio=SimpleNamespace(live=FakeLive()))
    callbacks = ProviderCallbacks(
        on_report=lambda _: None,
        on_status=lambda _state, _detail: None,
        on_malformed=lambda _: None,
    )
    connection = GeminiLiveConnection(
        client,
        approved_template=_template(),
        callbacks=callbacks,
        connect_timeout_seconds=1,
    )

    connection._connect_context()  # Inspect SDK configuration without connecting.

    assert captured["model"] == "gemini-3.1-flash-live-preview"
    assert "APPROVED TASK TEMPLATE" in captured["config"].system_instruction
    declaration = captured["config"].tools[0].function_declarations[0]
    assert declaration.name == "report_robot_state"
