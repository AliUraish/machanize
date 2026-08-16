from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from fastapi.testclient import TestClient

from machanize.adapters import LeRobotAdapter
from machanize.providers.base import ProviderCallbacks
from machanize.runtime.act import ACTActionSource
from machanize.runtime.api import RuntimeSettings, create_runtime_app
from machanize.runtime.control import PiControlLoop
from machanize.runtime.integration import RuntimeBridgeHook
from machanize.runtime.schemas import (
    ConnectionState,
    DecisionEvidence,
    RobotStateReport,
    RuntimeMode,
    RuntimeSample,
)
from machanize.runtime.state import RuntimeStateHub
from machanize.runtime.store import RuntimeStore
from machanize.runtime.supervisor import RuntimeSupervisor, RuntimeSupervisorConfig
from machanize.safety import StopLatch


def _install_fake_act_modules(
    monkeypatch,
    *,
    act_config: bool,
) -> dict[str, object]:
    calls: dict[str, object] = {}

    class FakeACTConfig:
        device = "cuda"

    loaded_config = FakeACTConfig() if act_config else object()

    class FakePreTrainedConfig:
        @classmethod
        def from_pretrained(cls, checkpoint, *, local_files_only):
            del cls
            assert checkpoint == "/tmp/act-checkpoint"
            assert local_files_only is True
            calls["config_loaded"] = True
            return loaded_config

    class FakeACTPolicy:
        @classmethod
        def from_pretrained(cls, checkpoint, *, config, local_files_only):
            assert checkpoint == "/tmp/act-checkpoint"
            assert config is loaded_config
            assert local_files_only is True
            calls["policy_loaded"] = True
            return cls()

        def to(self, device):
            calls["policy_device"] = str(device)
            return self

        def eval(self):
            return self

        def reset(self):
            calls["policy_reset"] = True

    def make_pre_post_processors(**kwargs):
        calls["processor_config"] = kwargs["policy_cfg"]
        return object(), object()

    fake_modules = {
        "lerobot.configs.policies": {"PreTrainedConfig": FakePreTrainedConfig},
        "lerobot.policies": {"make_pre_post_processors": make_pre_post_processors},
        "lerobot.policies.act.configuration_act": {"ACTConfig": FakeACTConfig},
        "lerobot.policies.act.modeling_act": {"ACTPolicy": FakeACTPolicy},
    }
    for name, attributes in fake_modules.items():
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_act_action_source_loads_through_pretrained_config(monkeypatch) -> None:
    calls = _install_fake_act_modules(monkeypatch, act_config=True)
    source = ACTActionSource(
        "/tmp/act-checkpoint",
        state_keys=("joint.pos",),
        action_keys=("joint.pos",),
        task="test",
    )

    source.start()

    assert calls["config_loaded"] is True
    assert calls["policy_loaded"] is True
    assert calls["processor_config"] is not None
    assert calls["policy_device"] == "cpu"
    assert calls["policy_reset"] is True


def test_act_action_source_rejects_non_act_config(monkeypatch) -> None:
    _install_fake_act_modules(monkeypatch, act_config=False)
    source = ACTActionSource(
        "/tmp/act-checkpoint",
        state_keys=("joint.pos",),
        action_keys=("joint.pos",),
        task="test",
    )

    try:
        source.start()
    except TypeError as error:
        assert "requires an ACT checkpoint" in str(error)
    else:
        raise AssertionError("A non-ACT checkpoint config must be rejected.")


class FakeRobot:
    def __init__(self) -> None:
        self.connected = False
        self.observations = 0
        self.sent_actions: list[dict[str, float]] = []

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def observation_features(self):
        return {"front": (8, 8, 3), "wrist": (8, 8, 3), "joint.pos": float}

    @property
    def action_features(self):
        return {"joint.pos": float}

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_observation(self):
        self.observations += 1
        return {
            "front": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist": np.full((8, 8, 3), 64, dtype=np.uint8),
            "joint.pos": float(self.observations),
        }

    def send_action(self, action):
        sent = dict(action)
        self.sent_actions.append(sent)
        return sent


class FakeActionSource:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True

    def propose(self, observation):
        return {"joint.pos": float(observation["joint.pos"]) + 0.1}

    def stop(self) -> None:
        self.started = False


class FakeConnection:
    def __init__(self, callbacks: ProviderCallbacks) -> None:
        self.callbacks = callbacks
        self.samples: list[RuntimeSample] = []

    async def send(self, sample: RuntimeSample) -> None:
        self.samples.append(sample)
        await self.callbacks.on_report(
            RobotStateReport(
                current_stage="carrying",
                progress=0.5,
                correct=True,
                confidence=0.95,
                evidence=[
                    DecisionEvidence(timestamp=sample.timestamp, description="Object is carried.")
                ],
                recommend_stop=False,
            )
        )

    async def close(self) -> None:
        await self.callbacks.on_status(ConnectionState.OFF, None)


class FakeProvider:
    model_id = "fake-observer"

    def __init__(self) -> None:
        self.connection: FakeConnection | None = None

    async def connect(self, *, approved_template, callbacks) -> FakeConnection:
        assert approved_template["approval_status"] == "approved"
        self.connection = FakeConnection(callbacks)
        await callbacks.on_status(ConnectionState.CONNECTED, None)
        return self.connection


def approved_template(*, status: str = "approved") -> dict:
    return {
        "template_version": 3,
        "task_description": "Place the blue object in the glass.",
        "ordered_task_stages": [
            {
                "name": "carrying",
                "description": "Carry the object.",
                "start_time_seconds": 0,
                "end_time_seconds": 1,
                "expected_object_relationships": ["object in gripper"],
                "expected_robot_behavior": "Move toward the glass.",
                "expected_gripper_behavior": "Remain closed.",
                "evidence": [],
                "confidence": 0.9,
                "uncertainty": [],
            }
        ],
        "success_conditions": ["Object is in glass."],
        "possible_failure_types": [
            {
                "failure_type": "object dropped",
                "description": "Object leaves gripper.",
                "related_stage_names": ["carrying"],
                "detectable_evidence": ["Object separates from gripper."],
            }
        ],
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
        "approval_status": status,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:01:00Z",
        "approved_at": "2026-01-01T00:01:00Z" if status == "approved" else None,
    }


def runtime_settings(tmp_path: Path) -> RuntimeSettings:
    root = Path(__file__).parent.parent
    return RuntimeSettings(
        repository_root=root,
        project_config=root / "configs/projects/so101_blue_object_to_glass.yaml",
        task_templates_root=tmp_path / "training-templates",
        runtime_storage_root=tmp_path / "runtime/sessions",
        runtime_templates_root=tmp_path / "runtime/templates",
        reset_token="manual-secret",
    )


def test_native_control_loop_and_five_fps_monitor_are_separate(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = FakeProvider()
        latch = StopLatch()
        supervisor = RuntimeSupervisor(
            provider,
            RuntimeStore(tmp_path / "sessions"),
            latch,
            RuntimeSupervisorConfig(max_video_fps=5),
        )
        await supervisor.create_session(approved_template())
        await supervisor.set_mode(RuntimeMode.MONITOR)
        state = RuntimeStateHub()
        hook = RuntimeBridgeHook(
            supervisor,
            asyncio.get_running_loop(),
            stop_latch=latch,
            state_hub=state,
            monitor_sample_fps=5,
        )
        robot = FakeRobot()
        loop = PiControlLoop(
            LeRobotAdapter(robot),
            FakeActionSource(),
            hook,
            state,
            latch,
            control_fps=50,
        )
        loop.start_preview()
        await asyncio.sleep(0.12)
        assert robot.observations >= 3
        assert robot.sent_actions == []
        assert not loop.act_running
        loop.start_act()
        await asyncio.sleep(0.48)
        loop.close()
        await asyncio.sleep(0.05)
        await supervisor.stop()

        assert provider.connection is not None
        assert len(robot.sent_actions) >= 10
        assert 1 <= len(provider.connection.samples) <= 4
        assert len(robot.sent_actions) > len(provider.connection.samples) * 3
        _, frame = state.latest_frame()
        assert frame is not None and frame.startswith(b"\xff\xd8")

    asyncio.run(scenario())


def test_latched_stop_blocks_every_action_until_manual_clear(tmp_path: Path) -> None:
    async def scenario() -> None:
        latch = StopLatch()
        latch.request("operator stop")
        supervisor = RuntimeSupervisor(
            FakeProvider(),
            RuntimeStore(tmp_path / "sessions"),
            latch,
            RuntimeSupervisorConfig(),
        )
        state = RuntimeStateHub()
        hook = RuntimeBridgeHook(
            supervisor,
            asyncio.get_running_loop(),
            stop_latch=latch,
            state_hub=state,
        )
        robot = FakeRobot()
        loop = PiControlLoop(
            LeRobotAdapter(robot),
            FakeActionSource(),
            hook,
            state,
            latch,
            control_fps=50,
        )
        loop.start_preview()
        await asyncio.sleep(0.12)
        assert robot.sent_actions == []
        with pytest.raises(RuntimeError, match="operator stop"):
            loop.start_act()
        latch.clear_by_operator()
        loop.start_act()
        await asyncio.sleep(0.08)
        loop.close()
        assert robot.sent_actions

    asyncio.run(scenario())


def test_pi_api_health_websocket_template_validation_and_reset(tmp_path: Path) -> None:
    app = create_runtime_app(runtime_settings(tmp_path), provider=FakeProvider())
    with TestClient(app) as client:
        health = client.get("/health")
        draft = client.post(
            "/api/runtime/sessions",
            json={"task_template": approved_template(status="draft")},
        )
        session = client.post(
            "/api/runtime/sessions",
            json={"task_template": approved_template()},
        )

        assert health.status_code == 200
        assert health.json()["service_role"] == "pi_runtime"
        assert health.json()["training_backend_controls_robot"] is False
        assert health.json()["monitor_sample_fps"] == 5
        assert draft.status_code == 409
        assert session.status_code == 200
        assert session.json()["template_version"] == 3

        app.state.services.stop_latch.request("test stop")
        wrong_reset = client.post(
            "/stop-latch/reset",
            headers={"X-Machanize-Reset-Token": "wrong"},
            json={"confirm": True},
        )
        correct_reset = client.post(
            "/stop-latch/reset",
            headers={"X-Machanize-Reset-Token": "manual-secret"},
            json={"confirm": True},
        )
        assert wrong_reset.status_code == 401
        assert correct_reset.status_code == 200
        assert not app.state.services.stop_latch.is_latched

        with client.websocket_connect("/ws/runtime") as websocket:
            telemetry = websocket.receive_json()
        assert telemetry["decision"] == "CONTINUE"
        assert telemetry["recommend_stop"] is False
        assert telemetry["stop_latched"] is False


def test_pi_api_never_executes_act_before_explicit_confirmed_start(tmp_path: Path) -> None:
    robot = FakeRobot()
    action_source = FakeActionSource()
    app = create_runtime_app(
        runtime_settings(tmp_path),
        provider=FakeProvider(),
        robot_adapter=LeRobotAdapter(robot),
        action_source=action_source,
    )

    with TestClient(app) as client:
        time.sleep(0.08)
        initial_health = client.get("/health").json()
        assert initial_health["act_state"] == "ready"
        assert initial_health["robot_connected"] is True
        assert robot.observations > 0
        assert robot.sent_actions == []
        assert action_source.started is False

        missing_confirmation = client.post(
            "/api/runtime/control/start",
            json={"confirm": False},
        )
        assert missing_confirmation.status_code == 422
        assert robot.sent_actions == []

        session = client.post(
            "/api/runtime/sessions",
            json={"task_template": approved_template()},
        ).json()
        without_monitor = client.post(
            "/api/runtime/control/start",
            json={"confirm": True},
        )
        assert without_monitor.status_code == 409
        assert robot.sent_actions == []

        monitor = client.put(
            f"/api/runtime/sessions/{session['session_id']}/mode",
            json={"mode": "monitor"},
        )
        assert monitor.status_code == 200
        started = client.post(
            "/api/runtime/control/start",
            json={"confirm": True},
        )
        assert started.status_code == 200
        assert started.json()["act_state"] == "running"
        time.sleep(0.08)
        assert robot.sent_actions

        stopped = client.post(
            "/api/runtime/control/stop",
            json={"confirm": True},
        )
        assert stopped.status_code == 200
        assert stopped.json()["act_state"] == "stopped"
        action_count = len(robot.sent_actions)
        observation_count = robot.observations
        time.sleep(0.08)
        assert len(robot.sent_actions) == action_count
        assert robot.observations > observation_count
        assert client.get("/health").json()["robot_connected"] is True


def test_combined_mjpeg_endpoint_emits_latest_fake_camera_frame(tmp_path: Path) -> None:
    app = create_runtime_app(runtime_settings(tmp_path), provider=FakeProvider())
    app.state.services.state.update_observation(
        timestamp="2026-01-01T00:00:00Z",
        joints={"joint.pos": 1},
        proposed_actions={"joint.pos": 2},
        combined_jpeg=b"\xff\xd8fake-jpeg\xff\xd9",
    )
    route = next(
        route for route in app.routes if getattr(route, "path", None) == "/stream/combined.mjpeg"
    )

    async def first_chunk() -> bytes:
        response = await route.endpoint()
        chunk = await anext(response.body_iterator)
        await response.body_iterator.aclose()
        return chunk

    chunk = asyncio.run(first_chunk())
    assert b"multipart" not in chunk
    assert b"Content-Type: image/jpeg" in chunk
    assert b"fake-jpeg" in chunk
