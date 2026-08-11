from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from machanize.adapters import LeRobotAdapter
from machanize.projects import load_project_config
from machanize.recording import EpisodeOutcome, EpisodeRecorder
from machanize.runtime import MachanizeLeRobotBridge


class FakeRobot:
    def __init__(self) -> None:
        self._connected = False
        self.sent_actions: list[dict[str, float]] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def observation_features(self) -> dict[str, type]:
        return {"shoulder.pos": float, "front": (2, 2, 3)}

    @property
    def action_features(self) -> dict[str, type]:
        return {"shoulder.pos": float}

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> dict[str, Any]:
        return {"shoulder.pos": 1.0, "front": [[[0, 0, 0]] * 2] * 2}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.sent_actions.append(action)
        return dict(action)


class FakeSink:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.saved_episodes = 0
        self.cleared_episodes = 0
        self.finalized = False

    def add_frame(self, observation, proposed_action, executed_action, task) -> None:
        self.frames.append(
            {
                "observation": observation,
                "proposed_action": proposed_action,
                "executed_action": executed_action,
                "task": task,
            }
        )

    def save_episode(self) -> int:
        index = self.saved_episodes
        self.saved_episodes += 1
        return index

    def clear_episode(self) -> None:
        self.cleared_episodes += 1
        self.frames.clear()

    def finalize(self) -> None:
        self.finalized = True


class FakeMotorBus:
    def __init__(self, *, torque_disable_fails: bool = False) -> None:
        self.is_connected = False
        self.torque_enabled = False
        self.torque_disable_fails = torque_disable_fails
        self.disable_torque_calls = 0
        self.disconnect_calls: list[bool] = []

    def connect(self) -> None:
        self.is_connected = True
        self.torque_enabled = True

    def disable_torque(self, *, num_retry: int = 0) -> None:
        self.disable_torque_calls += 1
        if self.torque_disable_fails:
            raise OSError("torque write failed")
        self.torque_enabled = False

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls.append(disable_torque)
        if disable_torque:
            self.disable_torque()
        self.is_connected = False


class FailingCamera:
    is_connected = False

    def connect(self) -> None:
        raise RuntimeError("camera failed")

    def disconnect(self) -> None:
        self.is_connected = False


class WorkingCamera:
    def __init__(self) -> None:
        self.is_connected = False
        self.disconnect_calls = 0

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


class PartiallyConnectingRobot(FakeRobot):
    def __init__(self, *, torque_disable_fails: bool = False) -> None:
        super().__init__()
        self.bus = FakeMotorBus(torque_disable_fails=torque_disable_fails)
        self.cameras = {"front": FailingCamera()}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(
            camera.is_connected for camera in self.cameras.values()
        )

    def connect(self) -> None:
        self.bus.connect()
        self.cameras["front"].connect()

    def disconnect(self) -> None:
        if not self.is_connected:
            raise RuntimeError("robot-level disconnect rejected partial connection")
        self.bus.disconnect()


class WristCameraFailureRobot(PartiallyConnectingRobot):
    def __init__(self) -> None:
        super().__init__()
        self.cameras = {"front": WorkingCamera(), "wrist": FailingCamera()}

    def connect(self) -> None:
        self.bus.connect()
        self.cameras["front"].connect()
        self.cameras["wrist"].connect()


def test_bridge_records_and_registers_episode() -> None:
    robot = FakeRobot()
    sink = FakeSink()
    with TemporaryDirectory() as directory:
        recorder = EpisodeRecorder(
            sink,
            manifest_directory=directory,
            project_name="demo",
            robot_type="so101",
        )
        bridge = MachanizeLeRobotBridge(
            LeRobotAdapter(robot),
            recorder,
            task="Pick up a blue object.",
        )

        bridge.connect()
        episode_id = bridge.start_episode()
        result = bridge.step({"shoulder.pos": 2.0})
        manifest_path = bridge.finish_episode(outcome=EpisodeOutcome.SUCCESS)
        bridge.close()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["episode_id"] == episode_id
        assert manifest["dataset_episode_index"] == 0
        assert manifest["frame_count"] == 1
        assert manifest["review_status"] == "pending"
        assert result.executed_action == {"shoulder.pos": 2.0}
        assert robot.sent_actions == [{"shoulder.pos": 2.0}]
        assert not robot.is_connected
        assert sink.finalized


def test_camera_connect_failure_disables_torque_and_disconnects_bus() -> None:
    robot = PartiallyConnectingRobot()
    adapter = LeRobotAdapter(robot)

    with pytest.raises(RuntimeError, match="camera failed"):
        adapter.connect()

    assert robot.bus.disable_torque_calls == 1
    assert not robot.bus.torque_enabled
    assert robot.bus.disconnect_calls == [False]
    assert not robot.bus.is_connected


def test_wrist_camera_failure_disconnects_front_camera_and_motor_bus() -> None:
    robot = WristCameraFailureRobot()
    adapter = LeRobotAdapter(robot)

    with pytest.raises(RuntimeError, match="camera failed"):
        adapter.connect()

    front_camera = robot.cameras["front"]
    assert isinstance(front_camera, WorkingCamera)
    assert front_camera.disconnect_calls == 1
    assert not front_camera.is_connected
    assert robot.bus.disable_torque_calls == 1
    assert not robot.bus.is_connected


def test_bus_is_closed_even_when_torque_disable_reports_an_error() -> None:
    robot = PartiallyConnectingRobot(torque_disable_fails=True)
    adapter = LeRobotAdapter(robot)

    with pytest.raises(RuntimeError, match="camera failed") as raised:
        adapter.connect()

    assert robot.bus.disable_torque_calls == 1
    assert robot.bus.disconnect_calls == [False]
    assert not robot.bus.is_connected
    assert any("torque write failed" in note for note in raised.value.__notes__)


def test_abort_clears_unsaved_episode() -> None:
    sink = FakeSink()
    with TemporaryDirectory() as directory:
        recorder = EpisodeRecorder(
            sink,
            manifest_directory=directory,
            project_name="demo",
            robot_type="so101",
        )
        recorder.start_episode(task="Demo")
        recorder.record_step(
            observation={"state": 1},
            proposed_action={"action": 2},
            executed_action={"action": 2},
        )
        recorder.abort_episode()

        assert not recorder.is_recording
        assert sink.cleared_episodes == 1
        assert not list(Path(directory).glob("*.json"))


def test_empty_episode_cannot_be_finished() -> None:
    with TemporaryDirectory() as directory:
        recorder = EpisodeRecorder(
            FakeSink(),
            manifest_directory=directory,
            project_name="demo",
            robot_type="so101",
        )
        recorder.start_episode(task="Demo")

        with pytest.raises(RuntimeError, match="no recorded frames"):
            recorder.finish_episode(outcome=EpisodeOutcome.UNKNOWN)


def test_project_config_loads_phase2_fields() -> None:
    config = load_project_config("configs/projects/so101_blue_object_to_glass.yaml")

    assert config.name == "blue-object-to-glass-demo"
    assert config.robot_type == "so101"
    assert config.camera_names == ("front", "wrist")
    assert config.camera_devices == {"front": "/dev/video0", "wrist": "/dev/video2"}
    assert config.episode_directory == Path("data/episodes")
