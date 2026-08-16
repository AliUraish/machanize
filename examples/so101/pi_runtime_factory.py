"""SO-101 + local ACT factory loaded only by the Raspberry Pi runtime process."""

from __future__ import annotations

import os
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from machanize.runtime.act import ACTActionSource


def create_runtime_hardware():
    control_fps = int(os.environ.get("MACHANIZE_CONTROL_FPS", "30"))
    follower = SO101Follower(
        SO101FollowerConfig(
            port=_required("MACHANIZE_FOLLOWER_PORT"),
            id="machanize_runtime_follower",
            cameras={
                "front": OpenCVCameraConfig(
                    index_or_path=_camera_source(os.environ.get("MACHANIZE_FRONT_CAMERA", "0")),
                    width=640,
                    height=480,
                    fps=control_fps,
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=_camera_source(os.environ.get("MACHANIZE_WRIST_CAMERA", "2")),
                    width=640,
                    height=480,
                    fps=control_fps,
                ),
            },
        )
    )
    state_keys = tuple(
        key for key, feature in follower.observation_features.items() if feature is float
    )
    action_keys = tuple(follower.action_features)
    action_source = ACTActionSource(
        Path(_required("MACHANIZE_ACT_CHECKPOINT")),
        state_keys=state_keys,
        action_keys=action_keys,
        task=os.environ.get(
            "MACHANIZE_TASK",
            "Pick up a blue object and place it inside a glass.",
        ),
        device=os.environ.get("MACHANIZE_ACT_DEVICE", "cpu"),
    )
    return follower, action_source


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required by the SO-101 Pi runtime factory.")
    return value


def _camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value
