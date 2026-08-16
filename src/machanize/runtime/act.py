"""Local LeRobot ACT action source; never exposed to the cloud provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class ACTActionSource:
    """Load a LeRobot ACT checkpoint and return named local robot actions."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        state_keys: tuple[str, ...],
        action_keys: tuple[str, ...],
        front_key: str = "front",
        wrist_key: str = "wrist",
        task: str,
        robot_type: str = "so101",
        device: str = "cpu",
    ) -> None:
        if not state_keys or not action_keys:
            raise ValueError("ACT state and action keys must not be empty.")
        self.checkpoint = str(checkpoint)
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.front_key = front_key
        self.wrist_key = wrist_key
        self.task = task
        self.robot_type = robot_type
        self.device_name = device
        self._policy: Any = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None
        self._device: Any = None

    def start(self) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        config = PreTrainedConfig.from_pretrained(
            self.checkpoint,
            local_files_only=True,
        )
        if not isinstance(config, ACTConfig):
            raise TypeError(
                "ACTActionSource requires an ACT checkpoint, but LeRobot loaded "
                f"{type(config).__name__}."
            )
        config.device = self.device_name
        policy = ACTPolicy.from_pretrained(
            self.checkpoint,
            config=config,
            local_files_only=True,
        )
        self._device = torch.device(self.device_name)
        self._policy = policy.to(self._device).eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": self.device_name}},
        )
        self._policy.reset()

    def propose(self, observation: Mapping[str, Any]) -> Mapping[str, float]:
        if self._policy is None or self._device is None:
            raise RuntimeError("ACT action source has not started.")
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        policy_observation = {
            "observation.state": np.asarray(
                [float(observation[key]) for key in self.state_keys],
                dtype=np.float32,
            ),
            "observation.images.front": np.asarray(observation[self.front_key]),
            "observation.images.wrist": np.asarray(observation[self.wrist_key]),
        }
        prepared = prepare_observation_for_inference(
            policy_observation,
            self._device,
            self.task,
            self.robot_type,
        )
        with torch.inference_mode():
            action = self._policy.select_action(self._preprocessor(prepared))
            action = self._postprocessor(action)
        values = action.detach().cpu().reshape(-1).tolist()
        if len(values) != len(self.action_keys):
            raise ValueError(
                f"ACT produced {len(values)} values for {len(self.action_keys)} robot actions."
            )
        return {key: float(value) for key, value in zip(self.action_keys, values, strict=True)}

    def stop(self) -> None:
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._device = None
