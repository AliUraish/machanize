"""LeRobotDataset-backed storage for Machanize episodes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from machanize.adapters.lerobot import LeRobotAdapter


class LeRobotDatasetSink:
    """Store observations and actions using LeRobot's Parquet/MP4 format."""

    def __init__(
        self,
        dataset: Any,
        dataset_features: Mapping[str, Any],
    ) -> None:
        self.dataset = dataset
        self.dataset_features = dataset_features

    @classmethod
    def create(
        cls,
        adapter: LeRobotAdapter,
        *,
        repo_id: str,
        root: str | Path,
        fps: int = 30,
    ) -> LeRobotDatasetSink:
        """Create a local dataset without uploading it to the Hugging Face Hub."""

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.utils.feature_utils import hw_to_dataset_features
        except ImportError as error:
            raise RuntimeError(
                "LeRobot dataset support is not installed. Install Machanize with the "
                "'lerobot' optional dependency."
            ) from error

        observation_features = adapter.observation_features
        action_features = adapter.action_features
        observation_dataset_features = hw_to_dataset_features(observation_features, "observation")
        action_dataset_features = hw_to_dataset_features(action_features, "action")
        proposed_action_features = {
            "machanize.proposed_action": {
                **action_dataset_features["action"],
                "names": list(action_dataset_features["action"]["names"]),
            }
        }
        dataset_features = {
            **observation_dataset_features,
            **proposed_action_features,
            **action_dataset_features,
        }
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=dataset_features,
            root=str(root),
            use_videos=True,
        )
        return cls(dataset, dataset_features)

    def add_frame(
        self,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
        executed_action: Mapping[str, Any],
        task: str,
    ) -> None:
        from lerobot.utils.feature_utils import build_dataset_frame

        frame = {
            **build_dataset_frame(self.dataset_features, observation, prefix="observation"),
            **build_dataset_frame(
                self.dataset_features,
                proposed_action,
                prefix="machanize.proposed_action",
            ),
            **build_dataset_frame(self.dataset_features, executed_action, prefix="action"),
            "task": task,
        }
        self.dataset.add_frame(frame)

    def save_episode(self) -> int:
        episode_index = int(self.dataset.num_episodes)
        self.dataset.save_episode()
        return episode_index

    def clear_episode(self) -> None:
        self.dataset.clear_episode_buffer()

    def finalize(self) -> None:
        self.dataset.finalize()
