"""Project configuration loading and basic validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    task_description: str
    robot_type: str
    camera_names: tuple[str, ...]
    episode_directory: Path
    raw: dict[str, Any]


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Project configuration must be a YAML mapping.")

    for section in ("project", "robot", "task", "storage"):
        if section not in raw or not isinstance(raw[section], dict):
            raise ValueError(f"Missing configuration section: {section}")

    return ProjectConfig(
        name=_required_string(raw["project"], "name"),
        task_description=_required_string(raw["project"], "description"),
        robot_type=_required_string(raw["robot"], "type"),
        camera_names=tuple(raw["robot"].get("camera_names", ())),
        episode_directory=Path(_required_string(raw["storage"], "episode_directory")),
        raw=raw,
    )


def _required_string(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration value must be a non-empty string: {key}")
    return value
