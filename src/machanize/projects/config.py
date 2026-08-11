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
    camera_devices: dict[str, str]
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

    camera_names = tuple(raw["robot"].get("camera_names", ()))
    camera_settings = raw["robot"].get("cameras", {})
    if not isinstance(camera_settings, dict):
        raise TypeError("Robot cameras configuration must be a mapping.")
    camera_devices = {
        name: _required_string(camera_settings[name], "device")
        for name in camera_names
        if name in camera_settings
    }
    if set(camera_devices) != set(camera_names):
        missing = sorted(set(camera_names) - set(camera_devices))
        raise ValueError(f"Missing camera device configuration: {', '.join(missing)}")

    return ProjectConfig(
        name=_required_string(raw["project"], "name"),
        task_description=_required_string(raw["project"], "description"),
        robot_type=_required_string(raw["robot"], "type"),
        camera_names=camera_names,
        camera_devices=camera_devices,
        episode_directory=Path(_required_string(raw["storage"], "episode_directory")),
        raw=raw,
    )


def _required_string(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration value must be a non-empty string: {key}")
    return value
