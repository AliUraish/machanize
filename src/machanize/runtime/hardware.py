"""Lazy Pi hardware/action factory loading with no imports in the Mac analysis service."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from machanize.adapters import LeRobotAdapter
from machanize.runtime.control import ActionSource, CallableActionSource


@dataclass(frozen=True)
class RuntimeHardware:
    adapter: LeRobotAdapter
    action_source: ActionSource


def load_runtime_hardware(factory_spec: str) -> RuntimeHardware:
    """Load `module:function`; the factory returns `(LeRobot robot, action source)`."""

    module_name, separator, function_name = factory_spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Runtime hardware factory must use the form 'module:function'.")
    factory = getattr(importlib.import_module(module_name), function_name)
    robot, action_source = factory()
    adapter = robot if isinstance(robot, LeRobotAdapter) else LeRobotAdapter(robot)
    if callable(action_source) and not hasattr(action_source, "propose"):
        action_source = CallableActionSource(action_source)
    for method in ("start", "propose", "stop"):
        if not callable(getattr(action_source, method, None)):
            raise TypeError(f"Runtime action source is missing {method}().")
    return RuntimeHardware(adapter=adapter, action_source=action_source)
