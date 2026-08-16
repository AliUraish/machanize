"""Deterministic local joint and proposed-action limit checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JointLimit:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.minimum >= self.maximum:
            raise ValueError("Joint limit minimum must be less than maximum.")


class LocalSafetyMonitor:
    def __init__(
        self,
        *,
        observation_limits: Mapping[str, JointLimit] | None = None,
        action_limits: Mapping[str, JointLimit] | None = None,
    ) -> None:
        self.observation_limits = dict(observation_limits or {})
        self.action_limits = dict(action_limits or {})

    def violation(
        self,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
    ) -> str | None:
        return self._check(
            observation, self.observation_limits, "joint observation"
        ) or self._check(
            proposed_action,
            self.action_limits,
            "proposed action",
        )

    @staticmethod
    def _check(
        values: Mapping[str, Any],
        limits: Mapping[str, JointLimit],
        label: str,
    ) -> str | None:
        for name, limit in limits.items():
            if name not in values:
                continue
            try:
                value = float(values[name])
            except (TypeError, ValueError):
                return f"Malformed {label}: {name}"
            if not limit.minimum <= value <= limit.maximum:
                return f"{label.title()} limit exceeded: {name}={value}"
        return None
