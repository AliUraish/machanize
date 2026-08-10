"""Storage contract used by the Machanize episode recorder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class EpisodeSink(Protocol):
    def add_frame(
        self,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
        executed_action: Mapping[str, Any],
        task: str,
    ) -> None: ...

    def save_episode(self) -> int: ...

    def clear_episode(self) -> None: ...

    def finalize(self) -> None: ...
