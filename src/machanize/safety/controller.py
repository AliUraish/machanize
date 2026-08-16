"""Safe-stop interfaces kept separate from cloud-provider output."""

from __future__ import annotations

from threading import RLock
from typing import Protocol


class SafeStopController(Protocol):
    def request_safe_stop(self, reason: str) -> None: ...


class StopLatch:
    """Thread-safe, idempotent stop latch that only a local operator can clear."""

    def __init__(self, controller: SafeStopController | None = None) -> None:
        self.controller = controller
        self._latched = False
        self._reason: str | None = None
        self._lock = RLock()

    @property
    def is_latched(self) -> bool:
        with self._lock:
            return self._latched

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def request(self, reason: str) -> bool:
        with self._lock:
            if self._latched:
                return False
            self._latched = True
            self._reason = reason
            if self.controller is not None:
                self.controller.request_safe_stop(reason)
            return True

    def clear_by_operator(self) -> None:
        with self._lock:
            self._latched = False
            self._reason = None
