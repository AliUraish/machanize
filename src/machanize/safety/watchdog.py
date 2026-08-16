"""Simple monotonic watchdog for local control loops."""

from __future__ import annotations

import time
from threading import RLock


class Watchdog:
    def __init__(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Watchdog timeout must be positive.")
        self.timeout_seconds = timeout_seconds
        self._last_heartbeat: float | None = None
        self._lock = RLock()

    def heartbeat(self, *, now: float | None = None) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic() if now is None else now

    def disarm(self) -> None:
        with self._lock:
            self._last_heartbeat = None

    def expired(self, *, now: float | None = None) -> bool:
        with self._lock:
            if self._last_heartbeat is None:
                return False
            current = time.monotonic() if now is None else now
            return current - self._last_heartbeat > self.timeout_seconds
