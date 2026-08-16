"""Provider interfaces for live robot-state monitoring."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from machanize.runtime.schemas import ConnectionState, RobotStateReport, RuntimeSample

ReportCallback = Callable[[RobotStateReport], Awaitable[None]]
StatusCallback = Callable[[ConnectionState, str | None], Awaitable[None]]
MalformedCallback = Callable[[str], Awaitable[None]]


class ProviderCallbacks:
    def __init__(
        self,
        *,
        on_report: ReportCallback,
        on_status: StatusCallback,
        on_malformed: MalformedCallback,
    ) -> None:
        self.on_report = on_report
        self.on_status = on_status
        self.on_malformed = on_malformed


class MonitoringConnection(Protocol):
    async def send(self, sample: RuntimeSample) -> None: ...

    async def close(self) -> None: ...


class MonitoringProvider(Protocol):
    model_id: str

    async def connect(
        self,
        *,
        approved_template: Mapping[str, object],
        callbacks: ProviderCallbacks,
    ) -> MonitoringConnection: ...
