"""Provider-neutral runtime monitoring schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeMode(StrEnum):
    OFF = "off"
    MONITOR = "monitor"
    ACTIVE = "active"


class ACTState(StrEnum):
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class ConnectionState(StrEnum):
    OFF = "off"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class DecisionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(min_length=1)
    description: str = Field(min_length=1)


class RobotStateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_stage: str = Field(min_length=1)
    progress: float = Field(ge=0, le=1)
    correct: bool
    failure_type: str | None = None
    confidence: float = Field(ge=0, le=1)
    evidence: list[DecisionEvidence]
    recommend_stop: bool

    @model_validator(mode="after")
    def validate_stop_evidence(self) -> RobotStateReport:
        if self.recommend_stop and not self.evidence:
            raise ValueError("Stop recommendations require timestamped evidence.")
        if self.recommend_stop and self.correct:
            raise ValueError("A correct state cannot recommend a stop.")
        return self


@dataclass(frozen=True)
class RuntimeSample:
    session_id: str
    sample_id: int
    timestamp: str
    monotonic_time: float
    joint_observations: dict[str, float]
    act_proposed_action: dict[str, float]
    combined_jpeg: bytes

    def metadata(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sample_id": self.sample_id,
            "timestamp": self.timestamp,
            "joint_observations": self.joint_observations,
            "act_proposed_action": self.act_proposed_action,
        }


class RuntimeDecision(BaseModel):
    decision_id: str
    session_id: str
    sample_id: int
    sample_timestamp: str
    received_at: str
    model_id: str
    mode: RuntimeMode
    report: RobotStateReport
    latency_ms: float = Field(ge=0)
    validation_status: Literal["valid"] = "valid"
    stop_streak: int = Field(ge=0)
    local_result: Literal["continue", "alert", "stop_requested"]
    safety_reason: str | None = None


class RuntimeSessionRecord(BaseModel):
    session_id: str
    template_episode_id: str
    template_version: int = Field(default=1, ge=1)
    template_revision: str
    model_id: str
    mode: RuntimeMode = RuntimeMode.OFF
    connection_state: ConnectionState = ConnectionState.OFF
    created_at: str
    started_at: str | None = None
    stopped_at: str | None = None
    last_sample_at: str | None = None
    last_decision_at: str | None = None
    last_latency_ms: float | None = None
    stop_latched: bool = False
    stop_reason: str | None = None
    thresholds: dict[str, Any]
