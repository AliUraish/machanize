"""Convert advisory cloud reports into deterministic local decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from machanize.runtime.schemas import RobotStateReport, RuntimeMode


class GateResult(StrEnum):
    CONTINUE = "continue"
    ALERT = "alert"
    STOP_REQUESTED = "stop_requested"


@dataclass(frozen=True)
class DecisionGateConfig:
    alert_threshold: float = 0.60
    stop_threshold: float = 0.85
    consecutive_stop_predictions: int = 3
    recommendation_window_seconds: float = 4.0

    def __post_init__(self) -> None:
        if not 0 <= self.alert_threshold <= self.stop_threshold <= 1:
            raise ValueError("Decision thresholds must satisfy 0 <= alert <= stop <= 1.")
        if self.consecutive_stop_predictions < 1:
            raise ValueError("Consecutive stop predictions must be at least one.")
        if self.recommendation_window_seconds <= 0:
            raise ValueError("Recommendation window must be positive.")


class DecisionGate:
    def __init__(self, config: DecisionGateConfig) -> None:
        self.config = config
        self.stop_streak = 0
        self._last_failure: str | None = None
        self._last_qualifying_at: float | None = None

    def reset(self) -> None:
        self.stop_streak = 0
        self._last_failure = None
        self._last_qualifying_at = None

    def evaluate(
        self,
        report: RobotStateReport,
        *,
        mode: RuntimeMode,
        monotonic_time: float,
    ) -> GateResult:
        if mode is RuntimeMode.OFF:
            self.reset()
            return GateResult.CONTINUE

        qualifies = report.recommend_stop and report.confidence >= self.config.stop_threshold
        failure = (report.failure_type or "unspecified").strip().lower()
        within_window = (
            self._last_qualifying_at is not None
            and monotonic_time - self._last_qualifying_at
            <= self.config.recommendation_window_seconds
        )
        if qualifies:
            if within_window and failure == self._last_failure:
                self.stop_streak += 1
            else:
                self.stop_streak = 1
            self._last_failure = failure
            self._last_qualifying_at = monotonic_time
        else:
            self.reset()

        if mode is RuntimeMode.ACTIVE and (
            self.stop_streak >= self.config.consecutive_stop_predictions
        ):
            return GateResult.STOP_REQUESTED
        if (
            not report.correct or report.recommend_stop
        ) and report.confidence >= self.config.alert_threshold:
            return GateResult.ALERT
        return GateResult.CONTINUE
