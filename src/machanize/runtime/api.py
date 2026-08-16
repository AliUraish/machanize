"""Raspberry Pi FastAPI service that owns LeRobot and local action gating."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator

from machanize.adapters import LeRobotAdapter
from machanize.analysis import TaskTemplateRecord, TaskTemplateStore
from machanize.decision import DecisionGateConfig
from machanize.projects import load_project_config
from machanize.providers.base import MonitoringProvider
from machanize.providers.gemini_live import GEMINI_LIVE_MODEL, GeminiLiveProvider
from machanize.runtime.control import ActionSource, PiControlLoop
from machanize.runtime.hardware import load_runtime_hardware
from machanize.runtime.integration import RuntimeBridgeHook
from machanize.runtime.schemas import ConnectionState, RuntimeMode
from machanize.runtime.state import RuntimeStateHub
from machanize.runtime.store import ApprovedRuntimeTemplateStore, RuntimeStore
from machanize.runtime.supervisor import RuntimeSupervisor, RuntimeSupervisorConfig
from machanize.safety import StopLatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeSettings:
    repository_root: Path
    project_config: Path
    task_templates_root: Path
    runtime_storage_root: Path
    runtime_templates_root: Path | None = None
    reset_token: str | None = None
    frontend_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )
    hardware_factory: str | None = None

    @classmethod
    def from_repository(cls, root: str | Path) -> RuntimeSettings:
        repository_root = Path(root).resolve()
        origins = tuple(
            origin.strip()
            for origin in os.environ.get(
                "MACHANIZE_FRONTEND_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        )
        return cls(
            repository_root=repository_root,
            project_config=(
                repository_root / "configs" / "projects" / "so101_blue_object_to_glass.yaml"
            ),
            task_templates_root=repository_root / "data" / "analysis" / "task_templates",
            runtime_storage_root=repository_root / "data" / "runtime" / "sessions",
            runtime_templates_root=repository_root / "data" / "runtime" / "task_templates",
            reset_token=os.environ.get("MACHANIZE_RESET_TOKEN"),
            frontend_origins=origins,
            hardware_factory=os.environ.get("MACHANIZE_RUNTIME_FACTORY"),
        )


class CreateRuntimeSessionRequest(BaseModel):
    template_episode_id: str | None = None
    task_template: TaskTemplateRecord | None = None

    @model_validator(mode="after")
    def validate_template_reference(self) -> CreateRuntimeSessionRequest:
        if self.template_episode_id is None and self.task_template is None:
            raise ValueError("Provide an approved task template or its episode ID.")
        if (
            self.template_episode_id is not None
            and self.task_template is not None
            and self.template_episode_id != self.task_template.source_episode.episode_id
        ):
            raise ValueError("Template episode ID does not match the supplied template.")
        return self


class RuntimeModeRequest(BaseModel):
    mode: RuntimeMode
    confirm_active: bool = False


class ClearStopRequest(BaseModel):
    confirm: Literal[True]


@dataclass
class RuntimeServices:
    supervisor: RuntimeSupervisor
    training_templates: TaskTemplateStore
    runtime_templates: ApprovedRuntimeTemplateStore
    store: RuntimeStore
    stop_latch: StopLatch
    state: RuntimeStateHub
    control_loop: PiControlLoop | None = None


def create_runtime_services(
    settings: RuntimeSettings,
    *,
    provider: MonitoringProvider | None = None,
    stop_latch: StopLatch | None = None,
) -> RuntimeServices:
    project = load_project_config(settings.project_config)
    runtime = project.raw.get("runtime", {})
    gate = DecisionGateConfig(
        alert_threshold=float(runtime.get("alert_threshold", 0.60)),
        stop_threshold=float(runtime.get("stop_threshold", 0.85)),
        consecutive_stop_predictions=int(runtime.get("consecutive_high_risk_predictions", 3)),
        recommendation_window_seconds=float(runtime.get("recommendation_window_seconds", 4)),
    )
    config = RuntimeSupervisorConfig(
        model_id=str(runtime.get("live_model", GEMINI_LIVE_MODEL)),
        max_video_fps=float(runtime.get("monitor_sample_fps", 5)),
        cloud_timeout_seconds=float(runtime.get("cloud_timeout_seconds", 4)),
        active_enabled=bool(runtime.get("active_enabled", False)),
        stop_on_cloud_failure=bool(runtime.get("fail_safe_on_inference_error", True)),
        decision_gate=gate,
    )
    store = RuntimeStore(settings.runtime_storage_root)
    latch = stop_latch or StopLatch()
    state = RuntimeStateHub()
    supervisor = RuntimeSupervisor(provider or GeminiLiveProvider(), store, latch, config)
    runtime_templates_root = settings.runtime_templates_root or (
        settings.runtime_storage_root.parent / "task_templates"
    )
    return RuntimeServices(
        supervisor=supervisor,
        training_templates=TaskTemplateStore(settings.task_templates_root),
        runtime_templates=ApprovedRuntimeTemplateStore(runtime_templates_root),
        store=store,
        stop_latch=latch,
        state=state,
    )


def create_runtime_app(
    settings: RuntimeSettings | None = None,
    *,
    provider: MonitoringProvider | None = None,
    stop_latch: StopLatch | None = None,
    robot_adapter: LeRobotAdapter | None = None,
    action_source: ActionSource | None = None,
) -> FastAPI:
    settings = settings or RuntimeSettings.from_repository(
        os.environ.get("MACHANIZE_ROOT", Path.cwd())
    )
    services = create_runtime_services(settings, provider=provider, stop_latch=stop_latch)
    project = load_project_config(settings.project_config)
    runtime_config = project.raw.get("runtime", {})

    if (robot_adapter is None) != (action_source is None):
        raise ValueError("Robot adapter and action source must be provided together.")
    if robot_adapter is None and settings.hardware_factory:
        hardware = load_runtime_hardware(settings.hardware_factory)
        robot_adapter = hardware.adapter
        action_source = hardware.action_source

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        if robot_adapter is not None and action_source is not None:
            hook = RuntimeBridgeHook(
                services.supervisor,
                asyncio.get_running_loop(),
                stop_latch=services.stop_latch,
                state_hub=services.state,
                monitor_sample_fps=services.supervisor.config.max_video_fps,
                front_key=str(runtime_config.get("front_camera_key", "front")),
                wrist_key=str(runtime_config.get("wrist_camera_key", "wrist")),
                state_key=str(runtime_config.get("joint_state_key", "observation.state")),
                control_watchdog_seconds=float(runtime_config.get("control_watchdog_seconds", 1)),
            )
            services.control_loop = PiControlLoop(
                robot_adapter,
                action_source,
                hook,
                services.state,
                services.stop_latch,
                control_fps=float(runtime_config.get("control_fps", 30)),
            )
            try:
                services.control_loop.start_preview()
            except Exception:
                # Keep the API available so the dashboard can display the hardware error.
                logger.exception("Robot preview failed to start; runtime API remains available.")
        try:
            yield
        finally:
            if services.control_loop is not None:
                services.control_loop.close()
            try:
                services.supervisor.status()
            except RuntimeError:
                pass
            else:
                await services.supervisor.stop()

    app = FastAPI(title="Machanize Pi Runtime", version="0.2.0", lifespan=lifespan)
    app.state.services = services
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.frontend_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def health_payload() -> dict:
        config = services.supervisor.config
        _, local = services.state.snapshot()
        return {
            "status": "ok",
            "service_role": "pi_runtime",
            "training_backend_controls_robot": False,
            "model_id": services.supervisor.provider.model_id,
            "api_key_configured": bool(os.environ.get("GEMINI_API_KEY")) or provider is not None,
            "active_enabled": config.active_enabled,
            "default_mode": RuntimeMode.OFF.value,
            "monitor_sample_fps": config.max_video_fps,
            "control_fps": float(runtime_config.get("control_fps", 30)),
            "cloud_timeout_seconds": config.cloud_timeout_seconds,
            "stop_threshold": config.decision_gate.stop_threshold,
            "consecutive_stop_predictions": config.decision_gate.consecutive_stop_predictions,
            "robot_configured": robot_adapter is not None,
            "robot_connected": local["robot_connected"],
            "control_loop_status": local["control_loop_status"],
            "act_state": local["act_state"],
            "reset_auth_configured": settings.reset_token is not None,
            "stop_latched": services.stop_latch.is_latched,
        }

    @app.get("/health")
    @app.get("/api/runtime/health")
    def health() -> dict:
        return health_payload()

    @app.get("/stream/combined.mjpeg")
    async def combined_stream() -> StreamingResponse:
        async def frames():
            last_sequence = -1
            while True:
                sequence, jpeg = services.state.latest_frame()
                if jpeg is not None and sequence != last_sequence:
                    last_sequence = sequence
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        + jpeg
                        + b"\r\n"
                    )
                await asyncio.sleep(0.03)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.websocket("/ws/runtime")
    async def runtime_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        previous_key: tuple[object, ...] | None = None
        try:
            while True:
                sequence, _ = services.state.snapshot()
                latest = services.supervisor.latest_decision
                try:
                    session = services.supervisor.status()
                except RuntimeError:
                    session = None
                key = (
                    sequence,
                    latest.decision_id if latest else None,
                    services.stop_latch.is_latched,
                    session.connection_state if session else None,
                )
                if key != previous_key:
                    await websocket.send_json(_runtime_snapshot(services))
                    previous_key = key
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return

    @app.get("/api/runtime/templates")
    def approved_templates() -> list[dict]:
        return [
            record.model_dump(mode="json")
            for record in services.training_templates.list_records(approved_only=True)
        ]

    @app.post("/api/runtime/sessions")
    async def create_session(request: CreateRuntimeSessionRequest) -> dict:
        template = _resolve_template(services, request)
        try:
            record = await services.supervisor.create_session(template.model_dump(mode="json"))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return record.model_dump(mode="json")

    @app.get("/api/runtime/sessions")
    def sessions() -> list[dict]:
        return [record.model_dump(mode="json") for record in services.store.list_sessions()]

    @app.get("/api/runtime/sessions/{session_id}")
    def session_status(session_id: str) -> dict:
        try:
            record = services.supervisor.status()
        except RuntimeError:
            record = None
        if record is None or record.session_id != session_id:
            try:
                record = services.store.get_session(session_id)
            except KeyError as error:
                raise HTTPException(status_code=404, detail="Runtime session not found.") from error
        return record.model_dump(mode="json")

    @app.put("/api/runtime/sessions/{session_id}/mode")
    async def set_mode(session_id: str, request: RuntimeModeRequest) -> dict:
        _require_session(services, session_id)
        if request.mode is not RuntimeMode.MONITOR and services.control_loop is not None:
            services.control_loop.stop_act()
        try:
            record = await services.supervisor.set_mode(
                request.mode,
                confirm_active=request.confirm_active,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except (ConnectionError, RuntimeError, TimeoutError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return record.model_dump(mode="json")

    @app.post("/api/runtime/sessions/{session_id}/stop")
    async def stop_session(session_id: str) -> dict:
        _require_session(services, session_id)
        if services.control_loop is not None:
            services.control_loop.stop_act()
        return (await services.supervisor.stop()).model_dump(mode="json")

    @app.post("/api/runtime/control/start")
    def start_control(request: ClearStopRequest) -> dict:
        del request
        control_loop = _require_control_loop(services)
        try:
            session = services.supervisor.status()
        except RuntimeError as error:
            raise HTTPException(
                status_code=409,
                detail="Create a session with an approved task template before starting ACT.",
            ) from error
        if session.stopped_at is not None or session.mode is not RuntimeMode.MONITOR:
            raise HTTPException(
                status_code=409,
                detail="Start ACT requires an active MONITOR session.",
            )
        if session.connection_state is not ConnectionState.CONNECTED:
            raise HTTPException(
                status_code=409,
                detail="Start ACT requires a connected MONITOR provider.",
            )
        try:
            control_loop.start_act()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"ACT failed to start: {type(error).__name__}",
            ) from error
        return _control_payload(services)

    @app.post("/api/runtime/control/stop")
    def stop_control(request: ClearStopRequest) -> dict:
        del request
        control_loop = _require_control_loop(services)
        try:
            control_loop.stop_act()
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"ACT failed to stop: {type(error).__name__}",
            ) from error
        return _control_payload(services)

    @app.get("/api/runtime/sessions/{session_id}/decisions")
    def decisions(session_id: str) -> list[dict]:
        try:
            stored = services.store.list_decisions(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Runtime session not found.") from error
        return [decision.model_dump(mode="json") for decision in stored]

    @app.post("/stop-latch/reset")
    def reset_stop_latch(
        request: ClearStopRequest,
        reset_token: str | None = Header(default=None, alias="X-Machanize-Reset-Token"),
    ) -> dict:
        del request
        _authenticate_reset(settings, reset_token)
        return _reset_latch(services)

    @app.post("/api/runtime/sessions/{session_id}/stop-latch/clear")
    def clear_stop_latch(
        session_id: str,
        request: ClearStopRequest,
        reset_token: str | None = Header(default=None, alias="X-Machanize-Reset-Token"),
    ) -> dict:
        del request
        _require_session(services, session_id)
        _authenticate_reset(settings, reset_token)
        return _reset_latch(services)

    return app


def _resolve_template(
    services: RuntimeServices,
    request: CreateRuntimeSessionRequest,
) -> TaskTemplateRecord:
    if request.task_template is not None:
        try:
            return services.runtime_templates.import_approved(request.task_template)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
    assert request.template_episode_id is not None
    template = services.runtime_templates.latest_for_episode(request.template_episode_id)
    if template is None:
        template = services.training_templates.get(request.template_episode_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Task template not found on the Pi.")
    try:
        return services.runtime_templates.import_approved(template)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _runtime_snapshot(services: RuntimeServices) -> dict:
    sequence, local = services.state.snapshot()
    latest = services.supervisor.latest_decision
    try:
        session = services.supervisor.status()
    except RuntimeError:
        session = None
    report = latest.report if latest else None
    incorrect = bool(report and (not report.correct or report.recommend_stop))
    recommend_stop = bool(report and report.recommend_stop)
    stop_latched = services.stop_latch.is_latched
    return {
        "sequence": sequence,
        **local,
        "current_stage": report.current_stage if report else "unknown",
        "progress": report.progress if report else 0,
        "risk": report.confidence if incorrect else 0,
        "confidence": report.confidence if report else 0,
        "evidence": [item.model_dump(mode="json") for item in report.evidence] if report else [],
        "decision": "STOP" if stop_latched or recommend_stop else "CONTINUE",
        "recommend_stop": recommend_stop,
        "monitor_result": latest.local_result if latest else "unavailable",
        "stop_latched": stop_latched,
        "stop_reason": services.stop_latch.reason,
        "provider_connection": session.connection_state.value if session else "off",
        "mode": session.mode.value if session else "off",
        "latency_ms": latest.latency_ms if latest else None,
    }


def _authenticate_reset(settings: RuntimeSettings, supplied: str | None) -> None:
    if not settings.reset_token:
        raise HTTPException(status_code=503, detail="Authenticated reset is not configured.")
    if supplied is None or not secrets.compare_digest(supplied, settings.reset_token):
        raise HTTPException(status_code=401, detail="Invalid reset credential.")


def _reset_latch(services: RuntimeServices) -> dict:
    try:
        record = services.supervisor.clear_stop_latch_by_operator()
    except RuntimeError as error:
        if str(error) != "No runtime session exists.":
            raise HTTPException(status_code=409, detail=str(error)) from error
        services.stop_latch.clear_by_operator()
        return {"status": "cleared", "stop_latched": False}
    return record.model_dump(mode="json")


def _require_session(services: RuntimeServices, session_id: str):
    try:
        record = services.supervisor.status()
    except RuntimeError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if record.session_id != session_id:
        raise HTTPException(status_code=404, detail="Runtime session not found.")
    return record


def _require_control_loop(services: RuntimeServices) -> PiControlLoop:
    if services.control_loop is None:
        raise HTTPException(status_code=409, detail="Robot hardware is not configured.")
    return services.control_loop


def _control_payload(services: RuntimeServices) -> dict:
    _, local = services.state.snapshot()
    return {
        "act_state": local["act_state"],
        "robot_connected": local["robot_connected"],
        "stop_latched": services.stop_latch.is_latched,
        "block_reason": local["block_reason"],
    }
