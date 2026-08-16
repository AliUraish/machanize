"""Gemini 3.1 Flash Live monitoring provider."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

from machanize.providers.base import MonitoringConnection, MonitoringProvider, ProviderCallbacks
from machanize.runtime.schemas import ConnectionState, RobotStateReport, RuntimeSample

GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"


class GeminiLiveProvider(MonitoringProvider):
    model_id = GEMINI_LIVE_MODEL

    def __init__(
        self,
        *,
        client_factory: Callable[[], Any] | None = None,
        connect_timeout_seconds: float = 15,
    ) -> None:
        self._client_factory = client_factory
        self.connect_timeout_seconds = connect_timeout_seconds

    async def connect(
        self,
        *,
        approved_template: Mapping[str, object],
        callbacks: ProviderCallbacks,
    ) -> MonitoringConnection:
        client = self._create_client()
        connection = GeminiLiveConnection(
            client,
            approved_template=approved_template,
            callbacks=callbacks,
            connect_timeout_seconds=self.connect_timeout_seconds,
        )
        await connection.start()
        return connection

    def _create_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the backend environment.")
        from google import genai

        return genai.Client(api_key=api_key)


class GeminiLiveConnection(MonitoringConnection):
    def __init__(
        self,
        client: Any,
        *,
        approved_template: Mapping[str, object],
        callbacks: ProviderCallbacks,
        connect_timeout_seconds: float,
    ) -> None:
        self.client = client
        self.approved_template = approved_template
        self.callbacks = callbacks
        self.connect_timeout_seconds = connect_timeout_seconds
        self._closed = asyncio.Event()
        self._connected = asyncio.Event()
        self._session: Any = None
        self._runner: asyncio.Task[None] | None = None
        self._session_handle: str | None = None
        self._send_lock = asyncio.Lock()

    async def start(self) -> None:
        self._runner = asyncio.create_task(self._run(), name="machanize-gemini-live")
        try:
            await asyncio.wait_for(self._connected.wait(), self.connect_timeout_seconds)
        except TimeoutError:
            await self.close()
            raise TimeoutError("Gemini Live connection timed out.") from None

    async def send(self, sample: RuntimeSample) -> None:
        from google.genai import types

        if self._session is None or not self._connected.is_set():
            raise ConnectionError("Gemini Live is not connected.")
        metadata = {
            **sample.metadata(),
            "instruction": "Evaluate this sample and call report_robot_state exactly once.",
        }
        async with self._send_lock:
            await self._session.send_realtime_input(
                video=types.Blob(data=sample.combined_jpeg, mime_type="image/jpeg")
            )
            await self._session.send_realtime_input(
                text=json.dumps(metadata, separators=(",", ":"))
            )

    async def close(self) -> None:
        self._closed.set()
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
        self._session = None
        self._connected.clear()
        await self.callbacks.on_status(ConnectionState.OFF, None)

    async def _run(self) -> None:
        reconnecting = False
        while not self._closed.is_set():
            await self.callbacks.on_status(
                ConnectionState.RECONNECTING if reconnecting else ConnectionState.CONNECTING,
                None,
            )
            try:
                async with self._connect_context() as session:
                    self._session = session
                    self._connected.set()
                    await self.callbacks.on_status(ConnectionState.CONNECTED, None)
                    async for response in session.receive():
                        await self._handle_response(response)
                        if self._closed.is_set():
                            return
            except asyncio.CancelledError:
                return
            except Exception as error:  # noqa: BLE001 - provider failures become local status
                self._connected.clear()
                self._session = None
                await self.callbacks.on_status(
                    ConnectionState.DISCONNECTED,
                    type(error).__name__,
                )
                reconnecting = True
                try:
                    await asyncio.wait_for(self._closed.wait(), timeout=1.0)
                except TimeoutError:
                    continue

    def _connect_context(self) -> AbstractAsyncContextManager[Any]:
        from google.genai import types

        declaration = types.FunctionDeclaration(
            name="report_robot_state",
            description=(
                "Report the observed task stage and whether the robot is progressing correctly. "
                "This function records a monitoring assessment and cannot control motors."
            ),
            parameters_json_schema=RobotStateReport.model_json_schema(),
        )
        template_json = json.dumps(self.approved_template, separators=(",", ":"))
        system_instruction = f"""
You are a read-only robot monitoring observer. Never issue motor commands or recovery actions.
For each timestamped sample, call report_robot_state exactly once. Do not answer with prose.
Use only stages and failure types supported by this approved task template. A stop recommendation
is advisory and will be evaluated by deterministic local safety logic.

APPROVED TASK TEMPLATE:
{template_json}
""".strip()
        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=[declaration])],
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
            session_resumption=types.SessionResumptionConfig(handle=self._session_handle),
            history_config=types.HistoryConfig(initial_history_in_client_content=True),
        )
        return self.client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config)

    async def _handle_response(self, response: Any) -> None:
        from google.genai import types

        update = getattr(response, "session_resumption_update", None)
        if update and getattr(update, "resumable", False) and getattr(update, "new_handle", None):
            self._session_handle = update.new_handle
        if getattr(response, "go_away", None) is not None:
            await self.callbacks.on_status(ConnectionState.RECONNECTING, "go_away")

        tool_call = getattr(response, "tool_call", None)
        if tool_call is None:
            if getattr(response, "text", None):
                await self.callbacks.on_malformed(
                    "Gemini returned text instead of a function call."
                )
            return

        for call in tool_call.function_calls:
            accepted = False
            if call.name != "report_robot_state":
                await self.callbacks.on_malformed(f"Unexpected function call: {call.name}")
            else:
                try:
                    report = RobotStateReport.model_validate(call.args)
                except ValueError:
                    await self.callbacks.on_malformed("Malformed report_robot_state arguments.")
                else:
                    accepted = True
                    await self.callbacks.on_report(report)
            if self._session is not None:
                await self._session.send_tool_response(
                    function_responses=types.FunctionResponse(
                        id=call.id,
                        name=call.name,
                        response={"accepted": accepted},
                    )
                )
