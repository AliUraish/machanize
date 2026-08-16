# Raspberry Pi Runtime

Machanize is split across two machines and three processes:

```text
Mac browser :5173 ───────────────→ Mac training/analysis API :8000
       │                                  robot access: impossible by design
       │
       └──── HTTP + WebSocket ───→ Raspberry Pi runtime :8001
                                          │ owns LeRobot connection
                                          ├ native-FPS observation/action loop
                                          ├ side-by-side MJPEG + telemetry
                                          ├ 5 FPS AI sampling
                                          └ local supervision gate + STOP latch
```

The Phase 3 server remains a read-only Mac process. It does not import the Pi runtime factory,
connect to LeRobot, or expose movement routes. The Pi service is the only process that constructs
the runtime robot and local ACT action source.

## Pi control and monitoring flow

```text
LeRobot observation at native FPS
→ publish Front/Wrist and joints for preview while ACT is stopped
→ explicit Start ACT after approved template + connected MONITOR session
→ local ACT action source proposes an action
→ publish the proposal to local GUI state
→ sample the latest state at approximately 5 FPS for the AI observer
→ local supervision gate checks limits, latch, freshness, and validated monitor state
→ adapter.execute(proposed_action) OR block
```

The AI sample loop is separate from the native control loop. It does not reduce the robot loop to
5 FPS. Because cloud inference is asynchronous, each native-FPS proposal is gated with the latest
fresh, locally validated monitoring decision; the cloud is never placed inline as a motor driver.

The monitoring provider receives combined Front/Wrist JPEGs plus the timestamp, joint state, and
proposed ACT action. `gemini-3.1-flash-live-preview` is implemented behind `MonitoringProvider`.
Gemini is given exactly one function:

```text
report_robot_state(
  current_stage,
  progress,
  correct,
  failure_type,
  confidence,
  evidence,
  recommend_stop
)
```

It receives no action or motor tool. Text responses, unknown functions, malformed arguments,
unknown stages, and failure types outside the approved template are rejected on the Pi.

## Approved template transfer

Every training draft has a monotonically increasing `template_version`. Generation and editing
produce `draft`; only the explicit approval endpoint produces `approved` with `approved_at`.

When the user starts runtime monitoring, the frontend sends the complete approved template from
the Mac to the Pi. The Pi validates it, refuses drafts/missing approval metadata, and saves an
immutable versioned snapshot under:

```text
data/runtime/task_templates/<episode>-v<version>-<sha256>.json
```

The session stores both `template_version` and the full SHA-256 template revision. An episode ID
alone works only when that approved snapshot already exists on the Pi.

## Pi API

- `GET /health` — Pi role, robot/control status, provider configuration, and latch state.
- `GET /stream/combined.mjpeg` — side-by-side Front/Wrist MJPEG stream.
- `WS /ws/runtime` — timestamps, joints, proposals, current stage, risk, evidence, decision,
  provider/robot health, and STOP-latch state.
- `POST /api/runtime/sessions` — imports an approved template and creates an OFF session.
- `PUT /api/runtime/sessions/{id}/mode` — changes OFF/MONITOR/ACTIVE.
- `POST /api/runtime/control/start` — starts ACT only with `{"confirm": true}` and a connected
  MONITOR session backed by an approved template.
- `POST /api/runtime/control/stop` — immediately disables further action execution with
  `{"confirm": true}` while preserving preview.
- `GET /api/runtime/sessions/{id}/decisions` — durable validated decision history.
- `POST /stop-latch/reset` — authenticated manual latch reset.

The compatibility session/history endpoints remain available so existing Phase 3 GUI behavior is
preserved.

## Safety behavior

- The local gate runs before every `adapter.execute()`.
- Backend startup and preview never load ACT or call `adapter.execute()`.
- Stop ACT clears execution permission before synchronizing with the action gate.
- A latched STOP blocks every subsequent action, including after provider recovery.
- Reset requires `X-Machanize-Reset-Token`, a matching Pi-only `MACHANIZE_RESET_TOKEN`, explicit
  confirmation, and an OFF runtime session.
- ACTIVE requires a fresh valid decision. Timeout, malformed output, send failure, disconnect, or
  missing required camera input fails closed and latches STOP.
- Repeated matching high-confidence recommendations are required before an AI recommendation can
  latch STOP. Thresholds, count, and the time window remain local configuration.
- Joint/action limits and the control watchdog remain local. Configure real robot-specific limits
  before enabling ACTIVE.
- The physical emergency stop remains independent and authoritative.
- There are no recovery actions.

ACTIVE remains `false` in the checked-in project configuration. Do not enable it until the stop
controller, limits, watchdog, reset procedure, and emergency stop have been physically validated.

## Start on the Mac

Install and start the analysis server. This process must remain on the Mac:

```text
cd /path/to/machanize
uv sync --extra dev --extra lerobot --extra vision
GEMINI_API_KEY=your-key uv run python training/serve_phase3.py
```

It listens on `127.0.0.1:8000` and cannot connect to the robot.

Configure and start the dashboard in another Mac terminal:

```text
cd /path/to/machanize/dashboard
cp .env.example .env.local
# Edit .env.local:
# VITE_RUNTIME_BASE_URL=http://raspberrypi.local:8001
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Training requests continue to use `/api` on port 8000. Only runtime
requests use `VITE_RUNTIME_BASE_URL`.

## Start on the Raspberry Pi

Copy the repository and the approved local ACT checkpoint to the Pi, calibrate the SO-101, and
install the runtime dependencies:

```text
cd /path/to/machanize
uv sync --extra lerobot --extra vision
```

Set the Pi-only environment. Replace the example addresses and device paths:

```text
export GEMINI_API_KEY=your-key
export MACHANIZE_RESET_TOKEN=choose-a-long-local-reset-secret
export MACHANIZE_FRONTEND_ORIGINS=http://MAC_IP:5173
export MACHANIZE_RUNTIME_FACTORY=examples.so101.pi_runtime_factory:create_runtime_hardware
export MACHANIZE_FOLLOWER_PORT=/dev/ttyACM0
export MACHANIZE_FRONT_CAMERA=/dev/video0
export MACHANIZE_WRIST_CAMERA=/dev/video2
export MACHANIZE_CONTROL_FPS=30
export MACHANIZE_ACT_CHECKPOINT=/path/to/pretrained_model
export MACHANIZE_ACT_DEVICE=cpu

uv run python training/serve_runtime.py
```

The runtime listens on `0.0.0.0:8001`. When a runtime factory is configured, backend startup connects
the robot cameras and starts an observation-only preview loop. It does not load ACT, propose actions,
or call `adapter.execute()`. Without a runtime factory, the API starts in diagnostic mode and
`/health` reports that no robot is connected.

In the dashboard, create a session from an approved template and switch it to MONITOR. **Start ACT**
then calls `POST /api/runtime/control/start` with `{"confirm": true}`. **Stop ACT** calls
`POST /api/runtime/control/stop` and places the runtime in `stopped` while leaving the preview loop
connected. ACT state is always one of `ready`, `running`, `stopped`, or `error`.

The supplied SO-101 factory uses the official LeRobot follower, OpenCV Front/Wrist cameras, and the
local `ACTActionSource`. Alternative policies or teleoperation sources can implement the small
`ActionSource` interface and be returned by another `module:function` runtime factory.

## Storage and secrets

```text
data/runtime/task_templates/       # immutable approved template versions
data/runtime/sessions/<id>/session.json
data/runtime/sessions/<id>/decisions.jsonl
data/runtime/sessions/<id>/events.jsonl
```

Raw runtime JPEGs are not stored in the decision log. `GEMINI_API_KEY` and
`MACHANIZE_RESET_TOKEN` are backend/Pi environment values only. Neither is returned by an endpoint,
sent over the runtime WebSocket, written to storage, or placed in Vite variables.

## Limitations and hardware verification

- Pi ACT inference and camera capture must be benchmarked on the target Pi. If policy inference is
  slower than the configured native control rate, use an appropriate local chunked/asynchronous
  action source; do not lower safety timeouts without measurement.
- The 5 FPS AI rate is approximate and network/provider latency is nondeterministic. Cloud
  monitoring is not a real-time interlock.
- Camera images, task text, joints, and proposed actions leave the Pi for Google in MONITOR/ACTIVE.
- Live sessions consume tokens and retained context can compound cost. Consult current
  [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and
  [Live session guidance](https://ai.google.dev/gemini-api/docs/live-api/session-management).
- Preview models, lighting, occlusion, blur, and novel failures can cause false alerts or misses.

## Tests

```text
uv run ruff check src tests training
uv run pytest -q

cd dashboard
npm run typecheck
npm test
npm run build
npm run test:e2e
```

Tests inject fake robots, cameras, action sources, approved templates, and monitoring providers.
They never connect to hardware or make paid Gemini requests.
