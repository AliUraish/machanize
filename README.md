# Machanize

Machanize is a supervisory SDK for robot policies. V1 observes a frozen LeRobot policy, learns task-specific failure patterns from reviewed episodes, and can alert or safely stop the robot at runtime.

## V1 modes

- **Training/analysis:** Analyze one successful demonstration into a reviewed task template, review
  labels, train YOLO and GRU, and approve models.
- **Runtime OFF:** Machanize does not monitor or intervene.
- **Runtime MONITOR:** Machanize detects and reports risk without controlling the robot.
- **Runtime ACTIVE:** Machanize detects risk and can block an action or safely stop the robot.

## V1 pipeline

```text
Successful LeRobot demonstration
                    ↓
  Gemini structured task template
                    ↓
        Explicit user approval
                    ↓
LeRobot camera + sensors + proposed actions
          ┌─────────┴─────────┐
          ↓                   ↓
 YOLO/GRU local path   Gemini Live monitor
          └─────────┬─────────┘
                    ↓
       Local decision gate → alert or stop
                    ↓
                   GUI
```

## Current status

Phases 1–3 are implemented. Runtime OFF/MONITOR and the disabled-by-default ACTIVE safety gate are
implemented for integration and local validation. Machanize can wrap a LeRobot-compatible robot and automatically
record observations, proposed actions, executed actions, task text, and episode outcomes into a
local LeRobotDataset. Phase 2 records synchronized front (`/dev/video0`) and wrist (`/dev/video2`)
streams. Each completed episode receives a pending-review manifest.

## Local setup

Python 3.12 is required.

```text
uv sync --extra dev
uv sync --extra lerobot
uv sync --extra lerobot --extra vision
uv run pytest
```

The LeRobot commands install dataset support and the Feetech SDK required by SO-101; the `vision`
extra adds the Phase 3 API and YOLO tooling.

## Phase 3 dashboard

Phase 3 never connects to the robot and exposes no movement routes. Its analysis workflow selects
one successful episode, combines synchronized Front and Wrist evidence at approximately 5 FPS,
and uses `gemini-robotics-er-1.6-preview` through the official Google GenAI SDK's normal
`generateContent` API. Gemini analyzes the demonstration; its weights are never trained or changed.

```text
GEMINI_API_KEY=your-key uv run --extra lerobot --extra vision python training/serve_phase3.py
cd dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Select exactly one successful episode to generate an editable task
template. A native episode with an `unknown` outcome can also be used after explicitly confirming it
as a successful demonstration. Recorded failures remain blocked. Every generated template starts as
a draft and requires an explicit user approval action.

## Runtime monitoring

Runtime is a separate Raspberry Pi service. It owns the LeRobot connection and keeps the local
observation/action loop at its configured native FPS while sampling combined Front/Wrist frames,
joints, and ACT proposals for the AI observer at approximately 5 FPS. The local Pi gate runs before
every `adapter.execute()` and is the only component allowed to block an action. Gemini receives no
motor-control function.

```text
Mac                  Raspberry Pi
:5173 dashboard ───→ :8001 runtime API + LeRobot + local gate
     │
     └─────────────→ :8000 training/analysis API (no robot access)
```

On the Mac, set `VITE_RUNTIME_BASE_URL=http://raspberrypi.local:8001` in
`dashboard/.env.local`. On the Pi, configure the local LeRobot/ACT factory and start:

```text
MACHANIZE_RUNTIME_FACTORY=examples.so101.pi_runtime_factory:create_runtime_hardware \
uv run python training/serve_runtime.py
```

The Pi service binds to `0.0.0.0:8001`. It connects the cameras for preview but never starts ACT or
executes actions at backend startup. ACT requires an approved template, a connected MONITOR session,
and an explicit **Start ACT** click. **Stop ACT** blocks subsequent actions without closing the camera
preview. ACTIVE is disabled in the checked-in configuration. See [Raspberry Pi
runtime](docs/RUNTIME.md) for the complete Mac/Pi commands and safety prerequisites.

The existing object-detection workflow can also batch-label every frame from every available camera
with Grounding DINO. DINO labels are saved automatically without approval. You can optionally edit
them before training YOLO Nano on episode-isolated train and validation splits.

See [Architecture](docs/ARCHITECTURE.md), [Development phases](docs/PHASES.md), [Runtime cloud
monitoring](docs/RUNTIME.md), and the first [task
configuration](configs/projects/so101_blue_object_to_glass.yaml).
