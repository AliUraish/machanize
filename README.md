# Machanize

Machanize is a supervisory SDK for robot policies. V1 observes a frozen LeRobot policy, learns task-specific failure patterns from reviewed episodes, and can alert or safely stop the robot at runtime.

## V1 modes

- **Training:** Record episodes, review labels, train YOLO and GRU, and approve models.
- **Runtime OFF:** Machanize does not monitor or intervene.
- **Runtime MONITOR:** Machanize detects and reports risk without controlling the robot.
- **Runtime ACTIVE:** Machanize detects risk and can block an action or safely stop the robot.

## V1 pipeline

```text
LeRobot camera + sensors + proposed actions
                    ↓
              YOLO detections
                    ↓
         GRU state and risk prediction
                    ↓
       Decision engine → alert or stop
                    ↓
                   GUI
```

## Current status

Phases 1 and 2 are complete, and Phase 3 is implemented for local visual review. Machanize can wrap a LeRobot-compatible robot and automatically
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

Phase 3 never connects to the robot and exposes no movement routes.

```text
uv run python training/serve_phase3.py
cd dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, select multiple episodes, and batch-label every frame from every available camera
with Grounding DINO. DINO labels are saved automatically without approval. You can optionally edit
them before training YOLO Nano on episode-isolated train and validation splits.

See [Architecture](docs/ARCHITECTURE.md), [Development phases](docs/PHASES.md), and the first [task configuration](configs/projects/so101_blue_object_to_glass.yaml).
