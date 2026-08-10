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

Phases 1 and 2 are complete. Machanize can wrap a LeRobot-compatible robot and automatically
record observations, proposed actions, executed actions, task text, and episode outcomes into a
local LeRobotDataset. Each completed episode receives a pending-review manifest.

## Local setup

Python 3.12 is required.

```text
uv sync --extra dev
uv sync --extra lerobot
uv run pytest
```

The second command installs LeRobot dataset support and the Feetech SDK required by SO-101.

See [Architecture](docs/ARCHITECTURE.md), [Development phases](docs/PHASES.md), and the first [task configuration](configs/projects/so101_pencil_to_glass.yaml).
