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

Phase 1 foundation is defined. No model training or robot-control implementation exists yet.

See [Architecture](docs/ARCHITECTURE.md), [Development phases](docs/PHASES.md), and the first [task configuration](configs/projects/so101_pencil_to_glass.yaml).

