Robot Supervisor SDK — V1 Plan

## Goal

Build an SDK that supervises a robot's existing frozen policy. The SDK watches the camera, robot sensors, and proposed actions, detects mistakes, and decides whether the robot should continue or stop.

V1 will only monitor and stop the robot. Automatic recovery and correction are future versions.

## Architecture

```text
Camera → YOLO → GRU → Error probability → Stop/Continue → GUI
                    ↑
       Sensors + proposed actions
```

- **YOLO:** Detects the task objects, target, and gripper.
- **GRU:** Watches how detections, sensors, and actions change over time.
- **Decision layer:** Compares the predicted risk with a threshold.
- **GUI:** Shows the video, detections, predicted state, risk score, and decision.
- **Original policy:** Remains frozen and performs the normal robot task.

## Modes

### 1. Training mode

Training mode records robot episodes and trains YOLO and the GRU offline.

Each episode should contain:

- Camera frames
- Timestamps
- Joint positions and velocities
- Gripper state
- Original policy's proposed actions
- Executed actions
- Task outcome
- Failure type and failure timestamp, when applicable

### 2. Runtime mode

Runtime mode loads the trained models and supervises the live robot.

```text
Robot observation
    ↓
Original policy proposes an action
    ↓
SDK evaluates recent observations and the action
    ├── Low risk → execute the action
    └── High risk → block the action and stop the robot
```

## Training YOLO

1. Record videos from successful and failed task attempts.
2. Extract representative frames.
3. Label bounding boxes for the required classes.
4. Split data by episode into training, validation, and test sets.
5. Train YOLO and measure detection quality on unseen episodes.

Example classes for a blue-object-to-glass task:

```text
blue_object
glass
gripper
```

## Training the GRU

Convert every frame into features such as:

```text
YOLO class, position, size, and confidence
joint positions and velocities
gripper state
original policy's proposed action
```

Group features into short sequences, initially 1–3 seconds, and label each sequence:

```text
normal
missed_blue_object
blue_object_dropped
wrong_direction
stuck
```

Train the GRU to output:

- Predicted state
- Failure probability

The dataset must contain normal behavior, failures, and the moments immediately before failures.

## Demonstration Analysis

Before downstream task-specific training, select one successful LeRobot episode with synchronized
Front and Wrist video. Sample it at approximately 5 FPS, render the views side by side, and include
timestamps, joint observations, recorded actions, proposed actions when available, and the task
description.

Send this evidence to `gemini-robotics-er-1.6-preview` with the official Google GenAI SDK and the
normal `generateContent` API. The model produces a structured task-template draft containing stages,
expected relationships and behavior, success conditions, possible failures, evidence, confidence,
and uncertainty. The GUI must allow editing and require explicit user approval; generated or edited
templates are never approved automatically. This analyzes the demonstration and does not train or
modify Gemini's weights.

## Runtime Decision Logic

```python
observation = robot.get_observation()
proposed_action = policy.select_action(observation)

risk = sdk.evaluate(
    observation=observation,
    proposed_action=proposed_action,
)

if risk < STOP_THRESHOLD:
    robot.send_action(proposed_action)
else:
    robot.stop()
```

The SDK should require several consecutive high-risk predictions before stopping, except when a hard safety rule is triggered.

## LeRobot Integration

The SDK will wrap the LeRobot control loop:

```text
LeRobot observation → original policy → proposed action
                              ↓
                         SDK monitor
                         ├── approve → send action
                         └── reject  → stop
```

The SDK must not modify the original policy's weights.

## Cloud Design

Recommended setup:

- **Cloud:** Model training, dataset storage, experiment tracking, and GUI history.
- **Robot computer:** YOLO/GRU inference and the stop/continue decision.
- **Hardware:** Independent emergency-stop system.

Cloud inference may be used for an early slow demo, but the robot must stop safely when the connection fails or the response exceeds a timeout.

## GUI Requirements

Display:

- Live camera feed
- YOLO bounding boxes
- Current predicted state
- Failure probability
- Proposed robot action
- SDK decision: `CONTINUE` or `STOP`
- Recent event log

## Development Milestones

1. Connect the SDK recorder to LeRobot.
2. Record successful and failed episodes for one task.
3. Label and train YOLO.
4. Generate sequence features and labels.
5. Train and evaluate the GRU.
6. Build the runtime monitor in shadow mode without stopping the robot.
7. Build the monitoring GUI.
8. Measure false alarms and missed failures.
9. Enable stopping with conservative thresholds.

## V1 Success Criteria

V1 is complete when it can:

- Integrate with a frozen LeRobot policy.
- Detect the required task objects.
- Recognize the defined normal and failure states.
- Display live predictions in the GUI.
- Stop the robot when failure risk crosses the threshold.
- Default to a safe stop if model inference or the network fails.

## Not Included in V1

- Automatic recovery
- Rewinding physical actions
- Learned correction policy
- Online modification of policy weights
- Autonomous retraining during runtime

These will be added after V1 reliably detects mistakes and safely stops the robot.
