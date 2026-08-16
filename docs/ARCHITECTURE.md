# Machanize V1 Architecture

## Components

| Component | Responsibility |
|---|---|
| Project manager | Stores the task, objects, mistakes, models, and thresholds for one project. |
| LeRobot adapter | Receives observations, actions, and episode events from LeRobot. |
| Episode recorder | Registers video, sensors, proposed actions, executed actions, and outcomes. |
| Evidence composer | Synchronizes Front and Wrist at about 5 FPS and renders side-by-side video with timestamps, joints, actions, and task text. |
| Gemini analyzer | Sends one successful demonstration to Gemini Robotics-ER 1.6 using the normal `generateContent` API and returns a structured draft. |
| Task-template review | Lets the operator edit the generated stages, relationships, behaviors, success criteria, failures, evidence, confidence, and uncertainty. |
| YOLO pipeline | Detects task objects and exposes detections for human review. |
| Sequence builder | Combines YOLO detections, robot sensors, and actions into time windows. |
| GRU pipeline | Predicts the current task state and failure risk from each time window. |
| Pi control loop | Owns LeRobot, keeps an observation-only camera preview active, and evaluates/gates ACT only after an explicit Start ACT request. |
| Runtime state hub | Publishes combined MJPEG and WebSocket telemetry without slowing the control loop. |
| Live monitoring provider | Holds the Gemini 3.1 Flash Live WebSocket, samples approximately 5 FPS plus telemetry, and accepts only the monitoring function call. |
| Decision engine | Validates task semantics and applies mode, confidence threshold, matching-failure window, and consecutive-prediction rules. |
| Safety controller | Keeps deterministic limits, watchdog, stop latch, and emergency-stop behavior local; Gemini cannot command motors. |
| Dashboard | Shows episodes, labels, detections, state, risk, and mode controls. |

## Training mode

```text
One recorded-success or user-confirmed unknown-outcome LeRobot episode
→ synchronized Front + Wrist samples at approximately 5 FPS
→ side-by-side evidence video + timestamped joint/action telemetry + task description
→ Gemini Robotics-ER 1.6 (`generateContent`, not Live API)
→ structured task-template draft
→ GUI review and editing
→ explicit user approval
→ approved task template
→ downstream visual and sequence training

LeRobot episodes
→ local MP4 and episode records
→ YOLO frame review
→ YOLO training
→ sequence creation
→ GRU state review
→ GRU training
→ one new untrained episode
→ runtime-ready model approval
```

Gemini analyzes the demonstration but is not trained or fine-tuned. Its output is never approved
automatically: generation and editing always produce `draft`, and only the explicit approval API can
produce `approved`.

The structured task template records:

- Task description and ordered stages.
- Expected object relationships, robot motion, and gripper behavior for each stage.
- Success conditions and possible failure types.
- Important timestamps and evidence.
- Confidence, uncertainty, source episode, model version, and user approval status.

## Phase 2 episode flow

```text
SO-101 follower + front camera (`/dev/video0`) + wrist camera (`/dev/video2`)
→ LeRobotAdapter
→ MachanizeLeRobotBridge.step(action)
→ robot.send_action(action)
→ one synchronized observation and `LeRobotDatasetSink.add_frame(...)`
→ EpisodeRecorder.finish_episode(...)
→ two MP4 streams + Parquet records + pending-review JSON manifest
```

Machanize stores both the policy's proposed action and the action actually accepted by the robot.
During teleoperation they are normally identical. Later, this distinction allows interventions to
be recorded without changing the original policy data.

Both camera frames receive the same LeRobot frame index and timestamp. This is software-level
synchronization at the observation loop; ordinary USB cameras are not hardware-triggered.

## Runtime mode

```text
Mac :5173 UI ─────────────────────────────────────────────┐
Mac :8000 analysis (never imports/owns robot) ────────────┤ approved template
                                                        ↓
Pi :8001 → LeRobot observation/preview ── explicit Start ACT ─→ local ACT proposal → gate → execute/block
             │                         ↑                       │
             ├→ combined MJPEG         └─ latest validated ────┤
             ├→ WebSocket telemetry           decision         │
             └→ separate ~5 FPS Gemini observer ───────────────┘
```

Backend startup never starts ACT or executes an action. Start ACT requires an approved template and
a connected MONITOR session. Stop ACT clears the execution permission before waiting for any
in-flight gate operation, then leaves the observation/camera loop connected.

The provider and local action-source boundaries are replaceable. Provider output is advisory; only
the Pi decision gate and stop latch can block execution. The STOP latch survives provider recovery
and requires an authenticated local reset while OFF. ACTIVE begins disabled, requires explicit
confirmation, and never enables recovery behavior. See [Raspberry Pi runtime](RUNTIME.md).

The physical emergency stop remains independent of Machanize.

## Local-first deployment

V1 stores and processes training videos, labels, and models on the development Mac. Runtime template
snapshots, decisions, gating, timeouts, watchdogs, and STOP state remain on the Raspberry Pi.
