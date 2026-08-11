# SO-101 Phase 2 Recording

This example connects a leader and follower SO-101, records every observation and action through
Machanize, and registers each completed episode for later YOLO/GRU review.

## Setup

```text
uv sync --extra lerobot
```

Calibrate the leader and follower through LeRobot before using this example.

## Record

```text
uv run python examples/so101/record_with_machanize.py \
  --follower-port /dev/tty.usbmodem_FOLLOWER \
  --leader-port /dev/tty.usbmodem_LEADER \
  --front-camera /dev/video0 \
  --wrist-camera /dev/video2 \
  --episodes 1
```

Each run creates a timestamped local session inside `data/episodes/blue-object-to-glass/`. Raw camera
and sensor data uses LeRobot's MP4/Parquet format. Machanize manifests are written beside it.
The front and wrist frames are captured during the same observation loop and stored with the same
LeRobot frame index and timestamp so Phase 3 can review them as a synchronized pair.
