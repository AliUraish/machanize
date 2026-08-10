"""Record SO-101 teleoperation episodes through Machanize.

This example requires Machanize's ``lerobot`` optional dependency and connected
SO-101 follower and leader arms. Run with ``--help`` for required device ports.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader

from machanize.adapters import LeRobotAdapter
from machanize.recording import EpisodeOutcome, EpisodeRecorder, LeRobotDatasetSink
from machanize.runtime import MachanizeLeRobotBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--camera", default="0")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--session-id")
    parser.add_argument("--data-root", type=Path, default=Path("data/episodes/pencil-to-glass"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_id = args.session_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session_root = args.data_root / session_id
    camera_source: int | str = int(args.camera) if args.camera.isdigit() else args.camera
    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id="machanize_follower",
            cameras={
                "front": OpenCVCameraConfig(
                    index_or_path=camera_source,
                    width=640,
                    height=480,
                    fps=args.fps,
                )
            },
        )
    )
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id="machanize_leader"))
    adapter = LeRobotAdapter(follower)
    sink = LeRobotDatasetSink.create(
        adapter,
        repo_id="machanize/pencil-to-glass",
        root=session_root / "lerobot",
        fps=args.fps,
    )
    recorder = EpisodeRecorder(
        sink,
        manifest_directory=session_root / "manifests",
        project_name="pencil-to-glass-demo",
        robot_type="so101",
    )
    bridge = MachanizeLeRobotBridge(
        adapter,
        recorder,
        task="Pick up a pencil and place it inside a glass.",
    )

    try:
        bridge.connect()
        leader.connect()
        for episode_number in range(1, args.episodes + 1):
            bridge.start_episode({"control": "teleoperation", "session_id": session_id})
            started_at = time.monotonic()
            try:
                while time.monotonic() - started_at < args.duration:
                    loop_started_at = time.monotonic()
                    bridge.step(leader.get_action())
                    time.sleep(max((1 / args.fps) - (time.monotonic() - loop_started_at), 0.0))
            except KeyboardInterrupt:
                pass

            prompt = f"Episode {episode_number} outcome [success/failure/unknown]: "
            outcome = EpisodeOutcome(input(prompt).strip().lower())
            manifest_path = bridge.finish_episode(outcome=outcome)
            print(f"Episode registered for review: {manifest_path}")
    except BaseException:
        if recorder.is_recording:
            bridge.abort_episode()
        raise
    finally:
        bridge.close()
        if leader.is_connected:
            leader.disconnect()


if __name__ == "__main__":
    main()
