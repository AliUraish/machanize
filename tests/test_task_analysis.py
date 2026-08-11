import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from machanize.analysis.task_template import (
    EpisodeEvidenceBuilder,
    PossibleFailure,
    TaskAnalysisJobManager,
    TaskStage,
    TaskTemplateDraft,
    TaskTemplateStore,
    TimestampEvidence,
)
from machanize.perception.episodes import EpisodeRecord


def _episode(tmp_path: Path, *, outcome: str = "success") -> EpisodeRecord:
    dataset_root = tmp_path / "lerobot"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text(
        """{
          "fps": 30,
          "features": {
            "observation.state": {"names": ["joint", "gripper"]},
            "action": {"names": ["joint", "gripper"]},
            "machanize.proposed_action": {"names": ["joint", "gripper"]}
          }
        }""",
        encoding="utf-8",
    )
    return EpisodeRecord(
        episode_id="episode-success",
        dataset_root=dataset_root,
        dataset_episode_index=0,
        project_name="demo",
        robot_type="so101",
        task="Place the blue object in the glass.",
        outcome=outcome,
        review_status="pending",
        processing_status="pending",
        frame_count=12,
        camera_keys=("observation.images.front", "observation.images.wrist"),
    )


def _sample(index: int) -> dict:
    return {
        "frame_index": index,
        "timestamp": index / 30,
        "observation.images.front": np.full((20, 30, 3), 20 + index, dtype=np.uint8),
        "observation.images.wrist": np.full((20, 30, 3), 80 + index, dtype=np.uint8),
        "observation.state": np.array([index, 0.5], dtype=np.float32),
        "action": np.array([index + 1, 0.6], dtype=np.float32),
        "machanize.proposed_action": np.array([index + 2, 0.7], dtype=np.float32),
    }


def _draft() -> TaskTemplateDraft:
    evidence = TimestampEvidence(timestamp_seconds=1.0, description="The gripper closes.")
    return TaskTemplateDraft(
        task_description="Place the blue object in the glass.",
        ordered_task_stages=[
            TaskStage(
                name="grasping",
                description="Close the gripper around the object.",
                start_time_seconds=0.5,
                end_time_seconds=1.2,
                expected_object_relationships=["gripper surrounds blue object"],
                expected_robot_behavior="Hold position over the object.",
                expected_gripper_behavior="Close around the object.",
                evidence=[evidence],
                confidence=0.9,
                uncertainty=[],
            )
        ],
        success_conditions=["The blue object is inside the glass."],
        possible_failure_types=[
            PossibleFailure(
                failure_type="missed grasp",
                description="The gripper closes without the object.",
                related_stage_names=["grasping"],
                detectable_evidence=["Object does not move with gripper."],
            )
        ],
        important_timestamps_and_evidence=[evidence],
        confidence=0.88,
        uncertainty=["Wrist view is briefly occluded."],
    )


def test_evidence_builder_combines_front_wrist_and_telemetry_at_five_fps(
    tmp_path: Path,
) -> None:
    encoded: dict[str, object] = {}

    def encode(frames: Path, destination: Path, fps: float) -> None:
        images = sorted(frames.glob("*.png"))
        encoded["count"] = len(images)
        encoded["fps"] = fps
        encoded["size"] = Image.open(images[0]).size
        destination.write_bytes(b"video")

    builder = EpisodeEvidenceBuilder(
        tmp_path / "evidence",
        dataset_factory=lambda _: [_sample(index) for index in range(12)],
        video_encoder=encode,
    )
    evidence = builder.build(_episode(tmp_path), fps=5)

    assert encoded == {"count": 2, "fps": 5.0, "size": (1280, 620)}
    assert evidence.video_path.read_bytes() == b"video"
    assert [row["frame_index"] for row in evidence.telemetry] == [0, 6]
    assert evidence.telemetry[1]["joint_observations"]["joint"] == pytest.approx(6)
    assert evidence.telemetry[1]["recorded_actions"]["gripper"] == pytest.approx(0.6)
    assert evidence.telemetry[1]["proposed_actions"]["joint"] == pytest.approx(8)


def test_evidence_builder_rejects_failed_episode(tmp_path: Path) -> None:
    builder = EpisodeEvidenceBuilder(tmp_path / "evidence", dataset_factory=lambda _: [])
    with pytest.raises(ValueError, match="successful episode"):
        builder.build(_episode(tmp_path, outcome="failure"))


def test_task_template_requires_explicit_approval(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    store = TaskTemplateStore(tmp_path / "templates")

    generated = store.create_model_draft(
        episode,
        _draft(),
        model_version="gemini-robotics-er-1.6-preview",
        video_fps=5,
    )
    assert generated.approval_status == "draft"
    assert generated.approved_at is None

    approved = store.approve(episode.episode_id)
    assert approved.approval_status == "approved"
    assert approved.approved_at is not None

    edited = _draft().model_copy(update={"confidence": 0.75})
    saved = store.save_user_draft(episode.episode_id, edited)
    assert saved.approval_status == "draft"
    assert saved.approved_at is None
    assert saved.confidence == pytest.approx(0.75)


def test_unknown_episode_requires_success_confirmation(tmp_path: Path) -> None:
    analyzed = []

    class FakeService:
        def analyze(self, episode: EpisodeRecord):
            analyzed.append(episode)
            return SimpleNamespace(model_dump=lambda **_: {"approval_status": "draft"})

    manager = TaskAnalysisJobManager(FakeService())
    episode = _episode(tmp_path, outcome="unknown")

    with pytest.raises(ValueError, match="Confirm"):
        manager.submit(episode)

    job = manager.submit(episode, confirm_unknown_as_success=True)
    deadline = time.monotonic() + 2
    while manager.to_api(job.job_id)["status"] in {"queued", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert manager.to_api(job.job_id)["status"] == "completed"
    assert analyzed[0].outcome == "success"
