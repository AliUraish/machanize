from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from fastapi.testclient import TestClient

from machanize.adapters import LeRobotAdapter
from machanize.analysis.task_template import TaskTemplateDraft, TaskTemplateStore
from machanize.perception.annotations import AnnotationStore, BoundingBox, PredictionStore
from machanize.perception.episodes import EpisodeRepository, ExtractedFrame, FrameExtractor
from machanize.perception.yolo import (
    YoloDatasetExporter,
    YoloExport,
    YoloModelRecord,
    YoloPredictor,
    YoloTrainer,
)
from machanize.phase3.api import Phase3Settings, create_app
from machanize.phase3.jobs import LabelingJobManager
from machanize.recording import EpisodeOutcome, EpisodeRecorder, LeRobotDatasetSink


class DatasetRobot:
    is_connected = False
    observation_features: ClassVar = {
        "shoulder.pos": float,
        "front": (64, 64, 3),
        "wrist": (64, 64, 3),
    }
    action_features: ClassVar = {"shoulder.pos": float}


@pytest.fixture(scope="module")
def real_pi_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_root = tmp_path_factory.mktemp("phase3") / "episodes"
    session_root = data_root / "20260810T120000Z"
    adapter = LeRobotAdapter(DatasetRobot())
    sink = LeRobotDatasetSink.create(
        adapter,
        repo_id="machanize/phase3-test",
        root=session_root / "lerobot",
        fps=10,
    )
    recorder = EpisodeRecorder(
        sink,
        manifest_directory=session_root / "manifests",
        project_name="blue-object-to-glass-demo",
        robot_type="so101",
    )
    recorder.start_episode(task="Pick up a blue object and place it inside a glass.")
    for index in range(6):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[12:28, 5 + index * 4 : 20 + index * 4, 0] = 255
        wrist_image = np.zeros((64, 64, 3), dtype=np.uint8)
        wrist_image[32:48, 8 + index * 3 : 23 + index * 3, 1] = 255
        recorder.record_step(
            observation={
                "shoulder.pos": float(index),
                "front": image,
                "wrist": wrist_image,
            },
            proposed_action={"shoulder.pos": float(index + 1)},
            executed_action={"shoulder.pos": float(index + 1)},
        )
    recorder.finish_episode(outcome=EpisodeOutcome.SUCCESS)
    recorder.close()
    return data_root


def test_real_lerobot_episode_is_discovered_and_extracted(
    real_pi_data_root: Path,
    tmp_path: Path,
) -> None:
    repository = EpisodeRepository(real_pi_data_root)
    episodes = repository.list()

    assert len(episodes) == 1
    assert episodes[0].robot_type == "so101"
    assert episodes[0].camera_keys == (
        "observation.images.front",
        "observation.images.wrist",
    )

    extractor = FrameExtractor(tmp_path / "frames")
    synchronized = extractor.extract_synchronized(episodes[0], stride=2)
    front_frames = synchronized["observation.images.front"]
    wrist_frames = synchronized["observation.images.wrist"]

    assert [frame.frame_index for frame in front_frames] == [0, 2, 4]
    assert [frame.frame_index for frame in wrist_frames] == [0, 2, 4]
    assert [frame.timestamp for frame in front_frames] == [
        frame.timestamp for frame in wrist_frames
    ]
    assert all(frame.image_path.is_file() for frame in front_frames + wrist_frames)
    assert all((frame.width, frame.height) == (64, 64) for frame in front_frames + wrist_frames)


def test_approved_annotations_export_to_yolo_format(
    real_pi_data_root: Path,
    tmp_path: Path,
) -> None:
    episode = EpisodeRepository(real_pi_data_root).list()[0]
    frames_by_camera = FrameExtractor(tmp_path / "frames").extract_synchronized(
        episode,
        stride=2,
    )
    store = AnnotationStore(tmp_path / "labels")
    for frames in frames_by_camera.values():
        for frame in frames[:2]:
            store.save(
                episode_id=episode.episode_id,
                camera_key=frame.camera_key,
                frame_id=frame.frame_id,
                image_path=frame.image_path,
                boxes=[BoundingBox("blue_object", 0.4, 0.4, 0.2, 0.2)],
                approved=True,
            )

    export = YoloDatasetExporter(store, tmp_path / "exports").export(
        class_names=["blue_object", "glass", "gripper"],
        export_id="test",
    )

    assert export.training_images == 2
    assert export.validation_images == 2
    assert export.camera_images == {
        "observation.images.front": 2,
        "observation.images.wrist": 2,
    }
    assert len(list((export.root / "images" / "train").glob("*.jpg"))) == 2
    assert len(list((export.root / "labels" / "val").glob("*.txt"))) == 2
    assert "blue_object" in export.data_yaml.read_text()
    export_manifest = json.loads((export.root / "machanize_export.json").read_text())
    frame_splits: dict[str, set[str]] = {}
    for frame in export_manifest["frames"]:
        frame_splits.setdefault(frame["frame_id"], set()).add(frame["split"])
    assert all(len(splits) == 1 for splits in frame_splits.values())


class FakeTrainingModel:
    def train(self, **kwargs):
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        weights = save_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"trained")
        return SimpleNamespace(save_dir=save_dir, results_dict={"map50": 0.8})


def test_yolo_trainer_registers_best_model(tmp_path: Path) -> None:
    export_root = tmp_path / "export"
    export_root.mkdir()
    data_yaml = export_root / "data.yaml"
    data_yaml.write_text("names: {0: blue_object}\n")
    export = YoloExport(export_root, data_yaml, ("blue_object",), 1, 1)

    trainer = YoloTrainer(tmp_path / "models", model_factory=lambda _: FakeTrainingModel())
    record = trainer.train(export, epochs=1, image_size=320, device="cpu")

    assert record.model_path.read_bytes() == b"trained"
    assert record.metrics == {"map50": 0.8}
    assert trainer.get_model(record.model_id).model_path == record.model_path


class FakeTensorValues:
    def __init__(self, values) -> None:
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakePredictionModel:
    names: ClassVar = {0: "blue_object"}

    def predict(self, source, conf, verbose):
        boxes = SimpleNamespace(
            xywhn=FakeTensorValues([[0.5, 0.5, 0.25, 0.25]]),
            cls=FakeTensorValues([0]),
            conf=FakeTensorValues([0.91]),
        )
        return [SimpleNamespace(boxes=boxes) for _ in source]


class FakeGroundingDino:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, float]] = []

    def detect(self, image_path, *, threshold):
        path = Path(image_path)
        self.calls.append((path, threshold))
        return [
            BoundingBox("blue_object", 0.5, 0.5, 0.25, 0.25, 0.92),
            BoundingBox("gripper", 0.25, 0.25, 0.2, 0.2, 0.81),
        ]


def test_prediction_review_is_saved_unapproved(
    real_pi_data_root: Path,
    tmp_path: Path,
) -> None:
    episode = EpisodeRepository(real_pi_data_root).list()[0]
    frames = FrameExtractor(tmp_path / "frames").extract(episode, stride=3)
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")
    model = YoloModelRecord("model-1", model_path, "yolo26n.pt", "now", {})
    store = PredictionStore(tmp_path / "predictions")

    count = YoloPredictor(store, model_factory=lambda _: FakePredictionModel()).predict(
        model,
        frames,
    )
    prediction = store.for_model("model-1").get(
        episode.episode_id,
        frames[0].camera_key,
        frames[0].frame_id,
    )

    assert count == 2
    assert prediction is not None
    assert prediction["approved"] is False
    assert prediction["boxes"][0]["confidence"] == pytest.approx(0.91)


def test_phase3_api_is_read_only_for_robot(
    real_pi_data_root: Path,
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parent.parent
    settings = Phase3Settings(
        repository_root=repository_root,
        data_root=real_pi_data_root,
        frame_cache_root=tmp_path / "cache",
        labels_root=tmp_path / "labels",
        predictions_root=tmp_path / "predictions",
        yolo_exports_root=tmp_path / "exports",
        models_root=tmp_path / "models",
        project_config=repository_root / "configs/projects/so101_blue_object_to_glass.yaml",
    )
    grounding_dino = FakeGroundingDino()
    client = TestClient(create_app(settings, grounding_dino=grounding_dino))

    health = client.get("/api/health")
    episodes = client.get("/api/episodes")
    openapi = client.get("/openapi.json").json()
    episode = episodes.json()[0]
    camera_key = "observation.images.wrist"
    extracted = client.post(
        f"/api/episodes/{episode['episode_id']}/extract-synchronized",
        json={"camera_keys": episode["camera_keys"], "stride": 3},
    )
    assert extracted.json()["synchronized_frame_count"] == 2
    assert set(extracted.json()["frames"]) == set(episode["camera_keys"])
    frame = extracted.json()["frames"][camera_key][0]
    suggested = client.post(
        f"/api/episodes/{episode['episode_id']}/frames/{frame['frame_id']}/auto-label",
        json={"camera_key": camera_key, "confidence": 0.4},
    )
    assert len(list(settings.labels_root.rglob("*.json"))) == 1
    saved = client.put(
        f"/api/episodes/{episode['episode_id']}/frames/{frame['frame_id']}/annotation",
        json={
            "camera_key": camera_key,
            "boxes": suggested.json()["boxes"][:1],
            "approved": True,
            "source": "grounding_dino",
        },
    )

    assert health.status_code == 200
    assert health.json()["robot_movement_enabled"] is False
    assert episodes.status_code == 200
    assert len(episodes.json()) == 1
    assert extracted.status_code == 200
    assert suggested.status_code == 200
    assert suggested.json()["approved"] is True
    assert suggested.json()["source"] == "grounding_dino"
    assert [box["class_name"] for box in suggested.json()["boxes"]] == [
        "blue_object",
        "gripper",
    ]
    assert grounding_dino.calls[0][0].parent.name == "observation.images.wrist"
    assert grounding_dino.calls[0][1] == pytest.approx(0.4)
    assert saved.status_code == 200
    assert saved.json()["approved"] is True
    saved_annotation = next(settings.labels_root.rglob("*.json"))
    saved_record = json.loads(saved_annotation.read_text())
    assert saved_record["source"] == "grounding_dino"
    assert [box["class_name"] for box in saved_record["boxes"]] == ["blue_object"]
    assert not any("/robot" in path or "/control" in path for path in openapi["paths"])


def test_task_template_api_never_approves_without_explicit_confirmation(
    real_pi_data_root: Path,
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parent.parent
    settings = Phase3Settings(
        repository_root=repository_root,
        data_root=real_pi_data_root,
        frame_cache_root=tmp_path / "cache",
        labels_root=tmp_path / "labels",
        predictions_root=tmp_path / "predictions",
        yolo_exports_root=tmp_path / "exports",
        models_root=tmp_path / "models",
        project_config=repository_root / "configs/projects/so101_blue_object_to_glass.yaml",
    )
    episode = EpisodeRepository(real_pi_data_root).list()[0]
    store = TaskTemplateStore(tmp_path / "templates")
    draft = TaskTemplateDraft.model_validate(
        {
            "task_description": episode.task,
            "ordered_task_stages": [
                {
                    "name": "success",
                    "description": "The object is inside the glass.",
                    "start_time_seconds": 0.4,
                    "end_time_seconds": 0.5,
                    "expected_object_relationships": ["object inside glass"],
                    "expected_robot_behavior": "Stop moving.",
                    "expected_gripper_behavior": "Release the object.",
                    "evidence": [{"timestamp_seconds": 0.5, "description": "Object is in glass."}],
                    "confidence": 0.9,
                    "uncertainty": [],
                }
            ],
            "success_conditions": ["Object remains inside glass."],
            "possible_failure_types": [
                {
                    "failure_type": "incorrect placement",
                    "description": "Object lands outside the glass.",
                    "related_stage_names": ["success"],
                    "detectable_evidence": ["Object is not inside glass."],
                }
            ],
            "important_timestamps_and_evidence": [
                {"timestamp_seconds": 0.5, "description": "Placement completed."}
            ],
            "confidence": 0.9,
            "uncertainty": [],
        }
    )
    store.create_model_draft(
        episode,
        draft,
        model_version="gemini-robotics-er-1.6-preview",
        video_fps=5,
    )
    analysis_service = SimpleNamespace(store=store, analyze=lambda _: None)
    client = TestClient(create_app(settings, task_analysis=analysis_service))

    expired_job = client.get("/api/analysis/jobs/process-restarted-job")
    generated = client.get(f"/api/analysis/templates/{episode.episode_id}")
    edited_payload = generated.json()
    edited_payload["confidence"] = 0.8
    saved = client.put(
        f"/api/analysis/templates/{episode.episode_id}",
        json=edited_payload,
    )
    rejected_approval = client.post(
        f"/api/analysis/templates/{episode.episode_id}/approve",
        json={"confirm": False},
    )
    approved = client.post(
        f"/api/analysis/templates/{episode.episode_id}/approve",
        json={"confirm": True},
    )

    assert expired_job.status_code == 410
    assert "API process may have restarted" in expired_job.json()["detail"]
    assert generated.json()["approval_status"] == "draft"
    assert saved.json()["approval_status"] == "draft"
    assert saved.json()["confidence"] == pytest.approx(0.8)
    assert rejected_approval.status_code == 422
    assert approved.json()["approval_status"] == "approved"
    assert approved.json()["approved_at"] is not None


def test_bounding_box_rejects_image_boundary_crossing() -> None:
    with pytest.raises(ValueError, match="horizontal boundary"):
        BoundingBox("blue_object", 0.05, 0.5, 0.2, 0.2).validate()


class FakeEpisodeRepository:
    def __init__(self) -> None:
        self.records = {
            episode_id: SimpleNamespace(
                episode_id=episode_id,
                camera_keys=(
                    "observation.images.front",
                    "observation.images.wrist",
                    "observation.images.overhead",
                ),
            )
            for episode_id in ("episode-1", "episode-2")
        }

    def get(self, episode_id: str):
        try:
            return self.records[episode_id]
        except KeyError as error:
            raise KeyError(f"Unknown episode: {episode_id}") from error


class FakeBatchFrameExtractor:
    def __init__(self, root: Path) -> None:
        self.root = root

    def extract_synchronized(self, episode, *, camera_keys, stride, overwrite):
        assert stride == 1
        assert overwrite is True
        frames = {}
        for camera_key in camera_keys:
            camera_frames = []
            for index in range(2):
                path = self.root / episode.episode_id / camera_key / f"{index:06d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frame")
                camera_frames.append(
                    ExtractedFrame(
                        episode_id=episode.episode_id,
                        camera_key=camera_key,
                        frame_id=f"{index:06d}",
                        frame_index=index,
                        timestamp=index / 10,
                        image_path=path,
                        width=64,
                        height=64,
                    )
                )
            frames[camera_key] = camera_frames
        return frames


def test_batch_dino_auto_approves_multiple_episodes_and_exports_without_leakage(
    tmp_path: Path,
) -> None:
    episodes = FakeEpisodeRepository()
    frames = FakeBatchFrameExtractor(tmp_path / "frames")
    annotations = AnnotationStore(tmp_path / "labels")
    detector = FakeGroundingDino()
    manager = LabelingJobManager(episodes, frames, annotations, detector)

    job = manager.submit(["episode-1", "episode-2"], confidence=0.5)
    deadline = time.monotonic() + 3
    while manager.get(job.job_id).status in {"queued", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    completed = manager.get(job.job_id)
    assert completed.status == "completed"
    assert completed.processed_frames == 12
    assert completed.labeled_frames == 12
    assert completed.total_boxes == 24
    assert completed.errors == []
    records = annotations.approved()
    assert len(records) == 12
    assert all(record["source"] == "grounding_dino" for record in records)
    assert all(record["approved"] is True for record in records)
    assert all(record["model_id"] == "IDEA-Research/grounding-dino-tiny" for record in records)

    export = YoloDatasetExporter(annotations, tmp_path / "exports").export(
        class_names=["blue_object", "glass", "gripper"],
        export_id="selected-episodes",
        episode_ids=["episode-1", "episode-2"],
    )
    manifest = json.loads((export.root / "machanize_export.json").read_text())
    episode_splits: dict[str, set[str]] = {}
    for frame in manifest["frames"]:
        episode_splits.setdefault(frame["episode_id"], set()).add(frame["split"])
    assert episode_splits == {"episode-1": {"train"}, "episode-2": {"val"}}
    assert export.camera_images == {
        "observation.images.front": 4,
        "observation.images.overhead": 4,
        "observation.images.wrist": 4,
    }
    assert export.training_boxes == 12
    assert export.validation_boxes == 12


def test_manifest_stays_pending_until_label_review(real_pi_data_root: Path) -> None:
    manifest = next(real_pi_data_root.rglob("manifests/*.json"))
    data = json.loads(manifest.read_text())

    assert data["review_status"] == "pending"
    assert data["processing_status"] == "pending"
