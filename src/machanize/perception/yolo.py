"""Approved-label export, YOLO Nano training, and prediction review."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from machanize.perception.annotations import AnnotationStore, BoundingBox, PredictionStore
from machanize.perception.episodes import ExtractedFrame


@dataclass(frozen=True)
class YoloExport:
    root: Path
    data_yaml: Path
    class_names: tuple[str, ...]
    training_images: int
    validation_images: int
    camera_images: dict[str, int] = field(default_factory=dict)
    training_boxes: int = 0
    validation_boxes: int = 0


@dataclass(frozen=True)
class YoloModelRecord:
    model_id: str
    model_path: Path
    base_model: str
    created_at: str
    metrics: dict[str, Any]

    def to_api(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_path"] = str(self.model_path)
        return data


class YoloDatasetExporter:
    def __init__(self, annotation_store: AnnotationStore, export_root: str | Path) -> None:
        self.annotation_store = annotation_store
        self.export_root = Path(export_root).resolve()

    def export(
        self,
        *,
        class_names: list[str],
        export_id: str | None = None,
        episode_ids: list[str] | None = None,
    ) -> YoloExport:
        if len(set(class_names)) != len(class_names):
            raise ValueError("YOLO class names must be unique.")
        selected_episode_ids = set(episode_ids) if episode_ids else None
        annotations = self.annotation_store.approved(selected_episode_ids)
        if len(annotations) < 2:
            raise ValueError("Approve at least two labeled frames before YOLO training.")

        export_id = export_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        root = self.export_root / export_id
        class_to_id = {name: index for index, name in enumerate(class_names)}
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for annotation in annotations:
            synchronized_frame = (annotation["episode_id"], annotation["frame_id"])
            grouped.setdefault(synchronized_frame, []).append(annotation)
        if len(grouped) < 2:
            raise ValueError(
                "Approve labels from at least two different frame timestamps before YOLO training."
            )

        ordered_groups = sorted(grouped.items())
        episode_splits: dict[str, str] | None = None
        if selected_episode_ids is not None:
            labeled_episodes = sorted({str(item[0][0]) for item in ordered_groups})
            if len(labeled_episodes) < 2:
                raise ValueError("Auto-label at least two selected episodes before YOLO training.")
            validation_episode_count = max(1, round(len(labeled_episodes) * 0.2))
            validation_episodes = set(labeled_episodes[-validation_episode_count:])
            episode_splits = {
                episode_id: "val" if episode_id in validation_episodes else "train"
                for episode_id in labeled_episodes
            }
        validation_group_count = max(1, round(len(ordered_groups) * 0.2))
        training_count = 0
        validation_count = 0
        training_boxes = 0
        validation_boxes = 0
        camera_images: dict[str, int] = {}
        export_manifest: list[dict[str, Any]] = []

        for group_index, ((episode_id, _), group_annotations) in enumerate(ordered_groups):
            split = episode_splits.get(episode_id) if episode_splits else (
                "val" if group_index >= len(ordered_groups) - validation_group_count else "train"
            )
            for annotation in sorted(group_annotations, key=lambda item: item["camera_key"]):
                if split == "train":
                    training_count += 1
                    training_boxes += len(annotation["boxes"])
                else:
                    validation_count += 1
                    validation_boxes += len(annotation["boxes"])
                camera_key = str(annotation["camera_key"])
                camera_images[camera_key] = camera_images.get(camera_key, 0) + 1
                image_path = Path(annotation["image_path"])
                if not image_path.is_file():
                    raise FileNotFoundError(f"Approved frame is missing: {image_path}")
                stem = _annotation_stem(annotation)
                image_destination = root / "images" / split / f"{stem}{image_path.suffix.lower()}"
                label_destination = root / "labels" / split / f"{stem}.txt"
                image_destination.parent.mkdir(parents=True, exist_ok=True)
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, image_destination)
                label_destination.write_text(
                    _to_yolo_labels(annotation["boxes"], class_to_id),
                    encoding="utf-8",
                )
                export_manifest.append(
                    {
                        "episode_id": annotation["episode_id"],
                        "frame_id": annotation["frame_id"],
                        "camera_key": camera_key,
                        "split": split,
                        "image": str(image_destination.relative_to(root)),
                        "label": str(label_destination.relative_to(root)),
                    }
                )

        data_yaml = root / "data.yaml"
        if validation_boxes == 0:
            raise ValueError("Validation labels contain zero boxes; YOLO training was blocked.")
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(root),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {index: name for index, name in enumerate(class_names)},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (root / "machanize_export.json").write_text(
            json.dumps(
                {
                    "camera_images": camera_images,
                    "training_boxes": training_boxes,
                    "validation_boxes": validation_boxes,
                    "frames": export_manifest,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return YoloExport(
            root=root,
            data_yaml=data_yaml,
            class_names=tuple(class_names),
            training_images=training_count,
            validation_images=validation_count,
            camera_images=camera_images,
            training_boxes=training_boxes,
            validation_boxes=validation_boxes,
        )


class YoloTrainer:
    """Train and register an Ultralytics Nano detector."""

    def __init__(
        self,
        models_root: str | Path,
        *,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.models_root = Path(models_root).resolve()
        self._model_factory = model_factory

    def train(
        self,
        export: YoloExport,
        *,
        base_model: str = "yolo26n.pt",
        epochs: int = 50,
        image_size: int = 640,
        device: str | None = None,
    ) -> YoloModelRecord:
        if epochs < 1:
            raise ValueError("YOLO training epochs must be positive.")
        model_id = f"yolo-nano-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
        run_root = self.models_root / model_id
        factory = self._model_factory or _ultralytics_factory
        model = factory(base_model)
        results = model.train(
            data=str(export.data_yaml),
            epochs=epochs,
            imgsz=image_size,
            device=device,
            project=str(run_root / "training"),
            name="run",
            exist_ok=True,
        )
        source_model = Path(results.save_dir) / "weights" / "best.pt"
        if not source_model.is_file():
            raise RuntimeError(f"YOLO training completed without best.pt: {source_model}")
        run_root.mkdir(parents=True, exist_ok=True)
        model_path = run_root / "best.pt"
        shutil.copy2(source_model, model_path)
        metrics = _json_safe(getattr(results, "results_dict", {}))
        record = YoloModelRecord(
            model_id=model_id,
            model_path=model_path,
            base_model=base_model,
            created_at=datetime.now(UTC).isoformat(),
            metrics=metrics,
        )
        (run_root / "model.json").write_text(
            json.dumps(record.to_api(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return record

    def list_models(self) -> list[YoloModelRecord]:
        records = []
        for path in sorted(self.models_root.glob("*/model.json"), reverse=True):
            raw = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                YoloModelRecord(
                    model_id=str(raw["model_id"]),
                    model_path=Path(raw["model_path"]),
                    base_model=str(raw["base_model"]),
                    created_at=str(raw["created_at"]),
                    metrics=dict(raw.get("metrics", {})),
                )
            )
        return records

    def get_model(self, model_id: str) -> YoloModelRecord:
        for model in self.list_models():
            if model.model_id == model_id:
                return model
        raise KeyError(f"Unknown YOLO model: {model_id}")


class YoloPredictor:
    def __init__(
        self,
        prediction_store: PredictionStore,
        *,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.prediction_store = prediction_store
        self._model_factory = model_factory

    def predict(
        self,
        model: YoloModelRecord,
        frames: list[ExtractedFrame],
        *,
        confidence: float = 0.25,
    ) -> int:
        if not frames:
            raise ValueError("Extract frames before running predictions.")
        detector = (self._model_factory or _ultralytics_factory)(str(model.model_path))
        results = detector.predict(
            source=[str(frame.image_path) for frame in frames],
            conf=confidence,
            verbose=False,
        )
        names = detector.names
        store = self.prediction_store.for_model(model.model_id)
        for frame, result in zip(frames, results, strict=True):
            boxes = []
            if result.boxes is not None:
                for xywhn, class_id, score in zip(
                    result.boxes.xywhn.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    strict=True,
                ):
                    boxes.append(
                        BoundingBox(
                            class_name=str(names[int(class_id)]),
                            x_center=float(xywhn[0]),
                            y_center=float(xywhn[1]),
                            width=float(xywhn[2]),
                            height=float(xywhn[3]),
                            confidence=float(score),
                        )
                    )
            store.save(
                episode_id=frame.episode_id,
                camera_key=frame.camera_key,
                frame_id=frame.frame_id,
                image_path=frame.image_path,
                boxes=boxes,
                approved=False,
                source="prediction",
                model_id=model.model_id,
            )
        return len(frames)


def _ultralytics_factory(model_path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path)


def _annotation_stem(annotation: dict[str, Any]) -> str:
    identity = f"{annotation['episode_id']}:{annotation['camera_key']}:{annotation['frame_id']}"
    camera_name = _safe_camera_name(str(annotation["camera_key"]).rsplit(".", 1)[-1])
    return f"{camera_name}-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


def _safe_camera_name(value: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in value)
    return safe.strip("_") or "camera"


def _to_yolo_labels(boxes: list[dict[str, Any]], class_to_id: dict[str, int]) -> str:
    lines = []
    for box in boxes:
        class_name = str(box["class_name"])
        if class_name not in class_to_id:
            raise ValueError(f"Unknown class in approved annotation: {class_name}")
        lines.append(
            " ".join(
                [
                    str(class_to_id[class_name]),
                    f"{float(box['x_center']):.8f}",
                    f"{float(box['y_center']):.8f}",
                    f"{float(box['width']):.8f}",
                    f"{float(box['height']):.8f}",
                ]
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
