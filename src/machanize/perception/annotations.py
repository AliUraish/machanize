"""YOLO annotation and prediction-review persistence."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SAFE_PART = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class BoundingBox:
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float | None = None

    def validate(self) -> None:
        if not self.class_name.strip():
            raise ValueError("Bounding-box class name cannot be empty.")
        values = (self.x_center, self.y_center, self.width, self.height)
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("Bounding-box coordinates must be normalized between 0 and 1.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Bounding-box width and height must be positive.")
        if self.x_center - self.width / 2 < 0 or self.x_center + self.width / 2 > 1:
            raise ValueError("Bounding box crosses the image's horizontal boundary.")
        if self.y_center - self.height / 2 < 0 or self.y_center + self.height / 2 > 1:
            raise ValueError("Bounding box crosses the image's vertical boundary.")


class AnnotationStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def save(
        self,
        *,
        episode_id: str,
        camera_key: str,
        frame_id: str,
        image_path: str | Path,
        boxes: list[BoundingBox],
        approved: bool,
        source: str = "manual",
        model_id: str | None = None,
    ) -> Path:
        for box in boxes:
            box.validate()
        record = {
            "episode_id": episode_id,
            "camera_key": camera_key,
            "frame_id": frame_id,
            "image_path": str(Path(image_path).resolve()),
            "boxes": [asdict(box) for box in boxes],
            "approved": approved,
            "source": source,
            "model_id": model_id,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        path = self.path_for(episode_id, camera_key, frame_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def get(self, episode_id: str, camera_key: str, frame_id: str) -> dict[str, Any] | None:
        path = self.path_for(episode_id, camera_key, frame_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def approved(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.root.rglob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("approved") is True:
                records.append(record)
        return records

    def path_for(self, episode_id: str, camera_key: str, frame_id: str) -> Path:
        return self.root / _safe(episode_id) / _safe(camera_key) / f"{_safe(frame_id)}.json"


class PredictionStore(AnnotationStore):
    def for_model(self, model_id: str) -> PredictionStore:
        return PredictionStore(self.root / _safe(model_id))


def _safe(value: str) -> str:
    return _SAFE_PART.sub("_", value).strip("._") or "item"
