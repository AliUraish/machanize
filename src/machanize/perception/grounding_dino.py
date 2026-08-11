"""Grounding DINO labels for Machanize Phase 3."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from PIL import Image

from machanize.perception.annotations import BoundingBox

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
CLASS_PROMPTS = ("a blue object", "a drinking glass", "a robot gripper")


class GroundingDinoBackend(Protocol):
    def predict(
        self,
        image: Image.Image,
        prompts: tuple[str, ...],
        threshold: float,
    ) -> list[dict[str, Any]]: ...


class GroundingDinoDetector:
    """Lazily load Grounding DINO and return normalized detections."""

    def __init__(
        self,
        *,
        backend_factory: Callable[[], GroundingDinoBackend] | None = None,
    ) -> None:
        self._backend_factory = backend_factory or TransformersGroundingDinoBackend
        self._backend: GroundingDinoBackend | None = None
        self._lock = Lock()

    def detect(self, image_path: str | Path, *, threshold: float) -> list[BoundingBox]:
        if not 0 < threshold <= 1:
            raise ValueError("Grounding DINO confidence must be between 0 and 1.")
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Frame image is missing: {path}")

        with Image.open(path) as source:
            image = source.convert("RGB")
        with self._lock:
            backend = self._backend_instance()
            raw_detections = backend.predict(image, CLASS_PROMPTS, threshold)
        return _normalize_detections(raw_detections, image.width, image.height, threshold)

    def _backend_instance(self) -> GroundingDinoBackend:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend


class TransformersGroundingDinoBackend:
    """Transformers implementation kept behind a lazy boundary for local startup speed."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Grounding DINO dependencies are missing. Install Machanize with the "
                "'vision' optional dependency."
            ) from error

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._model.to(self._device)
        self._model.eval()

    def predict(
        self,
        image: Image.Image,
        prompts: tuple[str, ...],
        threshold: float,
    ) -> list[dict[str, Any]]:
        text_labels = [list(prompts)]
        inputs = self._processor(images=image, text=text_labels, return_tensors="pt").to(
            self._device
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = result["text_labels"] if "text_labels" in result else result.get("labels", [])
        return [
            {
                "label": str(label),
                "score": float(score.item() if hasattr(score, "item") else score),
                "box": box.tolist() if hasattr(box, "tolist") else list(box),
            }
            for box, score, label in zip(
                result.get("boxes", []),
                result.get("scores", []),
                labels,
                strict=True,
            )
        ]


def _normalize_detections(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    threshold: float,
) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    for detection in detections:
        score = float(detection["score"])
        class_name = _canonical_class(str(detection["label"]))
        if class_name is None or score < threshold:
            continue
        x_min, y_min, x_max, y_max = (float(value) for value in detection["box"])
        x_min = min(max(x_min, 0), image_width)
        x_max = min(max(x_max, 0), image_width)
        y_min = min(max(y_min, 0), image_height)
        y_max = min(max(y_max, 0), image_height)
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        if width <= 0 or height <= 0:
            continue
        box = BoundingBox(
            class_name=class_name,
            x_center=((x_min + x_max) / 2) / image_width,
            y_center=((y_min + y_max) / 2) / image_height,
            width=width,
            height=height,
            confidence=score,
        )
        box.validate()
        boxes.append(box)
    return boxes


def _canonical_class(label: str) -> str | None:
    normalized = label.lower()
    if "blue object" in normalized or "blue_object" in normalized:
        return "blue_object"
    if "glass" in normalized:
        return "glass"
    if "gripper" in normalized:
        return "gripper"
    return None
