from __future__ import annotations

from pathlib import Path

from PIL import Image

from machanize.perception.grounding_dino import GroundingDinoDetector


class FakeBackend:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, image, prompts, threshold):
        self.calls.append((image.size, prompts, threshold))
        return [
            {"label": "a blue object", "score": 0.91, "box": [10, 20, 50, 60]},
            {"label": "a drinking glass", "score": 0.84, "box": [40, 10, 90, 70]},
            {"label": "a robot gripper", "score": 0.78, "box": [0, 0, 20, 30]},
            {"label": "a cat", "score": 0.99, "box": [1, 1, 5, 5]},
            {"label": "a blue object", "score": 0.20, "box": [5, 5, 10, 10]},
        ]


def test_grounding_dino_maps_supported_classes_and_normalizes_boxes(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 80)).save(image_path)
    backend = FakeBackend()
    detector = GroundingDinoDetector(backend_factory=lambda: backend)

    boxes = detector.detect(image_path, threshold=0.35)

    assert [box.class_name for box in boxes] == ["blue_object", "glass", "gripper"]
    assert boxes[0].x_center == 0.3
    assert boxes[0].y_center == 0.5
    assert boxes[0].width == 0.4
    assert boxes[0].height == 0.5
    assert boxes[0].confidence == 0.91
    assert backend.calls == [
        (
            (100, 80),
            ("a blue object", "a drinking glass", "a robot gripper"),
            0.35,
        )
    ]
