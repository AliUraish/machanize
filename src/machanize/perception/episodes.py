"""Discover Phase 2 Pi episodes and extract frames without robot access."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_SAFE_PART = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    dataset_root: Path
    dataset_episode_index: int
    project_name: str
    robot_type: str
    task: str
    outcome: str
    review_status: str
    processing_status: str
    frame_count: int | None
    camera_keys: tuple[str, ...]
    manifest_path: Path | None = None

    def to_api(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset_root"] = str(self.dataset_root)
        data["manifest_path"] = str(self.manifest_path) if self.manifest_path else None
        return data


@dataclass(frozen=True)
class ExtractedFrame:
    episode_id: str
    camera_key: str
    frame_id: str
    frame_index: int
    timestamp: float
    image_path: Path
    width: int
    height: int

    def to_api(self, cache_root: Path) -> dict[str, Any]:
        relative = self.image_path.relative_to(cache_root)
        return {
            **asdict(self),
            "image_path": str(self.image_path),
            "image_url": f"/files/{relative.as_posix()}",
        }


class EpisodeRepository:
    """Index Machanize manifests and native LeRobot v3 datasets copied from the Pi."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).resolve()

    def list(self) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        claimed: set[tuple[Path, int]] = set()

        for manifest_path in sorted(self.data_root.rglob("manifests/*.json")):
            record = self._from_manifest(manifest_path)
            records.append(record)
            claimed.add((record.dataset_root, record.dataset_episode_index))

        for info_path in sorted(self.data_root.rglob("lerobot/meta/info.json")):
            dataset_root = info_path.parent.parent.resolve()
            info = _read_json(info_path)
            for episode_index in range(int(info.get("total_episodes", 0))):
                if (dataset_root, episode_index) in claimed:
                    continue
                records.append(self._native_record(dataset_root, episode_index, info))

        return sorted(records, key=lambda record: record.episode_id)

    def get(self, episode_id: str) -> EpisodeRecord:
        for record in self.list():
            if record.episode_id == episode_id:
                return record
        raise KeyError(f"Unknown episode: {episode_id}")

    def _from_manifest(self, path: Path) -> EpisodeRecord:
        manifest = _read_json(path)
        dataset_root = (path.parent.parent / "lerobot").resolve()
        info = _read_json(dataset_root / "meta" / "info.json")
        return EpisodeRecord(
            episode_id=str(manifest["episode_id"]),
            dataset_root=dataset_root,
            dataset_episode_index=int(manifest["dataset_episode_index"]),
            project_name=str(manifest.get("project_name", "machanize")),
            robot_type=str(manifest.get("robot_type", info.get("robot_type") or "unknown")),
            task=str(manifest.get("task", "")),
            outcome=str(manifest.get("outcome", "unknown")),
            review_status=str(manifest.get("review_status", "pending")),
            processing_status=str(manifest.get("processing_status", "pending")),
            frame_count=_optional_int(manifest.get("frame_count")),
            camera_keys=_camera_keys(info),
            manifest_path=path.resolve(),
        )

    def _native_record(
        self,
        dataset_root: Path,
        episode_index: int,
        info: dict[str, Any],
    ) -> EpisodeRecord:
        digest = hashlib.sha256(str(dataset_root).encode()).hexdigest()[:10]
        return EpisodeRecord(
            episode_id=f"native-{digest}-{episode_index:06d}",
            dataset_root=dataset_root,
            dataset_episode_index=episode_index,
            project_name=dataset_root.parent.name,
            robot_type=str(info.get("robot_type") or "unknown"),
            task="",
            outcome="unknown",
            review_status="pending",
            processing_status="pending",
            frame_count=None,
            camera_keys=_camera_keys(info),
        )


class FrameExtractor:
    """Decode episode frames through LeRobot's v3 metadata-aware loader."""

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root).resolve()

    def extract(
        self,
        episode: EpisodeRecord,
        *,
        camera_key: str | None = None,
        stride: int = 1,
        overwrite: bool = False,
    ) -> list[ExtractedFrame]:
        selected_camera = _select_camera(episode.camera_keys, camera_key)
        return self.extract_synchronized(
            episode,
            camera_keys=[selected_camera],
            stride=stride,
            overwrite=overwrite,
        )[selected_camera]

    def extract_synchronized(
        self,
        episode: EpisodeRecord,
        *,
        camera_keys: list[str] | tuple[str, ...] | None = None,
        stride: int = 1,
        overwrite: bool = False,
    ) -> dict[str, list[ExtractedFrame]]:
        """Extract every selected camera from the same dataset samples in one pass."""

        if stride < 1:
            raise ValueError("Frame stride must be at least 1.")
        selected_cameras = _select_cameras(episode.camera_keys, camera_keys)
        cached = {
            camera: self.list_frames(episode.episode_id, camera) for camera in selected_cameras
        }
        if not overwrite and all(cached[camera] for camera in selected_cameras):
            _validate_synchronized_frames(cached)
            return cached

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        output_directories = {
            camera: self._frame_directory(episode.episode_id, camera) for camera in selected_cameras
        }
        for directory in output_directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        repo_id = (
            f"machanize/local-{hashlib.sha256(str(episode.dataset_root).encode()).hexdigest()[:12]}"
        )
        dataset = LeRobotDataset(
            repo_id,
            root=episode.dataset_root,
            episodes=[episode.dataset_episode_index],
        )

        frames_by_camera: dict[str, list[ExtractedFrame]] = {
            camera: [] for camera in selected_cameras
        }
        for local_index in range(0, len(dataset), stride):
            sample = dataset[local_index]
            frame_index = int(_scalar(sample["frame_index"]))
            timestamp = float(_scalar(sample["timestamp"]))
            frame_id = f"{frame_index:06d}"
            for camera in selected_cameras:
                image = _to_image(sample[camera])
                image_path = output_directories[camera] / f"{frame_id}.jpg"
                image.save(image_path, format="JPEG", quality=92)
                frames_by_camera[camera].append(
                    ExtractedFrame(
                        episode_id=episode.episode_id,
                        camera_key=camera,
                        frame_id=frame_id,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        image_path=image_path,
                        width=image.width,
                        height=image.height,
                    )
                )

        for camera, frames in frames_by_camera.items():
            index_path = output_directories[camera] / "index.json"
            index_path.write_text(
                json.dumps([frame.to_api(self.cache_root) for frame in frames], indent=2),
                encoding="utf-8",
            )
        _validate_synchronized_frames(frames_by_camera)
        return frames_by_camera

    def list_frames(self, episode_id: str, camera_key: str) -> list[ExtractedFrame]:
        index_path = self._frame_directory(episode_id, camera_key) / "index.json"
        if not index_path.exists():
            return []
        frames = _read_json(index_path)
        return [
            ExtractedFrame(
                episode_id=str(item["episode_id"]),
                camera_key=str(item["camera_key"]),
                frame_id=str(item["frame_id"]),
                frame_index=int(item["frame_index"]),
                timestamp=float(item["timestamp"]),
                image_path=Path(item["image_path"]).resolve(),
                width=int(item["width"]),
                height=int(item["height"]),
            )
            for item in frames
        ]

    def _frame_directory(self, episode_id: str, camera_key: str) -> Path:
        return self.cache_root / _safe_part(episode_id) / _safe_part(camera_key)


def _to_image(value: Any) -> Image.Image:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


def _select_camera(camera_keys: tuple[str, ...], requested: str | None) -> str:
    if not camera_keys:
        raise ValueError("Episode has no camera features.")
    if requested is None:
        return camera_keys[0]
    if requested in camera_keys:
        return requested
    full_name = f"observation.images.{requested}"
    if full_name in camera_keys:
        return full_name
    raise ValueError(f"Unknown camera {requested!r}. Available cameras: {camera_keys}")


def _select_cameras(
    available: tuple[str, ...],
    requested: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if not available:
        raise ValueError("Episode has no camera features.")
    if requested is None:
        return available
    if not requested:
        raise ValueError("Select at least one camera.")
    selected = tuple(_select_camera(available, camera) for camera in requested)
    if len(set(selected)) != len(selected):
        raise ValueError("Each camera may only be selected once.")
    return selected


def _validate_synchronized_frames(frames_by_camera: dict[str, list[ExtractedFrame]]) -> None:
    signatures = {
        camera: [(frame.frame_index, frame.timestamp) for frame in frames]
        for camera, frames in frames_by_camera.items()
    }
    reference = next(iter(signatures.values()), [])
    if any(signature != reference for signature in signatures.values()):
        raise ValueError("Camera streams do not share frame indices and timestamps.")


def _camera_keys(info: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") in {"video", "image"}
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_part(value: str) -> str:
    return _SAFE_PART.sub("_", value).strip("._") or "item"


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
