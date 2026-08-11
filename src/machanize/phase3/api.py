"""FastAPI application for Phase 3. This module has no robot-control imports."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from machanize.analysis.task_template import (
    ANALYSIS_FPS,
    GEMINI_ROBOTICS_MODEL,
    EpisodeEvidenceBuilder,
    GeminiRoboticsAnalyzer,
    TaskAnalysisJobManager,
    TaskAnalysisService,
    TaskTemplateDraft,
    TaskTemplateStore,
)
from machanize.perception.annotations import AnnotationStore, BoundingBox, PredictionStore
from machanize.perception.episodes import EpisodeRepository, FrameExtractor
from machanize.perception.grounding_dino import MODEL_ID, GroundingDinoDetector
from machanize.perception.yolo import YoloDatasetExporter, YoloPredictor, YoloTrainer
from machanize.phase3.jobs import LabelingJobManager, TrainingJobManager
from machanize.projects import load_project_config


@dataclass(frozen=True)
class Phase3Settings:
    repository_root: Path
    data_root: Path
    frame_cache_root: Path
    labels_root: Path
    predictions_root: Path
    yolo_exports_root: Path
    models_root: Path
    project_config: Path
    robot_movement_enabled: Literal[False] = False
    analysis_evidence_root: Path | None = None
    task_templates_root: Path | None = None

    @classmethod
    def from_repository(cls, root: str | Path) -> Phase3Settings:
        repository_root = Path(root).resolve()
        return cls(
            repository_root=repository_root,
            data_root=repository_root / "data" / "episodes",
            frame_cache_root=repository_root / "data" / "cache" / "frames",
            labels_root=repository_root / "data" / "labels" / "yolo",
            predictions_root=repository_root / "data" / "predictions" / "yolo",
            yolo_exports_root=repository_root / "data" / "yolo",
            models_root=repository_root / "models" / "yolo",
            project_config=(
                repository_root / "configs" / "projects" / "so101_blue_object_to_glass.yaml"
            ),
            analysis_evidence_root=repository_root / "data" / "cache" / "analysis",
            task_templates_root=repository_root / "data" / "analysis" / "task_templates",
        )


class ExtractRequest(BaseModel):
    camera_key: str | None = None
    stride: int = Field(default=5, ge=1, le=300)
    overwrite: bool = False


class SynchronizedExtractRequest(BaseModel):
    camera_keys: list[str] | None = None
    stride: int = Field(default=5, ge=1, le=300)
    overwrite: bool = False


class BoxPayload(BaseModel):
    class_name: str
    x_center: float = Field(ge=0, le=1)
    y_center: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class AnnotationPayload(BaseModel):
    camera_key: str
    boxes: list[BoxPayload]
    approved: bool = False
    source: Literal["manual", "prediction", "grounding_dino"] = "manual"
    model_id: str | None = None


class TrainRequest(BaseModel):
    base_model: str = "yolo26n.pt"
    epochs: int = Field(default=50, ge=1, le=1000)
    image_size: int = Field(default=640, ge=128, le=2048)
    device: str | None = None
    episode_ids: list[str] | None = None


class PredictionRequest(BaseModel):
    camera_key: str
    confidence: float = Field(default=0.25, ge=0, le=1)


class GroundingDinoRequest(BaseModel):
    camera_key: str
    confidence: float = Field(default=0.35, gt=0, le=1)


class BatchLabelRequest(BaseModel):
    episode_ids: list[str] = Field(min_length=1)
    confidence: float = Field(default=0.50, gt=0, le=1)


class StartTaskAnalysisRequest(BaseModel):
    episode_id: str
    confirm_unknown_as_success: bool = False


class ApproveTaskTemplateRequest(BaseModel):
    confirm: Literal[True]


@dataclass
class Phase3Services:
    settings: Phase3Settings
    episodes: EpisodeRepository
    frames: FrameExtractor
    annotations: AnnotationStore
    predictions: PredictionStore
    grounding_dino: GroundingDinoDetector
    exporter: YoloDatasetExporter
    trainer: YoloTrainer
    jobs: TrainingJobManager
    labeling_jobs: LabelingJobManager
    task_templates: TaskTemplateStore
    task_analysis_jobs: TaskAnalysisJobManager


def create_services(
    settings: Phase3Settings,
    grounding_dino: GroundingDinoDetector | None = None,
    task_analysis: TaskAnalysisService | None = None,
) -> Phase3Services:
    settings.frame_cache_root.mkdir(parents=True, exist_ok=True)
    settings.labels_root.mkdir(parents=True, exist_ok=True)
    settings.predictions_root.mkdir(parents=True, exist_ok=True)
    settings.models_root.mkdir(parents=True, exist_ok=True)
    annotations = AnnotationStore(settings.labels_root)
    trainer = YoloTrainer(settings.models_root)
    episodes = EpisodeRepository(settings.data_root)
    frames = FrameExtractor(settings.frame_cache_root)
    detector = grounding_dino or GroundingDinoDetector()
    task_template_root = settings.task_templates_root or (
        settings.repository_root / "data" / "analysis" / "task_templates"
    )
    analysis_evidence_root = settings.analysis_evidence_root or (
        settings.repository_root / "data" / "cache" / "analysis"
    )
    task_templates = TaskTemplateStore(task_template_root)
    task_analysis = task_analysis or TaskAnalysisService(
        EpisodeEvidenceBuilder(analysis_evidence_root),
        GeminiRoboticsAnalyzer(),
        task_templates,
    )
    return Phase3Services(
        settings=settings,
        episodes=episodes,
        frames=frames,
        annotations=annotations,
        predictions=PredictionStore(settings.predictions_root),
        grounding_dino=detector,
        exporter=YoloDatasetExporter(annotations, settings.yolo_exports_root),
        trainer=trainer,
        jobs=TrainingJobManager(trainer),
        labeling_jobs=LabelingJobManager(episodes, frames, annotations, detector),
        task_templates=task_analysis.store,
        task_analysis_jobs=TaskAnalysisJobManager(task_analysis),
    )


def create_app(
    settings: Phase3Settings | None = None,
    *,
    grounding_dino: GroundingDinoDetector | None = None,
    task_analysis: TaskAnalysisService | None = None,
) -> FastAPI:
    settings = settings or Phase3Settings.from_repository(
        os.environ.get("MACHANIZE_ROOT", Path.cwd())
    )
    services = create_services(settings, grounding_dino, task_analysis)
    config = load_project_config(settings.project_config)
    class_names = list(config.raw["task"]["objects"])

    app = FastAPI(title="Machanize Phase 3", version="0.1.0")
    app.state.services = services
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/files", StaticFiles(directory=settings.frame_cache_root), name="frames")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "phase": 3,
            "mode": "training",
            "robot_movement_enabled": settings.robot_movement_enabled,
            "classes": class_names,
            "task_analysis_model": GEMINI_ROBOTICS_MODEL,
            "task_analysis_fps": ANALYSIS_FPS,
            "task_template_approval": "manual",
        }

    @app.get("/api/episodes")
    def list_episodes() -> list[dict]:
        return [episode.to_api() for episode in services.episodes.list()]

    @app.post("/api/analysis/start")
    def start_task_analysis(request: StartTaskAnalysisRequest) -> dict:
        try:
            episode = services.episodes.get(request.episode_id)
            job = services.task_analysis_jobs.submit(
                episode,
                confirm_unknown_as_success=request.confirm_unknown_as_success,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return services.task_analysis_jobs.to_api(job.job_id)

    @app.get("/api/analysis/jobs/{job_id}")
    def task_analysis_status(job_id: str) -> dict:
        try:
            return services.task_analysis_jobs.to_api(job_id)
        except KeyError as error:
            raise HTTPException(
                status_code=410,
                detail=(
                    "Analysis job is no longer available. The API process may have restarted; "
                    "start the analysis again."
                ),
            ) from error

    @app.get("/api/analysis/templates/{episode_id}")
    def get_task_template(episode_id: str) -> dict:
        try:
            services.episodes.get(episode_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        record = services.task_templates.get(episode_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Task analysis not found.")
        return record.model_dump(mode="json")

    @app.put("/api/analysis/templates/{episode_id}")
    def save_task_template(episode_id: str, draft: TaskTemplateDraft) -> dict:
        try:
            services.episodes.get(episode_id)
            record = services.task_templates.save_user_draft(episode_id, draft)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return record.model_dump(mode="json")

    @app.post("/api/analysis/templates/{episode_id}/approve")
    def approve_task_template(
        episode_id: str,
        request: ApproveTaskTemplateRequest,
    ) -> dict:
        del request  # Literal[True] makes approval an explicit, validated user action.
        try:
            services.episodes.get(episode_id)
            record = services.task_templates.approve(episode_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return record.model_dump(mode="json")

    @app.post("/api/episodes/{episode_id}/extract")
    def extract_frames(episode_id: str, request: ExtractRequest) -> list[dict]:
        try:
            episode = services.episodes.get(episode_id)
            frames = services.frames.extract(
                episode,
                camera_key=request.camera_key,
                stride=request.stride,
                overwrite=request.overwrite,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return [frame.to_api(settings.frame_cache_root) for frame in frames]

    @app.post("/api/episodes/{episode_id}/extract-synchronized")
    def extract_synchronized_frames(
        episode_id: str,
        request: SynchronizedExtractRequest,
    ) -> dict:
        try:
            episode = services.episodes.get(episode_id)
            frames_by_camera = services.frames.extract_synchronized(
                episode,
                camera_keys=request.camera_keys,
                stride=request.stride,
                overwrite=request.overwrite,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "camera_keys": list(frames_by_camera),
            "synchronized_frame_count": min(
                (len(frames) for frames in frames_by_camera.values()),
                default=0,
            ),
            "frames": {
                camera: [frame.to_api(settings.frame_cache_root) for frame in frames]
                for camera, frames in frames_by_camera.items()
            },
        }

    @app.get("/api/episodes/{episode_id}/frames")
    def list_frames(episode_id: str, camera_key: str) -> list[dict]:
        try:
            services.episodes.get(episode_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        frames = services.frames.list_frames(episode_id, camera_key)
        return [frame.to_api(settings.frame_cache_root) for frame in frames]

    @app.get("/api/episodes/{episode_id}/frames/{frame_id}/annotation")
    def get_annotation(episode_id: str, frame_id: str, camera_key: str) -> dict:
        record = services.annotations.get(episode_id, camera_key, frame_id)
        return record or {
            "episode_id": episode_id,
            "camera_key": camera_key,
            "frame_id": frame_id,
            "boxes": [],
            "approved": False,
            "source": "manual",
        }

    @app.put("/api/episodes/{episode_id}/frames/{frame_id}/annotation")
    def save_annotation(episode_id: str, frame_id: str, payload: AnnotationPayload) -> dict:
        boxes = [BoundingBox(**box.model_dump()) for box in payload.boxes]
        try:
            services.episodes.get(episode_id)
            frames = services.frames.list_frames(episode_id, payload.camera_key)
            frame = next((item for item in frames if item.frame_id == frame_id), None)
            if frame is None:
                raise ValueError("Extracted frame not found.")
            path = services.annotations.save(
                episode_id=episode_id,
                camera_key=payload.camera_key,
                frame_id=frame_id,
                image_path=frame.image_path,
                boxes=boxes,
                approved=payload.approved,
                source=payload.source,
                model_id=payload.model_id,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"status": "saved", "path": str(path), "approved": payload.approved}

    @app.post("/api/episodes/{episode_id}/frames/{frame_id}/auto-label")
    def auto_label_frame(
        episode_id: str,
        frame_id: str,
        request: GroundingDinoRequest,
    ) -> dict:
        try:
            services.episodes.get(episode_id)
            frames = services.frames.list_frames(episode_id, request.camera_key)
            frame = next((item for item in frames if item.frame_id == frame_id), None)
            if frame is None:
                raise ValueError("Extracted frame not found.")
            boxes = services.grounding_dino.detect(
                frame.image_path,
                threshold=request.confidence,
            )
            services.annotations.save(
                episode_id=episode_id,
                camera_key=request.camera_key,
                frame_id=frame_id,
                image_path=frame.image_path,
                boxes=boxes,
                approved=True,
                source="grounding_dino",
                model_id=MODEL_ID,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "episode_id": episode_id,
            "camera_key": request.camera_key,
            "frame_id": frame_id,
            "image_path": str(frame.image_path),
            "boxes": [asdict(box) for box in boxes],
            "approved": True,
            "source": "grounding_dino",
            "model_id": MODEL_ID,
        }

    @app.post("/api/training/start")
    def start_training(request: TrainRequest) -> dict:
        try:
            if request.episode_ids:
                for episode_id in request.episode_ids:
                    services.episodes.get(episode_id)
            export = services.exporter.export(
                class_names=class_names,
                episode_ids=request.episode_ids,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        job = services.jobs.submit(
            export,
            base_model=request.base_model,
            epochs=request.epochs,
            image_size=request.image_size,
            device=request.device,
        )
        return {
            "job_id": job.job_id,
            "status": job.status,
            "training_images": export.training_images,
            "validation_images": export.validation_images,
            "camera_images": export.camera_images,
            "training_boxes": export.training_boxes,
            "validation_boxes": export.validation_boxes,
        }

    @app.post("/api/labeling/start")
    def start_batch_labeling(request: BatchLabelRequest) -> dict:
        try:
            job = services.labeling_jobs.submit(
                request.episode_ids,
                confidence=request.confidence,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return services.labeling_jobs.to_api(job.job_id)

    @app.get("/api/labeling/{job_id}")
    def labeling_status(job_id: str) -> dict:
        try:
            return services.labeling_jobs.to_api(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/training/{job_id}")
    def training_status(job_id: str) -> dict:
        try:
            return services.jobs.to_api(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/models")
    def list_models() -> list[dict]:
        return [model.to_api() for model in services.trainer.list_models()]

    @app.post("/api/models/{model_id}/predict/{episode_id}")
    def predict_episode(model_id: str, episode_id: str, request: PredictionRequest) -> dict:
        try:
            model = services.trainer.get_model(model_id)
            episode = services.episodes.get(episode_id)
            frames = services.frames.list_frames(episode.episode_id, request.camera_key)
            count = YoloPredictor(services.predictions).predict(
                model,
                frames,
                confidence=request.confidence,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"status": "completed", "frames": count, "model_id": model_id}

    @app.get("/api/models/{model_id}/predictions/{episode_id}/{frame_id}")
    def get_prediction(model_id: str, episode_id: str, frame_id: str, camera_key: str) -> dict:
        prediction = services.predictions.for_model(model_id).get(
            episode_id,
            camera_key,
            frame_id,
        )
        if prediction is None:
            raise HTTPException(status_code=404, detail="Prediction not found.")
        return prediction

    return app
