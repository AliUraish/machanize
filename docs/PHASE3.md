# Phase 3 — Demonstration Analysis and Visual Training

Phase 3 is a read-only training application. It does not import the LeRobot robot adapter, connect
to SO-101, send actions, or expose robot-control API routes.

## Successful-demonstration analysis

The analysis input is exactly one episode whose recorded outcome is `success`, or one native
LeRobot episode with outcome `unknown` that the user explicitly confirms as a successful
demonstration. Recorded failures are rejected. Both `observation.images.front` and
`observation.images.wrist` are required.

```text
Successful episode
→ sample synchronized dataset rows at approximately 5 FPS
→ render Front and Wrist side by side
→ overlay task description, timestamps, joint observations, and actions
→ attach the sampled telemetry as structured prompt evidence
→ upload the evidence video with the official Google GenAI SDK
→ call gemini-robotics-er-1.6-preview with models.generate_content
→ validate the structured task-template response
→ save it with approval_status=draft
→ edit/save in the GUI (still draft)
→ explicitly approve in the GUI
```

The call uses the normal `generateContent` API, not the Live API. `VideoMetadata.fps` is set to the
actual sampling rate, targeted at 5 FPS for detailed motion analysis. The upload is deleted from the
Gemini Files API after the response is received; the local evidence package remains cached.

The template contains the task description, ordered stages, expected object relationships, expected
robot and gripper behavior, success conditions, possible failures, timestamped evidence, confidence,
uncertainty, source episode, model version, and user approval status. Gemini only analyzes the
demonstration and its weights are not trained or modified.

Generated output is never approved automatically. Saving an edit resets an approved template to
`draft`; approval requires the dedicated endpoint with an explicit confirmation value.

## Input format

Machanize reads the same local structure produced on the Pi by Phase 2:

```text
data/episodes/<session>/
├── lerobot/
│   ├── meta/info.json
│   ├── meta/episodes/**/*.parquet
│   ├── data/**/*.parquet
│   └── videos/**/*.mp4
└── manifests/<episode-id>.json
```

Native LeRobot v3 datasets without Machanize manifests are also discovered. Both
`observation.images.front` and `observation.images.wrist` are read from each dataset sample.

## Start

```text
uv sync --extra dev --extra lerobot --extra vision
GEMINI_API_KEY=your-key uv run python training/serve_phase3.py
```

The server disables Uvicorn auto-reload by default because analysis and training jobs run in the
server process. Restarting that process interrupts active jobs. For API-only development,
`MACHANIZE_DEV_RELOAD=1` opts back into reload behavior; do not use it during analysis or training.

In another terminal:

```text
cd dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Workflow

1. Select exactly one successful episode and start demonstration analysis. For an `unknown`
   episode, use the confirmation action to declare that it is a successful demonstration.
2. Review and edit the structured Gemini draft, then explicitly approve it when correct.
3. For object-detection training, select one or more Pi episodes with the episode checkboxes.
4. Set the Grounding DINO confidence threshold (default `0.50`).
5. Start batch labeling. Machanize extracts every frame from every camera stream available in each
   selected LeRobot episode, and Grounding DINO labels
   `blue_object`, `glass`, and `gripper`.
6. DINO labels are saved and approved automatically. This is separate from task-template approval.
7. Select at least two labeled episodes and train `yolo26n.pt` using MPS on the Mac or CPU.
8. Machanize keeps complete episodes in either train or validation; an episode is never split.
9. Select the trained model and run predictions for review.

The first batch-label request lazily downloads `IDEA-Research/grounding-dino-tiny`. Grounding DINO
runs on all selected episode images and has no robot connection.

Complete episodes, including synchronized images from every available camera, always remain in the
same train or validation split.

## Local outputs

```text
data/cache/frames/       # Extracted JPEG review frames
data/labels/yolo/        # Approved and draft annotations
data/yolo/               # Exported YOLO train/validation datasets
data/predictions/yolo/   # Unapproved model predictions
data/cache/analysis/     # Side-by-side evidence video and timestamped telemetry
data/analysis/task_templates/ # Editable task-template records and approval status
models/yolo/             # Registered best.pt models and metadata
```

All generated artifacts are ignored by Git.
