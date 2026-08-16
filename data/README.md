# Local Data

V1 data remains local:

```text
data/
├── episodes/  # LeRobot episode videos and sensor/action records
├── analysis/  # Editable task-template drafts and explicit approval status
├── runtime/   # Session metadata, provider events, and validated decisions
├── labels/    # Approved YOLO and GRU labels
└── cache/     # Derived frames, evidence videos, detections, and sequences
```

Large and generated data is excluded from Git.

Runtime frames are sent from the Pi to the configured provider but are not copied into
`data/runtime`. The Pi stores immutable approved template versions under
`data/runtime/task_templates/`. Each session stores `session.json`, append-only `decisions.jsonl`,
and append-only `events.jsonl` so a stopped session can be reviewed after a backend restart.
Task-template approval changes remain on the Mac in
`data/analysis/task_templates/approval-history.jsonl`.
