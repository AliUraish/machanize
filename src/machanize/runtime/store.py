"""Durable local runtime session and decision storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

from machanize.analysis.task_template import TaskTemplateRecord
from machanize.runtime.schemas import RuntimeDecision, RuntimeSessionRecord


class RuntimeStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._lock = RLock()

    def save_session(self, record: RuntimeSessionRecord) -> None:
        with self._lock:
            directory = self._directory(record.session_id)
            directory.mkdir(parents=True, exist_ok=True)
            _atomic_write(directory / "session.json", record.model_dump_json(indent=2))

    def get_session(self, session_id: str) -> RuntimeSessionRecord:
        path = self._directory(session_id) / "session.json"
        if not path.is_file():
            raise KeyError(f"Unknown runtime session: {session_id}")
        return RuntimeSessionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[RuntimeSessionRecord]:
        if not self.root.is_dir():
            return []
        records = [
            RuntimeSessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.root.glob("*/session.json")
        ]
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    def append_decision(self, decision: RuntimeDecision) -> None:
        self._append(
            self._directory(decision.session_id) / "decisions.jsonl",
            decision.model_dump(mode="json"),
        )

    def list_decisions(self, session_id: str) -> list[RuntimeDecision]:
        self.get_session(session_id)
        path = self._directory(session_id) / "decisions.jsonl"
        if not path.is_file():
            return []
        return [
            RuntimeDecision.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        self.get_session(session_id)
        self._append(self._directory(session_id) / "events.jsonl", event)

    def _append(self, path: Path, value: dict[str, Any]) -> None:
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
                output.flush()
                os.fsync(output.fileno())

    def _directory(self, session_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in session_id)
        return self.root / (safe.strip("._") or "session")


class ApprovedRuntimeTemplateStore:
    """Immutable Pi-side snapshots imported from the Mac training backend."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._lock = RLock()

    def import_approved(self, record: TaskTemplateRecord) -> TaskTemplateRecord:
        if record.approval_status != "approved" or not record.approved_at:
            raise ValueError("Runtime requires a human-approved task template.")
        content = record.model_dump_json(indent=2)
        digest = hashlib.sha256(content.encode()).hexdigest()
        episode = _safe_part(record.source_episode.episode_id)
        path = self.root / f"{episode}-v{record.template_version}-{digest[:12]}.json"
        with self._lock:
            if path.is_file():
                existing = TaskTemplateRecord.model_validate_json(path.read_text(encoding="utf-8"))
                if existing != record:
                    raise ValueError(
                        "Approved template version conflicts with its stored snapshot."
                    )
                return existing
            self.root.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, content)
        return record

    def latest_for_episode(self, episode_id: str) -> TaskTemplateRecord | None:
        if not self.root.is_dir():
            return None
        records = [
            TaskTemplateRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.root.glob(f"{_safe_part(episode_id)}-v*.json")
        ]
        return max(records, key=lambda item: item.template_version, default=None)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
        temporary = Path(output.name)
    temporary.replace(path)


def _safe_part(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "_.-" else "_" for character in value
    )
    return safe.strip("._") or "template"
