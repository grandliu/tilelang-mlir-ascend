"""Read workflow Stage durations from statectl task timelines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


STAGE_NAMES = {
    0: "Scaffold / 脚手架",
    1: "Design / 设计",
    2: "Review / 检视",
    3: "Develop / 开发",
    4: "Tuning / 调优",
    5: "Integrate / 集成",
}


def load_stage_timing(sources: list[str | Path]) -> dict[str, Any]:
    """Aggregate completed attempts; keep unfinished attempts visible without a duration.

    A source is an operator/function directory or its .task_timeline.jsonl file.
    Each completed/failed event already has statectl's start-to-end duration_s.
    """
    if not sources:
        raise ValueError("--stage-timing workflow requires --timing-source")

    paths = []
    for source in sources:
        path = Path(source).resolve()
        if path.is_dir():
            path /= ".task_timeline.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"stage timing source does not exist: {path}")
        if path in paths:
            raise ValueError(f"duplicate stage timing source: {path}")
        paths.append(path)

    stages: dict[int, dict[str, Any]] = {}
    for path in paths:
        scope = path.parent.name
        starts: dict[int, list[str | None]] = {}
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid timeline JSON at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"timeline event must be an object at {path}:{line_no}"
                )
            stage = event.get("stage")
            if type(stage) is not int or stage not in STAGE_NAMES:
                continue
            action = event.get("action")
            if action not in {"start", "complete", "fail"}:
                continue
            row = stages.setdefault(
                stage,
                {
                    "stage": stage,
                    "name": STAGE_NAMES[stage],
                    "duration_s_total": 0.0,
                    "attempts": [],
                    "measured_attempts": 0,
                },
            )
            pending = starts.setdefault(stage, [])
            if action == "start":
                pending.append(event.get("ts"))
                continue
            started_at = pending.pop(0) if pending else None
            duration = event.get("duration_s")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < 0
            ):
                duration = None
            attempt = {
                "scope": scope,
                "outcome": "completed" if action == "complete" else "failed",
                "started_at": started_at,
                "ended_at": event.get("ts"),
                "duration_s": duration,
                "verdict": event.get("verdict") or event.get("reason"),
            }
            row["attempts"].append(attempt)
            if duration is not None:
                row["duration_s_total"] += duration
                row["measured_attempts"] += 1
        for stage, pending in starts.items():
            row = stages[stage]
            row["attempts"].extend(
                {
                    "scope": scope,
                    "outcome": "running",
                    "started_at": ts,
                    "ended_at": None,
                    "duration_s": None,
                    "verdict": None,
                }
                for ts in pending
            )

    rows = [stages[stage] for stage in sorted(stages)]
    for row in rows:
        row["duration_s_total"] = (
            round(row["duration_s_total"], 3) if row.pop("measured_attempts") else None
        )
        row["attempt_count"] = len(row["attempts"])
    return {"source": "statectl .task_timeline.jsonl", "stages": rows}
