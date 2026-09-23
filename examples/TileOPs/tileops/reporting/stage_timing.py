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


def _split_source(
    source: str | Path, default_operator: str | None, base_dir: Path | None
) -> tuple[str | None, Path]:
    """Return an optional operator label and the timeline source path.

    ``OperatorName=path`` labels sources for multi-operator reports. Plain paths
    remain backward compatible and belong to ``default_operator``.
    """
    raw = str(source)
    operator = default_operator
    path_text = raw
    if "=" in raw:
        candidate, candidate_path = raw.split("=", 1)
        if candidate and candidate_path:
            operator = candidate
            path_text = candidate_path
    path = Path(path_text)
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return operator, path.resolve()


def _aggregate_sources(sources: list[tuple[Path, str]]) -> dict[str, Any]:
    """Aggregate a resolved set of timeline paths and their display scopes."""
    stages: dict[int, dict[str, Any]] = {}
    for path, scope in sources:
        starts: dict[int, list[str | None]] = {}
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid timeline JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"timeline event must be an object at {path}:{line_no}")
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
    measured_total = 0.0
    measured_stages = 0
    for row in rows:
        if row.pop("measured_attempts"):
            row["duration_s_total"] = round(row["duration_s_total"], 3)
            measured_total += row["duration_s_total"]
            measured_stages += 1
        else:
            row["duration_s_total"] = None
        row["attempt_count"] = len(row["attempts"])
    return {
        "duration_s_total": round(measured_total, 3) if measured_stages else None,
        "stages": rows,
    }


def load_stage_timing(
    sources: list[str | Path],
    *,
    operator: str | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate completed attempts; keep unfinished attempts visible without a duration.

    A source is an operator/function directory or its .task_timeline.jsonl file.
    Each completed/failed event already has statectl's start-to-end duration_s.
    """
    if not sources:
        raise ValueError("--stage-timing workflow requires --timing-source")

    if operator == "all" and any("=" not in str(source) for source in sources):
        raise ValueError(
            "multi-operator stage timing requires OPERATOR=PATH for every --timing-source"
        )

    resolved_base = Path(base_dir).resolve() if base_dir is not None else None
    paths: list[tuple[Path, str, str | None]] = []
    for source in sources:
        source_operator, path = _split_source(source, operator, resolved_base)
        if path.is_dir():
            path /= ".task_timeline.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"stage timing source does not exist: {path}")
        if any(existing == path for existing, _, _ in paths):
            raise ValueError(f"duplicate stage timing source: {path}")
        paths.append((path, path.parent.name, source_operator))

    aggregate = _aggregate_sources([(path, scope) for path, scope, _ in paths])
    grouped: dict[str, list[tuple[Path, str]]] = {}
    for path, scope, source_operator in paths:
        if source_operator:
            grouped.setdefault(source_operator, []).append((path, scope))
    operators = []
    for name, operator_sources in grouped.items():
        operator_timing = _aggregate_sources(operator_sources)
        operators.append({"operator": name, **operator_timing})
    return {
        "source": "statectl .task_timeline.jsonl",
        **aggregate,
        "operators": operators,
    }
