"""Draft evaluation cases from the application's own sample inputs (V1.6.0-c).

The field run recorded a validation method against a contract test that was
never wired, and nothing pushed toward the evaluation stage. This module
drafts evaluation cases deterministically from sample inputs the
application already ships — files under conventionally named sample,
fixture, example, or evaluation/test-data folders — so a BYOK evaluation is
one review away instead of a blank page. Contents stay local (nothing is
fetched or sent), every case is marked DRAFT for review, and the toolkit
never runs the evaluation itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from llm_migrate.core.evaluation_models import EvaluationCase
from llm_migrate.core.models import StrictModel

SAMPLE_DIRECTORY_NAMES = {
    "eval",
    "evals",
    "evaluation",
    "evaluations",
    "test_data",
    "testdata",
    "test-data",
    "samples",
    "sample_data",
    "sample-data",
    "examples",
    "fixtures",
}
_SAMPLE_SUFFIXES = {".txt", ".md", ".json", ".jsonl"}
_INPUT_KEYS = ("input", "prompt", "question", "query", "text", "document", "message")
_EXPECTED_KEYS = ("expected_output", "expected", "answer", "output", "label")
MAX_CASES = 50
MAX_INPUT_CHARS = 20_000
_IGNORED = {".git", ".llm-migrate", ".venv", "venv", "node_modules", "__pycache__", "build", "dist"}


class EvaluationScaffold(StrictModel):
    """Outcome of scaffold_evaluation: DRAFT cases and a plan-bound suite."""

    schema_version: Literal["1"] = "1"
    run_id: str
    case_count: int = 0
    sample_files: list[str] = Field(default_factory=list)
    cases_path: str | None = None
    suite_path: str | None = None
    bound_manifest_sha256: str | None = None
    message: str
    next_steps: list[str] = Field(default_factory=list)


def sample_files(application: Path, exclude: set[str]) -> list[Path]:
    """Sample-input files under conventionally named folders, deterministically ordered."""
    found: list[Path] = []
    for item in sorted(application.rglob("*")):
        if not item.is_file() or item.suffix.casefold() not in _SAMPLE_SUFFIXES:
            continue
        parts = item.relative_to(application).parts
        if any(part in _IGNORED for part in parts):
            continue
        if not any(part.casefold() in SAMPLE_DIRECTORY_NAMES for part in parts[:-1]):
            continue
        if item.relative_to(application).as_posix() in exclude:
            continue
        found.append(item)
    return found


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")[:60] or "case"


def _record_case(record: Any) -> tuple[str, Any | None] | None:
    if isinstance(record, str) and record.strip():
        return record, None
    if isinstance(record, dict):
        text = next((record[key] for key in _INPUT_KEYS if isinstance(record.get(key), str)), None)
        if text and text.strip():
            expected = next((record[key] for key in _EXPECTED_KEYS if key in record), None)
            return text, expected
    return None


def _file_records(path: Path) -> list[tuple[str, Any | None]]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.casefold()
    if suffix in {".txt", ".md"}:
        return [(text, None)] if text.strip() else []
    records: list[Any] = []
    if suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        records = data if isinstance(data, list) else [data]
    return [item for item in (_record_case(record) for record in records) if item is not None]


def draft_cases(application: Path, exclude: set[str]) -> tuple[list[EvaluationCase], list[str]]:
    """(draft cases, sample files used), bounded to MAX_CASES."""
    cases: list[EvaluationCase] = []
    used: list[str] = []
    for path in sample_files(application, exclude):
        relative = path.relative_to(application).as_posix()
        try:
            records = _file_records(path)
        except (OSError, UnicodeError):
            continue
        for index, (text, expected) in enumerate(records):
            if len(cases) >= MAX_CASES:
                return cases, used
            if relative not in used:
                used.append(relative)
            suffix = f"-{index + 1}" if len(records) > 1 else ""
            cases.append(
                EvaluationCase(
                    id=f"draft-{len(cases) + 1:03d}-{_slug(path.stem)}{suffix}",
                    input=text[:MAX_INPUT_CHARS],
                    # Exact-match goldens are rarely right across models; keep
                    # the expectation as review metadata, not a validator.
                    tags=["draft"],
                    metadata={
                        "draft": True,
                        "source_file": relative,
                        "expected_hint": (
                            json.dumps(expected, sort_keys=True)[:2000]
                            if expected is not None
                            else None
                        ),
                    },
                )
            )
    return cases, used
