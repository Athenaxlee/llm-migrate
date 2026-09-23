"""Run-scoped empirical observations (V1.6.0-b).

An observation records what the user saw the target actually do for one
plan unknown (typically the printed result of an emitted probe script). It
is a RUN-SCOPED artifact (`observations.yaml` in the run workspace): it
closes the unknown it names in this run's plan and renders in the report,
and it never mutates the reviewed registry. Promoting an observation to
canonical knowledge is an explicit `propose_registry_update` with the
observation as evidence, reviewed like any other proposal.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError

from llm_migrate.core.models import StrictModel
from llm_migrate.core.runstate import atomic_write_text

OBSERVATIONS_FILENAME = "observations.yaml"


class ObservationError(ValueError):
    """The observation log cannot be read."""


class RunObservation(StrictModel):
    """One recorded empirical result for one plan unknown."""

    unknown_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    recorded_on: date


class RunObservationLog(StrictModel):
    schema_version: Literal["1"] = "1"
    run_id: str
    observations: list[RunObservation] = Field(default_factory=list)


class ObservationResult(StrictModel):
    """Outcome of one record_observation call."""

    schema_version: Literal["1"] = "1"
    run_id: str
    accepted: bool
    observation: RunObservation | None = None
    problems: list[str] = Field(default_factory=list)
    open_unknowns: int = 0
    message: str


def load_observations(run_dir: Path, run_id: str) -> RunObservationLog:
    path = Path(run_dir) / OBSERVATIONS_FILENAME
    if not path.is_file():
        return RunObservationLog(run_id=run_id)
    try:
        return RunObservationLog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise ObservationError(f"invalid observation log {path}: {exc}") from exc


def save_observations(run_dir: Path, log: RunObservationLog) -> Path:
    path = Path(run_dir) / OBSERVATIONS_FILENAME
    atomic_write_text(path, yaml.safe_dump(log.model_dump(mode="json"), sort_keys=False))
    return path


def upsert_observation(log: RunObservationLog, observation: RunObservation) -> RunObservationLog:
    """The latest observation for an unknown replaces an earlier one."""
    kept = [item for item in log.observations if item.unknown_id != observation.unknown_id]
    return log.model_copy(update={"observations": [*kept, observation]})


def observation_reasons(log: RunObservationLog) -> dict[str, str]:
    """Unknown id -> the closed reason its observation gives."""
    return {
        item.unknown_id: (
            f"observation recorded {item.recorded_on.isoformat()}: {item.outcome} "
            f"(evidence: {item.evidence})"
        )
        for item in log.observations
    }


def render_observations_section(log: RunObservationLog) -> str:
    lines = ["## Observations (run-scoped)", ""]
    if not log.observations:
        return "\n".join([*lines, "- None recorded.", ""])
    lines.append(
        "Empirical results recorded for this run only. They close the unknowns they "
        "name here and never change the reviewed registry; promoting one is an "
        "explicit propose_registry_update with the observation as evidence."
    )
    lines.append("")
    lines.extend(
        f"- `{item.unknown_id}` {item.subject} — **{item.outcome}** "
        f"(evidence: {item.evidence}; recorded {item.recorded_on.isoformat()})"
        for item in log.observations
    )
    lines.append("")
    return "\n".join(lines)
