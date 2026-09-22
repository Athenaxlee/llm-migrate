"""V1.5.1: alias-aware reference detection, reviewed new files, honest status."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_migrate.core.workspace import (
    MigrationRunConfig,
    source_detection_spellings,
)
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import insert_change

AS_OF = date(2026, 9, 22)


@pytest.fixture
def bedrock_app(tmp_path: Path, project_root: Path) -> Path:
    app = tmp_path / "bedrock_app"
    shutil.copytree(project_root / "tests/fixtures/applications/bedrock_app", app)
    return app


def _start(service: MigrationService, app: Path, research: str = "skip"):  # type: ignore[no-untyped-def]
    start = service.start_migration_run(
        app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research=research,
    )
    assert start.status == "ready" and start.paths is not None
    return start


def test_run_config_records_reference_spellings_with_aliases(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = _start(service, bedrock_app)
    config = yaml.safe_load(
        (Path(start.paths.run_dir) / "migration.yaml").read_text(encoding="utf-8")
    )
    source_refs = config["source_model_reference_spellings"]
    # Invocable spellings, the canonical name, and the reviewed aliases.
    assert set(config["source_model_spellings"]).issubset(set(source_refs))
    assert "claude-sonnet-4-6" in source_refs
    assert "sonnet 4.6" in source_refs
    # Aliases never leak into the INVOCABLE spelling set.
    assert "sonnet 4.6" not in config["source_model_spellings"]
    assert "sonnet 5" not in config["target_model_spellings"]


def test_unchanged_claim_rejected_when_file_names_a_source_alias(
    service: MigrationService, bedrock_app: Path
) -> None:
    """A registry alias of the source model is a source reference (v1.5.0 hole)."""
    helper = bedrock_app / "notes.py"
    helper.write_text(
        '"""Ops notes."""\n\n# Pricing assumes sonnet 4.6 on-demand rates.\nRATE = 1\n',
        encoding="utf-8",
    )
    start = _start(service, bedrock_app)
    result = service.submit_adapted_file(
        start.paths.run_dir,
        "notes.py",
        "",
        "Reviewed; nothing references the model.",
        [],
        submitted_on=AS_OF,
        unchanged=True,
    )
    assert not result.accepted
    assert "sonnet 4.6" in result.message
    assert "cannot be recorded as unchanged" in result.message


def test_detection_spellings_exclude_target_substrings() -> None:
    """A source spelling contained in a target spelling must not be detected.

    Cross-platform migration of the SAME model: the direct-API id is a
    substring of the Bedrock selector-qualified id, and the shared aliases
    are excluded exactly. Content naming only the target is never flagged.
    """
    config = MigrationRunConfig(
        run_id="run-x",
        application_root="/tmp/app",
        source={
            "provider": "anthropic",
            "platform": "anthropic-api",
            "model": "claude-sonnet-5",
        },
        target={
            "provider": "anthropic",
            "platform": "amazon-bedrock",
            "model": "claude-sonnet-5",
        },
        source_model_id="claude-sonnet-5",
        target_model_id="anthropic.claude-sonnet-5",
        source_model_spellings=["claude-sonnet-5"],
        target_model_spellings=["anthropic.claude-sonnet-5", "us.anthropic.claude-sonnet-5"],
        source_model_reference_spellings=["claude-sonnet-5", "sonnet 5"],
        target_model_reference_spellings=[
            "anthropic.claude-sonnet-5",
            "us.anthropic.claude-sonnet-5",
            "claude-sonnet-5",
            "sonnet 5",
        ],
        created_on=AS_OF,
    )
    assert source_detection_spellings(config) == []


def test_new_file_requires_an_annotated_change(
    service: MigrationService, bedrock_app: Path
) -> None:
    """Invented content cannot enter the deliverable set unreviewed."""
    start = _start(service, bedrock_app)
    rejected = service.submit_adapted_file(
        start.paths.run_dir,
        "helpers/invented.py",
        "TEMPERATURE = 0.9\n",
        "Adds tuned sampling defaults.",
        ["Added tuned defaults."],
        submitted_on=AS_OF,
        new_file=True,
    )
    assert not rejected.accepted
    assert "at least one annotated change" in rejected.message
    assert "submission-format requirements" in rejected.message
    accepted = service.submit_adapted_file(
        start.paths.run_dir,
        "helpers/invented.py",
        "TEMPERATURE = 0.9\n",
        "Adds tuned sampling defaults.",
        ["Added tuned defaults."],
        submitted_on=AS_OF,
        new_file=True,
        annotated_changes=[
            insert_change(
                "TEMPERATURE = 0.9",
                why="Carries the sampling default the plan requires.",
                kind="analysis_finding",
                reference="configuration coupling",
            )
        ],
    )
    assert accepted.accepted, accepted.message


def test_run_status_does_not_flip_research_pending_by_itself(
    service: MigrationService, bedrock_app: Path
) -> None:
    """Status is read-only: only explicit host activity moves past research."""
    start = _start(service, bedrock_app, research="auto")
    run_dir = Path(start.paths.run_dir)
    if not (run_dir / "request.yaml").is_file():
        pytest.skip("research not recommended for this pair; scenario needs request.yaml")
    first = service.get_run_status(run_dir)
    second = service.get_run_status(run_dir)
    assert first.state == "research_pending"
    assert second.state == "research_pending", (
        "a second status call must not steer past the research question"
    )
    service.list_adaptation_tasks(run_dir)
    after_list = service.get_run_status(run_dir)
    assert after_list.state != "research_pending"
