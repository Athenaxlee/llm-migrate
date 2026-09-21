"""Guided migration run workspace: start, research prompts, adaptation, finalize."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_migrate.core.workspace import load_run_config
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import (
    dispose_all,
    edit_change,
    insert_change,
    restructure_change,
)

AS_OF = date(2026, 9, 15)


@pytest.fixture
def bedrock_app(tmp_path: Path, project_root: Path) -> Path:
    app = tmp_path / "bedrock_app"
    shutil.copytree(project_root / "tests/fixtures/applications/bedrock_app", app)
    return app


@pytest.fixture
def anthropic_app(tmp_path: Path, project_root: Path) -> Path:
    app = tmp_path / "anthropic_app"
    shutil.copytree(project_root / "tests/fixtures/applications/anthropic_app", app)
    return app


def _start(service: MigrationService, app: Path):  # type: ignore[no-untyped-def]
    return service.start_migration_run(
        app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="bedrock",
        target_platform="bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
    )


def test_start_requires_confirmation_for_ambiguous_target(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="bedrock",
        target_platform="bedrock",
        as_of=AS_OF,
    )
    assert start.status == "needs_confirmation"
    assert start.run is None
    assert not (bedrock_app / ".llm-migrate").exists()
    assert {candidate.endpoint for candidate in start.target_match.candidates} == {
        "bedrock-runtime",
        "bedrock-mantle",
    }
    assert any("ask" in step.casefold() for step in start.next_steps)


def test_start_creates_default_workspace_and_next_steps(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = _start(service, bedrock_app)
    assert start.status == "ready"
    assert start.run is not None and start.paths is not None
    assert start.run.run_id == "claude-sonnet-4-6-to-claude-sonnet-5-2026-09-15"
    expected_run_dir = bedrock_app / ".llm-migrate" / "runs" / start.run.run_id
    assert Path(start.paths.run_dir) == expected_run_dir
    assert (expected_run_dir / "migration.yaml").is_file()
    config = load_run_config(expected_run_dir)
    assert config.source.model == "claude-sonnet-4-6"
    assert config.target_model_id == "anthropic.claude-sonnet-5"
    assert start.research is not None and start.research.level == "recommended"
    assert start.research.request_path is not None
    assert Path(start.research.request_path).is_file()
    assert any("finalize_migration" in step for step in start.next_steps)


def test_start_honors_explicit_output_dir_and_skip_research(
    service: MigrationService, bedrock_app: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "My Migration Output"
    start = service.start_migration_run(
        bedrock_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        output_dir=output_dir,
        research="skip",
    )
    assert start.status == "ready"
    assert start.paths is not None
    assert Path(start.paths.run_dir) == output_dir.resolve()
    assert start.research is not None
    assert start.research.level == "recommended"
    assert start.research.request_path is None
    assert not (output_dir / "request.yaml").exists()


def test_research_prompts_are_scope_isolated(service: MigrationService, bedrock_app: Path) -> None:
    start = _start(service, bedrock_app)
    assert start.paths is not None
    pack = service.get_research_prompts(start.paths.run_dir)
    scopes = {item.scope.value for item in pack.scopes}
    assert scopes == {"source", "target"}
    for item in pack.scopes:
        assert item.status == "research_pending"
        assert "Research ONLY the exact model identity" in item.researcher_prompt
        assert "INDEPENDENT" in item.reviewer_prompt
        assert item.research_output_path.endswith(f"research/{item.scope.value}.yaml")
    assert any("ONE agent per scope" in line for line in pack.orchestration_guidance)
    request = yaml.safe_load(
        (Path(start.paths.run_dir) / "request.yaml").read_text(encoding="utf-8")
    )
    assert request["run_id"] == start.run.run_id  # type: ignore[union-attr]


def test_adaptation_tasks_cover_affected_files_and_prompts(
    service: MigrationService, anthropic_app: Path
) -> None:
    start = service.start_migration_run(
        anthropic_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
    )
    assert start.status == "ready" and start.paths is not None
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    assert [task.source_path for task in tasks.prompt_tasks] == ["prompts/system.txt"]
    assert [task.source_path for task in tasks.file_tasks] == ["app.py"]
    assert tasks.file_tasks[0].status == "pending"
    assert tasks.target_model_id == "claude-sonnet-5"
    assert tasks.guidance


def test_submit_adapted_file_fails_closed_then_accepts(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = _start(service, bedrock_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir

    broken = service.submit_adapted_file(
        run_dir, "app.py", "def broken(:\n", "why", ["change"], submitted_on=AS_OF
    )
    assert not broken.accepted
    assert any("does not parse" in problem for problem in broken.problems)

    with pytest.raises(ValueError, match="escapes the application root"):
        service.submit_adapted_file(
            run_dir, "../escape.py", "x = 1\n", "why", ["change"], submitted_on=AS_OF
        )

    original = (bedrock_app / "app.py").read_text(encoding="utf-8")
    unchanged = service.submit_adapted_file(
        run_dir, "app.py", original, "why", ["change"], submitted_on=AS_OF
    )
    assert not unchanged.accepted

    adapted = original.replace("anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-5")
    accepted = service.submit_adapted_file(
        run_dir,
        "app.py",
        adapted,
        "Sonnet 5 uses a new Bedrock model id.",
        ["Replaced the modelId value."],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, run_dir, "app.py"),
        annotated_changes=[
            edit_change(
                "anthropic.claude-sonnet-4-6",
                "anthropic.claude-sonnet-5",
                why="Sonnet 5 has its own Bedrock model id.",
            )
        ],
    )
    assert accepted.accepted, accepted.message
    assert accepted.output_path is not None
    written = Path(accepted.output_path)
    assert written == Path(run_dir) / "output" / "files" / "app.py"
    assert written.read_text(encoding="utf-8") == adapted
    # The application tree itself is untouched.
    assert (bedrock_app / "app.py").read_text(encoding="utf-8") == original


def test_submit_adapted_file_warns_when_source_model_id_remains(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = _start(service, bedrock_app)
    assert start.paths is not None
    original = (bedrock_app / "app.py").read_text(encoding="utf-8")
    result = service.submit_adapted_file(
        start.paths.run_dir,
        "app.py",
        original + "\nTIMEOUT = 30\n",
        "why",
        ["Added a timeout."],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, start.paths.run_dir, "app.py"),
        annotated_changes=[insert_change("TIMEOUT = 30", why="Added a request timeout.")],
    )
    assert result.accepted, result.message
    assert any("still appears" in warning for warning in result.warnings)


def test_submit_adapted_prompt_validates_and_records(
    service: MigrationService, anthropic_app: Path
) -> None:
    start = service.start_migration_run(
        anthropic_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
    )
    assert start.paths is not None
    run_dir = start.paths.run_dir

    empty = service.submit_adapted_prompt(
        run_dir, "prompts/system.txt", "   ", "why", submitted_on=AS_OF
    )
    assert not empty.accepted

    undisposed = service.submit_adapted_prompt(
        run_dir,
        "prompts/system.txt",
        "Return a concise weather report for the requested city.",
        "Sonnet 5 follows short direct instructions without scaffolding.",
        ["Tightened the instruction to one sentence."],
        submitted_on=AS_OF,
    )
    assert not undisposed.accepted
    assert "every guidance item must be disposed" in undisposed.message

    result = service.submit_adapted_prompt(
        run_dir,
        "prompts/system.txt",
        "Return a concise weather report for the requested city.",
        "Sonnet 5 follows short direct instructions without scaffolding.",
        ["Tightened the instruction to one sentence."],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, run_dir, "prompts/system.txt"),
        annotated_changes=[restructure_change(why="Rewrote the prompt as one direct instruction.")],
    )
    assert result.accepted, result.message
    assert result.output_path is not None
    assert (
        Path(result.output_path) == Path(run_dir) / "output" / "prompts" / "prompts" / "system.txt"
    )
    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.prompt_tasks[0].status == "submitted"


def test_submit_adapted_prompt_accepts_multi_endpoint_bedrock_target(
    service: MigrationService, bedrock_app: Path
) -> None:
    """Endpoint context from the run config must reach prompt validation."""
    (bedrock_app / "prompts").mkdir()
    (bedrock_app / "prompts" / "system.txt").write_text(
        "Answer using the supplied tool.\n", encoding="utf-8"
    )
    start = _start(service, bedrock_app)
    assert start.paths is not None
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/system.txt",
        "Use the find_order tool for every order lookup and answer concisely.",
        "Sonnet 5 needs the tool policy stated explicitly.",
        ["Added an explicit tool-use instruction."],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, start.paths.run_dir, "prompts/system.txt"),
        annotated_changes=[
            edit_change(
                "Answer using the supplied tool.",
                "Use the find_order tool for every order lookup and answer concisely.",
                why="Sonnet 5 needs the tool policy stated explicitly.",
            )
        ],
    )
    assert result.accepted, result.message
    assert not any(issue.code == "ambiguous_target_platform" for issue in result.validation.issues)


def test_finalize_writes_manifest_report_and_coverage(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = _start(service, bedrock_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir

    first = service.finalize_migration_run(run_dir)
    assert first.coverage_gaps == ["app.py"]
    assert Path(first.manifest_path).is_file()
    assert "no adapted version was submitted" in Path(first.report_path).read_text(encoding="utf-8")

    original = (bedrock_app / "app.py").read_text(encoding="utf-8")
    adapted = original.replace("anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-5")
    submitted = service.submit_adapted_file(
        run_dir,
        "app.py",
        adapted,
        "Sonnet 5 uses a new Bedrock model id.",
        ["Replaced the modelId value."],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, run_dir, "app.py"),
        annotated_changes=[
            edit_change(
                "anthropic.claude-sonnet-4-6",
                "anthropic.claude-sonnet-5",
                why="Sonnet 5 has its own Bedrock model id.",
            )
        ],
    )
    assert submitted.accepted, submitted.message
    final = service.finalize_migration_run(run_dir)
    assert final.coverage_gaps == []
    assert final.adapted_files == 1
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert "## Adaptation deliverables" in report
    assert "Sonnet 5 uses a new Bedrock model id." in report
    assert "Replaced the modelId value." in report
    manifest = yaml.safe_load(Path(final.manifest_path).read_text(encoding="utf-8"))
    assert manifest["migration"]["target"]["model_id"] == "anthropic.claude-sonnet-5"
    # The run workspace never leaks into the scan of the application itself.
    assert manifest["migration"]["application"]["files_scanned"] == 1


def test_workspace_is_excluded_from_scanning(service: MigrationService, bedrock_app: Path) -> None:
    start = _start(service, bedrock_app)
    assert start.paths is not None
    (Path(start.paths.run_dir) / "output" / "files").mkdir(parents=True, exist_ok=True)
    (Path(start.paths.run_dir) / "output" / "files" / "app.py").write_text(
        "model = 'anthropic.claude-sonnet-5'\n", encoding="utf-8"
    )
    analysis = service.scan_application(bedrock_app)
    assert analysis.files_scanned == 1


def test_unchanged_deliverables_are_reported_explicitly(
    service: MigrationService, anthropic_app: Path
) -> None:
    """A reviewed no-change prompt must be stated in the report, never silent."""
    start = service.start_migration_run(
        anthropic_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
    )
    assert start.paths is not None
    run_dir = start.paths.run_dir
    result = service.submit_adapted_prompt(
        run_dir,
        "prompts/system.txt",
        "",
        "The prompt is already short, direct, and format-neutral for the target.",
        [],
        unchanged=True,
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(
            service, run_dir, "prompts/system.txt", disposition="not_applicable"
        ),
    )
    assert result.accepted, result.message
    # The deliverable stays clean: the original bytes, no annotations inside.
    original = (anthropic_app / "prompts" / "system.txt").read_text(encoding="utf-8")
    output = Path(run_dir) / "output" / "prompts" / "prompts" / "system.txt"
    assert output.read_text(encoding="utf-8") == original

    final = service.finalize_migration_run(run_dir)
    assert final.adapted_prompts == 0
    assert final.reviewed_unchanged == 1
    assert "reviewed and needed no change" in final.message
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert "### `prompts/system.txt` (prompt) — no change needed" in report
    assert "Why no change: The prompt is already short" in report
    assert "- Guidance dispositions:" in report
    assert "  - not applicable — " in report
    assert "0 file(s) adapted; 1 reviewed with no change needed." in report


def test_worklist_names_verbatim_source_and_disposition_rules(
    service: MigrationService, anthropic_app: Path
) -> None:
    start = service.start_migration_run(
        anthropic_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
    )
    assert start.paths is not None
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    assert tasks.schema_version == "3"
    assert tasks.prompt_tasks[0].verbatim_source
    assert any("verbatim_source` is the ORIGINAL" in line for line in tasks.guidance)
    assert any("dispose EVERY guidance item" in line for line in tasks.guidance)
