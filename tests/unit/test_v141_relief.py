"""V1.4.1 immediate relief: durable run state, UTC boundaries, tone, path validation."""

from __future__ import annotations

import asyncio
import importlib
import shutil
import threading
from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_migrate.core.agent_research import artifact_sha256
from llm_migrate.core.workspace import load_adaptation_log
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import dispose_all, edit_change
from tests.unit.v11_scenarios import (
    NOW,
    standard_host,
    standard_request,
    supportive_review,
    target_research,
)

AS_OF = date(2026, 9, 15)


@pytest.fixture
def bedrock_app(tmp_path: Path, project_root: Path) -> Path:
    app = tmp_path / "bedrock_app"
    shutil.copytree(project_root / "tests/fixtures/applications/bedrock_app", app)
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


def test_parallel_submissions_never_lose_adaptation_log_entries(
    service: MigrationService, bedrock_app: Path
) -> None:
    """Concurrent submissions each land in changes.yaml (locked read-modify-write)."""
    start = _start(service, bedrock_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    paths = [f"helpers/util_{index}.py" for index in range(4)]
    failures: list[str] = []

    def submit(source_path: str) -> None:
        result = service.submit_adapted_file(
            run_dir,
            source_path,
            f"VALUE = {source_path!r}\n",
            "New helper introduced by the migration.",
            ["Added the helper module."],
            submitted_on=AS_OF,
            new_file=True,
        )
        if not result.accepted:
            failures.append(result.message)

    threads = [threading.Thread(target=submit, args=(path,)) for path in paths]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    log = load_adaptation_log(Path(run_dir), start.run.run_id if start.run else "")
    assert {entry.source_path for entry in log.entries} == set(paths)


def test_disposition_rejections_state_they_are_format_requirements(
    service: MigrationService, bedrock_app: Path
) -> None:
    """Format rejections must not read as judgements on the adaptation content."""
    start = _start(service, bedrock_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    original = (bedrock_app / "app.py").read_text(encoding="utf-8")
    adapted = original.replace("anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5")
    result = service.submit_adapted_file(
        run_dir,
        "app.py",
        adapted,
        "Sonnet 5 uses a new Bedrock model id.",
        ["Replaced the modelId value."],
        submitted_on=AS_OF,
        annotated_changes=[
            edit_change(
                "anthropic.claude-sonnet-4-6",
                "us.anthropic.claude-sonnet-5",
                why="Sonnet 5 on Bedrock is invoked through the US inference profile.",
            )
        ],
    )
    assert not result.accepted
    assert "submission-format requirements" in result.message
    assert "not a judgement on the adaptation content" in result.message
    # The same submission with the format fields filled in is accepted.
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
                "us.anthropic.claude-sonnet-5",
                why="Sonnet 5 on Bedrock is invoked through the US inference profile.",
            )
        ],
    )
    assert accepted.accepted, accepted.message


def test_validate_research_artifact_reads_the_workspace(
    tmp_path: Path, service: MigrationService, project_root: Path
) -> None:
    application = project_root / "tests/fixtures/applications/anthropic_app"
    analysis = service.scan_application(application)
    request = standard_request(artifact_sha256(analysis.requirements))
    run_dir = tmp_path / "run"
    (run_dir / "research").mkdir(parents=True)
    (run_dir / "request.yaml").write_text(
        yaml.safe_dump(request.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    research = target_research()
    (run_dir / "research" / "target.yaml").write_text(
        yaml.safe_dump(research.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    research_only = service.validate_research_artifact(run_dir, "target")
    assert research_only.valid, research_only.problems
    assert research_only.research_checked and not research_only.review_checked

    review = supportive_review(
        research, reviewer_id="reviewer-x", researcher_id="researcher-y", provider="openai"
    )
    (run_dir / "review").mkdir()
    (run_dir / "review" / "target.yaml").write_text(
        yaml.safe_dump(review.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    both = service.validate_research_artifact(run_dir, "target")
    assert both.valid, both.problems
    assert both.review_checked

    missing = service.validate_research_artifact(run_dir, "source")
    assert not missing.valid
    assert any("does not exist" in problem for problem in missing.problems)

    unknown = service.validate_research_artifact(run_dir, "bogus")
    assert not unknown.valid
    assert any("unknown scope" in problem for problem in unknown.problems)

    (run_dir / "research" / "target.yaml").write_text("subject: [broken\n", encoding="utf-8")
    broken = service.validate_research_artifact(run_dir, "target")
    assert not broken.valid
    assert any("is invalid" in problem for problem in broken.problems)


def test_naive_now_is_interpreted_as_utc_at_the_mcp_boundary(
    tmp_path: Path, project_root: Path
) -> None:
    from llm_migrate.mcp.server import (
        generate_session_migration_plan as mcp_generate_session_migration_plan,
    )

    service = MigrationService.from_directory(project_root / "registry")
    application = project_root / "tests/fixtures/applications/anthropic_app"
    analysis = service.scan_application(application)
    request = standard_request(artifact_sha256(analysis.requirements))
    service.run_agent_research(
        request,
        standard_host(),
        tmp_path / "runs",
        now=NOW,
        shadow_canonical=["claude-sonnet-5"],
    )
    run_dir = tmp_path / "runs" / request.run_id
    naive_now = NOW.replace(tzinfo=None).isoformat()
    plan = mcp_generate_session_migration_plan(
        str(application),
        "claude-sonnet-5",
        "gpt-6.0-nova",
        str(run_dir),
        source_platform="anthropic-api",
        target_platform="openai-api",
        now=naive_now,
    )
    assert plan["target"]["model"] == "gpt-6.0-nova"


def test_guided_toolset_exposes_only_the_guided_workflow_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_migrate.mcp.server as server

    monkeypatch.setenv("LLM_MIGRATE_TOOLSET", "guided")
    try:
        reloaded = importlib.reload(server)
        names = {tool.name for tool in asyncio.run(reloaded.mcp.list_tools())}
        assert names == set(reloaded._GUIDED_TOOL_NAMES)
        assert "submit_adapted_prompt" in names
        assert "run_migration_eval" not in names
    finally:
        monkeypatch.delenv("LLM_MIGRATE_TOOLSET")
        importlib.reload(server)
