"""V1.4 interactive blocker resolution: questions, options, durable decisions."""

from __future__ import annotations

import shutil
import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_migrate.core.blockers import load_decision_log
from llm_migrate.core.models import ResolutionKind
from llm_migrate.core.workspace import load_run_config
from llm_migrate.service import MigrationService

AS_OF = date(2026, 9, 18)

NATIVE_APP = textwrap.dedent(
    """\
    import json

    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "extract"}],
        max_tokens=500,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {}},
            }
        },
    )
    payload = json.loads(response.content[0].text)
    """
)


@pytest.fixture
def native_app(tmp_path: Path) -> Path:
    """An application using native structured output on the Anthropic API."""
    app = tmp_path / "native_app"
    app.mkdir()
    (app / "app.py").write_text(NATIVE_APP, encoding="utf-8")
    return app


def _start_blocked_run(service: MigrationService, app: Path) -> Path:
    """Target claude-sonnet-5 on Bedrock, whose override declares no structured output."""
    start = service.start_migration_run(
        app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
    )
    assert start.status == "ready" and start.paths is not None
    return Path(start.paths.run_dir)


def _blocker_id(service: MigrationService, run_dir: Path, code: str) -> str:
    resolutions = service.get_blocker_resolutions(run_dir)
    return next(item.blocker.id for item in resolutions.resolutions if item.blocker.code == code)


def test_blocked_run_produces_questions_with_evidence_backed_options(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.blockers, "the run must be blocked for this scenario"
    assert any("get_blocker_resolutions" in line for line in tasks.guidance)
    assert any("never retry submissions" in line.casefold() for line in tasks.guidance)

    resolutions = service.get_blocker_resolutions(run_dir)
    assert resolutions.run_id == load_run_config(run_dir).run_id
    assert {item.blocker.code for item in resolutions.resolutions} == {
        "structured_output_unsupported",
        "structured_output_payload_unsupported",
        "structured_output_mapping_missing",
    }
    assert any("VERBATIM" in line for line in resolutions.guidance)
    for item in resolutions.resolutions:
        assert item.question.endswith("?")
        assert 2 <= len(item.options) <= 5
        accept = item.options[-1]
        assert accept.kind is ResolutionKind.ACCEPT_WITH_RATIONALE
        assert accept.id == "accept"
        assert any("never a default" in text for text in accept.consequences)
        for option in item.options:
            assert option.next_step.tool == "record_blocker_decision"
            assert option.next_step.arguments["blocker_id"] == item.blocker.id
            assert option.next_step.arguments["option_id"] == option.id

    unsupported = next(
        item
        for item in resolutions.resolutions
        if item.blocker.code == "structured_output_unsupported"
    )
    # The blocker itself cites the platform-scoped registry fact (AWS card).
    assert any("docs.aws.amazon.com" in url for url in unsupported.blocker.evidence_urls)
    option_ids = [option.id for option in unsupported.options]
    assert "retarget-anthropic-api" in option_ids
    assert "redesign" in option_ids
    # An alternative model on Bedrock cannot clear the missing deterministic
    # structured-output mapping, so no alternative option may be offered here.
    assert not any(option_id.startswith("alternative-") for option_id in option_ids)
    retarget = next(o for o in unsupported.options if o.id == "retarget-anthropic-api")
    assert retarget.evidence_urls and retarget.evidence_urls[0].startswith("https://")
    assert retarget.target_change is not None
    assert retarget.target_change.platform == "anthropic-api"
    redesign = next(o for o in unsupported.options if o.id == "redesign")
    assert redesign.task is not None
    assert redesign.task.category == "output_contract"
    assert any("json.loads" in line for line in redesign.task.guidance)
    assert redesign.evidence_urls, "redesign guidance must cite registry evidence"


def test_retarget_decision_updates_run_identity_and_unblocks(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    blocker_id = _blocker_id(service, run_dir, "structured_output_unsupported")
    result = service.record_blocker_decision(
        run_dir,
        blocker_id,
        "retarget-anthropic-api",
        decided_on=AS_OF,
    )
    assert result.accepted, result.problems
    assert result.run_config_updated
    config = load_run_config(run_dir)
    assert config.target.platform == "anthropic-api"
    assert config.target_model_id == "claude-sonnet-5"
    # Retargeting to a structured-output-capable endpoint clears every
    # structured-output blocker at once.
    assert result.unresolved_blockers == []
    assert (run_dir / "decisions.yaml").is_file()
    log = load_decision_log(run_dir, config.run_id)
    assert [item.kind for item in log.decisions] == [ResolutionKind.RETARGET]

    finalization = service.finalize_migration_run(run_dir)
    assert finalization.migration_complexity != "blocked"
    assert finalization.unresolved_blockers == []
    assert len(finalization.resolved_blockers) == 1
    assert finalization.stale_decisions == []
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "## Decisions" in report
    assert "**Retarget**" in report


def test_redesign_decision_injects_required_task_with_evidence(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    blocker_id = _blocker_id(service, run_dir, "structured_output_unsupported")
    result = service.record_blocker_decision(run_dir, blocker_id, "redesign", decided_on=AS_OF)
    assert result.accepted, result.problems
    assert not result.run_config_updated
    # Only the decided blocker is suppressed; the sibling blockers stay live.
    assert len(result.unresolved_blockers) == 2
    assert not any(blocker_id in line for line in result.unresolved_blockers)

    tasks = service.list_adaptation_tasks(run_dir)
    app_task = next(task for task in tasks.file_tasks if task.source_path == "app.py")
    redesign_changes = [
        change
        for change in app_task.required_changes
        if "Redesign off native structured output" in change
    ]
    assert redesign_changes
    assert "Guidance:" in redesign_changes[0]
    assert "evidence:" in redesign_changes[0]
    # Fail closed: the remaining blockers keep the plan blocked.
    finalization = service.finalize_migration_run(run_dir)
    assert finalization.migration_complexity == "blocked"
    assert len(finalization.unresolved_blockers) == 2
    assert len(finalization.resolved_blockers) == 1


def test_accept_requires_user_rationale_and_stays_reported(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    resolutions = service.get_blocker_resolutions(run_dir)
    blocker_ids = [item.blocker.id for item in resolutions.resolutions]

    refused = service.record_blocker_decision(run_dir, blocker_ids[0], "accept")
    assert not refused.accepted
    assert any("rationale" in problem for problem in refused.problems)
    assert not (run_dir / "decisions.yaml").is_file()

    rationale = "We ship without native structured output until Bedrock supports it."
    for blocker_id in blocker_ids:
        result = service.record_blocker_decision(
            run_dir, blocker_id, "accept", rationale, decided_on=AS_OF
        )
        assert result.accepted, result.problems
    assert result.unresolved_blockers == []

    finalization = service.finalize_migration_run(run_dir)
    assert finalization.migration_complexity != "blocked"
    assert finalization.unresolved_blockers == []
    assert len(finalization.resolved_blockers) == 3
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "**ACCEPTED RISK**" in report
    assert rationale in report
    # The manifest records the decisions durably alongside the plan.
    manifest = yaml.safe_load(Path(finalization.manifest_path).read_text(encoding="utf-8"))
    decisions = manifest["migration"]["decisions"]
    assert len(decisions) == 3
    assert all(item["decision"]["rationale"] == rationale for item in decisions)
    assert manifest["migration"]["blockers"] == []


def test_wrong_ids_fail_closed(service: MigrationService, native_app: Path) -> None:
    run_dir = _start_blocked_run(service, native_app)
    unknown = service.record_blocker_decision(run_dir, "nope:0000000000", "accept", "why")
    assert not unknown.accepted
    assert any("not an unresolved blocker" in problem for problem in unknown.problems)
    blocker_id = _blocker_id(service, run_dir, "structured_output_unsupported")
    bad_option = service.record_blocker_decision(run_dir, blocker_id, "invent-something")
    assert not bad_option.accepted
    assert any("valid options" in problem for problem in bad_option.problems)
    assert not (run_dir / "decisions.yaml").is_file()


def test_consistency_blocker_resolved_by_source_correction(
    service: MigrationService, project_root: Path, tmp_path: Path
) -> None:
    app = tmp_path / "openai_app"
    shutil.copytree(project_root / "tests/fixtures/applications/openai_app", app)
    start = service.start_migration_run(
        app,
        "claude-sonnet-5",  # wrong: the code actually uses gpt-5.6-sol
        "claude-sonnet-4-6",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        research="skip",
    )
    assert start.status == "ready" and start.paths is not None
    run_dir = Path(start.paths.run_dir)
    resolutions = service.get_blocker_resolutions(run_dir)
    codes = {item.blocker.code for item in resolutions.resolutions}
    assert {
        "source_model_mismatch",
        "source_provider_mismatch",
        "source_platform_mismatch",
    } <= codes

    model_mismatch = next(
        item for item in resolutions.resolutions if item.blocker.code == "source_model_mismatch"
    )
    correction = next(
        option
        for option in model_mismatch.options
        if option.id == "correct-source-model-gpt-5.6-sol"
    )
    assert correction.kind is ResolutionKind.CORRECTION
    assert correction.source_change is not None
    # The correction pairs the detected model with its detected platform.
    assert correction.source_change.platform == "openai-api"
    assert any("start_migration" in text for text in correction.consequences)

    # A provider mismatch alone has no registry-backed correction of its own.
    provider_mismatch = next(
        item for item in resolutions.resolutions if item.blocker.code == "source_provider_mismatch"
    )
    assert [option.id for option in provider_mismatch.options] == ["accept"]
    assert provider_mismatch.no_registry_backed_option is not None

    result = service.record_blocker_decision(
        run_dir, model_mismatch.blocker.id, correction.id, decided_on=AS_OF
    )
    assert result.accepted, result.problems
    assert result.run_config_updated
    config = load_run_config(run_dir)
    assert config.source.model == "gpt-5.6-sol"
    assert config.source.platform == "openai-api"
    assert config.source_model_id == "gpt-5.6-sol"
    # Correcting the source clears every consistency blocker at once.
    assert not any("mismatch" in line for line in result.unresolved_blockers)


def test_stale_decision_is_reported_not_applied(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    blocker_id = _blocker_id(service, run_dir, "structured_output_unsupported")
    result = service.record_blocker_decision(run_dir, blocker_id, "redesign", decided_on=AS_OF)
    assert result.accepted
    # The application changes: native structured output is removed entirely.
    (native_app / "app.py").write_text(
        NATIVE_APP.replace(
            """    output_config={
        "format": {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {}},
        }
    },
""",
            "",
        ),
        encoding="utf-8",
    )
    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.blockers == []
    app_task = next(task for task in tasks.file_tasks if task.source_path == "app.py")
    assert not any(
        "Redesign off native structured output" in change for change in app_task.required_changes
    ), "a stale decision must not inject its task"
    finalization = service.finalize_migration_run(run_dir)
    assert finalization.unresolved_blockers == []
    assert finalization.resolved_blockers == []
    assert len(finalization.stale_decisions) == 1
    assert "STALE" in finalization.message
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "**STALE decision**" in report


def test_guided_run_drives_blocked_to_finalized_without_relisting(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.blockers
    blocker_id = _blocker_id(service, run_dir, "structured_output_mapping_missing")
    decided = service.record_blocker_decision(
        run_dir, blocker_id, "retarget-anthropic-api", decided_on=AS_OF
    )
    assert decided.accepted and decided.unresolved_blockers == []
    # With zero unresolved blockers the workflow moves on; it never asks for
    # another blocker-resolution round.
    assert not any("get_blocker_resolutions" in step for step in decided.next_steps)
    assert any("list_adaptation_tasks" in step for step in decided.next_steps)

    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.blockers == []
    assert [task.source_path for task in tasks.file_tasks] == ["app.py"]
    submitted = service.submit_adapted_file(
        run_dir,
        "app.py",
        NATIVE_APP.replace("claude-sonnet-4-6", "claude-sonnet-5"),
        "Move the invocation to claude-sonnet-5 on the Anthropic API per the decision.",
        ["Replaced the model identifier with claude-sonnet-5."],
        submitted_on=AS_OF,
    )
    assert submitted.accepted, submitted.message

    finalization = service.finalize_migration_run(run_dir)
    assert finalization.migration_complexity != "blocked"
    assert finalization.unresolved_blockers == []
    assert finalization.coverage_gaps == []
    assert len(finalization.resolved_blockers) == 1
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "## Decisions" in report
    assert "Blocker decisions: 1 applied (0 accepted risk), 0 stale" in report


def test_mcp_blocker_tools_wrap_the_shared_workflow(
    service: MigrationService, native_app: Path
) -> None:
    from llm_migrate.mcp.server import (
        get_blocker_resolutions as mcp_get_blocker_resolutions,
    )
    from llm_migrate.mcp.server import (
        record_blocker_decision as mcp_record_blocker_decision,
    )

    run_dir = _start_blocked_run(service, native_app)
    payload = mcp_get_blocker_resolutions(str(run_dir))
    assert payload["resolutions"]
    first = payload["resolutions"][0]
    recorded = mcp_record_blocker_decision(
        str(run_dir),
        first["blocker"]["id"],
        "accept",
        rationale="Reviewed with the user; accepted for a staged rollout.",
    )
    assert recorded["accepted"] is True
    assert len(recorded["unresolved_blockers"]) == 2


def test_cli_blockers_and_decide_commands(service: MigrationService, native_app: Path) -> None:
    import json

    from typer.testing import CliRunner

    from llm_migrate.cli.main import app

    runner = CliRunner()
    run_dir = _start_blocked_run(service, native_app)
    listed = runner.invoke(app, ["run", "blockers", str(run_dir)])
    assert listed.exit_code == 0, listed.output
    resolutions = json.loads(listed.stdout)["resolutions"]
    assert resolutions
    blocker_id = resolutions[0]["blocker"]["id"]

    refused = runner.invoke(app, ["run", "decide", str(run_dir), blocker_id, "accept"])
    assert refused.exit_code == 1, refused.output

    decided = runner.invoke(
        app,
        [
            "run",
            "decide",
            str(run_dir),
            blocker_id,
            "accept",
            "--rationale",
            "User accepted the gap for now.",
        ],
    )
    assert decided.exit_code == 0, decided.output
    assert json.loads(decided.stdout)["accepted"] is True
