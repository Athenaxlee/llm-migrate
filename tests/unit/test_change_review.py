"""Per-change review: decisions, staleness, and deterministic regeneration."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_migrate.core.workspace import WorkspaceError, load_change_decision_log
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import dispose_all, edit_change

AS_OF = date(2026, 9, 16)

ORIGINAL_APP = textwrap.dedent(
    """\
    from anthropic import Anthropic

    client = Anthropic()
    MAX_TOKENS = 100
    client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=MAX_TOKENS,
    )
    """
)


@pytest.fixture
def file_app(tmp_path: Path) -> Path:
    app = tmp_path / "review_file_app"
    app.mkdir()
    (app / "app.py").write_text(ORIGINAL_APP, encoding="utf-8")
    return app


@pytest.fixture
def structured_app(tmp_path: Path) -> Path:
    app = tmp_path / "review_prompt_app"
    (app / "prompts").mkdir(parents=True)
    (app / "prompts" / "extract.yaml").write_text(
        'sys_prompt: "Extract the fields."\n'
        'user_prompt: "Process the document."\n'
        "temperature: 0.2\n",
        encoding="utf-8",
    )
    (app / "app.py").write_text(
        textwrap.dedent(
            """\
            import yaml
            from anthropic import Anthropic

            with open("prompts/extract.yaml") as handle:
                prompts = yaml.safe_load(handle)

            client = Anthropic()
            client.messages.create(
                model="claude-sonnet-4-6",
                system=prompts["sys_prompt"],
                messages=[{"role": "user", "content": prompts["user_prompt"]}],
                max_tokens=200,
            )
            """
        ),
        encoding="utf-8",
    )
    return app


def _start(service: MigrationService, app: Path):  # type: ignore[no-untyped-def]
    return service.start_migration_run(
        app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        research="skip",
    )


def _submit_file(service: MigrationService, run_dir, adapted: str):  # type: ignore[no-untyped-def]
    return service.submit_adapted_file(
        run_dir,
        "app.py",
        adapted,
        "Retargets the invocation and adjusts the output budget.",
        [],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, run_dir, "app.py"),
        annotated_changes=[
            edit_change(
                "claude-sonnet-4-6",
                "claude-sonnet-5",
                why="The migration targets claude-sonnet-5.",
            ),
            edit_change(
                "MAX_TOKENS = 100",
                "MAX_TOKENS = 130",
                why="The target model produces roughly 30 percent more tokens.",
            ),
        ],
    )


def test_review_decisions_regenerate_the_deliverable(
    service: MigrationService, file_app: Path
) -> None:
    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    submitted = _submit_file(service, run_dir, adapted)
    assert submitted.accepted, submitted.message

    review = service.get_change_review(run_dir)
    file_review = next(item for item in review.files if item.source_path == "app.py")
    assert file_review.pending == 2
    assert file_review.stale_reason is None
    assert "-MAX_TOKENS = 100" in file_review.diff and "+MAX_TOKENS = 130" in file_review.diff
    assert any("record_change_decision" in line for line in review.guidance)
    model_change = next(
        item for item in file_review.changes if "claude-sonnet-5" in item.change.why
    )
    budget_change = next(item for item in file_review.changes if "30 percent" in item.change.why)

    accepted = service.record_change_decision(
        run_dir, "app.py", model_change.change.id, "accepted", decided_on=AS_OF
    )
    assert accepted.accepted, accepted.problems
    assert accepted.pending_change_ids == [budget_change.change.id]

    rejected = service.record_change_decision(
        run_dir,
        "app.py",
        budget_change.change.id,
        "rejected",
        "Budget stays until we re-measure token usage.",
        decided_on=AS_OF,
    )
    assert rejected.accepted, rejected.problems
    assert not rejected.reverted_to_original
    assert rejected.pending_change_ids == []

    # The deliverable keeps the accepted edit and reverts the rejected one.
    deliverable = (Path(run_dir) / "output" / "files" / "app.py").read_text(encoding="utf-8")
    assert 'model="claude-sonnet-5"' in deliverable
    assert "MAX_TOKENS = 100" in deliverable and "MAX_TOKENS = 130" not in deliverable

    final = service.finalize_migration_run(run_dir)
    assert final.undecided_changes == []
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert "ACCEPTED in review" in report
    assert "REJECTED in review, reverted (note: Budget stays until" in report
    assert "Review outcome: 1 accepted, 1 rejected, 0 pending." in report

    # Re-deciding is possible because the as-submitted content is preserved.
    reinstated = service.record_change_decision(
        run_dir, "app.py", budget_change.change.id, "accepted", decided_on=AS_OF
    )
    assert reinstated.accepted, reinstated.problems
    deliverable = (Path(run_dir) / "output" / "files" / "app.py").read_text(encoding="utf-8")
    assert "MAX_TOKENS = 130" in deliverable


def test_rejecting_every_change_reverts_to_the_original(
    service: MigrationService, file_app: Path
) -> None:
    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    assert _submit_file(service, run_dir, adapted).accepted
    review = service.get_change_review(run_dir)
    change_ids = [item.change.id for item in review.files[0].changes]
    last = None
    for change_id in change_ids:
        last = service.record_change_decision(
            run_dir, "app.py", change_id, "rejected", "Not this release.", decided_on=AS_OF
        )
        assert last.accepted, last.problems
    assert last is not None and last.reverted_to_original
    deliverable = (Path(run_dir) / "output" / "files" / "app.py").read_text(encoding="utf-8")
    assert deliverable == ORIGINAL_APP
    report = Path(service.finalize_migration_run(run_dir).report_path).read_text(encoding="utf-8")
    assert "every change was rejected; no annotated adaptation remains" in report


def test_decisions_fail_closed_on_unknown_targets_and_unchanged_entries(
    service: MigrationService, file_app: Path
) -> None:
    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    missing = service.record_change_decision(run_dir, "app.py", "c:0000000000", "accepted")
    assert not missing.accepted
    assert any("no deliverable was submitted" in problem for problem in missing.problems)

    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    assert _submit_file(service, run_dir, adapted).accepted
    unknown = service.record_change_decision(run_dir, "app.py", "c:0000000000", "accepted")
    assert not unknown.accepted
    assert any("unknown change id" in problem for problem in unknown.problems)
    invalid = service.record_change_decision(run_dir, "app.py", "c:0000000000", "maybe")
    assert not invalid.accepted
    assert any("must be 'accepted' or 'rejected'" in problem for problem in invalid.problems)


def test_source_drift_freezes_the_review(service: MigrationService, file_app: Path) -> None:
    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    assert _submit_file(service, run_dir, adapted).accepted
    (file_app / "app.py").write_text(ORIGINAL_APP + "\nEXTRA = 1\n", encoding="utf-8")

    review = service.get_change_review(run_dir)
    file_review = next(item for item in review.files if item.source_path == "app.py")
    assert file_review.stale_reason is not None
    assert "changed since submission" in file_review.stale_reason

    change_id = file_review.changes[0].change.id
    refused = service.record_change_decision(run_dir, "app.py", change_id, "rejected")
    assert not refused.accepted
    assert any("changed since submission" in problem for problem in refused.problems)


def test_resubmission_makes_prior_decisions_stale(
    service: MigrationService, file_app: Path
) -> None:
    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    assert _submit_file(service, run_dir, adapted).accepted
    review = service.get_change_review(run_dir)
    change_id = review.files[0].changes[0].change.id
    assert service.record_change_decision(
        run_dir, "app.py", change_id, "rejected", "No.", decided_on=AS_OF
    ).accepted

    resubmitted = service.submit_adapted_file(
        run_dir,
        "app.py",
        ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5"),
        "Retargets the invocation only.",
        [],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, run_dir, "app.py"),
        annotated_changes=[
            edit_change(
                "claude-sonnet-4-6",
                "claude-sonnet-5",
                why="The migration targets claude-sonnet-5.",
            )
        ],
    )
    assert resubmitted.accepted, resubmitted.message
    review = service.get_change_review(run_dir)
    file_review = next(item for item in review.files if item.source_path == "app.py")
    assert all(item.status == "pending" for item in file_review.changes)
    assert review.stale_decisions and "NOT applied" in review.stale_decisions[0]


def test_structured_prompt_regeneration_preserves_non_prompt_values(
    service: MigrationService, structured_app: Path
) -> None:
    start = _start(service, structured_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    adapted = (
        'sys_prompt: "Extract every field from every row."\n'
        'user_prompt: "Process the supplied document faithfully."\n'
        "temperature: 0.2\n"
    )
    submitted = service.submit_adapted_prompt(
        run_dir,
        "prompts/extract.yaml",
        adapted,
        "Adapted both prompt components for the target model.",
        [],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, run_dir, "prompts/extract.yaml"),
        annotated_changes=[
            edit_change(
                "Extract the fields.",
                "Extract every field from every row.",
                why="Made the extraction scope explicit.",
            ),
            edit_change(
                "Process the document.",
                "Process the supplied document faithfully.",
                why="Made faithfulness explicit.",
            ),
        ],
    )
    assert submitted.accepted, submitted.message

    review = service.get_change_review(run_dir)
    file_review = next(item for item in review.files if item.source_path == "prompts/extract.yaml")
    faithfulness = next(item for item in file_review.changes if "faithfulness" in item.change.why)
    result = service.record_change_decision(
        run_dir,
        "prompts/extract.yaml",
        faithfulness.change.id,
        "rejected",
        "The original user prompt already performs well.",
        decided_on=AS_OF,
    )
    assert result.accepted, result.problems
    deliverable = yaml.safe_load(
        (Path(run_dir) / "output" / "prompts" / "prompts" / "extract.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert deliverable["sys_prompt"] == "Extract every field from every row."
    assert deliverable["user_prompt"] == "Process the document."
    assert deliverable["temperature"] == 0.2


def test_change_decision_log_from_another_run_is_refused(
    service: MigrationService, file_app: Path, tmp_path: Path
) -> None:
    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = Path(start.paths.run_dir)
    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    assert _submit_file(service, run_dir, adapted).accepted
    review = service.get_change_review(run_dir)
    change_id = review.files[0].changes[0].change.id
    assert service.record_change_decision(
        run_dir, "app.py", change_id, "accepted", decided_on=AS_OF
    ).accepted
    with pytest.raises(WorkspaceError, match="belongs to run"):
        load_change_decision_log(run_dir, "some-other-run")


def test_finalize_reports_undecided_changes(service: MigrationService, file_app: Path) -> None:
    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    assert _submit_file(service, run_dir, adapted).accepted
    final = service.finalize_migration_run(run_dir)
    assert final.undecided_changes == ["app.py: 2 change(s) pending review"]
    assert "awaiting review decisions" in final.message
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert report.count("pending review") >= 2


def test_cli_review_and_decide_change(service: MigrationService, file_app: Path) -> None:
    import json

    from typer.testing import CliRunner

    from llm_migrate.cli.main import app

    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    assert _submit_file(service, run_dir, adapted).accepted

    runner = CliRunner()
    listed = runner.invoke(app, ["run", "review", str(run_dir)])
    assert listed.exit_code == 0, listed.output
    payload = json.loads(listed.stdout)
    change_id = payload["files"][0]["changes"][0]["change"]["id"]

    decided = runner.invoke(
        app,
        [
            "run",
            "decide-change",
            str(run_dir),
            "app.py",
            change_id,
            "rejected",
            "--note",
            "Not this release.",
            "--decided-on",
            AS_OF.isoformat(),
        ],
    )
    assert decided.exit_code == 0, decided.output
    assert json.loads(decided.stdout)["accepted"] is True

    refused = runner.invoke(
        app, ["run", "decide-change", str(run_dir), "app.py", "c:0000000000", "accepted"]
    )
    assert refused.exit_code == 1, refused.output


def test_mcp_review_tools_wrap_the_shared_workflow(
    service: MigrationService, file_app: Path
) -> None:
    from llm_migrate.mcp.server import get_change_review as mcp_get_change_review
    from llm_migrate.mcp.server import record_change_decision as mcp_record_change_decision

    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    assert _submit_file(service, run_dir, adapted).accepted

    payload = mcp_get_change_review(str(run_dir))
    assert payload["files"][0]["pending"] == 2
    change_id = payload["files"][0]["changes"][0]["change"]["id"]
    recorded = mcp_record_change_decision(
        str(run_dir), "app.py", change_id, "accepted", decided_on=AS_OF.isoformat()
    )
    assert recorded["accepted"] is True
    assert len(recorded["pending_change_ids"]) == 1


def test_identical_resubmission_keeps_live_rejections_applied(
    service: MigrationService, file_app: Path
) -> None:
    """Review-hardening: an idempotent resubmit must not undo applied rejections."""
    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    assert _submit_file(service, run_dir, adapted).accepted
    review = service.get_change_review(run_dir)
    budget = next(
        item.change.id for item in review.files[0].changes if "30 percent" in item.change.why
    )
    assert service.record_change_decision(
        run_dir, "app.py", budget, "rejected", "No.", decided_on=AS_OF
    ).accepted
    deliverable = Path(run_dir) / "output" / "files" / "app.py"
    assert "MAX_TOKENS = 130" not in deliverable.read_text(encoding="utf-8")

    resubmitted = _submit_file(service, run_dir, adapted)
    assert resubmitted.accepted, resubmitted.message
    assert "re-applied to the deliverable" in resubmitted.message
    assert "MAX_TOKENS = 130" not in deliverable.read_text(encoding="utf-8")
    review = service.get_change_review(run_dir)
    budget_item = next(item for item in review.files[0].changes if item.change.id == budget)
    assert budget_item.status == "rejected"


def test_annotation_less_entries_are_marked_unreviewable(
    service: MigrationService, file_app: Path
) -> None:
    """A pre-annotation entry must not masquerade as a fully reviewed file."""
    from llm_migrate.core.workspace import (
        _save_adaptation_log,  # type: ignore[attr-defined]
        load_adaptation_log,
    )

    start = _start(service, file_app)
    assert start.paths is not None
    run_dir = Path(start.paths.run_dir)
    adapted = ORIGINAL_APP.replace("claude-sonnet-4-6", "claude-sonnet-5").replace(
        "MAX_TOKENS = 100", "MAX_TOKENS = 130"
    )
    assert _submit_file(service, run_dir, adapted).accepted
    log = load_adaptation_log(run_dir, start.run.run_id)  # type: ignore[union-attr]
    stripped = log.model_copy(
        update={
            "entries": [entry.model_copy(update={"annotated_changes": []}) for entry in log.entries]
        }
    )
    _save_adaptation_log(run_dir, stripped)

    review = service.get_change_review(run_dir)
    file_review = next(item for item in review.files if item.source_path == "app.py")
    assert file_review.stale_reason is not None
    assert "no annotated changes" in file_review.stale_reason
