"""V1.5.0-c: worklist snapshot, batching, disposition economics, run status."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_migrate.core.snapshot import SNAPSHOT_FILENAME, load_snapshot
from llm_migrate.core.workspace import load_adaptation_log
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import dispose_all, edit_change

AS_OF = date(2026, 9, 22)


@pytest.fixture
def bedrock_app(tmp_path: Path, project_root: Path) -> Path:
    app = tmp_path / "bedrock_app"
    shutil.copytree(project_root / "tests/fixtures/applications/bedrock_app", app)
    return app


def _start(service: MigrationService, app: Path):  # type: ignore[no-untyped-def]
    start = service.start_migration_run(
        app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
    )
    assert start.status == "ready" and start.paths is not None
    return start


def test_snapshot_reuse_and_complete_staleness_key(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = _start(service, bedrock_app)
    run_dir = Path(start.paths.run_dir)
    first = service.list_adaptation_tasks(run_dir)
    assert (run_dir / SNAPSHOT_FILENAME).is_file()
    status = service.get_run_status(run_dir)
    assert status.snapshot_reused, "an unchanged run must serve the snapshot"

    # Statuses are computed at read time, never baked into the snapshot.
    original = (bedrock_app / "app.py").read_text(encoding="utf-8")
    adapted = original.replace("anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5")
    submitted = service.submit_adapted_file(
        run_dir,
        "app.py",
        adapted,
        "Sonnet 5 uses the US inference-profile id.",
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
    assert submitted.accepted, submitted.message
    snapshot = load_snapshot(run_dir, first.run_id)
    assert snapshot is not None
    assert all(task.status == "pending" for task in snapshot.tasks.file_tasks)
    refreshed = service.list_adaptation_tasks(run_dir)
    app_task = next(task for task in refreshed.file_tasks if task.source_path == "app.py")
    assert app_task.status == "submitted"
    assert service.get_run_status(run_dir).snapshot_reused

    # Changing the scanned application invalidates the key and re-derives.
    (bedrock_app / "app.py").write_text(original + "\nTIMEOUT = 30\n", encoding="utf-8")
    assert not service.get_run_status(run_dir).snapshot_reused
    assert service.get_run_status(run_dir).snapshot_reused  # fresh again after re-derive

    # A recorded blocker decision (decisions.yaml) also invalidates the key.
    (run_dir / "decisions.yaml").write_text(
        yaml.safe_dump({"schema_version": "1", "run_id": first.run_id, "decisions": []}),
        encoding="utf-8",
    )
    assert not service.get_run_status(run_dir).snapshot_reused


def test_submit_adaptations_batches_with_per_item_results(
    service: MigrationService, bedrock_app: Path
) -> None:
    (bedrock_app / "helper.py").write_text("import anthropic\nVALUE = 1\n", encoding="utf-8")
    start = _start(service, bedrock_app)
    run_dir = start.paths.run_dir
    original = (bedrock_app / "app.py").read_text(encoding="utf-8")
    adapted = original.replace("anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5")
    dispositions = [item.model_dump() for item in dispose_all(service, run_dir, "app.py")]
    outcome = service.submit_adaptations(
        run_dir,
        [
            {
                "kind": "file",
                "source_path": "app.py",
                "content": adapted,
                "rationale": "Sonnet 5 uses the US inference-profile id.",
                "changes": ["Replaced the modelId value."],
                "guidance_dispositions": dispositions,
                "annotated_changes": [
                    edit_change(
                        "anthropic.claude-sonnet-4-6",
                        "us.anthropic.claude-sonnet-5",
                        why="Sonnet 5 on Bedrock is invoked through the US profile.",
                    ).model_dump()
                ],
            },
            {
                "kind": "file",
                "source_path": "new_module.py",
                "content": "def broken(:\n",
                "rationale": "why",
                "changes": ["added"],
                "new_file": True,
            },
        ],
        submitted_on=AS_OF,
    )
    assert outcome.accepted == 1
    assert [item.accepted for item in outcome.results] == [True, False]
    assert "does not parse" in outcome.results[1].message
    log = load_adaptation_log(Path(run_dir), outcome.run_id)
    assert {entry.source_path for entry in log.entries} == {"app.py"}


def test_default_disposition_expands_into_visible_records(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = _start(service, bedrock_app)
    run_dir = start.paths.run_dir
    original = (bedrock_app / "app.py").read_text(encoding="utf-8")
    adapted = original.replace("anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5")
    result = service.submit_adapted_file(
        run_dir,
        "app.py",
        adapted,
        "Sonnet 5 uses the US inference-profile id.",
        ["Replaced the modelId value."],
        submitted_on=AS_OF,
        default_disposition="applied",
        annotated_changes=[
            edit_change(
                "anthropic.claude-sonnet-4-6",
                "us.anthropic.claude-sonnet-5",
                why="Sonnet 5 on Bedrock is invoked through the US inference profile.",
            )
        ],
    )
    assert result.accepted, result.message
    log = load_adaptation_log(Path(run_dir), start.run.run_id)
    entry = log.entries[0]
    assert entry.guidance_dispositions, "defaults must expand into per-item records"
    assert all(item.defaulted for item in entry.guidance_dispositions)
    assert all(item.guidance for item in entry.guidance_dispositions)
    final = service.finalize_migration_run(run_dir)
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert "DEFAULTED" in report


def test_run_status_walks_the_state_machine(service: MigrationService, bedrock_app: Path) -> None:
    start = _start(service, bedrock_app)
    run_dir = start.paths.run_dir
    status = service.get_run_status(run_dir)
    assert status.state == "tasks_pending"
    assert status.pending_tasks == 1
    original = (bedrock_app / "app.py").read_text(encoding="utf-8")
    adapted = original.replace("anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5")
    service.submit_adapted_file(
        run_dir,
        "app.py",
        adapted,
        "Sonnet 5 uses the US inference-profile id.",
        ["Replaced the modelId value."],
        submitted_on=AS_OF,
        default_disposition="applied",
        annotated_changes=[
            edit_change(
                "anthropic.claude-sonnet-4-6",
                "us.anthropic.claude-sonnet-5",
                why="Sonnet 5 on Bedrock is invoked through the US inference profile.",
            )
        ],
    )
    status = service.get_run_status(run_dir)
    assert status.state == "review_pending"
    assert status.pending_review_changes == 1
    review = service.get_change_review(run_dir)
    change_id = review.files[0].changes[0].change.id
    batch = service.record_change_decisions(
        run_dir,
        [{"source_path": "app.py", "change_id": change_id, "decision": "accepted"}],
        decided_on=AS_OF,
    )
    assert batch.recorded == 1
    status = service.get_run_status(run_dir)
    assert status.state == "ready_to_finalize"
    assert "finalize_migration" in status.next_action


def test_shared_guidance_is_disposable_once_per_run(
    service: MigrationService, project_root: Path, tmp_path: Path
) -> None:
    app = tmp_path / "configured_prompt_app"
    shutil.copytree(project_root / "tests/fixtures/applications/configured_prompt_app", app)
    start = _start(service, app)
    run_dir = start.paths.run_dir
    tasks = service.list_adaptation_tasks(run_dir)
    assert len(tasks.prompt_tasks) >= 2 and tasks.shared_prompt_guidance
    first, second = tasks.prompt_tasks[0], tasks.prompt_tasks[1]

    def adapt(task):  # type: ignore[no-untyped-def]
        # Structured prompt documents: change only the sys_prompt component.
        data = yaml.safe_load(task.verbatim_source)
        data["sys_prompt"] = data["sys_prompt"].rstrip() + " Always answer in complete sentences.\n"
        return yaml.safe_dump(data, sort_keys=False)

    first_result = service.submit_adapted_prompt(
        run_dir,
        first.source_path,
        adapt(first),
        "Strengthened the instruction for the target model.",
        ["Added an explicit completeness instruction."],
        submitted_on=AS_OF,
        default_disposition="not_applicable",
        default_disposition_note="Reviewed; no other guidance applies to this prompt.",
        annotated_changes=[
            edit_change(
                "",
                "Always answer in complete sentences.",
                why="The target model needs the explicit instruction.",
            ).model_copy(update={"operation": "insert", "original_anchor": ""})
        ],
    )
    assert first_result.accepted, first_result.message
    # The second submission no longer owes the shared guidance: only its own
    # per-task guidance needs disposing.
    second_result = service.submit_adapted_prompt(
        run_dir,
        second.source_path,
        adapt(second),
        "Strengthened the instruction for the target model.",
        ["Added an explicit completeness instruction."],
        submitted_on=AS_OF,
        guidance_dispositions=[
            {"guidance_id": item.id, "disposition": "not_applicable", "note": ""}
            for item in second.guidance
        ],
        annotated_changes=[
            edit_change(
                "",
                "Always answer in complete sentences.",
                why="The target model needs the explicit instruction.",
            ).model_copy(update={"operation": "insert", "original_anchor": ""})
        ],
    )
    assert second_result.accepted, second_result.message
