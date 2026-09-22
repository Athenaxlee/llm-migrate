"""V1.5.0-d: strict production mode and the validation disposition."""

from __future__ import annotations

import ast
import shutil
from datetime import date
from pathlib import Path

import pytest

from llm_migrate.core.workspace import load_validation_disposition
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import dispose_all, edit_change

AS_OF = date(2026, 9, 22)


@pytest.fixture
def bedrock_app(tmp_path: Path, project_root: Path) -> Path:
    app = tmp_path / "bedrock_app"
    shutil.copytree(project_root / "tests/fixtures/applications/bedrock_app", app)
    return app


def _start(service: MigrationService, app: Path, *, strict: bool):  # type: ignore[no-untyped-def]
    start = service.start_migration_run(
        app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
        strict=strict,
    )
    assert start.status == "ready" and start.paths is not None
    return start


def test_strict_flag_is_recorded_and_default_stays_off(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = _start(service, bedrock_app, strict=True)
    assert start.run is not None and start.run.strict
    lax = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
        run_id="lax-run",
    )
    assert lax.run is not None and not lax.run.strict


def test_strict_missing_invocation_facts_becomes_a_blocker(
    service: MigrationService, bedrock_app: Path
) -> None:
    """A target platform without reviewed invocation facts blocks strict runs."""
    common = dict(
        source_platform="amazon-bedrock",
        target_platform="budget-platform",
        as_of=AS_OF,
        research="skip",
    )
    lax = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "fixture-gamma-cheap-v1",
        run_id="lax-gamma",
        **common,  # type: ignore[arg-type]
    )
    assert lax.status == "ready" and lax.paths is not None
    lax_tasks = service.list_adaptation_tasks(lax.paths.run_dir)
    assert not any("invocation_facts_unknown" in blocker for blocker in lax_tasks.blockers)

    strict = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "fixture-gamma-cheap-v1",
        run_id="strict-gamma",
        strict=True,
        **common,  # type: ignore[arg-type]
    )
    assert strict.status == "ready" and strict.paths is not None
    strict_tasks = service.list_adaptation_tasks(strict.paths.run_dir)
    assert any("invocation_facts_unknown" in blocker for blocker in strict_tasks.blockers)
    # The blocker rides the normal resolution flow (accept is always offered).
    resolutions = service.get_blocker_resolutions(strict.paths.run_dir)
    resolution = next(
        item for item in resolutions.resolutions if item.blocker.code == "invocation_facts_unknown"
    )
    assert resolution.options[-1].id == "accept"


def test_strict_rejects_unknown_evidence_urls(service: MigrationService, bedrock_app: Path) -> None:
    start = _start(service, bedrock_app, strict=True)
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
        guidance_dispositions=dispose_all(service, run_dir, "app.py"),
        annotated_changes=[
            edit_change(
                "anthropic.claude-sonnet-4-6",
                "us.anthropic.claude-sonnet-5",
                why="Sonnet 5 on Bedrock is invoked through the US inference profile.",
                kind="research",
                url="https://example.com/not-a-known-source",
            )
        ],
    )
    assert not result.accepted
    assert "strict mode: unknown evidence URLs are rejected" in result.message


def test_validation_disposition_and_contract_test_deliverable(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = _start(service, bedrock_app, strict=True)
    run_dir = Path(start.paths.run_dir)
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
    batch = service.record_change_decisions(
        run_dir,
        [
            {
                "source_path": "app.py",
                "change_id": review_item.change.id,
                "decision": "accepted",
            }
            for review_item in service.get_change_review(run_dir).files[0].changes
        ],
        decided_on=AS_OF,
    )
    assert batch.recorded == len(batch.results)

    # Strict finalize without a validation disposition: loud violation.
    final = service.finalize_migration_run(run_dir)
    assert final.contract_test_path is not None
    assert any("validation disposition" in item for item in final.strict_violations)
    assert final.message.startswith("STRICT MODE:")

    # The generated contract test is deterministic, parseable Python that
    # names the invocation id and forbids the source and bare target ids.
    contract = Path(final.contract_test_path).read_text(encoding="utf-8")
    ast.parse(contract)
    assert "EXPECTED_MODEL_ID = 'us.anthropic.claude-sonnet-5'" in contract
    assert "anthropic.claude-sonnet-4-6" in contract  # forbidden source id
    assert "never executes your application" in contract

    # Recording the disposition needs a real rationale for accepts.
    with pytest.raises(ValueError, match="user's own"):
        service.record_validation_disposition(run_dir, "accepted_without_validation", "  ")
    disposition = service.record_validation_disposition(
        run_dir,
        "generated_tests",
        "Ran output/validation/test_target_invocation.py against the adapted app.",
        decided_on=AS_OF,
    )
    assert load_validation_disposition(run_dir, disposition.run_id) is not None
    clean = service.finalize_migration_run(run_dir)
    assert clean.validation_disposition == "generated_tests"
    assert clean.strict_violations == []
    assert not clean.message.startswith("STRICT MODE:")
    report = Path(clean.report_path).read_text(encoding="utf-8")
    assert "## Validation" in report
    assert "generated contract test" in report.casefold()


def test_strict_incomplete_prompt_coverage_becomes_a_blocker(
    service: MigrationService, bedrock_app: Path
) -> None:
    (bedrock_app / "dynamic.py").write_text(
        "import boto3\n\n"
        'client = boto3.client("bedrock-runtime")\n\n'
        "def ask(history):\n"
        "    return client.converse(\n"
        '        modelId="anthropic.claude-sonnet-4-6",\n'
        "        messages=build(history),\n"
        "    )\n\n"
        "def build(history):\n"
        "    return list(history)\n",
        encoding="utf-8",
    )
    strict = _start(service, bedrock_app, strict=True)
    tasks = service.list_adaptation_tasks(strict.paths.run_dir)
    assert any("incomplete_prompt_coverage" in blocker for blocker in tasks.blockers)
