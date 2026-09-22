"""V1.5.0-b: semantic config couplings, difference propagation, diet, consistency."""

from __future__ import annotations

import shutil
import textwrap
from datetime import date
from pathlib import Path

import pytest

from llm_migrate.core.models import CouplingKind
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import dispose_all, edit_change

AS_OF = date(2026, 9, 22)

CONFIG_APP_CODE = textwrap.dedent(
    """\
    import boto3
    import yaml

    with open("config/model_profiles.yaml") as handle:
        profile = yaml.safe_load(handle)

    client = boto3.client("bedrock-runtime")
    response = client.converse(
        modelId=profile["model_id"],
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
    )
    """
)

CONFIG_DOC = textwrap.dedent(
    """\
    model_id: us.anthropic.claude-sonnet-4-6
    temperature: 0.2
    max_tokens: 1024
    region: us-east-1
    input_price_per_million: 3.0
    unrelated: value
    """
)

UNANCHORED_DOC = textwrap.dedent(
    """\
    temperature: 0.7
    max_tokens: 2048
    input_price_per_million: 9.0
    """
)


@pytest.fixture
def config_app(tmp_path: Path) -> Path:
    app = tmp_path / "config_app"
    (app / "config").mkdir(parents=True)
    (app / "app.py").write_text(CONFIG_APP_CODE, encoding="utf-8")
    (app / "config" / "model_profiles.yaml").write_text(CONFIG_DOC, encoding="utf-8")
    (app / "ci_settings.yaml").write_text(UNANCHORED_DOC, encoding="utf-8")
    (app / "helper.py").write_text("import anthropic\n", encoding="utf-8")
    return app


def test_anchored_config_values_become_couplings(
    service: MigrationService, config_app: Path
) -> None:
    analysis = service.scan_application(config_app)
    config_findings = [
        item for item in analysis.findings if item.location.path == "config/model_profiles.yaml"
    ]
    kinds = {(item.kind, (item.metadata or {}).get("value_kind")) for item in config_findings}
    assert (CouplingKind.MODEL_IDENTIFIER, "model_id") in kinds
    assert (CouplingKind.PARAMETER, "sampling") in kinds
    assert (CouplingKind.PARAMETER, "token_budget") in kinds
    assert (CouplingKind.CONFIGURATION, "region") in kinds
    assert (CouplingKind.CONFIGURATION, "pricing") in kinds
    model_finding = next(
        item for item in config_findings if item.kind is CouplingKind.MODEL_IDENTIFIER
    )
    assert model_finding.value == "us.anthropic.claude-sonnet-4-6"
    # The detected config model id feeds the source-model requirements.
    assert "us.anthropic.claude-sonnet-4-6" in analysis.requirements.source_models
    # The unanchored document (no model id, never traced into an invocation)
    # contributes nothing — pricing-shaped values alone are never couplings.
    assert not any(item.location.path == "ci_settings.yaml" for item in analysis.findings)


def test_usage_anchoring_without_model_id_promotes_non_pricing_values(
    service: MigrationService, tmp_path: Path
) -> None:
    app = tmp_path / "usage_app"
    app.mkdir()
    (app / "app.py").write_text(CONFIG_APP_CODE.replace("config/", ""), encoding="utf-8")
    (app / "model_profiles.yaml").write_text(
        "model_id: unregistered-model\ntemperature: 0.5\ninput_price_per_million: 4.0\n",
        encoding="utf-8",
    )
    analysis = service.scan_application(app)
    config_findings = [
        item for item in analysis.findings if item.location.path == "model_profiles.yaml"
    ]
    value_kinds = {(item.metadata or {}).get("value_kind") for item in config_findings}
    # Usage-anchored: the document's values reach the converse() call.
    assert "model_id" in value_kinds and "sampling" in value_kinds
    # Pricing still requires a registry-matched model anchor.
    assert "pricing" not in value_kinds


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


def test_config_file_gets_a_task_and_sdk_only_file_is_unaffected(
    service: MigrationService, config_app: Path
) -> None:
    start = _start(service, config_app)
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    task_paths = {task.source_path for task in tasks.file_tasks}
    assert "config/model_profiles.yaml" in task_paths
    assert tasks.unaffected_files == ["helper.py"]
    assert "helper.py" not in task_paths
    assert any("confirm_unaffected" in line for line in tasks.guidance)
    # Non-prompt model differences arrive as required changes on the governed
    # files, not as shared prompt guidance (the one propagation mechanism).
    config_task = next(
        task for task in tasks.file_tasks if task.source_path == "config/model_profiles.yaml"
    )
    assert any(
        "Model difference" in item.text and "temperature" in item.text
        for item in config_task.required_changes
    )
    assert not any("temperature" in item.text for item in tasks.shared_prompt_guidance)


def test_confirm_unaffected_records_reviewed_entries_per_file(
    service: MigrationService, config_app: Path
) -> None:
    start = _start(service, config_app)
    run_dir = start.paths.run_dir
    with pytest.raises(ValueError, match="requires a rationale"):
        service.confirm_unaffected(run_dir, ["helper.py"], "   ")
    confirmation = service.confirm_unaffected(
        run_dir,
        ["helper.py", "app.py", "nonexistent.py"],
        "Only an incidental SDK import; nothing invokes a model here.",
        submitted_on=AS_OF,
    )
    by_path = {item.source_path: item for item in confirmation.results}
    assert confirmation.confirmed == 1
    assert by_path["helper.py"].accepted
    assert not by_path["app.py"].accepted
    assert any("required changes" in problem for problem in by_path["app.py"].problems)
    assert not by_path["nonexistent.py"].accepted
    # The confirmed file leaves the unaffected list and never becomes a gap.
    refreshed = service.list_adaptation_tasks(run_dir)
    assert refreshed.unaffected_files == []
    final = service.finalize_migration_run(run_dir)
    assert "helper.py" not in final.coverage_gaps
    assert final.unconfirmed_unaffected == []
    assert final.reviewed_unchanged == 1
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert "helper.py" in report and "no change needed" in report


def test_consistency_gate_flags_reverted_source_reference(
    service: MigrationService, project_root: Path, tmp_path: Path
) -> None:
    app = tmp_path / "bedrock_app"
    shutil.copytree(project_root / "tests/fixtures/applications/bedrock_app", app)
    start = _start(service, app)
    run_dir = start.paths.run_dir
    original = (app / "app.py").read_text(encoding="utf-8")
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
    clean = service.finalize_migration_run(run_dir)
    assert clean.consistency_findings == []
    # The user rejects the id swap in review: the deliverable reverts to the
    # original and now references the source model again — only the
    # cross-surface gate sees that.
    review = service.get_change_review(run_dir)
    change_id = review.files[0].changes[0].change.id
    decided = service.record_change_decision(
        run_dir, "app.py", change_id, "rejected", decided_on=AS_OF
    )
    assert decided.accepted, decided.problems
    final = service.finalize_migration_run(run_dir)
    assert any("source_reference_remains" in item for item in final.consistency_findings)
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert "Cross-surface consistency" in report
    assert "source_reference_remains" in report


def test_consistency_gate_flags_dropped_coupling_and_mixed_selectors(
    service: MigrationService, config_app: Path
) -> None:
    start = _start(service, config_app)
    run_dir = start.paths.run_dir
    original = (config_app / "app.py").read_text(encoding="utf-8")
    # A global restructure covers the diff without accounting for the dropped
    # toolConfig-free rewrite; here it silently drops the temperature-free
    # invocation shape — the config file keeps its couplings.
    adapted_config = (
        "model_id: eu.anthropic.claude-sonnet-5\nregion: eu-central-1\nunrelated: value\n"
    )
    from tests.unit.adaptation_helpers import restructure_change

    confirmed = service.submit_adapted_file(
        run_dir,
        "config/model_profiles.yaml",
        adapted_config,
        "Retargeted the profile to Sonnet 5 in the EU profile.",
        ["Rewrote the model profile for the EU deployment."],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, run_dir, "config/model_profiles.yaml"),
        annotated_changes=[restructure_change(why="Rewrote the profile for the new deployment.")],
    )
    assert confirmed.accepted, confirmed.message
    adapted_code = original.replace('profile["model_id"]', '"us.anthropic.claude-sonnet-5"')
    coded = service.submit_adapted_file(
        run_dir,
        "app.py",
        adapted_code,
        "Hardcoded the US inference-profile id.",
        ["Inlined the model id."],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, run_dir, "app.py"),
        annotated_changes=[
            edit_change(
                'profile["model_id"]',
                '"us.anthropic.claude-sonnet-5"',
                why="Bedrock requires the inference-profile id for Sonnet 5.",
            )
        ],
    )
    assert coded.accepted, coded.message
    final = service.finalize_migration_run(run_dir)
    codes = {item.split("]", 1)[0].lstrip("[") for item in final.consistency_findings}
    assert "mixed_target_selectors" in codes
    assert "dropped_coupling_undisposed" in codes
