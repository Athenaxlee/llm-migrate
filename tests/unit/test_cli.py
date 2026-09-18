from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from llm_migrate.cli.main import app
from tests.unit.test_proposals import research_result

runner = CliRunner()


def test_models_resolve_smoke() -> None:
    result = runner.invoke(app, ["models", "resolve", "alpha large"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["canonical_name"] == "fixture-alpha-large-v1"


def test_compare_smoke() -> None:
    result = runner.invoke(app, ["compare", "alpha large", "beta balanced"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["target_model"] == "fixture-beta-balanced-v2"


def test_v1_lifecycle_and_cost_commands() -> None:
    lifecycle = runner.invoke(app, ["models", "lifecycle", "alpha large", "--as-of", "2026-08-24"])
    cost = runner.invoke(
        app,
        [
            "estimate-cost",
            "--from",
            "alpha large",
            "--to",
            "gamma cheap",
            "--requests",
            "100",
            "--input-tokens",
            "1000",
            "--output-tokens",
            "500",
        ],
    )
    assert lifecycle.exit_code == 0, lifecycle.output
    assert not json.loads(lifecycle.stdout)["suitable_for_new_migrations"]
    assert cost.exit_code == 0, cost.output
    assert json.loads(cost.stdout)["estimated_delta_usd"] == "-1.19"


def test_missing_model_exits_cleanly() -> None:
    result = runner.invoke(app, ["models", "resolve", "does not exist"])
    assert result.exit_code == 2
    assert "no model matches" in result.output


def test_registry_validate_and_stale_commands() -> None:
    validated = runner.invoke(app, ["registry", "validate"])
    stale = runner.invoke(app, ["registry", "stale", "--as-of", "2026-09-08"])
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.stdout)["migration_count"] == 3
    assert stale.exit_code == 0, stale.output
    assert any(item["stale"] for item in json.loads(stale.stdout)["sections"])


def test_registry_propose_update_json_and_output(tmp_path: Path) -> None:
    research_path = tmp_path / "research.yaml"
    research_path.write_text(
        yaml.safe_dump(research_result().model_dump(mode="json")), encoding="utf-8"
    )
    output = tmp_path / "review"
    result = runner.invoke(
        app,
        [
            "registry",
            "propose-update",
            str(research_path),
            "--json",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["changed_facts"]
    assert (output / "fixture-alpha-evidence-update" / "proposal.md").is_file()


def test_scan_application_smoke(project_root: Path) -> None:
    result = runner.invoke(
        app,
        [
            "scan-application",
            str(project_root / "tests" / "fixtures" / "applications" / "openai_app"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["requirements"]["required_capabilities"] == [
        "image_input",
        "structured_output",
        "tool_use",
    ]


def test_invocation_prepare_and_prompt_validate_commands(project_root: Path) -> None:
    application = project_root / "tests" / "fixtures" / "applications" / "anthropic_app"
    prepared = runner.invoke(
        app,
        [
            "invocation",
            "prepare-migration",
            str(application),
            "--from",
            "claude-sonnet-5",
            "--from-platform",
            "anthropic-api",
            "--to",
            "gpt-5.6-sol",
            "--to-platform",
            "openai-api",
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    assert json.loads(prepared.stdout)["target"]["operation"] == "responses.create"

    validated = runner.invoke(
        app,
        [
            "prompt",
            "validate",
            str(project_root / "examples" / "prompt.md"),
            "--to",
            "gamma cheap",
        ],
    )
    assert validated.exit_code == 0, validated.output
    validated_payload = json.loads(validated.stdout)
    assert validated_payload["valid"]  # prompt-lexical mismatches warn, never block
    assert any(item["code"].startswith("unsupported_") for item in validated_payload["issues"])

    prompt_path = project_root / "examples" / "prompt.md"
    prompt_before = prompt_path.read_bytes()
    prepared_prompt = runner.invoke(
        app,
        [
            "prompt",
            "prepare-migration",
            str(prompt_path),
            "--from",
            "claude-sonnet-5",
            "--from-platform",
            "anthropic-api",
            "--to",
            "gpt-5.6-sol",
            "--to-platform",
            "openai-api",
            "--role",
            "system",
        ],
    )
    assert prepared_prompt.exit_code == 0, prepared_prompt.output
    prompt_payload = json.loads(prepared_prompt.stdout)
    assert prompt_payload["source_role"] == "system"
    assert prompt_payload["target_platform"] == "openai-api"
    assert prompt_path.read_bytes() == prompt_before


def test_integrated_plan_and_report_commands(project_root: Path, tmp_path: Path) -> None:
    application = project_root / "tests" / "fixtures" / "applications" / "anthropic_app"
    before = {path: path.read_bytes() for path in application.rglob("*") if path.is_file()}
    manifest_path = tmp_path / "migration-manifest.yaml"
    report_path = tmp_path / "migration-report.md"
    common = [
        str(application),
        "--from",
        "claude-sonnet-5",
        "--from-platform",
        "anthropic-api",
        "--to",
        "gpt-5.6-sol",
        "--to-platform",
        "openai-api",
    ]
    planned = runner.invoke(app, ["plan", *common, "--output", str(manifest_path)])
    reported = runner.invoke(app, ["report", *common, "--output", str(report_path)])
    assert planned.exit_code == 0, planned.output
    assert reported.exit_code == 0, reported.output
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["migration"]["schema_version"] == "3"
    assert manifest["migration"]["affected_files"] == ["app.py", "prompts/system.txt"]
    tool_migration = manifest["migration"]["invocation_changes"][0]["tool_schema_migrations"][0]
    assert tool_migration["target_definition"]["type"] == "function"
    report_text = report_path.read_text()
    assert "## Required changes" in report_text
    assert "## Tool schema candidates" in report_text
    assert "## Target platform configuration" in report_text
    assert "candidate emitted" in report_text
    after = {path: path.read_bytes() for path in application.rglob("*") if path.is_file()}
    assert before == after


def test_integrated_artifacts_refuse_to_overwrite_analyzed_source(project_root: Path) -> None:
    application = project_root / "tests" / "fixtures" / "applications" / "anthropic_app"
    source_file = application / "app.py"
    before = source_file.read_bytes()
    common = [
        str(application),
        "--from",
        "claude-sonnet-5",
        "--from-platform",
        "anthropic-api",
        "--to",
        "gpt-5.6-sol",
        "--to-platform",
        "openai-api",
        "--output",
        str(source_file),
    ]
    planned = runner.invoke(app, ["plan", *common])
    reported = runner.invoke(app, ["report", *common])
    assert planned.exit_code == 2
    assert reported.exit_code == 2
    assert "Refusing to overwrite analyzed source file" in planned.output
    assert "Refusing to overwrite analyzed source file" in reported.output
    assert source_file.read_bytes() == before


def test_evaluation_suite_run_comparison_and_regression_artifacts(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    manifest = project_root / "tests" / "golden" / "v04" / "anthropic_to_openai.yaml"
    corpus = tmp_path / "corpus.yaml"
    corpus.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "id": "smoke",
                        "input": "Return ok",
                        "expected_output": "ok",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.yaml"
    generated = runner.invoke(
        app,
        [
            "eval",
            "generate-suite",
            str(manifest),
            str(corpus),
            "--output",
            str(suite_path),
        ],
    )
    assert generated.exit_code == 0, generated.output
    suite = yaml.safe_load(suite_path.read_text())
    assert suite["schema_version"] == "2"
    assert suite["cases"][0]["validators"][0]["kind"] == "exact_match"

    source_config = tmp_path / "source.yaml"
    target_config = tmp_path / "target.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "provider": "anthropic",
                "platform": "anthropic-api",
                "model": "claude-sonnet-5",
                "model_id": "claude-sonnet-5",
                "credential_env": "V05_TEST_SOURCE_KEY",
            }
        ),
        encoding="utf-8",
    )
    target_config.write_text(
        yaml.safe_dump(
            {
                "provider": "openai",
                "platform": "openai-api",
                "model": "gpt-5.6-sol",
                "model_id": "gpt-5.6-sol",
                "credential_env": "V05_TEST_TARGET_KEY",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("V05_TEST_SOURCE_KEY", raising=False)
    monkeypatch.delenv("V05_TEST_TARGET_KEY", raising=False)
    run_path = tmp_path / "run.yaml"
    executed = runner.invoke(
        app,
        [
            "eval",
            "run",
            str(suite_path),
            str(source_config),
            str(target_config),
            "--output",
            str(run_path),
        ],
    )
    assert executed.exit_code == 0, executed.output
    run = yaml.safe_load(run_path.read_text())
    assert run["source_results"][0]["error_kind"] == "authentication"
    assert run["target_results"][0]["error_kind"] == "authentication"

    compared = runner.invoke(app, ["eval", "compare-outputs", str(run_path)])
    report_path = tmp_path / "regressions.yaml"
    analyzed = runner.invoke(
        app,
        [
            "eval",
            "analyze-regressions",
            str(run_path),
            "--output",
            str(report_path),
        ],
    )
    assert compared.exit_code == 0, compared.output
    assert analyzed.exit_code == 0, analyzed.output
    assert json.loads(compared.stdout)[0]["case_id"] == "smoke"
    assert yaml.safe_load(report_path.read_text())["summary"]["case_count"] == 1


def test_evaluation_artifact_refuses_to_overwrite_inputs(
    project_root: Path, tmp_path: Path
) -> None:
    manifest = project_root / "tests" / "golden" / "v04" / "anthropic_to_openai.yaml"
    corpus = tmp_path / "corpus.yaml"
    corpus.write_text(
        yaml.safe_dump([{"id": "smoke", "input": "hello", "expected_output": "ok"}]),
        encoding="utf-8",
    )
    before = corpus.read_bytes()
    result = runner.invoke(
        app,
        [
            "eval",
            "generate-suite",
            str(manifest),
            str(corpus),
            "--output",
            str(corpus),
        ],
    )
    assert result.exit_code == 2
    assert "Refusing to overwrite an evaluation input" in result.output
    assert corpus.read_bytes() == before


def test_v11_research_stage_commands_drive_the_file_protocol(
    tmp_path: Path, project_root: Path
) -> None:
    """An agent host can drive every V1.1 stage through CLI file artifacts."""
    from tests.unit.v11_scenarios import (
        NOW,
        pair_research,
        source_research,
        supportive_review,
        target_research,
    )

    application = project_root / "tests/fixtures/applications/anthropic_app"
    run_dir = tmp_path / "runs" / "run-cli-001"
    created = runner.invoke(
        app,
        [
            "research",
            "create-request",
            str(application),
            "--run-id",
            "run-cli-001",
            "--source-provider",
            "anthropic",
            "--source-platform",
            "anthropic-api",
            "--source-model",
            "claude-sonnet-5",
            "--source-endpoint",
            "messages",
            "--target-provider",
            "openai",
            "--target-platform",
            "openai-api",
            "--target-model",
            "gpt-6.0-nova",
            "--target-endpoint",
            "responses",
            "--as-of",
            "2026-08-29",
            "--output",
            str(run_dir / "request.yaml"),
        ],
    )
    assert created.exit_code == 0, created.output

    artifacts = {
        "source": (source_research(), "researcher-source"),
        "target": (target_research(), "researcher-target"),
        "pair": (pair_research(), "researcher-pair"),
    }
    for scope, (research, researcher) in artifacts.items():
        research_path = run_dir / "research" / f"{scope}.yaml"
        research_path.parent.mkdir(parents=True, exist_ok=True)
        research_path.write_text(
            yaml.safe_dump(research.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        review = supportive_review(
            research,
            reviewer_id="reviewer-independent",
            researcher_id=researcher,
            provider="openai",
        )
        review_path = run_dir / "review" / f"{scope}.yaml"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(
            yaml.safe_dump(review.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

    validated = runner.invoke(
        app,
        [
            "research",
            "validate-result",
            str(run_dir / "research" / "target.yaml"),
            str(run_dir / "request.yaml"),
        ],
    )
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.stdout)["valid"] is True

    reviewed = runner.invoke(
        app,
        [
            "research",
            "validate-review",
            str(run_dir / "review" / "target.yaml"),
            str(run_dir / "research" / "target.yaml"),
        ],
    )
    assert reviewed.exit_code == 0, reviewed.output

    consensus = runner.invoke(
        app,
        [
            "research",
            "consensus",
            str(run_dir / "research" / "target.yaml"),
            str(run_dir / "review" / "target.yaml"),
            str(run_dir / "request.yaml"),
        ],
    )
    assert consensus.exit_code == 0, consensus.output
    assert json.loads(consensus.stdout)["suitability"] == "suitable"

    built = runner.invoke(
        app,
        [
            "research",
            "build-session",
            str(run_dir),
            "--now",
            NOW.isoformat(),
            "--shadow",
            "claude-sonnet-5",
        ],
    )
    assert built.exit_code == 0, built.output
    outcome = json.loads(built.stdout)
    assert outcome["manifest"]["trust_level"] == "session_agent_reviewed"

    status = runner.invoke(app, ["research", "status", str(run_dir)])
    assert status.exit_code == 0, status.output
    assert json.loads(status.stdout)["stopping_reason"] == "completed"

    planned = runner.invoke(
        app,
        [
            "plan",
            str(application),
            "--from",
            "claude-sonnet-5",
            "--to",
            "gpt-6.0-nova",
            "--from-platform",
            "anthropic-api",
            "--to-platform",
            "openai-api",
            "--session",
            str(run_dir),
            "--session-as-of",
            NOW.isoformat(),
            "--json",
        ],
    )
    assert planned.exit_code == 0, planned.output
    plan_payload = json.loads(planned.stdout)
    assert plan_payload["target"]["model"] == "gpt-6.0-nova"
    assert any("Session overlay run" in warning for warning in plan_payload["warnings"])

    without_session = runner.invoke(
        app,
        [
            "plan",
            str(application),
            "--from",
            "claude-sonnet-5",
            "--to",
            "gpt-6.0-nova",
        ],
    )
    assert without_session.exit_code == 2
