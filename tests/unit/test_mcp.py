from __future__ import annotations

import asyncio

from llm_migrate.mcp.server import analyze_regressions as mcp_analyze_regressions
from llm_migrate.mcp.server import compare_outputs as mcp_compare_outputs
from llm_migrate.mcp.server import generate_eval_suite as mcp_generate_eval_suite
from llm_migrate.mcp.server import generate_migration_plan as mcp_generate_migration_plan
from llm_migrate.mcp.server import generate_migration_report as mcp_generate_migration_report
from llm_migrate.mcp.server import mcp
from llm_migrate.mcp.server import prepare_prompt_migration as mcp_prepare_prompt_migration
from llm_migrate.mcp.server import propose_registry_update as mcp_propose_registry_update
from llm_migrate.mcp.server import run_migration_eval as mcp_run_migration_eval
from tests.unit.test_proposals import research_result


def test_all_service_operations_are_exposed_as_mcp_tools() -> None:
    tools = asyncio.run(mcp.list_tools())
    assert {tool.name for tool in tools} == {
        "resolve_model",
        "get_model_profile",
        "check_model_lifecycle",
        "compare_models",
        "recommend_models",
        "analyze_prompt",
        "analyze_invocation",
        "prepare_prompt_migration",
        "prepare_invocation_migration",
        "validate_prompt",
        "generate_migration_plan",
        "generate_migration_report",
        "propose_registry_update",
        "query_live_pricing",
        "estimate_migration_cost",
        "scan_application",
        "generate_eval_suite",
        "run_migration_eval",
        "compare_outputs",
        "analyze_regressions",
        "optimize_migration",
        "create_migration_research_request",
        "validate_research_result",
        "validate_research_artifact",
        "validate_evidence_review",
        "build_research_consensus",
        "build_session_registry",
        "generate_session_migration_plan",
        "start_migration",
        "get_research_prompts",
        "list_adaptation_tasks",
        "get_blocker_resolutions",
        "record_blocker_decision",
        "submit_adapted_prompt",
        "submit_adapted_file",
        "confirm_unaffected",
        "add_prompt_sources",
        "confirm_prompt_consumer",
        "record_observation",
        "scaffold_evaluation",
        "submit_adaptations",
        "record_change_decisions",
        "get_run_status",
        "record_validation_disposition",
        "finalize_migration",
        "get_change_review",
        "record_change_decision",
    }


def test_mcp_session_research_tools_finalize_and_plan(tmp_path, project_root) -> None:  # type: ignore[no-untyped-def]
    from llm_migrate.core.agent_research import artifact_sha256
    from llm_migrate.mcp.server import build_session_registry as mcp_build_session_registry
    from llm_migrate.mcp.server import (
        generate_session_migration_plan as mcp_generate_session_migration_plan,
    )
    from llm_migrate.service import MigrationService
    from tests.unit.v11_scenarios import NOW, standard_host, standard_request

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

    outcome = mcp_build_session_registry(
        str(run_dir), now=NOW.isoformat(), shadow_canonical=["claude-sonnet-5"]
    )
    assert outcome["manifest"] is not None
    assert outcome["manifest"]["trust_level"] == "session_agent_reviewed"

    plan = mcp_generate_session_migration_plan(
        str(application),
        "claude-sonnet-5",
        "gpt-6.0-nova",
        str(run_dir),
        source_platform="anthropic-api",
        target_platform="openai-api",
        now=NOW.isoformat(),
    )
    assert plan["target"]["model"] == "gpt-6.0-nova"
    assert any("Session overlay run" in warning for warning in plan["warnings"])
    assert any("session_agent_reviewed" in warning for warning in plan["warnings"])


def test_mcp_guided_run_covers_the_vague_bedrock_scenario(tmp_path, project_root) -> None:  # type: ignore[no-untyped-def]
    """A vague Bedrock migration resolves registry-first and produces deliverables."""
    import shutil

    from llm_migrate.mcp.server import finalize_migration as mcp_finalize_migration
    from llm_migrate.mcp.server import list_adaptation_tasks as mcp_list_adaptation_tasks
    from llm_migrate.mcp.server import resolve_model as mcp_resolve_model
    from llm_migrate.mcp.server import start_migration as mcp_start_migration
    from llm_migrate.mcp.server import submit_adapted_file as mcp_submit_adapted_file

    resolved = mcp_resolve_model("us.anthropic.claude-sonnet-4-6", "bedrock")
    assert resolved["status"] == "resolved"
    assert resolved["canonical_name"] == "claude-sonnet-4-6"
    assert "resolution" not in resolved

    app = tmp_path / "bedrock_app"
    shutil.copytree(project_root / "tests/fixtures/applications/bedrock_app", app)
    ambiguous = mcp_start_migration(
        str(app),
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="bedrock",
        target_platform="bedrock",
        as_of="2026-09-15",
    )
    assert ambiguous["status"] == "needs_confirmation"
    assert {item["endpoint"] for item in ambiguous["target_match"]["candidates"]} == {
        "bedrock-runtime",
        "bedrock-mantle",
    }

    start = mcp_start_migration(
        str(app),
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="bedrock",
        target_platform="bedrock",
        target_endpoint="bedrock-runtime",
        as_of="2026-09-15",
    )
    assert start["status"] == "ready"
    assert start["research"]["level"] == "recommended"
    run_dir = start["paths"]["run_dir"]
    assert ".llm-migrate" in run_dir

    tasks = mcp_list_adaptation_tasks(run_dir)
    assert [task["source_path"] for task in tasks["file_tasks"]] == ["app.py"]

    original = (app / "app.py").read_text(encoding="utf-8")
    submission = mcp_submit_adapted_file(
        run_dir,
        "app.py",
        original.replace("anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5"),
        "Sonnet 5 uses a new Bedrock model id.",
        ["Replaced the modelId value with the US inference-profile id."],
        guidance_dispositions=[
            {"guidance_id": item["id"], "disposition": "applied", "note": ""}
            for item in tasks["file_tasks"][0]["required_changes"]
        ],
        annotated_changes=[
            {
                "operation": "edit",
                "original_anchor": "anthropic.claude-sonnet-4-6",
                "adapted_anchor": "us.anthropic.claude-sonnet-5",
                "why": "Sonnet 5 on Bedrock is invoked through the US inference profile.",
                "evidence": [{"kind": "mechanical"}],
            }
        ],
    )
    assert submission["accepted"] is True, submission["message"]

    final = mcp_finalize_migration(run_dir)
    assert final["coverage_gaps"] == []
    report = (app / ".llm-migrate/runs" / final["run_id"] / "output/migration-report.md").read_text(
        encoding="utf-8"
    )
    assert "## Adaptation deliverables" in report
    assert "Replaced the modelId value with the US inference-profile id." in report
    assert (app / "app.py").read_text(encoding="utf-8") == original


def test_mcp_proposal_tool_accepts_structured_evidence() -> None:
    result = mcp_propose_registry_update(research_result().model_dump(mode="json"))
    assert result["changed_facts"]
    assert result["candidate_model"]["identity"]["canonical_name"] == "fixture-alpha-large-v1"


def test_mcp_prompt_preparation_preserves_platform_and_role_context() -> None:
    result = mcp_prepare_prompt_migration(
        "claude-sonnet-5",
        "gpt-5.6-sol",
        "Return JSON only.",
        source_path="prompts/system.txt",
        source_role="system",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    assert result["source_path"] == "prompts/system.txt"
    assert result["source_role"] == "system"
    assert result["target_platform"] == "openai-api"


def test_mcp_integrated_plan_and_report_use_the_shared_workflow() -> None:
    application = "tests/fixtures/applications/anthropic_app"
    plan = mcp_generate_migration_plan(
        application,
        "claude-sonnet-5",
        "gpt-5.6-sol",
        "anthropic-api",
        "openai-api",
    )
    report = mcp_generate_migration_report(
        application,
        "claude-sonnet-5",
        "gpt-5.6-sol",
        "anthropic-api",
        "openai-api",
    )
    assert plan["schema_version"] == "5"
    assert plan["affected_files"] == ["app.py", "prompts/system.txt"]
    assert (
        plan["invocation_changes"][0]["tool_schema_migrations"][0]["target_definition"]["type"]
        == "function"
    )
    assert "# Migration report" in report


def test_mcp_evaluation_tools_use_shared_contracts_without_inline_credentials(
    monkeypatch,
) -> None:
    plan = mcp_generate_migration_plan(
        "tests/fixtures/applications/anthropic_app",
        "claude-sonnet-5",
        "gpt-5.6-sol",
        "anthropic-api",
        "openai-api",
    )
    suite = mcp_generate_eval_suite(
        plan,
        [{"id": "smoke", "input": "Return ok", "expected_output": "ok"}],
        "mcp-smoke",
    )
    monkeypatch.delenv("V05_MCP_SOURCE_KEY", raising=False)
    monkeypatch.delenv("V05_MCP_TARGET_KEY", raising=False)
    run = mcp_run_migration_eval(
        suite,
        {
            "provider": "anthropic",
            "platform": "anthropic-api",
            "model": "claude-sonnet-5",
            "model_id": "claude-sonnet-5",
            "credential_env": "V05_MCP_SOURCE_KEY",
        },
        {
            "provider": "openai",
            "platform": "openai-api",
            "model": "gpt-5.6-sol",
            "model_id": "gpt-5.6-sol",
            "credential_env": "V05_MCP_TARGET_KEY",
        },
    )
    comparison = mcp_compare_outputs(run["source_results"][0], run["target_results"][0])
    report = mcp_analyze_regressions(run)
    assert run["source_results"][0]["error_kind"] == "authentication"
    assert comparison["case_id"] == "smoke"
    assert report["summary"]["case_count"] == 1
