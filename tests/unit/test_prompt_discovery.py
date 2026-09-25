"""Prompt provenance discovery: config-driven sources, coverage, overrides."""

from __future__ import annotations

import shutil
import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_migrate.core.models import (
    PromptDiscoveryCoverage,
    PromptSourceConfidence,
    PromptSourceOrigin,
)
from llm_migrate.scanners import scan_application
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import dispose_all, edit_change

AS_OF = date(2026, 9, 15)


@pytest.fixture
def configured_app(tmp_path: Path, project_root: Path) -> Path:
    app = tmp_path / "configured_prompt_app"
    shutil.copytree(project_root / "tests/fixtures/applications/configured_prompt_app", app)
    return app


def _write_app(tmp_path: Path, files: dict[str, str]) -> Path:
    app = tmp_path / "app"
    for name, content in files.items():
        target = app / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(content), encoding="utf-8")
    return app


def test_yaml_prompt_source_produces_structured_components(configured_app: Path) -> None:
    analysis = scan_application(configured_app)
    multi = next(
        item
        for item in analysis.prompt_sources
        if item.path == "prompt_lib/claude_prompt_multimodal.yaml"
    )
    assert multi.format.value == "yaml"
    assert [(item.role, item.key) for item in multi.components] == [
        ("system", "sys_prompt"),
        ("user", "user_prompt"),
    ]
    assert all(item.content.strip() for item in multi.components)


def test_config_driven_prompt_path_resolves_with_provenance(configured_app: Path) -> None:
    analysis = scan_application(configured_app)
    by_path = {item.path: item for item in analysis.prompt_sources}
    multi = by_path["prompt_lib/claude_prompt_multimodal.yaml"]
    assert multi.confidence is PromptSourceConfidence.HIGH
    assert multi.provenance == [
        "app.py loads model_profiles.yaml",
        "model_profiles.yaml: models.claude.prompts.multi -> "
        "prompt_lib/claude_prompt_multimodal.yaml (resolved relative to the configuration file)",
    ]
    for referenced in ("prompt_lib/claude_prompt.yaml", "prompt_lib/claude_prompt_colmap.yaml"):
        assert by_path[referenced].confidence is PromptSourceConfidence.MEDIUM
    assert "model_profiles.yaml" not in by_path
    discovery = analysis.prompt_discovery
    assert discovery.consumers == 2
    assert discovery.source_backed_consumers == 2
    assert discovery.resolved_sources == 3
    assert discovery.coverage is PromptDiscoveryCoverage.RESOLVED


def test_open_variable_and_path_composition_resolve(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        {
            "app.py": """
                from pathlib import Path

                import yaml
                from anthropic import Anthropic

                PROMPT_DIR = Path(__file__).parent / "prompts"
                prompt_path = PROMPT_DIR / "foo.yaml"
                with open(prompt_path) as handle:
                    data = yaml.safe_load(handle)

                client = Anthropic()
                client.messages.create(
                    model="claude-sonnet-5",
                    system=data["system_prompt"],
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=100,
                )
            """,
            "prompts/foo.yaml": "system_prompt: |\n  Stay grounded.\n",
        },
    )
    analysis = scan_application(app)
    assert [item.path for item in analysis.prompt_sources] == ["prompts/foo.yaml"]
    source = analysis.prompt_sources[0]
    assert source.confidence is PromptSourceConfidence.HIGH
    assert source.provenance == ["app.py loads prompts/foo.yaml"]
    assert analysis.prompt_discovery.source_backed_consumers == 1
    assert analysis.prompt_discovery.coverage is PromptDiscoveryCoverage.RESOLVED


def test_unresolved_dynamic_prompt_is_reported_not_hidden(
    tmp_path: Path, service: MigrationService
) -> None:
    app = _write_app(
        tmp_path,
        {
            "app.py": """
                from anthropic import Anthropic

                from state import build_chat_history

                client = Anthropic()

                def run(runtime_state):
                    messages = build_chat_history(runtime_state)
                    return client.messages.create(
                        model="claude-sonnet-5",
                        messages=messages,
                        max_tokens=100,
                    )
            """,
        },
    )
    analysis = scan_application(app)
    assert analysis.prompt_discovery.dynamic_consumers == 1
    assert analysis.prompt_discovery.coverage is PromptDiscoveryCoverage.UNRESOLVED
    plan = service.generate_migration_plan(
        app,
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    assert plan.prompt_changes == []
    assert any("Prompt adaptation coverage is incomplete" in item for item in plan.warnings)
    assert any(
        "built at runtime" in item for item in (u.message for u in plan.unknowns if u.is_open)
    )
    report = service.migration_report(plan)
    assert "## Prompt discovery" in report
    assert "Coverage: **unresolved**" in report
    assert "Prompt adaptation coverage is incomplete" in report


def test_prompt_like_yaml_without_llm_provenance_is_not_a_task(
    tmp_path: Path, service: MigrationService
) -> None:
    app = _write_app(
        tmp_path,
        {
            "app.py": """
                from anthropic import Anthropic

                client = Anthropic()
                client.messages.create(
                    model="claude-sonnet-5",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=100,
                )
            """,
            "ui_config.yaml": 'ui:\n  prompt: "Enter your username"\n',
            "standalone_prompt.yaml": 'prompt: "Say hello"\n',
        },
    )
    analysis = scan_application(app)
    by_path = {item.path: item for item in analysis.prompt_sources}
    assert "ui_config.yaml" not in by_path
    standalone = by_path["standalone_prompt.yaml"]
    assert standalone.confidence is PromptSourceConfidence.LOW
    assert analysis.prompt_discovery.low_confidence_sources == 1
    plan = service.generate_migration_plan(
        app,
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    assert plan.prompt_changes == []
    assert any(
        "standalone_prompt.yaml" in item for item in (u.message for u in plan.unknowns if u.is_open)
    )


def test_explicit_prompt_source_override(tmp_path: Path, service: MigrationService) -> None:
    app = _write_app(
        tmp_path,
        {
            "app.py": """
                from anthropic import Anthropic

                client = Anthropic()
                client.messages.create(
                    model="claude-sonnet-5",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=100,
                )
            """,
            "standalone_prompt.yaml": 'prompt: "Say hello"\n',
        },
    )
    analysis = scan_application(app, prompt_sources=["standalone_prompt.yaml", "missing.yaml"])
    source = next(item for item in analysis.prompt_sources if item.path == "standalone_prompt.yaml")
    assert source.origin is PromptSourceOrigin.OVERRIDE
    assert source.confidence is PromptSourceConfidence.HIGH
    assert source.provenance[0] == "explicit prompt source override"
    assert any("missing.yaml" in item for item in analysis.warnings)
    plan = service.generate_migration_plan(
        app,
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
        prompt_sources=["standalone_prompt.yaml"],
    )
    assert [item.source_path for item in plan.prompt_changes] == ["standalone_prompt.yaml"]


def test_plan_prepares_each_component_with_role(
    configured_app: Path, service: MigrationService
) -> None:
    plan = service.generate_migration_plan(
        configured_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
    )
    changes = {
        (item.source_path, item.source_component, item.source_role) for item in plan.prompt_changes
    }
    assert (
        "prompt_lib/claude_prompt_multimodal.yaml",
        "sys_prompt",
        "system",
    ) in changes
    assert ("prompt_lib/claude_prompt_multimodal.yaml", "user_prompt", "user") in changes
    assert len(plan.prompt_changes) == 6  # three sources x two components
    assert plan.prompt_discovery.coverage is PromptDiscoveryCoverage.RESOLVED
    report = service.migration_report(plan)
    assert "## Prompt discovery" in report
    assert "Resolved prompt sources: 3" in report


def _start_run(service: MigrationService, app: Path):  # type: ignore[no-untyped-def]
    return service.start_migration_run(
        app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="bedrock",
        target_platform="bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
    )


def test_run_workspace_groups_structured_prompt_tasks(
    configured_app: Path, service: MigrationService
) -> None:
    start = _start_run(service, configured_app)
    assert start.status == "ready" and start.paths is not None
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    by_path = {task.source_path: task for task in tasks.prompt_tasks}
    assert set(by_path) == {
        "prompt_lib/claude_prompt.yaml",
        "prompt_lib/claude_prompt_colmap.yaml",
        "prompt_lib/claude_prompt_multimodal.yaml",
    }
    task = by_path["prompt_lib/claude_prompt.yaml"]
    assert task.components == ["sys_prompt", "user_prompt"]
    all_guidance = [*task.guidance, *tasks.shared_prompt_guidance]
    assert any("structured prompt document" in item.text.casefold() for item in all_guidance)
    assert all(item.id.startswith("g:") for item in all_guidance)
    # Guidance shared by every prompt task is hoisted once instead of repeated.
    shared = set(tasks.shared_prompt_guidance)
    assert shared
    assert all(not shared & set(item.guidance) for item in tasks.prompt_tasks)
    candidate = yaml.safe_load(task.verbatim_source)
    assert candidate["temperature"] == 0.2
    assert candidate["metadata"] == {"owner": "team-a"}
    assert set(candidate) == {"temperature", "sys_prompt", "user_prompt", "metadata"}


def test_structured_prompt_submission_preserves_format(
    configured_app: Path, service: MigrationService
) -> None:
    start = _start_run(service, configured_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    original = yaml.safe_load(
        (configured_app / "prompt_lib/claude_prompt.yaml").read_text(encoding="utf-8")
    )
    adapted = dict(original)
    adapted["sys_prompt"] = "You are a careful single-document summarizer.\n"
    adapted["user_prompt"] = "Summarize the supplied document faithfully.\n"
    accepted = service.submit_adapted_prompt(
        run_dir,
        "prompt_lib/claude_prompt.yaml",
        yaml.safe_dump(adapted, sort_keys=False),
        "Adapted for the target model.",
        ["Rewrote both prompt components."],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, run_dir, "prompt_lib/claude_prompt.yaml"),
        annotated_changes=[
            edit_change(
                "You are a single-document summarizer.",
                "You are a careful single-document summarizer.",
                why="Tightened the role statement for the target model.",
            ),
            edit_change(
                "Summarize the supplied document.",
                "Summarize the supplied document faithfully.",
                why="Made faithfulness explicit for the target model.",
            ),
        ],
    )
    assert accepted.accepted, accepted.message
    assert accepted.output_path is not None
    written = yaml.safe_load(Path(accepted.output_path).read_text(encoding="utf-8"))
    assert written["temperature"] == 0.2
    assert written["metadata"] == {"owner": "team-a"}

    broken = dict(adapted)
    broken["temperature"] = 0.9
    rejected = service.submit_adapted_prompt(
        run_dir,
        "prompt_lib/claude_prompt.yaml",
        yaml.safe_dump(broken, sort_keys=False),
        "Changed a non-prompt value.",
        ["Changed temperature."],
        submitted_on=AS_OF,
    )
    assert not rejected.accepted
    assert "temperature" in rejected.message

    invalid = service.submit_adapted_prompt(
        run_dir,
        "prompt_lib/claude_prompt.yaml",
        "sys_prompt: [unclosed",
        "Broken document.",
        ["Broke the YAML."],
        submitted_on=AS_OF,
    )
    assert not invalid.accepted
    assert "not valid yaml" in invalid.message


def test_run_config_persists_prompt_source_overrides(
    tmp_path: Path, project_root: Path, service: MigrationService
) -> None:
    app = tmp_path / "anthropic_app"
    shutil.copytree(project_root / "tests/fixtures/applications/anthropic_app", app)
    (app / "extra_prompt.yaml").write_text('prompt: "Extra instructions."\n', encoding="utf-8")
    start = service.start_migration_run(
        app,
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        prompt_sources=["extra_prompt.yaml"],
    )
    assert start.status == "ready" and start.run is not None and start.paths is not None
    assert start.run.prompt_sources == ["extra_prompt.yaml"]
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    assert "extra_prompt.yaml" in {task.source_path for task in tasks.prompt_tasks}
