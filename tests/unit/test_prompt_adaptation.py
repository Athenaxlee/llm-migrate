"""Evidence-based prompt adaptation: minimal candidates, structural guardrails."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_migrate.service import MigrationService

AS_OF = date(2026, 9, 16)

STRUCTURED_PROMPT = textwrap.dedent(
    """\
    You are a document correction assistant.

    <correction_rules>
    Correct a value only when confidence is at least 0.9.
    </correction_rules>

    <validation_before_response>
    Verify that every cell identifier exists in the input.
    </validation_before_response>

    Return JSON only.
    """
)


@pytest.fixture
def xml_prompt_app(tmp_path: Path) -> Path:
    app = tmp_path / "xml_prompt_app"
    (app / "prompts").mkdir(parents=True)
    (app / "prompts" / "system.txt").write_text(STRUCTURED_PROMPT, encoding="utf-8")
    (app / "app.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path

            from anthropic import Anthropic

            client = Anthropic()
            SYSTEM_PROMPT = Path("prompts/system.txt").read_text()
            client.messages.create(
                model="claude-sonnet-4-6",
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": "Correct this document."}],
                max_tokens=800,
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
    )


def test_candidate_is_source_verbatim_and_advice_carries_evidence(
    service: MigrationService,
) -> None:
    spec = service.prepare_prompt_migration(
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        STRUCTURED_PROMPT,
        source_path="prompts/system.txt",
        source_role="system",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
    )
    assert spec.candidate_prompt == STRUCTURED_PROMPT
    evidenced = [item for item in spec.instructions_needing_strengthening if item.evidence_urls]
    assert evidenced, "target guidance advice must cite registry evidence"
    assert all(url.startswith("https://") for item in evidenced for url in item.evidence_urls)


def test_prompt_tasks_carry_structure_and_evidence_guidance(
    service: MigrationService, xml_prompt_app: Path
) -> None:
    start = _start(service, xml_prompt_app)
    assert start.status == "ready" and start.paths is not None
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    task = next(item for item in tasks.prompt_tasks if item.source_path == "prompts/system.txt")
    assert task.deterministic_candidate == STRUCTURED_PROMPT
    assert any(
        "<correction_rules>" in line and "<validation_before_response>" in line
        for line in task.guidance
    )
    knowledge_lines = [
        line for line in tasks.shared_prompt_guidance if line.startswith("Model difference (")
    ]
    assert knowledge_lines, "migration-knowledge differences must reach the prompt tasks"
    assert any("(evidence: https://" in line for line in knowledge_lines)
    assert any("(evidence: https://" in line for line in task.guidance)
    assert any("Adapt prompts minimally" in line for line in tasks.guidance)


def test_structural_drop_is_rejected_without_allow_restructure(
    service: MigrationService, xml_prompt_app: Path
) -> None:
    start = _start(service, xml_prompt_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    flattened = (
        "You are a document correction assistant. Correct a value only when confidence "
        "is at least 0.9. Verify that every cell identifier exists in the input. "
        "Return JSON only."
    )
    rejected = service.submit_adapted_prompt(
        run_dir,
        "prompts/system.txt",
        flattened,
        "Rewrote the prompt as plain prose.",
        ["Flattened the XML sections."],
        submitted_on=AS_OF,
    )
    assert not rejected.accepted
    assert "<correction_rules>" in rejected.message
    assert "allow_restructure" in rejected.message

    accepted = service.submit_adapted_prompt(
        run_dir,
        "prompts/system.txt",
        flattened,
        "Restructure justified by an evaluated regression.",
        ["Flattened the XML sections after evaluation evidence."],
        allow_restructure=True,
        submitted_on=AS_OF,
    )
    assert accepted.accepted, accepted.message
    changes = yaml.safe_load(
        (Path(run_dir) / "output" / "changes.yaml").read_text(encoding="utf-8")
    )
    entry = next(item for item in changes["entries"] if item["source_path"] == "prompts/system.txt")
    assert any("restructure accepted" in warning for warning in entry["warnings"])


def test_structure_preserving_adaptation_is_accepted(
    service: MigrationService, xml_prompt_app: Path
) -> None:
    start = _start(service, xml_prompt_app)
    assert start.paths is not None
    adapted = STRUCTURED_PROMPT.replace(
        "Correct a value only when confidence is at least 0.9.",
        "Correct a value only when confidence is at least 0.9; never lower this threshold.",
    )
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/system.txt",
        adapted,
        "Strengthened the correction threshold per target guidance.",
        ["Tightened the correction-rules wording; structure unchanged."],
        submitted_on=AS_OF,
    )
    assert result.accepted, result.message


def test_difference_table_links_claims_to_evidence(
    service: MigrationService, project_root: Path
) -> None:
    report = service.generate_migration_report(
        project_root / "tests" / "fixtures" / "applications" / "configured_prompt_app",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="anthropic-api",
    )
    row = next(
        line for line in report.splitlines() if line.startswith("| parameters (temperature) ")
    )
    assert "[supported](https://" in row
    assert "[unsupported](https://" in row
    pricing_row = next(line for line in report.splitlines() if line.startswith("| pricing "))
    assert "[input $" in pricing_row and "](https://" in pricing_row
    knowledge_row = next(
        line for line in report.splitlines() if line.startswith("| migration knowledge ")
    )
    assert "](https://" in knowledge_row


def test_dropping_one_of_two_duplicate_sections_is_rejected(
    service: MigrationService, tmp_path: Path
) -> None:
    prompt = (
        "<rules>\nStay grounded.\n</rules>\n\n"
        "<example>\nInput: a\nOutput: 1\n</example>\n\n"
        "<example>\nInput: b\nOutput: 2\n</example>\n"
    )
    app = tmp_path / "dup_app"
    (app / "prompts").mkdir(parents=True)
    (app / "prompts" / "system.txt").write_text(prompt, encoding="utf-8")
    (app / "app.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path

            from anthropic import Anthropic

            client = Anthropic()
            SYSTEM_PROMPT = Path("prompts/system.txt").read_text()
            client.messages.create(
                model="claude-sonnet-4-6",
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )
            """
        ),
        encoding="utf-8",
    )
    start = _start(service, app)
    assert start.paths is not None
    one_example = prompt.replace("<example>\nInput: b\nOutput: 2\n</example>\n", "")
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/system.txt",
        one_example,
        "Tightened the prompt.",
        ["Removed a redundant example."],
        submitted_on=AS_OF,
    )
    assert not result.accepted
    assert "1 of 2 <example> section(s)" in result.message


def test_unpaired_placeholders_are_not_protected_structure(
    service: MigrationService, tmp_path: Path
) -> None:
    prompt = "<rules>\nEscalate to <ops@example.com> and report dates as <YYYY-MM-DD>.\n</rules>\n"
    app = tmp_path / "placeholder_app"
    (app / "prompts").mkdir(parents=True)
    (app / "prompts" / "system.txt").write_text(prompt, encoding="utf-8")
    (app / "app.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path

            from anthropic import Anthropic

            client = Anthropic()
            SYSTEM_PROMPT = Path("prompts/system.txt").read_text()
            client.messages.create(
                model="claude-sonnet-4-6",
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )
            """
        ),
        encoding="utf-8",
    )
    start = _start(service, app)
    assert start.paths is not None
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    task = next(item for item in tasks.prompt_tasks if item.source_path == "prompts/system.txt")
    section_lines = [line for line in task.guidance if "Preserve the structural" in line]
    assert section_lines and "<rules>" in section_lines[0]
    assert "<ops" not in section_lines[0] and "<YYYY-MM-DD>" not in section_lines[0]
    adapted = (
        "<rules>\nEscalate to the operations mailbox and use ISO dates (YYYY-MM-DD).\n</rules>\n"
    )
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/system.txt",
        adapted,
        "Reworded placeholders; structure unchanged.",
        ["Replaced the placeholder tokens with plain wording."],
        submitted_on=AS_OF,
    )
    assert result.accepted, result.message


def test_allow_restructure_requires_recorded_changes(
    service: MigrationService, xml_prompt_app: Path
) -> None:
    start = _start(service, xml_prompt_app)
    assert start.paths is not None
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/system.txt",
        "Flattened prose without any sections. Return JSON only.",
        "Restructured.",
        [],
        allow_restructure=True,
        submitted_on=AS_OF,
    )
    assert not result.accepted
    assert "allow_restructure requires at least one `changes` entry" in result.message


def test_prompt_sources_cannot_bypass_the_gate_via_file_submission(
    service: MigrationService, xml_prompt_app: Path
) -> None:
    start = _start(service, xml_prompt_app)
    assert start.paths is not None
    result = service.submit_adapted_file(
        start.paths.run_dir,
        "prompts/system.txt",
        "Flattened prose without any sections. Return JSON only.",
        "Adapted the prompt.",
        ["Flattened it."],
        submitted_on=AS_OF,
    )
    assert not result.accepted
    assert "submit_adapted_prompt" in result.message


def test_message_reorder_is_not_a_structural_drop() -> None:
    from llm_migrate.core.models import PromptSourceFormat
    from llm_migrate.core.prompt_documents import evaluate_prompt_submission

    original = yaml.safe_dump(
        {
            "messages": [
                {"role": "system", "content": "<policy>Stay safe.</policy>"},
                {"role": "user", "content": "Question one."},
                {"role": "user", "content": "Question two."},
            ]
        }
    )
    reordered = yaml.safe_dump(
        {
            "messages": [
                {"role": "user", "content": "Question one. Question two."},
                {"role": "system", "content": "<policy>Stay safe.</policy>"},
            ]
        }
    )
    problems, drops = evaluate_prompt_submission(original, reordered, PromptSourceFormat.YAML)
    assert problems == []
    assert drops == []
    flattened = yaml.safe_dump(
        {"messages": [{"role": "user", "content": "Stay safe. Question one."}]}
    )
    _, drops = evaluate_prompt_submission(original, flattened, PromptSourceFormat.YAML)
    assert any("<policy>" in drop for drop in drops)


def test_missing_original_prompt_is_flagged_not_silent(
    service: MigrationService, xml_prompt_app: Path
) -> None:
    start = _start(service, xml_prompt_app)
    assert start.paths is not None
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/new_prompt.txt",
        "A brand new prompt for the target model.",
        "The migration introduces this prompt.",
        ["Added a new prompt."],
        submitted_on=AS_OF,
    )
    assert result.accepted
    assert "structural checks were skipped" in result.message


def test_override_derived_claims_are_not_linked_to_contradicting_docs(
    service: MigrationService, project_root: Path
) -> None:
    plan = service.generate_migration_plan(
        project_root / "tests" / "fixtures" / "applications" / "configured_prompt_app",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
    )
    structured = next(
        item for item in plan.model_differences.differences if item.category == "structured_output"
    )
    assert structured.target_value is False  # bedrock capability override
    assert structured.target_evidence_url is None


def test_unchanged_file_submission_closes_coverage_without_an_invented_edit(
    service: MigrationService, xml_prompt_app: Path
) -> None:
    (xml_prompt_app / "helpers.py").write_text(
        'def retry_delays():\n    return [1, 2, 4]\n\n\nTIMEOUT = "ANTHROPIC_TIMEOUT"\n',
        encoding="utf-8",
    )
    start = _start(service, xml_prompt_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir

    identical = service.submit_adapted_file(
        run_dir,
        "helpers.py",
        (xml_prompt_app / "helpers.py").read_text(encoding="utf-8"),
        "No change needed.",
        ["Nothing to change."],
        submitted_on=AS_OF,
    )
    assert not identical.accepted
    assert "unchanged=true" in identical.message

    unchanged = service.submit_adapted_file(
        run_dir,
        "helpers.py",
        "",
        "Retry helpers are provider-neutral; no target-model change needed.",
        [],
        unchanged=True,
        submitted_on=AS_OF,
    )
    assert unchanged.accepted, unchanged.message

    lazy = service.submit_adapted_file(
        run_dir,
        "app.py",  # references the source model id
        "",
        "Looks fine.",
        [],
        unchanged=True,
        submitted_on=AS_OF,
    )
    assert not lazy.accepted
    assert "source model id" in lazy.message
