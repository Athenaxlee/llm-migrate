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
    assessment = evaluate_prompt_submission(original, reordered, PromptSourceFormat.YAML)
    assert assessment.problems == []
    assert assessment.structural_drops == []
    assert assessment.runtime_changed
    flattened = yaml.safe_dump(
        {"messages": [{"role": "user", "content": "Stay safe. Question one."}]}
    )
    assessment = evaluate_prompt_submission(original, flattened, PromptSourceFormat.YAML)
    assert any("<policy>" in drop for drop in assessment.structural_drops)


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
    # An override-derived claim may cite only the platform entry's own sources
    # (the AWS model card), never a profile-level document that could assert
    # the opposite value.
    assert structured.target_evidence_url is not None
    assert "docs.aws.amazon.com" in structured.target_evidence_url


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


@pytest.fixture
def json_prompt_app(tmp_path: Path) -> Path:
    """App whose prompt enforces JSON through prompt text plus json.loads()."""
    app = tmp_path / "json_prompt_app"
    (app / "prompts").mkdir(parents=True)
    (app / "prompts" / "extract.yaml").write_text(
        'sys_prompt: "Extract the fields.\\nExtract the fields.\\nReturn valid JSON only."\n',
        encoding="utf-8",
    )
    (app / "app.py").write_text(
        textwrap.dedent(
            """\
            import json

            import yaml
            from anthropic import Anthropic

            with open("prompts/extract.yaml") as handle:
                prompts = yaml.safe_load(handle)

            client = Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                system=prompts["sys_prompt"],
                messages=[{"role": "user", "content": "document"}],
                max_tokens=800,
            )
            data = json.loads(response.content[0].text)
            """
        ),
        encoding="utf-8",
    )
    return app


def test_prompt_enforced_json_never_blocks_the_plan(
    service: MigrationService, json_prompt_app: Path
) -> None:
    plan = service.generate_migration_plan(
        json_prompt_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="amazon-bedrock",  # structured_output: false via override
        target_endpoint="bedrock-runtime",
    )
    unsupported = [
        item for item in plan.validation_results if item.code == "unsupported_structured_output"
    ]
    assert unsupported, "the mismatch must still be surfaced"
    assert all(item.level.value == "warning" for item in unsupported)
    assert not any("structured output" in blocker.message for blocker in plan.blockers)


def test_serialization_bypass_is_rejected(service: MigrationService, json_prompt_app: Path) -> None:
    start = service.start_migration_run(
        json_prompt_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        research="skip",
    )
    assert start.paths is not None
    # The exact incident: JSON decodes to JSON, so the runtime prompt is
    # unchanged; the submission must be rejected, not recorded as an adaptation.
    bypass = (
        'sys_prompt: "Extract the fields.\\nExtract the fields.\\nReturn valid \\u004aSON only."\n'
    )
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/extract.yaml",
        bypass,
        "Adapted the output-format instruction.",
        ["Reworded the JSON requirement."],
        submitted_on=AS_OF,
    )
    assert not result.accepted
    assert "decodes to the same runtime values" in result.message
    assert "unchanged=true" in result.message


def test_validation_runs_on_decoded_values(
    service: MigrationService, json_prompt_app: Path
) -> None:
    start = service.start_migration_run(
        json_prompt_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="bedrock",
        target_platform="bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
    )
    assert start.paths is not None
    # A real adaptation that hides the word JSON behind an escape in the raw
    # YAML: decoded-component validation must still see it.
    adapted = (
        'sys_prompt: "Extract every field from every row.\\n'
        'Return valid \\u004aSON only, with all required keys."\n'
    )
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/extract.yaml",
        adapted,
        "Clarified scope per literal-instruction guidance.",
        ["Made the extraction scope explicit."],
        submitted_on=AS_OF,
    )
    assert result.accepted, result.message
    codes = {issue.code for issue in result.validation.issues}
    assert "unsupported_structured_output" in codes  # seen despite the escape
    assert result.validation.valid  # ...but as a warning, not a blocker


def test_unchanged_prompt_and_cosmetic_only_rejection(
    service: MigrationService, json_prompt_app: Path
) -> None:
    start = service.start_migration_run(
        json_prompt_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        research="skip",
    )
    assert start.paths is not None
    run_dir = start.paths.run_dir
    unchanged = service.submit_adapted_prompt(
        run_dir,
        "prompts/extract.yaml",
        "",
        "Prompt already fits the target; JSON enforcement stays prompt-level.",
        [],
        unchanged=True,
        submitted_on=AS_OF,
    )
    assert unchanged.accepted, unchanged.message
    # Whitespace/case-only edits are no-ops in disguise and are rejected, so
    # the incident cannot recur one keystroke away from the escape bypass.
    cosmetic = service.submit_adapted_prompt(
        run_dir,
        "prompts/extract.yaml",
        'sys_prompt: "Extract the fields.\\nExtract the fields.\\nreturn valid json only."\n',
        "Adapted casing.",
        ["Lower-cased the format instruction."],
        submitted_on=AS_OF,
    )
    assert not cosmetic.accepted
    assert "whitespace or letter case" in cosmetic.message
    assert "unchanged=true" in cosmetic.message
    # Reordering top-level keys leaves every runtime value identical: a no-op.
    reordered = service.submit_adapted_prompt(
        run_dir,
        "prompts/extract.yaml",
        'user_prompt: "ignored"\n',
        "placeholder",
        ["x"],
        submitted_on=AS_OF,
    )
    assert not reordered.accepted  # dropped sys_prompt entirely


def test_in_prompt_findings_reach_the_task_guidance(
    service: MigrationService, json_prompt_app: Path
) -> None:
    start = service.start_migration_run(
        json_prompt_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        research="skip",
    )
    assert start.paths is not None
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    task = next(item for item in tasks.prompt_tasks if item.source_path == "prompts/extract.yaml")
    findings = [line for line in task.guidance if line.startswith("[sys_prompt] In-prompt finding")]
    assert any("duplicated_requirements" in line for line in findings)
    assert any("json_only_prompting" in line for line in findings)


def test_worklist_and_finalization_coverage_agree(
    service: MigrationService, json_prompt_app: Path
) -> None:
    start = service.start_migration_run(
        json_prompt_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        research="skip",
    )
    assert start.paths is not None
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    worklist = sorted(item.source_path for item in [*tasks.prompt_tasks, *tasks.file_tasks])
    final = service.finalize_migration_run(start.paths.run_dir)
    assert final.coverage_gaps == worklist  # nothing submitted: gaps == worklist, exactly


@pytest.fixture
def review_round_app(tmp_path: Path) -> Path:
    """Prompt document carrying its own model id plus multiple components."""
    app = tmp_path / "review_round_app"
    (app / "prompts").mkdir(parents=True)
    (app / "prompts" / "agent.yaml").write_text(
        "model: claude-sonnet-4-6\n"
        'sys_prompt: "You are a document assistant. Return valid JSON only."\n'
        "messages:\n"
        '  - {role: user, content: "First question."}\n'
        '  - {role: user, content: "Second question."}\n',
        encoding="utf-8",
    )
    (app / "app.py").write_text(
        textwrap.dedent(
            """\
            import yaml
            from anthropic import Anthropic

            with open("prompts/agent.yaml") as handle:
                prompts = yaml.safe_load(handle)

            client = Anthropic()
            client.messages.create(
                model="claude-sonnet-4-6",
                system=prompts["sys_prompt"],
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=800,
            )
            """
        ),
        encoding="utf-8",
    )
    return app


def _start_review_round(service: MigrationService, app: Path):  # type: ignore[no-untyped-def]
    return service.start_migration_run(
        app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        research="skip",
    )


def test_key_reorder_only_is_rejected_as_a_no_op(
    service: MigrationService, review_round_app: Path
) -> None:
    start = _start_review_round(service, review_round_app)
    assert start.paths is not None
    reordered = (
        'sys_prompt: "You are a document assistant. Return valid JSON only."\n'
        "model: claude-sonnet-4-6\n"
        "messages:\n"
        '  - {role: user, content: "First question."}\n'
        '  - {role: user, content: "Second question."}\n'
    )
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/agent.yaml",
        reordered,
        "Adapted the document layout.",
        ["Moved the system prompt first."],
        submitted_on=AS_OF,
    )
    assert not result.accepted
    assert "decodes to the same runtime values" in result.message


def test_blanking_all_message_content_is_a_structural_drop(
    service: MigrationService, review_round_app: Path
) -> None:
    start = _start_review_round(service, review_round_app)
    assert start.paths is not None
    gutted = (
        "model: claude-sonnet-4-6\n"
        'sys_prompt: "You are a document assistant. Return valid JSON only."\n'
        "messages:\n"
        '  - {role: user, content: "   "}\n'
    )
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/agent.yaml",
        gutted,
        "Trimmed the message history.",
        ["Removed the seed questions."],
        submitted_on=AS_OF,
    )
    assert not result.accepted
    assert "all message content was removed or emptied" in result.message


def test_model_id_swap_in_non_prompt_value_is_sanctioned(
    service: MigrationService, review_round_app: Path
) -> None:
    start = _start_review_round(service, review_round_app)
    assert start.paths is not None
    adapted = (
        "model: claude-sonnet-5\n"
        'sys_prompt: "You are a document assistant. Return valid JSON only, with every '
        'required key present."\n'
        "messages:\n"
        '  - {role: user, content: "First question."}\n'
        '  - {role: user, content: "Second question."}\n'
    )
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/agent.yaml",
        adapted,
        "Adapted the output contract and pointed the document at the target model.",
        ["Strengthened the JSON key requirement; swapped the model id."],
        submitted_on=AS_OF,
    )
    assert result.accepted, result.message
    assert "updated from the source to the target model id" in result.message
    # Any OTHER non-prompt change is still rejected.
    tampered = adapted.replace("model: claude-sonnet-5", "model: gpt-5.6-sol")
    rejected = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/agent.yaml",
        tampered,
        "Adapted.",
        ["x"],
        submitted_on=AS_OF,
    )
    assert not rejected.accepted
    assert "changed the non-prompt value of 'model'" in rejected.message


def test_unchanged_prompt_rejected_when_source_model_is_referenced(
    service: MigrationService, review_round_app: Path
) -> None:
    start = _start_review_round(service, review_round_app)
    assert start.paths is not None
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/agent.yaml",
        "",
        "Looks fine as-is.",
        [],
        unchanged=True,
        submitted_on=AS_OF,
    )
    assert not result.accepted
    assert "reference the source model" in result.message
    no_rationale = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/agent.yaml",
        "",
        "   ",
        [],
        unchanged=True,
        submitted_on=AS_OF,
    )
    assert not no_rationale.accepted
    assert "requires a rationale" in no_rationale.message


def test_unchanged_with_content_is_an_error(
    service: MigrationService, review_round_app: Path
) -> None:
    start = _start_review_round(service, review_round_app)
    assert start.paths is not None
    with pytest.raises(ValueError, match="cannot be combined"):
        service.submit_adapted_prompt(
            start.paths.run_dir,
            "prompts/agent.yaml",
            "sys_prompt: adapted",
            "r",
            ["c"],
            unchanged=True,
            submitted_on=AS_OF,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        service.submit_adapted_file(
            start.paths.run_dir,
            "app.py",
            "content",
            "r",
            ["c"],
            unchanged=True,
            submitted_on=AS_OF,
        )


def test_validation_issues_are_aggregated_not_per_component(
    service: MigrationService, review_round_app: Path
) -> None:
    start = service.start_migration_run(
        review_round_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="bedrock",
        target_platform="bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
    )
    assert start.paths is not None
    adapted = (
        "model: claude-sonnet-4-6\n"
        'sys_prompt: "You are a document assistant. Return valid JSON only, with every '
        'required key."\n'
        "messages:\n"
        '  - {role: user, content: "First question, in valid JSON."}\n'
        '  - {role: user, content: "Second question, in valid JSON."}\n'
    )
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/agent.yaml",
        adapted,
        "Adapted for the bedrock target.",
        ["Strengthened key requirements; swapped the model id."],
        submitted_on=AS_OF,
    )
    assert result.accepted, result.message
    codes = [issue.code for issue in result.validation.issues]
    # Three JSON-mentioning components, ONE aggregated issue - not one each.
    assert codes.count("unsupported_structured_output") == 1


def test_aggregate_context_budget_blocks_oversized_documents(
    service: MigrationService, tmp_path: Path
) -> None:
    app = tmp_path / "big_prompt_app"
    (app / "prompts").mkdir(parents=True)
    half = "word " * 20000  # ~25k tokens per component; gamma allows 32k total
    (app / "prompts" / "big.yaml").write_text(
        f'sys_prompt: "{half}"\nuser_prompt: "{half}"\n', encoding="utf-8"
    )
    (app / "app.py").write_text(
        textwrap.dedent(
            """\
            import yaml
            from anthropic import Anthropic

            with open("prompts/big.yaml") as handle:
                prompts = yaml.safe_load(handle)

            client = Anthropic()
            client.messages.create(
                model="alpha-large-v1",
                system=prompts["sys_prompt"],
                messages=[{"role": "user", "content": prompts["user_prompt"]}],
                max_tokens=500,
            )
            """
        ),
        encoding="utf-8",
    )
    start = service.start_migration_run(
        app,
        "alpha large",
        "gamma cheap",
        as_of=AS_OF,
        research="skip",
    )
    assert start.paths is not None
    adapted = (
        (app / "prompts" / "big.yaml")
        .read_text(encoding="utf-8")
        .replace("word word", "term word", 1)
    )
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/big.yaml",
        adapted,
        "Adapted.",
        ["Reworded the opening."],
        submitted_on=AS_OF,
    )
    assert not result.accepted
    # Each component alone fits gamma's 32k window; the JOINED payload must not.
    assert "context window" in result.message.casefold()


def test_unparseable_original_is_disclosed_not_silent(
    service: MigrationService, tmp_path: Path
) -> None:
    app = tmp_path / "broken_original_app"
    (app / "prompts").mkdir(parents=True)
    (app / "prompts" / "broken.yaml").write_text('sys_prompt: "unterminated\n', encoding="utf-8")
    (app / "app.py").write_text(
        textwrap.dedent(
            """\
            from anthropic import Anthropic

            client = Anthropic()
            client.messages.create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )
            """
        ),
        encoding="utf-8",
    )
    start = _start_review_round(service, app)
    assert start.paths is not None
    result = service.submit_adapted_prompt(
        start.paths.run_dir,
        "prompts/broken.yaml",
        'sys_prompt: "A complete adapted prompt."\n',
        "Repaired and adapted the prompt.",
        ["Fixed the document and adapted the wording."],
        submitted_on=AS_OF,
    )
    assert result.accepted, result.message
    assert "could not be parsed" in result.message


def test_dynamic_request_surfaces_as_unknown_and_native_use_blocks(
    service: MigrationService, tmp_path: Path
) -> None:
    dynamic_app = tmp_path / "dynamic_app"
    dynamic_app.mkdir()
    (dynamic_app / "app.py").write_text(
        textwrap.dedent(
            """\
            from anthropic import Anthropic

            from settings import build_request

            client = Anthropic()
            request = build_request()
            client.messages.create(**request)
            """
        ),
        encoding="utf-8",
    )
    plan = service.generate_migration_plan(
        dynamic_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
    )
    assert any("dynamically" in unknown for unknown in plan.unknowns)

    native_app = tmp_path / "native_app"
    native_app.mkdir()
    (native_app / "app.py").write_text(
        textwrap.dedent(
            """\
            from anthropic import Anthropic

            client = Anthropic()
            client.messages.create(
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
            """
        ),
        encoding="utf-8",
    )
    plan = service.generate_migration_plan(
        native_app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
    )
    assert any(
        "configures structured output" in blocker.message and "unsupported" in blocker.message
        for blocker in plan.blockers
    )
    assert plan.migration_complexity == "blocked"
