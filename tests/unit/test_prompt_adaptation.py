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
    knowledge_lines = [line for line in task.guidance if line.startswith("Model difference (")]
    assert knowledge_lines, "migration-knowledge differences must reach the prompt task"
    assert any("(evidence: https://" in line for line in knowledge_lines)
    assert any("(evidence: https://" in line for line in task.guidance if "guidance" not in line)
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
