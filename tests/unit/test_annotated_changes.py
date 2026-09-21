"""Hunk-anchored annotated changes: diff reconciliation, evidence, reporting."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_migrate.core.annotations import (
    AnnotatedChange,
    ChangeEvidence,
    annotation_problems,
    decoded_view,
)
from llm_migrate.core.models import PromptSourceFormat
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import (
    dispose_all,
    edit_change,
    insert_change,
    restructure_change,
)

AS_OF = date(2026, 9, 16)

ORIGINAL = "You are a helper.\nThink step by step.\nReturn JSON only.\n"
ADAPTED = "You are a helper.\nReturn JSON only, with every required key.\n"


def _delete(anchor: str, why: str = "Removed for the target model.") -> AnnotatedChange:
    return AnnotatedChange(
        operation="delete",
        original_anchor=anchor,
        why=why,
        evidence=[ChangeEvidence(kind="mechanical")],
    )


def test_every_hunk_needs_a_covering_annotation() -> None:
    problems, _ = annotation_problems(ORIGINAL, ADAPTED, [])
    assert any("undocumented" in problem for problem in problems)

    covering = [
        _delete("Think step by step."),
        edit_change(
            "Return JSON only.",
            "Return JSON only, with every required key.",
            why="Strengthened the output contract.",
        ),
    ]
    problems, warnings = annotation_problems(ORIGINAL, ADAPTED, covering)
    assert problems == []
    assert warnings == []


def test_phantom_and_unresolvable_annotations_are_rejected() -> None:
    covering = [
        _delete("Think step by step."),
        edit_change(
            "Return JSON only.",
            "Return JSON only, with every required key.",
        ),
    ]
    phantom = edit_change("You are a helper.", "You are a helper.", why="No real change.")
    problems, _ = annotation_problems(ORIGINAL, ADAPTED, [*covering, phantom])
    assert any("does not correspond to any actual change" in problem for problem in problems)

    unresolvable = [
        _delete("This text never existed."),
        covering[1],
    ]
    problems, _ = annotation_problems(ORIGINAL, ADAPTED, unresolvable)
    assert any("does not occur in the original content" in problem for problem in problems)


def test_operation_shapes_are_enforced() -> None:
    missing_adapted = AnnotatedChange(
        operation="edit",
        original_anchor="Think step by step.",
        why="Edit without an adapted anchor.",
        evidence=[ChangeEvidence(kind="mechanical")],
    )
    problems, _ = annotation_problems(ORIGINAL, ADAPTED, [missing_adapted])
    assert any("requires both" in problem for problem in problems)

    bad_insert = AnnotatedChange(
        operation="insert",
        original_anchor="Return JSON only.",
        adapted_anchor="with every required key",
        why="Insert carrying an original anchor.",
        evidence=[ChangeEvidence(kind="mechanical")],
    )
    problems, _ = annotation_problems(ORIGINAL, ADAPTED, [bad_insert])
    assert any("cannot carry original_anchor" in problem for problem in problems)


def test_evidence_is_required_and_checked_against_known_urls() -> None:
    no_evidence = AnnotatedChange(
        operation="delete",
        original_anchor="Think step by step.",
        why="Removed the chain-of-thought request.",
    )
    problems, _ = annotation_problems(ORIGINAL, ADAPTED, [no_evidence])
    assert any("carries no evidence entry" in problem for problem in problems)

    bare_claim = AnnotatedChange(
        operation="delete",
        original_anchor="Think step by step.",
        why="Removed the chain-of-thought request.",
        evidence=[ChangeEvidence(kind="model_guidance")],
    )
    problems, _ = annotation_problems(ORIGINAL, ADAPTED, [bare_claim])
    assert any("needs a url or a reference" in problem for problem in problems)

    unknown_url = AnnotatedChange(
        operation="delete",
        original_anchor="Think step by step.",
        why="Removed the chain-of-thought request.",
        evidence=[ChangeEvidence(kind="model_guidance", url="https://example.com/blog")],
    )
    covering = [
        unknown_url,
        edit_change("Return JSON only.", "Return JSON only, with every required key."),
    ]
    problems, warnings = annotation_problems(
        ORIGINAL, ADAPTED, covering, known_evidence_urls={"https://docs.example/known"}
    )
    assert problems == []
    assert any("not among the run's known evidence sources" in warning for warning in warnings)


def test_global_restructure_covers_every_hunk() -> None:
    problems, _ = annotation_problems(
        ORIGINAL,
        "A completely rewritten prompt.\n",
        [restructure_change(why="Rewrote the whole prompt for the target model.")],
    )
    assert problems == []


def test_decoded_view_is_serialization_independent() -> None:
    original = 'sys_prompt: "Hello.\\nReturn valid \\u004aSON."\n'
    assert decoded_view(original, PromptSourceFormat.YAML) == "Hello.\nReturn valid JSON."
    assert decoded_view("sys_prompt: [unclosed", PromptSourceFormat.YAML) is None
    assert decoded_view("plain text", None) == "plain text"


@pytest.fixture
def simple_app(tmp_path: Path) -> Path:
    app = tmp_path / "annotated_app"
    (app / "prompts").mkdir(parents=True)
    (app / "prompts" / "system.txt").write_text(ORIGINAL, encoding="utf-8")
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


def test_prompt_submission_reconciles_annotations_end_to_end(
    service: MigrationService, simple_app: Path
) -> None:
    start = _start(service, simple_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    dispositions = dispose_all(service, run_dir, "prompts/system.txt")

    undocumented = service.submit_adapted_prompt(
        run_dir,
        "prompts/system.txt",
        ADAPTED,
        "Adapted for the target model.",
        ["Removed the chain-of-thought request; strengthened the output contract."],
        submitted_on=AS_OF,
        guidance_dispositions=dispositions,
    )
    assert not undocumented.accepted
    assert "undocumented" in undocumented.message

    known_url = next(
        item.text.split("(evidence: ")[1].rstrip(")")
        for item in service.list_adaptation_tasks(run_dir).shared_prompt_guidance
        if "(evidence: " in item.text
    )
    annotations = [
        AnnotatedChange(
            operation="delete",
            original_anchor="Think step by step.",
            why="The target model reasons natively; explicit step instructions constrain it.",
            evidence=[ChangeEvidence(kind="model_guidance", url=known_url)],
        ),
        edit_change(
            "Return JSON only.",
            "Return JSON only, with every required key.",
            why="Strengthened the output contract per target guidance.",
        ),
    ]
    accepted = service.submit_adapted_prompt(
        run_dir,
        "prompts/system.txt",
        ADAPTED,
        "Adapted for the target model.",
        [],
        submitted_on=AS_OF,
        guidance_dispositions=dispositions,
        annotated_changes=annotations,
    )
    assert accepted.accepted, accepted.message
    # A plan-known evidence URL raises no unknown-source warning.
    assert not any("known evidence sources" in warning for warning in accepted.message.split(";"))

    changes = yaml.safe_load(
        (Path(run_dir) / "output" / "changes.yaml").read_text(encoding="utf-8")
    )
    assert changes["schema_version"] == "2"
    entry = next(item for item in changes["entries"] if item["source_path"] == "prompts/system.txt")
    recorded = entry["annotated_changes"]
    assert len(recorded) == 2
    assert all(item["id"].startswith("c:") for item in recorded)
    assert recorded[0]["evidence"][0]["url"] == known_url

    final = service.finalize_migration_run(run_dir)
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert "The target model reasons natively" in report
    assert f"evidence: model_guidance {known_url}" in report
    assert "- before: 'Think step by step.'" in report
    assert "- after: 'Return JSON only, with every required key.'" in report
    # The deliverable itself stays clean: exactly the adapted content.
    output = Path(run_dir) / "output" / "prompts" / "prompts" / "system.txt"
    assert output.read_text(encoding="utf-8") == ADAPTED


def test_unchanged_submission_cannot_carry_annotations(
    service: MigrationService, simple_app: Path
) -> None:
    start = _start(service, simple_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    result = service.submit_adapted_prompt(
        run_dir,
        "prompts/system.txt",
        "",
        "Reviewed; already fits the target.",
        [],
        unchanged=True,
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(
            service, run_dir, "prompts/system.txt", disposition="not_applicable"
        ),
        annotated_changes=[insert_change("anything")],
    )
    assert not result.accepted
    assert "cannot carry annotated_changes" in result.message


def test_file_submission_requires_dispositions_and_annotations(
    service: MigrationService, simple_app: Path
) -> None:
    start = _start(service, simple_app)
    assert start.paths is not None
    run_dir = start.paths.run_dir
    original = (simple_app / "app.py").read_text(encoding="utf-8")
    adapted = original.replace("claude-sonnet-4-6", "claude-sonnet-5")

    bare = service.submit_adapted_file(
        run_dir,
        "app.py",
        adapted,
        "Points the invocation at the target model.",
        ["Swapped the model id."],
        submitted_on=AS_OF,
    )
    assert not bare.accepted
    assert "every guidance item must be disposed" in bare.message
    assert "undocumented" in bare.message

    accepted = service.submit_adapted_file(
        run_dir,
        "app.py",
        adapted,
        "Points the invocation at the target model.",
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
    assert accepted.accepted, accepted.message
    changes = yaml.safe_load(
        (Path(run_dir) / "output" / "changes.yaml").read_text(encoding="utf-8")
    )
    entry = next(item for item in changes["entries"] if item["source_path"] == "app.py")
    assert entry["guidance_dispositions"]
    assert entry["guidance_dispositions"][0]["guidance"], "resolved text must be recorded"
    assert entry["annotated_changes"][0]["id"].startswith("c:")
