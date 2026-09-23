"""V1.5.2: consumer precision, scoping, answered unknowns, evidence, validation honesty."""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml

from llm_migrate.core.evaluation import evaluation_artifact_as_yaml
from llm_migrate.core.evaluation_models import (
    EvaluationCase,
    EvaluationRunConfig,
    EvaluationStatus,
    ProviderCallResult,
)
from llm_migrate.core.models import MigrationPlan
from llm_migrate.core.snapshot import load_snapshot
from llm_migrate.core.workspace import (
    ValidationDisposition,
    load_adaptation_log,
    save_validation_disposition,
)
from llm_migrate.mcp.server import start_migration as mcp_start_migration
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import edit_change

AS_OF = date(2026, 9, 22)
SOURCE, TARGET = "claude-sonnet-4-6", "claude-sonnet-5"

APP_SOURCE = """\
import anthropic
from pathlib import Path

from helpers import preprocess

SYSTEM_PROMPT = Path("prompts/system.txt").read_text()
client = anthropic.Anthropic()
cleaned = preprocess(input="raw user text")
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": "hi"}],
)
"""
HELPERS = """\
def preprocess(input=None, prompt=None):
    return (input or "").strip()


def parse(text):
    return preprocess(input=text, prompt="ignored")
"""


def _write(root: Path, files: dict[str, str]) -> Path:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def plain_app(tmp_path: Path) -> Path:
    """Resolved prompt coverage, no tools, no structured output, no reasoning."""
    return _write(
        tmp_path / "plain_app",
        {
            "app.py": APP_SOURCE,
            "helpers.py": HELPERS,
            "prompts/system.txt": "You are a concise assistant. Answer briefly.\n",
        },
    )


def _start(service: MigrationService, app: Path, **extra):  # type: ignore[no-untyped-def]
    start = service.start_migration_run(
        app,
        SOURCE,
        TARGET,
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        research="skip",
        **extra,
    )
    assert start.status == "ready" and start.paths is not None
    return start


def _plan(service: MigrationService, app: Path, **extra) -> MigrationPlan:  # type: ignore[no-untyped-def]
    return service.generate_migration_plan(
        app,
        SOURCE,
        TARGET,
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        **extra,
    )


# -- consumer precision (G4) -------------------------------------------------


def test_prompt_keywords_on_helpers_are_not_consumers(
    service: MigrationService, plain_app: Path
) -> None:
    analysis = service.scan_application(plain_app)
    consumer_files = {
        finding.location.path
        for finding in analysis.findings
        if finding.kind.value == "prompt" and finding.detail.startswith("supplies prompt content")
    }
    assert consumer_files == {"app.py"}, "input=/prompt= on helpers must not be consumers"
    assert analysis.prompt_discovery.consumers == 2  # system= and messages= only


# -- out-of-scope prompt files (G5) -----------------------------------------


def test_unreferenced_sibling_candidate_is_out_of_scope(
    service: MigrationService, plain_app: Path
) -> None:
    _write(
        plain_app,
        {
            "prompt_lib/claude_prompt.yaml": "sys_prompt: You classify rooms.\n",
            "prompt_lib/llama_prompt.yaml": "sys_prompt: You classify rooms (llama).\n",
        },
    )
    plan = _plan(service, plain_app, prompt_sources=["prompt_lib/claude_prompt.yaml"])
    assert any(item.startswith("prompt_lib/llama_prompt.yaml:") for item in plan.out_of_scope)
    assert not any("llama_prompt.yaml" in item for item in plan.unknowns)
    assert {spec.source_path for spec in plan.prompt_changes} >= {"prompt_lib/claude_prompt.yaml"}
    assert "llama_prompt.yaml" not in {spec.source_path for spec in plan.prompt_changes}


def test_candidate_without_selected_sibling_stays_unknown(
    service: MigrationService, plain_app: Path
) -> None:
    _write(plain_app, {"prompt_lib/claude_prompt.yaml": "sys_prompt: You classify rooms.\n"})
    plan = _plan(service, plain_app)
    assert any(
        "prompt_lib/claude_prompt.yaml contains prompt-like keys" in u for u in plan.unknowns
    )
    assert plan.out_of_scope == []


def test_prompt_referenced_only_by_a_foreign_profile_is_out_of_scope(
    service: MigrationService, plain_app: Path
) -> None:
    _write(
        plain_app,
        {
            "config/profiles.yaml": (
                "claude:\n  model_id: claude-sonnet-4-6\n"
                "  prompts: {multi: config/claude.yaml}\n"
                "llama:\n  model_id: meta.llama3-70b-instruct\n"
                "  prompts: {multi: config/llama.yaml}\n"
            ),
            "config/claude.yaml": "sys_prompt: Claude prompt.\n",
            "config/llama.yaml": "sys_prompt: Llama prompt.\n",
        },
    )
    plan = _plan(service, plain_app)
    prepared = {spec.source_path for spec in plan.prompt_changes}
    assert "config/claude.yaml" in prepared
    assert "config/llama.yaml" not in prepared
    assert any(
        item.startswith("config/llama.yaml:") and "meta.llama3-70b-instruct" in item
        for item in plan.out_of_scope
    )


# -- answered unknowns and same-provider noise (G9, G10) ---------------------


def test_scan_answered_unknowns_are_not_emitted(service: MigrationService, plain_app: Path) -> None:
    plan = _plan(service, plain_app)
    assert not any("parallel_tool_use" in item for item in plan.unknowns)
    report = service.migration_report(plan)
    assert "| Unknown | Action |" in report or "## Unresolved unknowns" in report


def test_same_provider_move_drops_the_xml_formatting_finding(
    service: MigrationService, plain_app: Path
) -> None:
    (plain_app / "prompts/system.txt").write_text(
        "<rules>Answer briefly.</rules>\n", encoding="utf-8"
    )
    plan = _plan(service, plain_app)
    categories = {
        finding.category
        for spec in plan.prompt_changes
        for finding in spec.source_prompt_analysis.findings
    }
    assert "section_structure" in categories
    assert "provider_specific_formatting" not in categories


# -- evidence widening and review visibility (G7) ----------------------------


def test_registry_urls_are_known_and_review_shows_evidence_state(
    service: MigrationService, plain_app: Path
) -> None:
    start = _start(service, plain_app)
    run_dir = Path(start.paths.run_dir)
    tasks = service.list_adaptation_tasks(run_dir)
    snapshot = load_snapshot(run_dir, start.run.run_id)
    assert snapshot is not None and snapshot.registry_evidence_urls
    registry_url = snapshot.registry_evidence_urls[0]
    original = (plain_app / "prompts/system.txt").read_text(encoding="utf-8")
    adapted = original.replace("Answer briefly.", "Answer in at most two sentences.")
    task = tasks.prompt_tasks[0]
    dispositions = [
        {"guidance_id": item.id, "disposition": "applied"}
        for item in [*task.guidance, *tasks.shared_prompt_guidance]
        if not item.not_applicable_reason
    ]
    result = service.submit_adapted_prompt(
        run_dir,
        "prompts/system.txt",
        adapted,
        "Tighter verbosity for the target.",
        [],
        submitted_on=AS_OF,
        guidance_dispositions=dispositions,
        annotated_changes=[
            edit_change(
                "Answer briefly.",
                "Answer in at most two sentences.",
                kind="model_guidance",
                url=registry_url,
            ).model_copy(
                update={
                    "evidence": [
                        *edit_change("a", "b", kind="model_guidance", url=registry_url).evidence,
                        *edit_change(
                            "a", "b", kind="model_guidance", url="https://example.com/made-up"
                        ).evidence,
                    ]
                }
            )
        ],
    )
    assert result.accepted, result.message
    review = service.get_change_review(run_dir)
    statuses = review.files[0].changes[0].evidence_status
    assert statuses[0] == "registry-recorded"
    assert statuses[1].startswith("UNKNOWN")


# -- validation honesty (G8, G17) --------------------------------------------


def _finalized(service: MigrationService, plain_app: Path) -> Path:
    start = _start(service, plain_app)
    run_dir = Path(start.paths.run_dir)
    service.finalize_migration_run(run_dir)
    return run_dir


def test_byok_evaluation_requires_a_bound_evaluation_run(
    service: MigrationService, plain_app: Path
) -> None:
    run_dir = _finalized(service, plain_app)
    with pytest.raises(ValueError, match="evaluation_run_path"):
        service.record_validation_disposition(run_dir, "byok_evaluation", decided_on=AS_OF)

    manifest = yaml.safe_load((run_dir / "output/migration-manifest.yaml").read_text())
    plan = MigrationPlan.model_validate(manifest["migration"])
    suite = service.generate_eval_suite(plan, [EvaluationCase(id="c1", input="hi")])
    configs = [
        EvaluationRunConfig(
            provider="anthropic",
            platform="anthropic-api",
            model=endpoint.model,
            model_id=endpoint.model_id,
            credential_env="ANTHROPIC_API_KEY",
        )
        for endpoint in (plan.source, plan.target)
    ]

    class Executor:
        def execute(self, config, case):  # type: ignore[no-untyped-def]
            return ProviderCallResult(
                status=EvaluationStatus.PASSED,
                output="ok",
                latency_ms=10,
                input_tokens=1,
                output_tokens=1,
            )

    times = iter([datetime(2026, 9, 22, tzinfo=UTC), datetime(2026, 9, 22, 0, 0, 1, tzinfo=UTC)])
    run = service.run_migration_eval(
        suite, configs[0], configs[1], executor=Executor(), clock=times.__next__
    )
    artifact = run_dir / "eval-run.yaml"
    artifact.write_text(evaluation_artifact_as_yaml(run), encoding="utf-8")
    disposition = service.record_validation_disposition(
        run_dir, "byok_evaluation", evaluation_run_path="eval-run.yaml", decided_on=AS_OF
    )
    assert disposition.bound_manifest_sha256 == suite.migration_manifest_sha256
    final = service.finalize_migration_run(run_dir)
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert "NOT VALIDATED" not in report

    # A later plan change supersedes the evaluation's binding.
    stale = run.model_copy(
        update={"suite": run.suite.model_copy(update={"migration_manifest_sha256": "0" * 64})}
    )
    artifact.write_text(evaluation_artifact_as_yaml(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="different manifest"):
        service.record_validation_disposition(
            run_dir, "byok_evaluation", evaluation_run_path="eval-run.yaml", decided_on=AS_OF
        )


def test_legacy_disposition_without_evidence_is_not_validation(
    service: MigrationService, plain_app: Path
) -> None:
    run_dir = _finalized(service, plain_app)
    config = yaml.safe_load((run_dir / "migration.yaml").read_text())
    save_validation_disposition(
        run_dir,
        ValidationDisposition(run_id=config["run_id"], method="generated_tests", decided_on=AS_OF),
    )
    final = service.finalize_migration_run(run_dir)
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert "NOT VALIDATED" in report and "outcome not recorded" in report


# -- whole-application sweep (G15) -------------------------------------------


def test_untraced_file_naming_the_source_model_is_a_finding_and_closable(
    service: MigrationService, plain_app: Path
) -> None:
    _write(plain_app, {"notes/ops.txt": "Rates assume claude-sonnet-4-6 on-demand.\n"})
    start = _start(service, plain_app, strict=True)
    run_dir = Path(start.paths.run_dir)
    final = service.finalize_migration_run(run_dir)
    uncovered = [
        item for item in final.consistency_findings if "[source_reference_uncovered]" in item
    ]
    assert any("notes/ops.txt" in item for item in uncovered)
    assert any("source_reference_uncovered" in item for item in final.strict_violations)
    assert "## Action required" in Path(final.report_path).read_text(encoding="utf-8")

    refused = service.confirm_unaffected(run_dir, ["notes/ops.txt"], "Historical note.")
    assert refused.confirmed == 0
    closed = service.confirm_unaffected(
        run_dir,
        ["notes/ops.txt"],
        "The user keeps this historical rate note unchanged.",
        acknowledge_source_references=True,
    )
    assert closed.confirmed == 1, closed.results[0].message
    log = load_adaptation_log(run_dir, start.run.run_id)
    assert any(entry.source_reference_acknowledged for entry in log.entries)
    again = service.finalize_migration_run(run_dir)
    assert not any("notes/ops.txt" in item for item in again.consistency_findings)
    report = Path(again.report_path).read_text(encoding="utf-8")
    assert "ACKNOWLEDGED SOURCE REFERENCE" in report


def test_worklist_tasks_cannot_be_acknowledged_away(
    service: MigrationService, plain_app: Path
) -> None:
    start = _start(service, plain_app)
    result = service.confirm_unaffected(
        start.paths.run_dir, ["app.py"], "no", acknowledge_source_references=True
    )
    assert result.confirmed == 0


# -- guidance applicability (G18) --------------------------------------------


def test_inapplicable_registry_guidance_is_pre_disposed(
    service: MigrationService, plain_app: Path
) -> None:
    start = _start(service, plain_app)
    run_dir = Path(start.paths.run_dir)
    tasks = service.list_adaptation_tasks(run_dir)
    items = [*tasks.prompt_tasks[0].guidance, *tasks.shared_prompt_guidance]
    pre = [item for item in items if item.not_applicable_reason]
    assert any("prefill" in item.text for item in pre)
    assert any("adaptive thinking" in item.text for item in pre)
    owed = [
        {"guidance_id": item.id, "disposition": "applied"}
        for item in items
        if not item.not_applicable_reason
    ]
    result = service.submit_adapted_prompt(
        run_dir,
        "prompts/system.txt",
        "",
        "No change needed for this short prompt.",
        [],
        submitted_on=AS_OF,
        unchanged=True,
        guidance_dispositions=[{**item, "disposition": "not_applicable"} for item in owed],
    )
    assert result.accepted, result.message
    entry = load_adaptation_log(run_dir, start.run.run_id).entries[0]
    recorded = [item for item in entry.guidance_dispositions if item.pre_disposed]
    assert recorded and all(item.disposition == "not_applicable" for item in recorded)


def test_partial_coverage_never_pre_disposes(service: MigrationService, plain_app: Path) -> None:
    (plain_app / "dynamic.py").write_text(
        "import anthropic\n"
        "client = anthropic.Anthropic()\n"
        "def ask(history):\n"
        "    return client.messages.create(model='claude-sonnet-4-6', max_tokens=9,"
        " system=history[0], messages=history[1:])\n",
        encoding="utf-8",
    )
    start = _start(service, plain_app)
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    assert tasks.prompt_coverage != "resolved"
    items = [*tasks.prompt_tasks[0].guidance, *tasks.shared_prompt_guidance]
    assert not any(item.not_applicable_reason for item in items)


# -- status honesty and two-pass legibility (G20, G14) -----------------------


def test_ready_status_states_the_two_pass_validation_flow(
    service: MigrationService, plain_app: Path
) -> None:
    start = _start(service, plain_app)
    run_dir = Path(start.paths.run_dir)
    tasks = service.list_adaptation_tasks(run_dir)
    prompt_owed = [
        {"guidance_id": item.id, "disposition": "not_applicable"}
        for item in [*tasks.prompt_tasks[0].guidance, *tasks.shared_prompt_guidance]
        if not item.not_applicable_reason
    ]
    original = (plain_app / "app.py").read_text(encoding="utf-8")
    outcome = service.submit_adaptations(
        run_dir,
        [
            {
                "kind": "prompt",
                "source_path": "prompts/system.txt",
                "content": "",
                "rationale": "Short prompt; no change needed.",
                "unchanged": True,
                "guidance_dispositions": prompt_owed,
            },
            {
                "kind": "file",
                "source_path": "app.py",
                "content": original.replace(SOURCE, TARGET),
                "rationale": "Target model id.",
                "default_disposition": "applied",
                "annotated_changes": [edit_change(SOURCE, TARGET).model_dump()],
            },
        ],
        submitted_on=AS_OF,
    )
    assert outcome.accepted == 2, [item.message for item in outcome.results]
    review = service.get_change_review(run_dir)
    service.record_change_decisions(
        run_dir,
        [
            {"source_path": item.source_path, "change_id": change.change.id, "decision": "accepted"}
            for item in review.files
            for change in item.changes
        ],
        decided_on=AS_OF,
    )
    status = service.get_run_status(run_dir)
    assert status.state == "ready_to_finalize"
    assert "two-pass" in status.next_action
    assert "record_validation_disposition" in status.next_action


def test_status_warns_on_incomplete_prompt_coverage(
    service: MigrationService, plain_app: Path
) -> None:
    (plain_app / "prompts/system.txt").unlink()
    _write(plain_app, {"prompt_lib/claude_prompt.yaml": "sys_prompt: You classify rooms.\n"})
    # v1.6.0-a: zero resolved sources with a candidate asks at start; deferring
    # lands the run in discovery_incomplete, which names the candidate.
    assert service.start_migration_run(
        plain_app, SOURCE, TARGET, as_of=AS_OF, research="skip"
    ).prompt_candidates
    start = _start(service, plain_app, defer_prompt_candidates=True)
    status = service.get_run_status(start.paths.run_dir)
    assert status.state == "discovery_incomplete"
    assert "prompt_lib/claude_prompt.yaml" in status.next_action


def test_mcp_start_returns_paths_relative_to_the_run_root(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = tmp_path / "bedrock_app"
    shutil.copytree(project_root / "tests/fixtures/applications/bedrock_app", app)
    monkeypatch.setenv("LLM_MIGRATE_REGISTRY", str(project_root / "registry"))
    result = mcp_start_migration(
        str(app),
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF.isoformat(),
        research="skip",
    )
    paths = result["paths"]
    assert Path(paths["run_dir"]).is_absolute()
    assert paths["report_path"] == "output/migration-report.md"
    assert not any(
        value.startswith(paths["run_dir"]) for key, value in paths.items() if key != "run_dir"
    )
