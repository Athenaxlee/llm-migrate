"""V1.6.0-a: prompt discovery that survives real repo layouts.

Multi-base (leading-segment stripping) resolution, separator- and
case-robust values, chained single-file loaders, unique key-match promotion,
start-time prompt-source confirmation, live-run source addition, consumer
confirmation, dismissals, and the discovery_incomplete run state.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_migrate.cli.main import app as cli_app
from llm_migrate.core.models import PromptDiscoveryCoverage, PromptSourceConfidence
from llm_migrate.core.snapshot import SNAPSHOT_FILENAME, load_snapshot
from llm_migrate.core.workspace import load_adaptation_log, load_run_config
from llm_migrate.mcp.server import add_prompt_sources as mcp_add_prompt_sources
from llm_migrate.mcp.server import confirm_prompt_consumer as mcp_confirm_prompt_consumer
from llm_migrate.scanners.config import PathResolver, detect_case_insensitive, resolve_reference
from llm_migrate.service import MigrationService

AS_OF = date(2026, 9, 23)
SOURCE, TARGET = "us.anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5"
BEDROCK = {
    "source_platform": "amazon-bedrock",
    "target_platform": "amazon-bedrock",
    "target_endpoint": "bedrock-runtime",
}
CLAUDE_FILES = [
    "analysis/prompt_lib/claude_prompt_colmap.yaml",
    "analysis/prompt_lib/claude_prompt_multi.yaml",
    "analysis/prompt_lib/claude_prompt_single.yaml",
]
LLAMA_FILES = [
    "analysis/prompt_lib/llama2_prompt_multi.yaml",
    "analysis/prompt_lib/llama2_prompt_single.yaml",
    "analysis/prompt_lib/llama_prompt_multi.yaml",
    "analysis/prompt_lib/llama_prompt_single.yaml",
]


@pytest.fixture
def field_app(tmp_path: Path, project_root: Path) -> Path:
    """The field pattern: repo-root-relative config paths, scanned from `app/`."""
    repo = tmp_path / "field_pattern_repo"
    shutil.copytree(project_root / "tests/fixtures/applications/field_pattern_repo", repo)
    return repo / "app"


@pytest.fixture
def unresolved_field_app(field_app: Path) -> Path:
    """The field pattern before the fix: config values the resolver cannot map.

    Prompt names are composed at runtime (`f"{name}.yaml"`), so every prompt
    file is an unreferenced low-confidence candidate, and every consumer reads
    keys all seven candidates share — nothing can be promoted.
    """
    config = field_app / "config/model_profiles.yaml"
    text = config.read_text(encoding="utf-8").replace(".yaml\n", "\n")
    config.write_text(text, encoding="utf-8")
    return field_app


@pytest.fixture
def probe_app(tmp_path: Path, project_root: Path) -> Path:
    app = tmp_path / "probe_forms_app"
    shutil.copytree(project_root / "tests/fixtures/applications/probe_forms_app", app)
    return app


def _start(service: MigrationService, app: Path, **extra):  # type: ignore[no-untyped-def]
    return service.start_migration_run(
        app, SOURCE, TARGET, as_of=AS_OF, research="skip", **BEDROCK, **extra
    )


def _run_dir(service: MigrationService, app: Path, **extra) -> Path:  # type: ignore[no-untyped-def]
    start = _start(service, app, **extra)
    assert start.status == "ready" and start.paths is not None, start.next_steps
    return Path(start.paths.run_dir)


# -- multi-base resolution (G1) ---------------------------------------------


def test_field_pattern_resolves_with_zero_configuration(
    service: MigrationService, field_app: Path
) -> None:
    analysis = service.scan_application(field_app)
    by_path = {item.path: item for item in analysis.prompt_sources}
    for path in CLAUDE_FILES:
        source = by_path[path]
        # A stripped resolution ranks medium, never high, and says so.
        assert source.confidence is PromptSourceConfidence.MEDIUM
        assert any("stripping leading 'app'" in line for line in source.provenance)
    discovery = analysis.prompt_discovery
    assert discovery.consumers == 3  # the preprocessing helpers' input= never count
    assert discovery.dynamic_consumers == 3

    plan = service.generate_migration_plan(field_app, SOURCE, TARGET, **BEDROCK)
    prepared = sorted({spec.source_path for spec in plan.prompt_changes})
    assert prepared == CLAUDE_FILES
    assert len(plan.prompt_changes) == 6  # sys_prompt + user_prompt per Claude file
    assert sorted(item.split(": ", 1)[0] for item in plan.out_of_scope) == LLAMA_FILES
    assert not any("llama" in item for item in (u.message for u in plan.unknowns if u.is_open))


def test_stripping_is_bounded_to_the_application_root(tmp_path: Path) -> None:
    known = {"prompts/p.yaml", "deep/x/y/p.yaml"}
    root = tmp_path / "repo" / "svc" / "app"
    root.mkdir(parents=True)
    resolver = PathResolver.for_files(known, root)

    def resolved(value: str) -> str | None:
        found = resolve_reference("config/c.yaml", value, resolver)
        return found.path if found is not None else None

    assert resolved("app/prompts/p.yaml") == "prompts/p.yaml"
    assert resolved("svc/app/prompts/p.yaml") == "prompts/p.yaml"
    assert resolved("repo/svc/app/prompts/p.yaml") == "prompts/p.yaml"
    # The dropped prefix must equal the root's trailing components.
    assert resolved("other/prompts/p.yaml") is None
    assert resolved("svc/prompts/p.yaml") is None
    # Never outside the application: traversal, absolute, and unknown files.
    assert resolved("../prompts/p.yaml") is None
    assert resolved("app/../../prompts/p.yaml") is None
    assert resolved("/repo/svc/app/prompts/p.yaml") is None
    assert resolved("app/secrets/p.yaml") is None
    # At most three stripped segments.
    deeper = PathResolver.for_files(known, tmp_path / "a" / "b" / "c" / "d")
    found = resolve_reference("", "a/b/c/d/prompts/p.yaml", deeper)
    assert found is None


def test_stripping_rule_never_reaches_a_file_outside_the_scan_root(
    service: MigrationService, field_app: Path
) -> None:
    outside = field_app.parent / "analysis/prompt_lib/claude_prompt_multi.yaml"
    outside.parent.mkdir(parents=True)
    outside.write_text("sys_prompt: outside the application\n", encoding="utf-8")
    analysis = service.scan_application(field_app)
    assert all(not source.path.startswith("..") for source in analysis.prompt_sources)
    assert {source.path for source in analysis.prompt_sources} >= set(CLAUDE_FILES)


# -- separators, case, and chained loaders (G15, G16) ------------------------


def test_backslash_values_and_chained_loaders_resolve(
    service: MigrationService, probe_app: Path
) -> None:
    analysis = service.scan_application(probe_app)
    by_path = {item.path: item for item in analysis.prompt_sources}
    # `prompt_lib\summary.yaml` (Windows-authored config value) resolves.
    assert by_path["prompt_lib/summary.yaml"].confidence is PromptSourceConfidence.HIGH
    # open(...).read(), handle.read(), and Path(...).read_text() passed directly.
    for path in ("prompts/system.txt", "prompts/guidelines.txt", "prompts/user.txt"):
        assert by_path[path].confidence is PromptSourceConfidence.HIGH
    consumers = {
        (item.location.path, str(item.value)): item.metadata
        for item in analysis.findings
        if item.detail.startswith("supplies prompt content")
    }
    assert consumers[("app.py", "system")]["resolution"] == "source"
    assert set(consumers[("app.py", "system")]["sources"]) == {
        "prompts/guidelines.txt",
        "prompts/system.txt",
    }
    assert consumers[("app.py", "messages")]["sources"] == ["prompts/user.txt"]
    assert consumers[("summarize.py", "system")]["sources"] == ["prompt_lib/summary.yaml"]


def test_case_folding_applies_only_on_case_insensitive_filesystems(tmp_path: Path) -> None:
    known = {"Prompts/System.yaml"}
    sensitive = PathResolver(frozenset(known), ("app",), case_insensitive=False)
    folding = PathResolver(frozenset(known), ("app",), case_insensitive=True)
    assert resolve_reference("", "prompts/system.yaml", sensitive) is None
    folded = resolve_reference("", "APP\\prompts\\SYSTEM.yaml", folding)
    assert folded is not None
    # The resolved path is always the known-files spelling.
    assert folded.path == "Prompts/System.yaml" and folded.rule == "stripped_prefix"
    (tmp_path / "Probe.txt").write_text("x", encoding="utf-8")
    assert isinstance(detect_case_insensitive(tmp_path, ["Probe.txt"]), bool)
    assert detect_case_insensitive(tmp_path, []) is False


# -- key-match promotion (G2) ------------------------------------------------


def test_unique_key_match_promotes_one_level_with_evidence(
    service: MigrationService, tmp_path: Path
) -> None:
    app = tmp_path / "keyed"
    (app / "lib").mkdir(parents=True)
    (app / "lib/loader.py").write_text(
        "import yaml\n\n\ndef load(path):\n    with open(path) as h:\n"
        "        return yaml.safe_load(h)\n",
        encoding="utf-8",
    )
    (app / "app.py").write_text(
        "import anthropic\n\nfrom lib.loader import load\n\n"
        "client = anthropic.Anthropic()\n"
        "PROMPTS = load(BASE + NAME)\n"
        "client.messages.create(\n"
        '    model="claude-sonnet-4-6",\n'
        "    max_tokens=100,\n"
        '    system=PROMPTS["classifier_prompt"]["sys_prompt"],\n'
        '    messages=[{"role": "user", "content": "x"}],\n'
        ")\n",
        encoding="utf-8",
    )
    (app / "cfg").mkdir()
    (app / "cfg/classifier.yaml").write_text(
        "classifier_prompt: {}\nsys_prompt: You classify.\n", encoding="utf-8"
    )
    (app / "cfg/other.yaml").write_text("sys_prompt: Something else.\n", encoding="utf-8")
    analysis = service.scan_application(app)
    by_path = {item.path: item for item in analysis.prompt_sources}
    promoted = by_path["cfg/classifier.yaml"]
    assert promoted.confidence is PromptSourceConfidence.MEDIUM
    assert any(
        "key-match promotion" in line and "app.py:10" in line for line in promoted.provenance
    )
    # `other.yaml` lacks `classifier_prompt`: no match, no promotion.
    assert by_path["cfg/other.yaml"].confidence is PromptSourceConfidence.LOW


def test_ambiguous_key_match_promotes_nothing(
    service: MigrationService, unresolved_field_app: Path
) -> None:
    analysis = service.scan_application(unresolved_field_app)
    assert {item.confidence for item in analysis.prompt_sources} == {PromptSourceConfidence.LOW}
    assert analysis.prompt_discovery.resolved_sources == 0


# -- start-time confirmation (G3) ---------------------------------------------


def test_start_asks_for_prompt_sources_and_writes_nothing(
    service: MigrationService, unresolved_field_app: Path
) -> None:
    start = _start(service, unresolved_field_app)
    assert start.status == "needs_confirmation"
    assert start.run is None and start.paths is None
    assert sorted(item.path for item in start.prompt_candidates) == sorted(
        [*CLAUDE_FILES, *LLAMA_FILES]
    )
    candidate = start.prompt_candidates[0]
    assert candidate.confidence == "low"
    assert candidate.components == ["sys_prompt", "user_prompt"]
    assert candidate.evidence
    assert any("defer_prompt_candidates" in step for step in start.next_steps)
    assert not (unresolved_field_app / ".llm-migrate").exists()

    retry = _start(service, unresolved_field_app, prompt_sources=CLAUDE_FILES)
    assert retry.status == "ready"
    tasks = service.list_adaptation_tasks(retry.paths.run_dir)
    assert sorted(item.source_path for item in tasks.prompt_tasks) == CLAUDE_FILES


def test_dynamic_chat_apps_without_candidates_start_directly(
    service: MigrationService, probe_app: Path
) -> None:
    run_dir = _run_dir(service, probe_app)
    status = service.get_run_status(run_dir)
    # summarize.py's messages=text is dynamic chat input: partial coverage,
    # no candidate, non-strict -> never forced into discovery_incomplete.
    assert status.prompt_coverage == "partial"
    assert status.state != "discovery_incomplete"
    assert status.dynamic_prompt_consumers == 1


# -- live-run source addition and dismissal (G6, G20) ------------------------


def test_add_prompt_sources_rederives_and_keeps_prior_entries(
    service: MigrationService, unresolved_field_app: Path
) -> None:
    run_dir = _run_dir(service, unresolved_field_app, defer_prompt_candidates=True)
    status = service.get_run_status(run_dir)
    assert status.state == "discovery_incomplete"
    assert set(status.prompt_candidates) == {*CLAUDE_FILES, *LLAMA_FILES}
    assert "add_prompt_sources(run_dir" in status.next_action

    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.prompt_tasks == []
    loader = "analysis/prompt_loader.py"
    if loader in tasks.unaffected_files:
        confirmation = service.confirm_unaffected(
            run_dir, [loader], "Loader has no model coupling."
        )
        assert confirmation.confirmed == 1
    prior = {
        (entry.kind, entry.source_path)
        for entry in load_adaptation_log(run_dir, load_run_config(run_dir).run_id).entries
    }

    update = service.add_prompt_sources(run_dir, CLAUDE_FILES, now=None)
    assert update.accepted, update.problems
    assert update.added_sources == CLAUDE_FILES
    assert update.new_prompt_tasks == CLAUDE_FILES
    config = load_run_config(run_dir)
    assert config.prompt_sources == CLAUDE_FILES
    after = service.list_adaptation_tasks(run_dir)
    assert all(item.status == "pending" for item in after.prompt_tasks)
    assert {
        (e.kind, e.source_path) for e in load_adaptation_log(run_dir, config.run_id).entries
    } == prior
    # The llama siblings of the selected files are now out of scope, not candidates.
    assert after.prompt_candidates == []
    assert service.get_run_status(run_dir).state != "discovery_incomplete"


def test_dismissal_needs_rationale_and_closes_the_state(
    service: MigrationService, unresolved_field_app: Path
) -> None:
    run_dir = _run_dir(service, unresolved_field_app, defer_prompt_candidates=True)
    before = (run_dir / "migration.yaml").read_text(encoding="utf-8")

    no_reason = service.add_prompt_sources(run_dir, [], dismiss=LLAMA_FILES)
    assert not no_reason.accepted and any("rationale" in item for item in no_reason.problems)
    unknown = service.add_prompt_sources(
        run_dir, [], dismiss=["analysis/not_a_candidate.yaml"], rationale="Not a prompt."
    )
    assert not unknown.accepted
    empty = service.add_prompt_sources(run_dir, [])
    assert not empty.accepted
    assert (run_dir / "migration.yaml").read_text(encoding="utf-8") == before

    added = service.add_prompt_sources(
        run_dir,
        CLAUDE_FILES[:1],
        dismiss=[*CLAUDE_FILES[1:], *LLAMA_FILES],
        rationale="Only the column-mapping prompt is live; the rest are retired.",
    )
    assert added.accepted, added.problems
    assert added.prompt_candidates == []
    config = load_run_config(run_dir)
    assert config.prompt_discovery_dismissals[0].rationale.startswith("Only the column")
    status = service.get_run_status(run_dir)
    assert status.state != "discovery_incomplete"
    finalization = service.finalize_migration_run(run_dir)
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "dismissed by the user as not a live prompt" in report
    assert "Only the column-mapping prompt is live" in report


def test_strict_runs_close_discovery_by_confirming_or_dismissing_consumers(
    service: MigrationService, field_app: Path
) -> None:
    run_dir = _run_dir(service, field_app, strict=True)
    status = service.get_run_status(run_dir)
    assert status.state == "discovery_incomplete"
    assert status.prompt_candidates == []
    assert "STRICT MODE" in status.next_action
    # Several files share the consumer's keys: the user must choose, so no
    # source is pre-filled.
    assert "<ask the user: one of" in status.next_action

    tasks = service.list_adaptation_tasks(run_dir)
    by_location = {item.location: item for item in tasks.dynamic_prompt_consumers}
    assert set(by_location) == {
        "analysis/invoke_multimodal.py:22",
        "analysis/invoke_multimodal.py:23",
        "analysis/invoke_single.py:22",
    }
    assert by_location["analysis/invoke_multimodal.py:22"].access_keys == ["sys_prompt"]
    assert by_location["analysis/invoke_multimodal.py:22"].matching_sources == CLAUDE_FILES

    wrong = service.confirm_prompt_consumer(run_dir, "analysis/nowhere.py:1", CLAUDE_FILES[1])
    assert not wrong.accepted
    missing = service.confirm_prompt_consumer(
        run_dir, "analysis/invoke_multimodal.py:22", "analysis/missing.yaml"
    )
    assert not missing.accepted

    multi = "analysis/prompt_lib/claude_prompt_multi.yaml"
    for location in ("analysis/invoke_multimodal.py:22", "analysis/invoke_multimodal.py:23"):
        update = service.confirm_prompt_consumer(run_dir, location, multi)
        assert update.accepted, update.problems
    assert update.dynamic_prompt_consumers == ["analysis/invoke_single.py:22"]
    assert service.get_run_status(run_dir).state == "discovery_incomplete"

    single = service.confirm_prompt_consumer(
        run_dir, "analysis/invoke_single.py:22", "analysis/prompt_lib/claude_prompt_single.yaml"
    )
    assert single.accepted and single.prompt_coverage == "resolved"
    assert single.dynamic_prompt_consumers == []
    status = service.get_run_status(run_dir)
    assert status.state != "discovery_incomplete"
    assert status.prompt_coverage == "resolved"
    analysis = service._scan_for_run(load_run_config(run_dir))
    by_path = {item.path: item for item in analysis.prompt_sources}
    assert by_path[multi].confidence is PromptSourceConfidence.HIGH
    assert any("user-confirmed prompt consumer" in line for line in by_path[multi].provenance)
    # The strict coverage blocker is gone once every consumer is closed.
    assert not any(
        "incomplete_prompt_coverage" in line
        for line in service.list_adaptation_tasks(run_dir).blockers
    )


def test_strict_consumer_dismissal_resolves_coverage(
    service: MigrationService, probe_app: Path
) -> None:
    run_dir = _run_dir(service, probe_app, strict=True)
    status = service.get_run_status(run_dir)
    assert status.state == "discovery_incomplete"
    update = service.add_prompt_sources(
        run_dir,
        [],
        dismiss=["summarize.py:20"],
        rationale="messages carries the end user's document text at runtime.",
    )
    assert update.accepted, update.problems
    assert update.prompt_coverage == "resolved"
    analysis = service._scan_for_run(load_run_config(run_dir))
    assert analysis.prompt_discovery.dismissed_consumers == 1
    assert analysis.prompt_discovery.coverage is PromptDiscoveryCoverage.RESOLVED
    assert service.get_run_status(run_dir).state != "discovery_incomplete"


def test_stale_consumer_decisions_are_reported_not_applied(
    service: MigrationService, probe_app: Path
) -> None:
    analysis = service.scan_application(
        probe_app,
        consumer_confirmations={"app.py:999": "prompts/system.txt"},
        dismissed_consumers=["summarize.py:1"],
    )
    assert sum("matches no current prompt consumer" in item for item in analysis.warnings) == 2
    assert analysis.prompt_discovery.dismissed_consumers == 0


# -- snapshot invalidation, CLI, and MCP --------------------------------------


def test_pre_v160a_snapshots_are_rederived(service: MigrationService, probe_app: Path) -> None:
    run_dir = _run_dir(service, probe_app)
    service.list_adaptation_tasks(run_dir)
    path = run_dir / SNAPSHOT_FILENAME
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["schema_version"] = "1"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_snapshot(run_dir, load_run_config(run_dir).run_id) is None
    assert service.get_run_status(run_dir).snapshot_reused is False


def test_cli_and_mcp_expose_live_run_discovery(
    service: MigrationService,
    unresolved_field_app: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = str(project_root / "registry")
    runner = CliRunner()
    base = ["run", "start", str(unresolved_field_app), "--from", SOURCE, "--to", TARGET]
    base += ["--from-platform", "amazon-bedrock", "--to-platform", "amazon-bedrock"]
    base += ["--to-endpoint", "bedrock-runtime", "--as-of", AS_OF.isoformat()]
    base += ["--skip-research", "--registry", registry]
    asked = runner.invoke(cli_app, base)
    assert asked.exit_code == 0, asked.output
    payload = json.loads(asked.output)
    assert payload["status"] == "needs_confirmation" and payload["prompt_candidates"]

    started = runner.invoke(cli_app, [*base, "--defer-prompt-candidates"])
    assert started.exit_code == 0, started.output
    run_dir = json.loads(started.output)["paths"]["run_dir"]
    monkeypatch.setenv("LLM_MIGRATE_REGISTRY", registry)
    result = mcp_add_prompt_sources(
        run_dir, [], dismiss=LLAMA_FILES, rationale="Llama profiles are retired."
    )
    assert result["accepted"], result["problems"]
    added = runner.invoke(
        cli_app,
        ["run", "add-prompt-source", run_dir, CLAUDE_FILES[0], "--registry", registry],
    )
    assert added.exit_code == 0, added.output
    assert json.loads(added.output)["added_sources"] == [CLAUDE_FILES[0]]
    refused = runner.invoke(
        cli_app,
        [
            "run",
            "confirm-consumer",
            run_dir,
            "nowhere.py:1",
            CLAUDE_FILES[0],
            "--registry",
            registry,
        ],
    )
    assert refused.exit_code == 1
    confirmed = mcp_confirm_prompt_consumer(
        run_dir, "analysis/invoke_multimodal.py:22", CLAUDE_FILES[0]
    )
    assert confirmed["accepted"], confirmed["problems"]
