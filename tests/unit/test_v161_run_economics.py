"""v1.6.1: optional research for stale-only facts, per-call economics, evidence refresh."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_migrate.cli.main import app as cli_app
from llm_migrate.core import snapshot as snapshot_module
from llm_migrate.core.refresh import (
    REFRESH_EVIDENCE_FILENAME,
    RefreshError,
    SourceRefreshStatus,
    load_baselines,
    recorded_sources,
    refresh_evidence,
    visible_text_digest,
)
from llm_migrate.scanners.python import scannable_files
from llm_migrate.service import MigrationService

# Well past every freshness window of the refreshed (2026-09-24) Anthropic profiles.
AS_OF = date(2026, 11, 15)
BEDROCK = {
    "source_platform": "amazon-bedrock",
    "target_platform": "amazon-bedrock",
    "target_endpoint": "bedrock-runtime",
}


@pytest.fixture
def bedrock_app(tmp_path: Path, project_root: Path) -> Path:
    app = tmp_path / "bedrock_app"
    shutil.copytree(project_root / "tests/fixtures/applications/bedrock_app", app)
    return app


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


# --- research by default, hands-free ---------------------------------------


def test_stale_facts_keep_research_recommended_and_hands_free(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        as_of=AS_OF,
        research="auto",
        **BEDROCK,
    )
    assert start.status == "ready" and start.paths is not None and start.research is not None
    assert start.research.level == "recommended"
    assert any("past their freshness window" in reason for reason in start.research.reasons)
    assert any("last checked 2026-09-24" in reason for reason in start.research.reasons)
    research_steps = [step for step in start.next_steps if "research" in step.casefold()]
    assert research_steps[0].startswith("Research is recommended (")
    assert "run it now by default" in research_steps[0]
    assert "NON-INTERACTIVE background agent" in research_steps[1]
    assert "never a visible browser" in research_steps[1]
    assert not any(step.startswith("Ask the user whether") for step in start.next_steps)
    run_dir = Path(start.paths.run_dir)
    assert (run_dir / "request.yaml").is_file()

    status = service.get_run_status(run_dir)
    assert status.state == "research_pending"
    assert status.research_scopes_pending == ["source", "target"]
    assert "no user action is needed" in status.next_action
    assert "non-interactive background agents" in status.next_action
    assert "ask the user" not in status.next_action.casefold()


def test_research_prompts_demand_non_interactive_tools(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        as_of=AS_OF,
        research="auto",
        **BEDROCK,
    )
    assert start.paths is not None
    pack = service.get_research_prompts(start.paths.run_dir)
    for scope in pack.scopes:
        for prompt in (scope.researcher_prompt, scope.reviewer_prompt):
            assert "never open or drive a visible interactive browser" in prompt
            assert "never wait for user input" in prompt
    guidance = " ".join(pack.orchestration_guidance)
    assert "NON-INTERACTIVE background subagents" in guidance
    assert "in parallel (up to 2 at a time)" in guidance
    assert "do not pause for confirmation" in guidance


def test_refreshed_target_contributes_no_staleness_reason(
    service: MigrationService, bedrock_app: Path
) -> None:
    """The 2026-09-24 evidence refresh: the target is fresh the day after.

    The source keeps two held categories (capabilities, prompting_guidance:
    the AWS card contradicts the recorded maximum output and the migration
    guide URL became an index page), so research is still recommended, but
    nothing about the target is stale.
    """
    start = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        as_of=date(2026, 9, 25),
        research="auto",
        **BEDROCK,
    )
    assert start.research is not None and start.research.level == "recommended"
    assert start.research.scopes == ["source"]
    assert not any("target model" in reason for reason in start.research.reasons)
    (reason,) = start.research.reasons
    assert "capabilities, prompting_guidance" in reason


# --- per-call economics ----------------------------------------------------


def test_scannable_files_prunes_ignored_directories_without_descending(tmp_path: Path) -> None:
    app = tmp_path / "app"
    (app / "pkg").mkdir(parents=True)
    (app / "pkg" / "main.py").write_text("x = 1\n")
    (app / "prompts").mkdir()
    (app / "prompts" / "system.yaml").write_text("system: hi\n")
    for ignored in (
        ".venv/lib/site-packages/dep",
        "node_modules/mod",
        ".llm-migrate/runs/r/output/prompts",
    ):
        directory = app / ignored
        directory.mkdir(parents=True)
        (directory / "noise.py").write_text("y = 2\n")
        (directory / "noise.yaml").write_text("a: b\n")
    outside = tmp_path / "outside.py"
    outside.write_text("z = 3\n")
    # Windows without symlink privilege: the pruning assertions still run.
    with contextlib.suppress(OSError):
        os.symlink(outside, app / "pkg" / "linked.py")

    python_files, candidates = scannable_files(app)
    assert [p.relative_to(app).as_posix() for p in python_files] == ["pkg/main.py"]
    assert [p.relative_to(app).as_posix() for p in candidates] == ["prompts/system.yaml"]
    assert scannable_files(app / "pkg" / "main.py") == ([app / "pkg" / "main.py"], [])


def test_file_digest_cache_is_keyed_by_stat_signature(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("model: a\n")
    first = snapshot_module._file_digest(target)
    assert first == hashlib.sha256(b"model: a\n").hexdigest()
    assert snapshot_module._file_digest(target) == first  # cache hit, same signature
    target.write_text("model: b\n")  # same size; force a distinct mtime explicitly
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1_000_000_000))
    second = snapshot_module._file_digest(target)
    assert second == hashlib.sha256(b"model: b\n").hexdigest()
    assert second != first
    assert snapshot_module._file_digest(tmp_path / "missing.yaml") == "unreadable"


def test_mcp_service_is_cached_per_registry_digest(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llm_migrate.mcp.server as server

    registry = tmp_path / "registry"
    shutil.copytree(project_root / "registry", registry)
    monkeypatch.setenv("LLM_MIGRATE_REGISTRY", str(registry))
    first = server._service()
    assert server._service() is first
    profile = registry / "models" / "anthropic" / "claude-sonnet-5.yaml"
    profile.write_text(profile.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")
    second = server._service()
    assert second is not first
    assert server._service() is second


# --- maintainer evidence refresh -------------------------------------------


def _fake_fetcher(pages: dict[str, bytes]):  # type: ignore[no-untyped-def]
    def fetch(url: str, timeout: float) -> bytes:
        if url not in pages:
            raise OSError(f"offline: {url}")
        return pages[url]

    return fetch


def test_visible_text_digest_ignores_markup_noise() -> None:
    a = (
        b"<html><head><script>nonce=1</script><style>.x{}</style></head>"
        b"<body><p>$2 / MTok</p></body></html>"
    )
    b = b"<html><head><script>nonce=2</script></head><body>\n<p>  $2 / MTok </p>\n</body></html>"
    assert visible_text_digest(a) == visible_text_digest(b)
    assert visible_text_digest(b"<p>$3 / MTok</p>") != visible_text_digest(a)


def test_refresh_evidence_records_baselines_then_proposes_only_for_unchanged_pages(
    service: MigrationService, project_root: Path, tmp_path: Path
) -> None:
    proposals = tmp_path / "proposals"
    # Start from bundles without baselines: the first refresh must record them.
    shutil.copytree(
        project_root / ".registry-proposals",
        proposals,
        ignore=shutil.ignore_patterns(REFRESH_EVIDENCE_FILENAME),
    )
    registry_before = _tree_digest(project_root / "registry")
    profile = service.registry.get("claude-sonnet-5")
    urls = sorted({str(source.url) for source in recorded_sources(profile) if source.url})
    assert urls, "the profile records source URLs"
    pages = {
        url: f"<html><body>Claude Sonnet 5 page for {url}</body></html>".encode() for url in urls
    }
    today = date(2026, 9, 24)

    first = refresh_evidence(
        service.registry,
        proposals,
        models=["claude-sonnet-5"],
        fetcher=_fake_fetcher(pages),
        today=today,
    )
    (report,) = first.models
    assert {o.status for o in report.sources} == {SourceRefreshStatus.BASELINE_RECORDED}
    assert report.proposed_freshness == []
    assert report.baselines_recorded == len(urls)
    assert any(
        "first refresh" in item and "verify the facts by hand" in item
        for item in report.held_categories
    )
    baseline_file = proposals / "claude-sonnet-5" / REFRESH_EVIDENCE_FILENAME
    assert baseline_file.is_file()
    assert {b.url for b in load_baselines(proposals, "claude-sonnet-5").baselines} == set(urls)

    # Second refresh: one page changed, one unreachable, the rest unchanged.
    deprecations = next(url for url in urls if "model-deprecations" in url)
    overview = next(url for url in urls if url.endswith("/models/overview"))
    changed_pages = dict(pages)
    changed_pages[deprecations] = b"<html><body>claude-sonnet-5 retired!</body></html>"
    del changed_pages[overview]
    second = refresh_evidence(
        service.registry,
        proposals,
        models=["claude-sonnet-5"],
        fetcher=_fake_fetcher(changed_pages),
        today=date(2026, 9, 25),
    )
    (report,) = second.models
    by_url = {o.url: o for o in report.sources}
    assert by_url[deprecations].status is SourceRefreshStatus.CHANGED
    assert by_url[overview].status is SourceRefreshStatus.UNREACHABLE
    proposed = {item.category.value: item for item in report.proposed_freshness}
    # lifecycle is supported only by the changed deprecations page (and the
    # AWS card, unchanged) -> held because a supporting page changed.
    assert "lifecycle" not in proposed
    assert any(
        item.startswith("lifecycle:") and "re-verify" in item for item in report.held_categories
    )
    # capabilities are supported by unchanged pages -> proposed with today's date.
    assert proposed["capabilities"].proposed_checked_at == date(2026, 9, 25)
    assert (
        proposed["capabilities"].current_checked_at == profile.freshness["capabilities"].checked_at
    )
    assert all(url != deprecations for url in proposed["capabilities"].supporting_urls)
    # A changed page is never rebaselined silently.
    baselines = {b.url: b for b in load_baselines(proposals, "claude-sonnet-5").baselines}
    assert baselines[deprecations].content_sha256 == visible_text_digest(pages[deprecations])[0]

    third = refresh_evidence(
        service.registry,
        proposals,
        models=["claude-sonnet-5"],
        fetcher=_fake_fetcher(changed_pages),
        today=date(2026, 9, 26),
        rebaseline=True,
    )
    assert third.models[0].rebaselined == 1
    baselines = {b.url: b for b in load_baselines(proposals, "claude-sonnet-5").baselines}
    assert (
        baselines[deprecations].content_sha256
        == visible_text_digest(b"<html><body>claude-sonnet-5 retired!</body></html>")[0]
    )
    assert _tree_digest(project_root / "registry") == registry_before, "canonical files untouched"


def test_refresh_holds_categories_whose_page_no_longer_names_the_model(
    service: MigrationService, project_root: Path, tmp_path: Path
) -> None:
    proposals = tmp_path / "proposals"
    shutil.copytree(
        project_root / ".registry-proposals",
        proposals,
        ignore=shutil.ignore_patterns(REFRESH_EVIDENCE_FILENAME),
    )
    profile = service.registry.get("claude-sonnet-4-6")
    urls = sorted({str(source.url) for source in recorded_sources(profile) if source.url})
    overview = next(url for url in urls if url.endswith("/models/overview"))
    pages = {url: f"<html><body>Claude Sonnet 4.6 {url}</body></html>".encode() for url in urls}
    pages[overview] = b"<html><body>Current models: Claude Sonnet 5 only</body></html>"
    kwargs = {"models": ["claude-sonnet-4-6"], "fetcher": _fake_fetcher(pages)}
    refresh_evidence(service.registry, proposals, today=date(2026, 9, 24), **kwargs)
    second = refresh_evidence(service.registry, proposals, today=date(2026, 9, 25), **kwargs)
    (report,) = second.models
    by_url = {o.url: o for o in report.sources}
    assert by_url[overview].status is SourceRefreshStatus.NO_MENTION
    assert "no longer names this model" in (by_url[overview].error or "")
    proposed = {item.category.value: item for item in report.proposed_freshness}
    # pricing is supported by the pricing page (names the model) -> proposed
    # without the silent overview page among its supporting URLs.
    assert overview not in proposed["pricing"].supporting_urls
    assert overview not in proposed.get("capabilities", proposed["pricing"]).supporting_urls


def test_refresh_honors_holds_recorded_in_the_bundle_review(
    service: MigrationService, project_root: Path, tmp_path: Path
) -> None:
    """The Sonnet 4.6 bundle holds capabilities (64K vs 128K) and prompt guidance."""
    proposals = tmp_path / "proposals"
    shutil.copytree(
        project_root / ".registry-proposals",
        proposals,
        ignore=shutil.ignore_patterns(REFRESH_EVIDENCE_FILENAME),
    )
    profile = service.registry.get("claude-sonnet-4-6")
    urls = sorted({str(source.url) for source in recorded_sources(profile) if source.url})
    pages = {url: f"<html><body>Claude Sonnet 4.6 {url}</body></html>".encode() for url in urls}
    kwargs = {"models": ["claude-sonnet-4-6"], "fetcher": _fake_fetcher(pages)}
    refresh_evidence(service.registry, proposals, today=date(2026, 9, 24), **kwargs)
    second = refresh_evidence(service.registry, proposals, today=date(2026, 9, 25), **kwargs)
    (report,) = second.models
    proposed = {item.category.value for item in report.proposed_freshness}
    assert "pricing" in proposed and "lifecycle" in proposed
    assert "capabilities" not in proposed and "prompting_guidance" not in proposed
    held = " ".join(report.held_categories)
    assert "capabilities: on hold by the bundle's review decision" in held
    assert "64K" in held


def test_refresh_evidence_refuses_unknown_models_and_missing_bundles(
    service: MigrationService, tmp_path: Path
) -> None:
    with pytest.raises(RefreshError, match="unknown canonical model"):
        refresh_evidence(service.registry, tmp_path, models=["nope"], fetcher=_fake_fetcher({}))
    with pytest.raises(RefreshError, match="no proposal bundle directory"):
        refresh_evidence(
            service.registry,
            tmp_path,
            models=["claude-sonnet-5"],
            fetcher=lambda url, timeout: b"page",
        )


def test_refresh_evidence_cli_reports_and_writes(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llm_migrate.core.refresh as refresh_module

    monkeypatch.setattr(refresh_module, "fetch_https_source", lambda url, timeout: b"<p>same</p>")
    proposals = tmp_path / "proposals"
    shutil.copytree(
        project_root / ".registry-proposals",
        proposals,
        ignore=shutil.ignore_patterns(REFRESH_EVIDENCE_FILENAME),
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        [
            "registry",
            "refresh-evidence",
            "--model",
            "claude-sonnet-4-6",
            "--proposals",
            str(proposals),
            "--as-of",
            "2026-09-24",
            "--output",
            str(tmp_path / "out"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["models"][0]["model"] == "claude-sonnet-4-6"
    assert payload["models"][0]["baselines_recorded"] > 0
    assert (tmp_path / "out" / "refresh-2026-09-24.md").is_file()
    assert (proposals / "claude-sonnet-4-6" / REFRESH_EVIDENCE_FILENAME).is_file()
    saved = yaml.safe_load(
        (proposals / "claude-sonnet-4-6" / REFRESH_EVIDENCE_FILENAME).read_text()
    )
    assert saved["model"] == "claude-sonnet-4-6"
