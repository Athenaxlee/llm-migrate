from __future__ import annotations

from pathlib import Path

from llm_migrate.core.agent_research import artifact_sha256
from llm_migrate.service import MigrationService
from tests.unit.v11_scenarios import NOW, standard_host, standard_request


def test_v11_research_run_artifacts_match_exact_goldens(
    service: MigrationService, project_root: Path, tmp_path: Path
) -> None:
    """Every stage artifact of the standard fake-agent run is byte-stable."""
    application = project_root / "tests/fixtures/applications/anthropic_app"
    analysis = service.scan_application(application)
    request = standard_request(artifact_sha256(analysis.requirements))
    outcome = service.run_agent_research(
        request,
        standard_host(),
        tmp_path,
        now=NOW,
        shadow_canonical=["claude-sonnet-5"],
    )
    assert outcome.manifest is not None

    golden_root = project_root / "tests" / "golden" / "v11" / request.run_id
    generated_root = tmp_path / request.run_id
    golden_files = sorted(
        path.relative_to(golden_root) for path in golden_root.rglob("*") if path.is_file()
    )
    generated_files = sorted(
        path.relative_to(generated_root) for path in generated_root.rglob("*") if path.is_file()
    )
    assert generated_files == golden_files
    for relative in golden_files:
        assert (generated_root / relative).read_text(encoding="utf-8") == (
            golden_root / relative
        ).read_text(encoding="utf-8"), f"{relative} no longer matches its golden snapshot"
