"""Regenerate the deterministic V1.1 golden research-run workspace."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_migrate.core.agent_research import artifact_sha256  # noqa: E402
from llm_migrate.service import MigrationService  # noqa: E402
from tests.unit.v11_scenarios import NOW, standard_host, standard_request  # noqa: E402


def build_golden_workspace(workspace: Path) -> None:
    service = MigrationService.from_directory(ROOT / "registry")
    application = ROOT / "tests/fixtures/applications/anthropic_app"
    analysis = service.scan_application(application)
    request = standard_request(artifact_sha256(analysis.requirements))
    outcome = service.run_agent_research(
        request,
        standard_host(),
        workspace,
        now=NOW,
        shadow_canonical=["claude-sonnet-5"],
    )
    if outcome.manifest is None:
        raise SystemExit("golden workflow run did not produce a session manifest")


def main() -> None:
    output = ROOT / "tests" / "golden" / "v11"
    if output.exists():
        shutil.rmtree(output)
    build_golden_workspace(output)
    print(f"Regenerated V1.1 golden run workspace under {output}")


if __name__ == "__main__":
    main()
