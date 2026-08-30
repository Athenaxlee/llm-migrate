from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_migrate.core.knowledge import ResearchResult
from llm_migrate.core.models import ModelProfile

REQUIRED_FILES = {
    "candidate-model.yaml",
    "evidence.yaml",
    "proposal.md",
    "proposal.yaml",
}


def _yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), f"{path} must contain a YAML mapping"
    return raw


def _canonical_profiles(project_root: Path) -> dict[str, ModelProfile]:
    profiles: dict[str, ModelProfile] = {}
    for path in (project_root / "registry" / "models").rglob("*.yaml"):
        profile = ModelProfile.model_validate(_yaml_mapping(path))
        profiles[profile.identity.canonical_name] = profile
    return profiles


def test_checked_in_proposal_bundles_cover_every_provider_profile(
    project_root: Path,
) -> None:
    proposal_root = project_root / ".registry-proposals"
    bundles = sorted(path for path in proposal_root.iterdir() if path.is_dir())
    assert bundles, "at least one checked-in proposal bundle is required"
    canonical = _canonical_profiles(project_root)
    provider_models = {name: profile for name, profile in canonical.items() if not profile.fixture}
    covered_models: set[str] = set()

    for bundle in bundles:
        assert {path.name for path in bundle.iterdir()} == REQUIRED_FILES

        candidate = ModelProfile.model_validate(_yaml_mapping(bundle / "candidate-model.yaml"))
        evidence = ResearchResult.model_validate(_yaml_mapping(bundle / "evidence.yaml"))
        proposal = _yaml_mapping(bundle / "proposal.yaml")
        report = (bundle / "proposal.md").read_text(encoding="utf-8")

        model_name = candidate.identity.canonical_name
        assert model_name not in covered_models, f"duplicate proposal bundle for {model_name}"
        covered_models.add(model_name)
        assert proposal["model"] == model_name
        assert evidence.canonical_model_candidate == model_name
        assert report.startswith(f"# {candidate.identity.display_name} Registry Proposal")

        source_ids = {source.id for source in evidence.sources}
        for item in [*proposal["changes"], *proposal["conflicts"]]:
            assert set(item["supporting_sources"]) <= source_ids

        assert all(isinstance(field, str) for field in proposal["unknown_fields"])
        review = proposal["review"]
        status = str(review["status"])
        assert model_name in provider_models
        if status.startswith("approved"):
            assert review["decisions"]
            assert candidate == provider_models[model_name]
        else:
            assert status == "pending_review"
            assert review["required_actions"]

    assert covered_models == set(provider_models), (
        "every non-fixture provider profile needs exactly one checked-in review bundle"
    )
