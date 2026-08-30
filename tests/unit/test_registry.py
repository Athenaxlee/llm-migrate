from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_migrate.core.registry import (
    DuplicateModelError,
    ModelRegistry,
    RegistryValidationError,
)
from llm_migrate.core.resolver import AmbiguousModelError, resolve_model
from llm_migrate.service import MigrationService


def test_valid_registry_loads_in_canonical_order(project_root: Path) -> None:
    registry = ModelRegistry.from_directory(project_root / "registry" / "models")
    names = [profile.identity.canonical_name for profile in registry.all()]
    assert names == sorted(names)
    assert len(names) == 9


def test_malformed_yaml_is_actionable(tmp_path: Path) -> None:
    (tmp_path / "broken.yaml").write_text("identity: [not: valid", encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="broken.yaml"):
        ModelRegistry.from_directory(tmp_path)


def test_duplicate_canonical_name_is_rejected(project_root: Path, tmp_path: Path) -> None:
    source = project_root / "registry" / "models" / "fixtures" / "alpha-large.yaml"
    text = source.read_text(encoding="utf-8")
    (tmp_path / "one.yaml").write_text(text, encoding="utf-8")
    (tmp_path / "two.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(DuplicateModelError, match="fixture-alpha-large-v1"):
        ModelRegistry.from_directory(tmp_path)


def test_alias_and_platform_id_resolution(service: MigrationService) -> None:
    alpha = service.resolve_model("Alpha--Large")
    beta = service.resolve_model("example.beta-balanced-v2")
    assert alpha.identity.canonical_name == "fixture-alpha-large-v1"
    assert beta.identity.canonical_name == "fixture-beta-balanced-v2"


def test_ambiguous_alias_is_never_guessed(project_root: Path) -> None:
    registry = ModelRegistry.from_directory(project_root / "registry" / "models")
    with pytest.raises(AmbiguousModelError, match="fixture-alpha-large-v1"):
        resolve_model(registry, "shared alias")


def test_unknown_fields_are_rejected(project_root: Path, tmp_path: Path) -> None:
    source = project_root / "registry" / "models" / "fixtures" / "alpha-large.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["made_up_field"] = True
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="made_up_field"):
        ModelRegistry.from_directory(tmp_path)
