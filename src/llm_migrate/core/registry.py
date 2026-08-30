"""Local YAML registry loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import ValidationError

from llm_migrate.core.knowledge import (
    MigrationKnowledge,
    ObservationRecord,
    RegistryValidationReport,
)
from llm_migrate.core.models import ModelProfile

RegistryDocument = TypeVar("RegistryDocument", ModelProfile, MigrationKnowledge, ObservationRecord)


class RegistryError(ValueError):
    """Base error for actionable registry failures."""


class RegistryValidationError(RegistryError):
    """A registry file is invalid."""


class DuplicateModelError(RegistryError):
    """Two files declare the same canonical model name."""


class ModelRegistry:
    """An immutable collection of validated model profiles."""

    def __init__(
        self,
        profiles: list[ModelProfile],
        migrations: list[MigrationKnowledge] | None = None,
        observations: list[ObservationRecord] | None = None,
    ) -> None:
        models: dict[str, ModelProfile] = {}
        for profile in profiles:
            name = profile.identity.canonical_name
            if name in models:
                raise DuplicateModelError(f"duplicate canonical model name: {name}")
            models[name] = profile
        self._models = models
        self._migrations = sorted(migrations or [], key=lambda item: item.id)
        self._observations = sorted(observations or [], key=lambda item: item.id)
        migration_ids = [item.id for item in self._migrations]
        if len(migration_ids) != len(set(migration_ids)):
            raise RegistryValidationError("duplicate migration registry ID")
        observation_ids = [item.id for item in self._observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise RegistryValidationError("duplicate observation registry ID")
        for migration in self._migrations:
            if migration.source_model not in self._models:
                raise RegistryValidationError(
                    f"migration {migration.id!r} references unknown source model "
                    f"{migration.source_model!r}"
                )
            if migration.target_model not in self._models:
                raise RegistryValidationError(
                    f"migration {migration.id!r} references unknown target model "
                    f"{migration.target_model!r}"
                )
        for observation in self._observations:
            if observation.model and observation.model not in self._models:
                raise RegistryValidationError(
                    f"observation {observation.id!r} references unknown model {observation.model!r}"
                )

    @classmethod
    def from_directory(cls, directory: Path | str) -> ModelRegistry:
        root = Path(directory)
        if not root.is_dir():
            raise RegistryError(f"registry model directory does not exist: {root}")
        profiles: list[ModelProfile] = []
        paths = sorted((*root.rglob("*.yaml"), *root.rglob("*.yml")))
        for path in paths:
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise RegistryValidationError("document must be a YAML mapping")
                profiles.append(ModelProfile.model_validate(raw))
            except (OSError, yaml.YAMLError, ValidationError, RegistryValidationError) as exc:
                raise RegistryValidationError(f"invalid registry file {path}: {exc}") from exc
        if not profiles:
            raise RegistryError(f"no model YAML files found under: {root}")
        return cls(profiles)

    @classmethod
    def from_root(cls, directory: Path | str) -> ModelRegistry:
        root = Path(directory)
        models = cls.from_directory(root / "models").all()
        migrations = _load_documents(root / "migrations", MigrationKnowledge)
        observations = _load_documents(root / "observations", ObservationRecord)
        return cls(models, migrations, observations)

    def all(self) -> list[ModelProfile]:
        return [self._models[name] for name in sorted(self._models)]

    def get(self, canonical_name: str) -> ModelProfile:
        try:
            return self._models[canonical_name]
        except KeyError as exc:
            raise RegistryError(f"unknown canonical model: {canonical_name}") from exc

    def migrations(self) -> list[MigrationKnowledge]:
        return list(self._migrations)

    def migrations_for(self, source_model: str, target_model: str) -> list[MigrationKnowledge]:
        return [
            item
            for item in self._migrations
            if item.source_model == source_model and item.target_model == target_model
        ]

    def observations(self) -> list[ObservationRecord]:
        return list(self._observations)

    def validation_report(self) -> RegistryValidationReport:
        aliases: dict[str, list[str]] = {}
        for profile in self.all():
            for alias in profile.identity.aliases:
                aliases.setdefault(alias.casefold(), []).append(profile.identity.canonical_name)
        warnings = [
            f"ambiguous alias {alias!r}: {', '.join(sorted(names))}"
            for alias, names in sorted(aliases.items())
            if len(names) > 1
        ]
        return RegistryValidationReport(
            valid=True,
            model_count=len(self._models),
            migration_count=len(self._migrations),
            observation_count=len(self._observations),
            warnings=warnings,
        )


def _load_documents(
    directory: Path,
    schema: type[RegistryDocument],
) -> list[RegistryDocument]:
    if not directory.is_dir():
        return []
    documents: list[RegistryDocument] = []
    paths = sorted((*directory.rglob("*.yaml"), *directory.rglob("*.yml")))
    for path in paths:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise RegistryValidationError("document must be a YAML mapping")
            documents.append(schema.model_validate(raw))
        except (OSError, yaml.YAMLError, ValidationError, RegistryValidationError) as exc:
            raise RegistryValidationError(f"invalid registry file {path}: {exc}") from exc
    return documents
