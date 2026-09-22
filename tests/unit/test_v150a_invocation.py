"""V1.5.0-a invocation identity: selectors, run seeding, enforcement, blockers."""

from __future__ import annotations

import shutil
import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llm_migrate.core.invocation_identity import (
    derive_invocation,
    platform_spelling_set,
    references_bare_alone,
    selector_for_spelling,
)
from llm_migrate.core.models import (
    PlatformAvailability,
    PlatformInvocation,
)
from llm_migrate.core.workspace import load_run_config
from llm_migrate.service import MigrationService

AS_OF = date(2026, 9, 22)


def _platform(**kwargs) -> PlatformAvailability:  # type: ignore[no-untyped-def]
    base = {"platform": "amazon-bedrock", "model_id": "anthropic.claude-sonnet-5"}
    return PlatformAvailability.model_validate({**base, **kwargs})


_SELECTORS = [
    {"name": "us", "model_id": "us.anthropic.claude-sonnet-5"},
    {"name": "eu", "model_id": "eu.anthropic.claude-sonnet-5"},
]


def test_selector_id_must_differ_from_the_bare_platform_id() -> None:
    with pytest.raises(ValidationError, match="must differ from the platform's"):
        _platform(
            invocation={
                "bare_on_demand_supported": False,
                "selectors": [{"name": "self", "model_id": "anthropic.claude-sonnet-5"}],
            }
        )


def test_invocation_schema_fails_closed() -> None:
    with pytest.raises(ValidationError, match="selector names must be unique"):
        PlatformInvocation.model_validate(
            {"selectors": [_SELECTORS[0], {**_SELECTORS[1], "name": "us"}]}
        )
    with pytest.raises(ValidationError, match="selector model ids must be unique"):
        PlatformInvocation.model_validate(
            {"selectors": [_SELECTORS[0], {**_SELECTORS[0], "name": "eu"}]}
        )
    with pytest.raises(ValidationError, match="must declare at least one invocation selector"):
        PlatformInvocation.model_validate({"bare_on_demand_supported": False})


def test_derive_invocation_rules() -> None:
    # No reviewed facts: fail-open on the bare id, with a warning.
    bare_only = derive_invocation(_platform(), "anthropic.claude-sonnet-5")
    assert bare_only.choice is not None
    assert bare_only.choice.invocation_model_id == "anthropic.claude-sonnet-5"
    assert any("No reviewed invocation facts" in w for w in bare_only.choice.warnings)

    facts = _platform(invocation={"bare_on_demand_supported": False, "selectors": _SELECTORS})
    # The user's spelling seeds the selector (full id or prefix form).
    spelled = derive_invocation(facts, "eu.anthropic.claude-sonnet-5")
    assert spelled.choice is not None and spelled.choice.selector == "eu"
    assert spelled.choice.requires_selector
    # Bare spelling with several selectors: the user must pick one.
    ambiguous = derive_invocation(facts, "anthropic.claude-sonnet-5")
    assert ambiguous.choice is None
    assert [s.name for s in ambiguous.selection_required] == ["us", "eu"]
    # A sole selector is chosen with a note.
    single = _platform(invocation={"bare_on_demand_supported": False, "selectors": _SELECTORS[:1]})
    sole = derive_invocation(single, None)
    assert sole.choice is not None and sole.choice.selector == "us"
    assert any("only invocation selector" in n for n in sole.choice.notes)
    # bare_on_demand_supported true: the bare id is the invocation id.
    supported = derive_invocation(_platform(invocation={"bare_on_demand_supported": True}), None)
    assert supported.choice is not None
    assert supported.choice.invocation_model_id == "anthropic.claude-sonnet-5"
    assert not supported.choice.warnings

    assert selector_for_spelling(facts, "us.anthropic.claude-sonnet-5") is not None
    assert selector_for_spelling(facts, "anthropic.claude-sonnet-5") is None
    # A selector prefix in front of some OTHER identifier never seeds a selector.
    assert selector_for_spelling(facts, "us.some-other-model") is None
    assert platform_spelling_set(facts) == [
        "anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
        "eu.anthropic.claude-sonnet-5",
    ]


def test_references_bare_alone_ignores_selector_qualified_forms() -> None:
    bare = "anthropic.claude-sonnet-5"
    qualified = ["us.anthropic.claude-sonnet-5", "eu.anthropic.claude-sonnet-5"]
    assert not references_bare_alone('modelId="us.anthropic.claude-sonnet-5"', bare, qualified)
    assert references_bare_alone('modelId="anthropic.claude-sonnet-5"', bare, qualified)
    assert references_bare_alone(
        'a="us.anthropic.claude-sonnet-5"\nb="anthropic.claude-sonnet-5"', bare, qualified
    )
    assert not references_bare_alone("nothing here", bare, qualified)


def test_selector_ids_resolve_exactly_with_their_platform(service: MigrationService) -> None:
    match = service.match_model("eu.anthropic.claude-sonnet-5", "amazon-bedrock", "bedrock-runtime")
    assert match.status.value == "resolved"
    assert match.canonical_name == "claude-sonnet-5"
    assert match.model_id == "anthropic.claude-sonnet-5"
    assert any("invocation selector" in note for note in match.notes)


@pytest.fixture
def bedrock_app(tmp_path: Path, project_root: Path) -> Path:
    app = tmp_path / "bedrock_app"
    shutil.copytree(project_root / "tests/fixtures/applications/bedrock_app", app)
    return app


def test_start_requires_selector_choice_for_bare_target(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
    )
    assert start.status == "needs_confirmation"
    assert start.paths is None  # nothing was written
    candidates = start.target_match.candidates
    assert {item.matched_identifier for item in candidates} == {
        "us.anthropic.claude-sonnet-5",
        "eu.anthropic.claude-sonnet-5",
        "au.anthropic.claude-sonnet-5",
        "global.anthropic.claude-sonnet-5",
    }
    assert any("not invocable on demand" in note for note in start.target_match.notes)


def test_start_seeds_selector_from_spelling_and_worklist_enforces_it(
    service: MigrationService, bedrock_app: Path
) -> None:
    start = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "eu.anthropic.claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
    )
    assert start.status == "ready" and start.run is not None and start.paths is not None
    config = start.run
    assert config.target_invocation_selector == "eu"
    assert config.target_invocation_model_id == "eu.anthropic.claude-sonnet-5"
    assert config.target_invocation_requires_selector
    assert "eu.anthropic.claude-sonnet-5" in config.target_model_spellings
    assert "us.anthropic.claude-sonnet-4-6" in config.source_model_spellings

    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    assert tasks.target_invocation_model_id == "eu.anthropic.claude-sonnet-5"
    assert tasks.target_invocation_selector == "eu"
    assert not tasks.blockers  # the selector is recorded; no invocation blocker
    file_task = next(task for task in tasks.file_tasks if task.source_path == "app.py")
    assert any("eu.anthropic.claude-sonnet-5" in item.text for item in file_task.required_changes)

    final = service.finalize_migration_run(start.paths.run_dir)
    report = Path(final.report_path).read_text(encoding="utf-8")
    assert "Invocation identity" in report
    assert "eu.anthropic.claude-sonnet-5" in report


def test_legacy_run_without_selector_gets_blocker_with_selector_options(
    service: MigrationService, bedrock_app: Path
) -> None:
    """A pre-v1.5 run config rides the blocker flow to a recorded selector."""
    start = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
    )
    assert start.status == "ready" and start.paths is not None
    run_dir = Path(start.paths.run_dir)
    # Strip the invocation fields, as a run started before v1.5.0-a would be.
    raw = yaml.safe_load((run_dir / "migration.yaml").read_text(encoding="utf-8"))
    for field in (
        "source_invocation_model_id",
        "target_invocation_model_id",
        "source_invocation_selector",
        "target_invocation_selector",
        "target_invocation_requires_selector",
        "source_model_spellings",
        "target_model_spellings",
    ):
        raw.pop(field, None)
    (run_dir / "migration.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    assert load_run_config(run_dir).target_invocation_selector is None

    tasks = service.list_adaptation_tasks(run_dir)
    assert any("invocation_selector_required" in blocker for blocker in tasks.blockers)

    resolutions = service.get_blocker_resolutions(run_dir)
    resolution = next(
        item
        for item in resolutions.resolutions
        if item.blocker.code == "invocation_selector_required"
    )
    option_ids = [option.id for option in resolution.options]
    assert "selector-us" in option_ids and option_ids[-1] == "accept"
    us_option = next(option for option in resolution.options if option.id == "selector-us")
    assert us_option.evidence_urls, "selector options must carry evidence"

    decided = service.record_blocker_decision(
        run_dir, resolution.blocker.id, "selector-us", decided_on=AS_OF
    )
    assert decided.accepted, decided.problems
    assert decided.run_config_updated
    config = load_run_config(run_dir)
    assert config.target_invocation_selector == "us"
    assert config.target_invocation_model_id == "us.anthropic.claude-sonnet-5"
    refreshed = service.list_adaptation_tasks(run_dir)
    assert not any("invocation_selector_required" in blocker for blocker in refreshed.blockers)

    final = service.finalize_migration_run(run_dir)
    assert not final.unresolved_blockers
    assert not final.stale_decisions


def test_invocation_blocker_offers_every_selector_uncapped(
    service: MigrationService, bedrock_app: Path
) -> None:
    """A 5-selector target (claude-sonnet-4-6) must not lose options to the cap."""
    from llm_migrate.core.blockers import _invocation_options
    from llm_migrate.core.models import BlockerCategory, migration_blocker

    start = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-4-6",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
    )
    assert start.status == "ready" and start.run is not None
    blocker = migration_blocker(
        code="invocation_selector_required",
        category=BlockerCategory.INVOCATION,
        message="test",
    )
    options = _invocation_options(service.registry, start.run, blocker, "run")
    assert [option.id for option in options] == [
        "selector-us",
        "selector-eu",
        "selector-au",
        "selector-jp",
        "selector-global",
    ]
    assert all(option.evidence_urls for option in options)


def test_prompt_submission_rejects_forbidden_bare_target_reference(
    service: MigrationService, bedrock_app: Path
) -> None:
    """The prompt path enforces the same bare-id invocation gate as files."""
    (bedrock_app / "prompts").mkdir()
    (bedrock_app / "prompts" / "profile.yaml").write_text(NATIVE_PROMPT_DOC, encoding="utf-8")
    start = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
        prompt_sources=["prompts/profile.yaml"],
    )
    assert start.status == "ready" and start.paths is not None
    run_dir = start.paths.run_dir
    from tests.unit.adaptation_helpers import dispose_all, edit_change

    bare_swap = NATIVE_PROMPT_DOC.replace(
        "us.anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-5"
    ).replace("careful assistant", "careful assistant for Claude Sonnet 5")
    rejected = service.submit_adapted_prompt(
        run_dir,
        "prompts/profile.yaml",
        bare_swap,
        "Adapted for Sonnet 5 on Bedrock.",
        ["Updated the model id and refreshed the prompt."],
        guidance_dispositions=dispose_all(service, run_dir, "prompts/profile.yaml"),
        annotated_changes=[
            edit_change(
                "careful assistant",
                "careful assistant for Claude Sonnet 5",
                why="Names the target model explicitly.",
            )
        ],
        submitted_on=AS_OF,
    )
    assert not rejected.accepted
    assert "not invocable on demand" in rejected.message


NATIVE_PROMPT_DOC = textwrap.dedent(
    """\
    model: us.anthropic.claude-sonnet-4-6
    sys_prompt: |
      You are a careful assistant.
    """
)


def test_sanctioned_swap_accepts_selector_qualified_target(
    service: MigrationService, bedrock_app: Path
) -> None:
    """A structured non-prompt model value may swap to the selector-qualified id."""
    (bedrock_app / "prompts").mkdir()
    (bedrock_app / "prompts" / "profile.yaml").write_text(NATIVE_PROMPT_DOC, encoding="utf-8")
    start = service.start_migration_run(
        bedrock_app,
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
        prompt_sources=["prompts/profile.yaml"],
    )
    assert start.status == "ready" and start.paths is not None
    run_dir = start.paths.run_dir
    adapted = NATIVE_PROMPT_DOC.replace(
        "us.anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5"
    ).replace("careful assistant", "careful assistant for Claude Sonnet 5")
    from tests.unit.adaptation_helpers import dispose_all, edit_change

    result = service.submit_adapted_prompt(
        run_dir,
        "prompts/profile.yaml",
        adapted,
        "Adapted for Sonnet 5 on Bedrock.",
        ["Updated the model id to the US inference profile and refreshed the prompt."],
        guidance_dispositions=dispose_all(service, run_dir, "prompts/profile.yaml"),
        annotated_changes=[
            edit_change(
                "careful assistant",
                "careful assistant for Claude Sonnet 5",
                why="Names the target model explicitly.",
            )
        ],
        submitted_on=AS_OF,
    )
    assert result.accepted, result.message
