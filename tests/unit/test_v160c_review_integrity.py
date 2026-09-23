"""V1.6.0-c: review integrity and validation teeth.

Unit-aware registry-contradiction gate for adapted pricing, CONTESTED marks
in change review, the evaluation scaffold and validation_pending state, and
pair `prompt_guidance` knowledge landed through a reviewed bundle.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from llm_migrate.core.evaluation import manifest_sha256
from llm_migrate.core.models import ModelPricing, PricePerMillionTokens, UnknownKind
from llm_migrate.core.pricing_gate import classify_price
from llm_migrate.core.workspace import load_run_config
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import dispose_all, edit_change

AS_OF = date(2026, 9, 23)
BEDROCK = {
    "source_platform": "amazon-bedrock",
    "target_platform": "amazon-bedrock",
    "target_endpoint": "bedrock-runtime",
}
BEDROCK_PAIR = ("us.anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5")
UNKNOWN_URL = "https://example.com/blog/regional-premium"
REGISTRY_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/models/overview"
TARGET = ModelPricing(
    input=PricePerMillionTokens(amount=Decimal(2)), output=PricePerMillionTokens(amount=Decimal(10))
)
SOURCE = ModelPricing(
    input=PricePerMillionTokens(amount=Decimal(3)), output=PricePerMillionTokens(amount=Decimal(15))
)

APP = """\
import boto3
import yaml

with open("config/model_profiles.yaml") as handle:
    PROFILES = yaml.safe_load(handle)

PROFILE = PROFILES["models"]["claude"]
client = boto3.client("bedrock-runtime")
client.converse(
    modelId=PROFILE["model_id"],
    messages=[{"role": "user", "content": [{"text": "hi"}]}],
    inferenceConfig={"maxTokens": 100},
)
"""
CONFIG = """\
models:
  claude:
    model_id: anthropic.claude-sonnet-4-6
    pricing:
      input_cost_per_1k: 0.003
      output_cost_per_1k: 0.015
  llama:
    model_id: meta.llama3-70b-instruct-v1:0
    pricing:
      input_cost_per_1k: 0.00265
      output_cost_per_1k: 0.0035
"""


def _write(root: Path, files: dict[str, str]) -> Path:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def priced_app(tmp_path: Path) -> Path:
    return _write(tmp_path / "priced_app", {"app.py": APP, "config/model_profiles.yaml": CONFIG})


def _run(service: MigrationService, app: Path, **extra) -> Path:  # type: ignore[no-untyped-def]
    start = service.start_migration_run(
        app, *BEDROCK_PAIR, as_of=AS_OF, research="skip", **BEDROCK, **extra
    )
    assert start.status == "ready" and start.paths is not None, start.next_steps
    return Path(start.paths.run_dir)


def _submit_config(
    service: MigrationService,
    run_dir: Path,
    app: Path,
    *,
    input_price: str,
    output_price: str,
    kind: str = "model_difference",
    url: str = UNKNOWN_URL,
) -> None:
    path = "config/model_profiles.yaml"
    original = (app / path).read_text(encoding="utf-8")
    adapted = original.replace(
        "model_id: anthropic.claude-sonnet-4-6", "model_id: us.anthropic.claude-sonnet-5"
    )
    changes = [
        edit_change(
            "model_id: anthropic.claude-sonnet-4-6",
            "model_id: us.anthropic.claude-sonnet-5",
            "Invoke the target through its reviewed inference profile.",
        )
    ]
    for key, old, new in (
        ("input_cost_per_1k", "0.003", input_price),
        ("output_cost_per_1k", "0.015", output_price),
    ):
        if new == old:
            continue
        adapted = adapted.replace(f"{key}: {old}", f"{key}: {new}", 1)
        changes.append(
            edit_change(
                f"{key}: {old}",
                f"{key}: {new}",
                "Price the target for cost telemetry.",
                kind=kind,  # type: ignore[arg-type]
                url=url,
            )
        )
    result = service.submit_adapted_file(
        run_dir,
        path,
        adapted,
        "Retarget the Claude profile.",
        ["model id", "pricing"],
        guidance_dispositions=dispose_all(service, run_dir, path),
        annotated_changes=changes,
        submitted_on=AS_OF,
    )
    assert result.accepted, result.problems


# -- unit-aware pricing classification ---------------------------------------


def test_field_pricing_edit_is_a_contradiction_with_its_factor() -> None:
    for key, value in (("input_cost_per_1k", 0.0022), ("output_cost_per_1k", 0.011)):
        check = classify_price(("pricing", key), value, target=TARGET, source=SOURCE, as_of=AS_OF)
        assert check is not None and check.verdict == "contradiction"
        assert check.factor == Decimal("1.1") and check.scale == "per_1k"
        assert "1.1x" in check.message
    # Without a unit in the key the closest standard scale is inferred.
    check = classify_price(("input_price",), "0.0022", target=TARGET, source=SOURCE, as_of=AS_OF)
    assert check is not None and check.verdict == "contradiction" and check.scale == "per_1k"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("input_price", 0.000003),
        ("input_price", 0.003),
        ("input_price", 3),
        ("input_usd_per_1m", 3),
    ],
)
def test_source_price_under_any_scale_is_stale(key: str, value: float) -> None:
    check = classify_price((key,), value, target=TARGET, source=SOURCE, as_of=AS_OF)
    assert check is not None and check.verdict == "stale"


def test_target_price_is_consistent_and_off_scale_is_unrecognized() -> None:
    for key, value in (
        ("input_price", 0.000002),
        ("output_per_1k", 0.01),
        ("input_per_million", 2),
    ):
        check = classify_price((key,), value, target=TARGET, source=SOURCE, as_of=AS_OF)
        assert check is not None and check.verdict == "consistent", (key, value)
    check = classify_price(("input_price",), 7.5, target=TARGET, source=SOURCE, as_of=AS_OF)
    assert check is not None and check.verdict == "unit_unrecognized"
    assert "matches no standard pricing scale" in check.message
    # A key that names its unit is read in that unit only: 2 per 1K is not $2/1M.
    named = classify_price(("input_per_1k",), 2, target=TARGET, source=SOURCE, as_of=AS_OF)
    assert named is not None and named.verdict == "unit_unrecognized"


def test_expired_registry_prices_say_so() -> None:
    expiring = ModelPricing(
        input=PricePerMillionTokens(
            amount=Decimal(2), valid_until=date(2026, 8, 31), notes="Introductory price."
        )
    )
    check = classify_price(("input_per_1k",), 0.0025, target=expiring, source=SOURCE, as_of=AS_OF)
    assert check is not None and "valid until 2026-08-31" in check.message


# -- the gate over a run's deliverables ---------------------------------------


def test_gate_flags_the_field_edit_and_ignores_foreign_profiles(
    service: MigrationService, priced_app: Path
) -> None:
    run_dir = _run(service, priced_app)
    _submit_config(service, run_dir, priced_app, input_price="0.0022", output_price="0.011")
    finalization = service.finalize_migration_run(run_dir)
    pricing = [item for item in finalization.consistency_findings if "pricing_" in item]
    assert len(pricing) == 2, finalization.consistency_findings
    assert all("pricing_contradicts_registry" in item and "1.1x" in item for item in pricing)
    assert not any("llama" in item for item in pricing)  # the other model's profile
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "pricing_contradicts_registry" in report


def test_registry_evidence_clears_a_departure_but_unknown_evidence_does_not(
    service: MigrationService, priced_app: Path
) -> None:
    run_dir = _run(service, priced_app)
    _submit_config(
        service,
        run_dir,
        priced_app,
        input_price="0.0022",
        output_price="0.011",
        url=REGISTRY_PRICING_URL,
    )
    finalization = service.finalize_migration_run(run_dir)
    assert not any("pricing_" in item for item in finalization.consistency_findings)


def test_unchanged_source_pricing_is_stale(service: MigrationService, priced_app: Path) -> None:
    run_dir = _run(service, priced_app)
    _submit_config(service, run_dir, priced_app, input_price="0.003", output_price="0.015")
    finalization = service.finalize_migration_run(run_dir)
    stale = [item for item in finalization.consistency_findings if "pricing_stale_source" in item]
    assert len(stale) == 2


def test_strict_mode_blocks_on_a_pricing_contradiction(
    service: MigrationService, priced_app: Path
) -> None:
    run_dir = _run(service, priced_app, strict=True)
    _submit_config(
        service,
        run_dir,
        priced_app,
        input_price="0.0022",
        output_price="0.011",
        kind="mechanical",
        url="",
    )
    finalization = service.finalize_migration_run(run_dir)
    assert any("pricing_contradicts_registry" in item for item in finalization.strict_violations)


# -- CONTESTED marks in change review ----------------------------------------


def test_change_depending_on_contested_fact_is_marked_until_observed(
    service: MigrationService, priced_app: Path
) -> None:
    run_dir = _run(service, priced_app)
    original = (priced_app / "app.py").read_text(encoding="utf-8")
    anchor = '    inferenceConfig={"maxTokens": 100},\n'
    disabled = anchor + '    additionalModelRequestFields={"thinking": {"type": "disabled"}},\n'
    result = service.submit_adapted_file(
        run_dir,
        "app.py",
        original.replace(anchor, disabled),
        "Keep the no-thinking behavior of the source.",
        ["disable adaptive thinking"],
        guidance_dispositions=dispose_all(service, run_dir, "app.py"),
        annotated_changes=[
            edit_change(
                anchor.strip(),
                disabled.strip(),
                "Sonnet 5 thinks by default; disable it to keep source behavior.",
                kind="model_difference",
                url="https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5",
            )
        ],
        submitted_on=AS_OF,
    )
    assert result.accepted, result.problems
    _submit_config(
        service,
        run_dir,
        priced_app,
        input_price="0.002",
        output_price="0.01",
        url="https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5",
    )
    review = service.get_change_review(run_dir)
    marks = {
        item.source_path: [change.contested for change in item.changes] for item in review.files
    }
    assert marks["app.py"][0] and "CONTESTED" in marks["app.py"][0][0]
    assert "AWS model card" in marks["app.py"][0][0]
    # Citing a conflict's source document without exercising the setting is not contested.
    assert not any(marks["config/model_profiles.yaml"])
    finalization = service.finalize_migration_run(run_dir)
    assert finalization.contested_changes
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "depend on a CONTESTED registry fact" in report

    plan = service._plan_for_run(load_run_config(run_dir), run_dir)
    contested = next(item for item in plan.unknowns if item.kind is UnknownKind.CONTESTED_EVIDENCE)
    assert service.record_observation(
        run_dir, contested.id, "ACCEPTED", "stopReason=end_turn; reasoning_returned=False"
    ).accepted
    review = service.get_change_review(run_dir)
    app_review = next(item for item in review.files if item.source_path == "app.py")
    assert app_review.changes[0].contested == []


# -- prompt_guidance pair knowledge -------------------------------------------


def _prompt_app(tmp_path: Path, *, temperature: bool) -> Path:
    call = (
        "client.messages.create(\n"
        '    model="claude-sonnet-4-6",\n'
        "    max_tokens=100,\n"
        + ("    temperature=0.2,\n" if temperature else "")
        + "    system=SYSTEM,\n"
        '    messages=[{"role": "user", "content": "hi"}],\n'
        ")\n"
    )
    return _write(
        tmp_path / ("sampled" if temperature else "unsampled"),
        {
            "app.py": (
                "from pathlib import Path\n\nimport anthropic\n\n"
                'SYSTEM = Path("prompts/system.txt").read_text()\n'
                "client = anthropic.Anthropic()\n" + call
            ),
            "prompts/system.txt": "You are a concise assistant.\n",
        },
    )


def _shared_guidance(service: MigrationService, app: Path):  # type: ignore[no-untyped-def]
    start = service.start_migration_run(
        app, "claude-sonnet-4-6", "claude-sonnet-5", as_of=AS_OF, research="skip"
    )
    assert start.paths is not None
    return service.list_adaptation_tasks(start.paths.run_dir).shared_prompt_guidance


def test_pair_prompt_guidance_reaches_prompt_tasks_with_evidence(
    service: MigrationService, tmp_path: Path
) -> None:
    sampled = _shared_guidance(service, _prompt_app(tmp_path, temperature=True))
    sampling = next(item for item in sampled if "system-prompt instructions" in item.text)
    assert sampling.not_applicable_reason is None
    assert "platform.claude.com" in sampling.text
    recount = next(item for item in sampled if "Recount each prompt's tokens" in item.text)
    assert recount.not_applicable_reason is None


def test_sampling_prompt_guidance_is_predisposed_when_unused(
    service: MigrationService, tmp_path: Path
) -> None:
    unsampled = _shared_guidance(service, _prompt_app(tmp_path, temperature=False))
    sampling = next(item for item in unsampled if "system-prompt instructions" in item.text)
    assert sampling.not_applicable_reason is not None
    assert "sampling parameter" in sampling.not_applicable_reason
    recount = next(item for item in unsampled if "Recount each prompt's tokens" in item.text)
    assert recount.not_applicable_reason is None  # no trigger: always owed


# -- evaluation scaffold and validation_pending --------------------------------


def test_scaffold_drafts_plan_bound_cases_and_status_pushes_validation(
    service: MigrationService, tmp_path: Path
) -> None:
    app = _write(
        tmp_path / "eval_app",
        {
            "app.py": (
                "import anthropic\n\nclient = anthropic.Anthropic()\n"
                "client.messages.create(\n"
                '    model="claude-sonnet-4-6",\n'
                "    max_tokens=100,\n"
                '    messages=[{"role": "user", "content": "hi"}],\n'
                ")\n"
            ),
            "samples/question.txt": "What changed in the Q3 report?\n",
            "test_data/cases.jsonl": (
                '{"input": "Summarize the memo.", "expected": "A short summary."}\n'
                '{"question": "List the risks."}\n'
                "not json\n"
            ),
            "docs/notes.txt": "Not a sample folder.\n",
        },
    )
    start = service.start_migration_run(
        app, "claude-sonnet-4-6", "claude-sonnet-5", as_of=AS_OF, research="skip"
    )
    assert start.paths is not None
    run_dir = Path(start.paths.run_dir)
    original = (app / "app.py").read_text(encoding="utf-8")
    submitted = service.submit_adapted_file(
        run_dir,
        "app.py",
        original.replace('model="claude-sonnet-4-6"', 'model="claude-sonnet-5"'),
        "Retarget the model.",
        ["model id"],
        guidance_dispositions=dispose_all(service, run_dir, "app.py"),
        annotated_changes=[
            edit_change('model="claude-sonnet-4-6"', 'model="claude-sonnet-5"', "Target id.")
        ],
        submitted_on=AS_OF,
    )
    assert submitted.accepted, submitted.problems
    review = service.get_change_review(run_dir)
    change_id = review.files[0].changes[0].change.id
    assert service.record_change_decision(
        run_dir, "app.py", change_id, "accepted", decided_on=AS_OF
    ).accepted
    assert service.get_run_status(run_dir).state == "ready_to_finalize"
    service.finalize_migration_run(run_dir)
    status = service.get_run_status(run_dir)
    assert status.state == "validation_pending"
    assert "scaffold_evaluation(" in status.next_action

    scaffold = service.scaffold_evaluation(run_dir)
    assert scaffold.case_count == 3
    assert scaffold.sample_files == ["samples/question.txt", "test_data/cases.jsonl"]
    plan = service._plan_for_run(load_run_config(run_dir), run_dir)
    assert scaffold.bound_manifest_sha256 == manifest_sha256(plan)
    assert scaffold.suite_path and Path(scaffold.suite_path).is_file()
    suite_text = Path(scaffold.suite_path).read_text(encoding="utf-8")
    assert "DRAFT" in suite_text and "Summarize the memo." in suite_text
    assert "Not a sample folder" not in suite_text

    service.record_validation_disposition(
        run_dir,
        "generated_tests",
        outcome="run_passed",
        outcome_summary="1 passed in 0.01s",
        decided_on=AS_OF,
    )
    assert service.get_run_status(run_dir).state == "ready_to_finalize"


def test_scaffold_without_samples_says_how_to_proceed(
    service: MigrationService, tmp_path: Path
) -> None:
    app = _write(
        tmp_path / "bare",
        {
            "app.py": (
                "import anthropic\n\nclient = anthropic.Anthropic()\n"
                'client.messages.create(model="claude-sonnet-4-6", max_tokens=10, '
                'messages=[{"role": "user", "content": "hi"}])\n'
            )
        },
    )
    start = service.start_migration_run(
        app, "claude-sonnet-4-6", "claude-sonnet-5", as_of=AS_OF, research="skip"
    )
    assert start.paths is not None
    scaffold = service.scaffold_evaluation(start.paths.run_dir)
    assert scaffold.case_count == 0 and scaffold.suite_path is None
    assert "generate_eval_suite" in scaffold.message
