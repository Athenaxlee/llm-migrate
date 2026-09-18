from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from llm_migrate.core.evaluation_models import (
    EvaluationCase,
    EvaluationRunConfig,
    EvaluationStatus,
    ProviderCallResult,
)
from llm_migrate.core.intelligence import (
    check_model_lifecycle,
    estimate_migration_cost,
)
from llm_migrate.core.models import LifecycleStatus, MigrationWorkload
from llm_migrate.service import MigrationService


class StableFakeExecutor:
    """Offline endpoint double for the V1 acceptance workflow."""

    def execute(self, config: EvaluationRunConfig, case: EvaluationCase) -> ProviderCallResult:
        return ProviderCallResult(
            status=EvaluationStatus.PASSED,
            output="ok",
            latency_ms=100 if config.provider == "anthropic" else 80,
            input_tokens=10,
            output_tokens=2,
        )


def test_v1_offline_workflow_reaches_bounded_optimization(
    service: MigrationService,
    project_root: Path,
) -> None:
    application = project_root / "tests/fixtures/applications/anthropic_app"
    before = {path: path.read_bytes() for path in application.rglob("*") if path.is_file()}
    plan = service.generate_migration_plan(
        application,
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    suite = service.generate_eval_suite(
        plan,
        [EvaluationCase(id="v1-smoke", input="Return ok", expected_output="ok")],
        name="v1-acceptance",
    )
    source = EvaluationRunConfig(
        provider="anthropic",
        platform="anthropic-api",
        model="claude-sonnet-5",
        model_id="claude-sonnet-5",
        credential_env="UNUSED_SOURCE_KEY",
        input_price_per_million="2",
        output_price_per_million="10",
    )
    target = EvaluationRunConfig(
        provider="openai",
        platform="openai-api",
        model="gpt-5.6-sol",
        model_id="gpt-5.6-sol",
        credential_env="UNUSED_TARGET_KEY",
        input_price_per_million="3",
        output_price_per_million="15",
    )
    ticks = iter(
        [
            datetime(2026, 8, 24, tzinfo=UTC),
            datetime(2026, 8, 24, 0, 0, 1, tzinfo=UTC),
        ]
    )
    run = service.run_migration_eval(
        suite,
        source,
        target,
        executor=StableFakeExecutor(),
        clock=ticks.__next__,
    )
    report = service.analyze_regressions(run)
    optimization = service.optimize_migration(report)

    assert plan.schema_version == "4"
    assert plan.invocation_changes[0].schema_version == "3"
    assert plan.invocation_changes[0].tool_schema_migrations[0].target_definition is not None
    assert plan.invocation_changes[0].configuration_migration is not None
    assert plan.invocation_changes[0].configuration_migration.credential_environment_variables == [
        "OPENAI_API_KEY"
    ]
    assert suite.migration_manifest_sha256 == run.suite.migration_manifest_sha256
    assert report.summary.target_pass_rate == 1
    assert optimization.regression_report_sha256
    assert before == {path: path.read_bytes() for path in application.rglob("*") if path.is_file()}


def test_v1_lifecycle_and_cost_intelligence_is_offline_and_provenanced(
    service: MigrationService,
) -> None:
    lifecycle = service.check_model_lifecycle("alpha large", as_of_date=date(2026, 8, 24))
    estimate = service.estimate_migration_cost(
        "alpha large",
        "gamma cheap",
        MigrationWorkload(
            requests=100,
            input_tokens_per_request=1_000,
            output_tokens_per_request=500,
        ),
    )

    assert not lifecycle.suitable_for_new_migrations
    assert lifecycle.lifecycle_facts_stale
    assert lifecycle.sources

    profile = service.get_model_profile("gamma cheap")
    deprecated = profile.model_copy(
        update={
            "lifecycle": profile.lifecycle.model_copy(
                update={
                    "status": LifecycleStatus.ACTIVE,
                    "deprecated_on": date(2026, 8, 1),
                }
            )
        }
    )
    dated_lifecycle = check_model_lifecycle(deprecated, as_of_date=date(2026, 8, 24))
    assert not dated_lifecycle.suitable_for_new_migrations
    assert "deprecated as of 2026-08-01" in dated_lifecycle.warnings[0]
    assert estimate.source.total_cost_usd == Decimal("1.25")
    assert estimate.target.total_cost_usd == Decimal("0.06")
    assert estimate.estimated_delta_usd == Decimal("-1.19")
    assert estimate.estimated_savings_percent == Decimal("95.20")
    assert estimate.source.sources

    unpriced = service.get_model_profile("gpt-5.6-luna").model_copy(update={"pricing": None})
    unknown = estimate_migration_cost(
        unpriced,
        service.get_model_profile("gamma cheap"),
        MigrationWorkload(
            requests=1,
            input_tokens_per_request=1,
            output_tokens_per_request=1,
        ),
    )
    assert unknown.source.total_cost_usd is None
    assert unknown.estimated_delta_usd is None
    assert unknown.source.unknown_price_components == ["input", "output"]
