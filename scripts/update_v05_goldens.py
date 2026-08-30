"""Regenerate deterministic V0.5 evaluation-suite and regression goldens."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml

from llm_migrate.core.evaluation_models import (
    EvaluationCase,
    EvaluationRunConfig,
    EvaluationStatus,
    EvaluationValidator,
    EvaluatorKind,
    ProviderCallResult,
)
from llm_migrate.core.models import MigrationPlan
from llm_migrate.service import MigrationService

ROOT = Path(__file__).resolve().parents[1]


class GoldenExecutor:
    def execute(self, config: EvaluationRunConfig, case: EvaluationCase) -> ProviderCallResult:
        source = config.model == "claude-sonnet-5"
        if case.id == "structured-extraction":
            return ProviderCallResult(
                status=EvaluationStatus.PASSED,
                output='{"answer":"Acme"}' if source else "not json",
                latency_ms=100 if source else 350,
                input_tokens=10,
                output_tokens=5 if source else 12,
            )
        return ProviderCallResult(
            status=EvaluationStatus.PASSED,
            output="done" if source else "cannot comply",
            tool_calls=[{"name": "lookup", "arguments": {"id": 1}}] if source else [],
            refusal_detected=not source,
            latency_ms=80 if source else 120,
            input_tokens=8,
            output_tokens=3 if source else 5,
        )


def build_artifacts() -> tuple[str, str]:
    service = MigrationService.from_directory(ROOT / "registry")
    manifest = yaml.safe_load(
        (ROOT / "tests" / "golden" / "v04" / "anthropic_to_openai.yaml").read_text()
    )
    plan = MigrationPlan.model_validate(manifest["migration"])
    cases = [
        EvaluationCase(
            id="structured-extraction",
            input="Extract Acme as JSON.",
            expected_output={"answer": "Acme"},
            validators=[
                EvaluationValidator(
                    kind=EvaluatorKind.JSON_SCHEMA,
                    name="output-schema",
                    schema_definition={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                )
            ],
        ),
        EvaluationCase(
            id="tool-use",
            input="Look up record one.",
            expected_output="done",
            expected_tool_calls=[{"name": "lookup", "arguments": {"id": 1}}],
        ),
    ]
    suite = service.generate_eval_suite(plan, cases, name="v05-golden")
    source_config = EvaluationRunConfig(
        provider="anthropic",
        platform="anthropic-api",
        model="claude-sonnet-5",
        model_id="claude-sonnet-5",
        credential_env="ANTHROPIC_API_KEY",
        input_price_per_million="2",
        output_price_per_million="10",
    )
    target_config = EvaluationRunConfig(
        provider="openai",
        platform="openai-api",
        model="gpt-5.6-sol",
        model_id="gpt-5.6-sol",
        credential_env="OPENAI_API_KEY",
        input_price_per_million="3",
        output_price_per_million="15",
    )
    times = iter([datetime(2026, 8, 23, tzinfo=UTC), datetime(2026, 8, 23, 0, 0, 1, tzinfo=UTC)])
    run = service.run_migration_eval(
        suite,
        source_config,
        target_config,
        executor=GoldenExecutor(),
        clock=times.__next__,
    )
    report = service.analyze_regressions(run)
    return (
        service.evaluation_artifact_as_yaml(suite),
        service.evaluation_artifact_as_yaml(report),
    )


def main() -> None:
    output = ROOT / "tests" / "golden" / "v05"
    output.mkdir(parents=True, exist_ok=True)
    suite, report = build_artifacts()
    (output / "evaluation-suite.yaml").write_text(suite, encoding="utf-8")
    (output / "regression-report.yaml").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
