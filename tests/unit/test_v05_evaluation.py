from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from llm_migrate.adapters.evaluation import HttpEvaluationExecutor
from llm_migrate.core.evaluation import evaluation_artifact_as_yaml
from llm_migrate.core.evaluation_models import (
    EvaluationCase,
    EvaluationRunConfig,
    EvaluationStatus,
    EvaluationSuite,
    EvaluationValidator,
    EvaluatorKind,
    EvaluatorResult,
    MigrationEvalRun,
    ProviderCallResult,
    RegressionCategory,
    RegressionReport,
)
from llm_migrate.service import MigrationService
from scripts.update_v05_goldens import build_artifacts


class FakeExecutor:
    def __init__(self, responses: dict[tuple[str, str], ProviderCallResult]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def execute(self, config: EvaluationRunConfig, case: EvaluationCase) -> ProviderCallResult:
        self.calls.append((config.model, case.id))
        return self.responses[(config.model, case.id)]


def _plan(service: MigrationService, project_root: Path):
    return service.generate_migration_plan(
        project_root / "tests" / "fixtures" / "applications" / "anthropic_app",
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )


def _configs() -> tuple[EvaluationRunConfig, EvaluationRunConfig]:
    return (
        EvaluationRunConfig(
            provider="anthropic",
            platform="anthropic-api",
            model="claude-sonnet-5",
            model_id="claude-sonnet-5",
            credential_env="ANTHROPIC_API_KEY",
            input_price_per_million="2",
            output_price_per_million="10",
        ),
        EvaluationRunConfig(
            provider="openai",
            platform="openai-api",
            model="gpt-5.6-sol",
            model_id="gpt-5.6-sol",
            credential_env="OPENAI_API_KEY",
            input_price_per_million="3",
            output_price_per_million="15",
        ),
    )


def _call(
    output: str,
    *,
    latency: float = 100,
    tokens: tuple[int, int] = (10, 5),
    tool_calls: list[dict[str, Any]] | None = None,
    refusal: bool = False,
):
    return ProviderCallResult(
        status=EvaluationStatus.PASSED,
        output=output,
        tool_calls=tool_calls or [],
        refusal_detected=refusal,
        latency_ms=latency,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
    )


def test_generate_suite_binds_manifest_and_adds_golden_validator(
    service: MigrationService, project_root: Path
) -> None:
    plan = _plan(service, project_root)
    case = EvaluationCase(
        id="extract-1",
        input="Extract Acme",
        expected_output={"answer": "Acme"},
        expected_tool_calls=[{"name": "lookup", "arguments": {"name": "Acme"}}],
    )
    first = service.generate_eval_suite(plan, [case], name="extraction")
    second = service.generate_eval_suite(plan, [case], name="extraction")
    assert first.migration_manifest_sha256 == second.migration_manifest_sha256
    assert {validator.kind for validator in first.cases[0].validators} == {
        EvaluatorKind.EXACT_MATCH,
        EvaluatorKind.TOOL_CALLS,
    }
    with pytest.raises(ValidationError, match="case ids must be unique"):
        EvaluationSuite(
            name="duplicate",
            migration_manifest_sha256=first.migration_manifest_sha256,
            source=first.source,
            target=first.target,
            cases=[case, case],
        )


def test_run_compare_and_analyze_detect_regression_and_persist_results(
    service: MigrationService, project_root: Path
) -> None:
    plan = _plan(service, project_root)
    case = EvaluationCase(
        id="extract-1",
        input="Extract Acme",
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
    )
    suite = service.generate_eval_suite(plan, [case], name="extraction")
    source_config, target_config = _configs()
    executor = FakeExecutor(
        {
            (source_config.model, case.id): _call('{"answer":"Acme"}'),
            (target_config.model, case.id): _call("not json", latency=400, tokens=(20, 20)),
        }
    )
    times = iter([datetime(2026, 8, 23, tzinfo=UTC), datetime(2026, 8, 23, 0, 0, 1, tzinfo=UTC)])
    run = service.run_migration_eval(
        suite,
        source_config,
        target_config,
        executor=executor,
        clock=times.__next__,
    )
    assert executor.calls == [
        (source_config.model, case.id),
        (target_config.model, case.id),
    ]
    assert run.source_results[0].status is EvaluationStatus.PASSED
    assert run.target_results[0].status is EvaluationStatus.FAILED
    assert run.source_results[0].estimated_cost_usd == Decimal("0.00007")
    assert run.target_results[0].estimated_cost_usd == Decimal("0.00036")
    comparison = service.compare_outputs(run.source_results[0], run.target_results[0])
    assert comparison.regression
    assert comparison.score_delta == -1
    assert comparison.cost_delta_usd == Decimal("0.00029")
    assert "latency_increased_over_50_percent" in comparison.signals
    report = service.analyze_regressions(run)
    assert report.summary.source_pass_rate == 1
    assert report.summary.target_pass_rate == 0
    assert {finding.category.value for finding in report.regressions} >= {
        "format_schema_parser_incompatibility",
        "latency_or_token_cost_regression",
    }
    rendered = evaluation_artifact_as_yaml(run)
    assert "ANTHROPIC_API_KEY" in rendered
    assert "sk-test-secret" not in rendered
    assert MigrationEvalRun.model_validate(yaml.safe_load(rendered)) == run
    tampered = yaml.safe_load(rendered)
    tampered["source_results"][0]["case_id"] = "wrong-case"
    with pytest.raises(ValidationError, match="source results must match suite case order"):
        MigrationEvalRun.model_validate(tampered)
    report_rendered = evaluation_artifact_as_yaml(report)
    assert RegressionReport.model_validate(yaml.safe_load(report_rendered)) == report


def test_partial_failure_and_unconfigured_custom_evaluator_remain_explicit(
    service: MigrationService, project_root: Path
) -> None:
    plan = _plan(service, project_root)
    case = EvaluationCase(
        id="custom",
        input="Answer",
        validators=[
            EvaluationValidator(
                kind=EvaluatorKind.CUSTOM,
                name="grounding",
                evaluator_name="grounding-check",
            )
        ],
    )
    suite = service.generate_eval_suite(plan, [case])
    source_config, target_config = _configs()
    executor = FakeExecutor(
        {
            (source_config.model, case.id): _call("grounded"),
            (target_config.model, case.id): ProviderCallResult(
                status=EvaluationStatus.ERROR,
                latency_ms=250,
                attempts=3,
                error_kind="rate_limit",
                error_message="Provider request failed after bounded retries.",
            ),
        }
    )
    run = service.run_migration_eval(suite, source_config, target_config, executor=executor)
    assert run.source_results[0].status is EvaluationStatus.SKIPPED
    assert run.target_results[0].status is EvaluationStatus.ERROR
    assert run.target_results[0].attempts == 3
    assert run.warnings == ["Unconfigured evaluators were skipped: grounding-check."]


def test_custom_evaluator_isolated_and_threshold_applied(
    service: MigrationService, project_root: Path
) -> None:
    plan = _plan(service, project_root)
    case = EvaluationCase(
        id="judge",
        input="Answer",
        validators=[
            EvaluationValidator(
                kind=EvaluatorKind.LLM_JUDGE,
                name="instruction judge",
                evaluator_name="judge",
                minimum_score=0.8,
            )
        ],
    )
    suite = service.generate_eval_suite(plan, [case])
    source_config, target_config = _configs()
    executor = FakeExecutor(
        {
            (source_config.model, case.id): _call("source"),
            (target_config.model, case.id): _call("target"),
        }
    )

    def judge(evaluation_case: EvaluationCase, output: str) -> EvaluatorResult:
        score = 1.0 if output == "source" else 0.5
        return EvaluatorResult(
            name="instruction judge",
            kind=EvaluatorKind.LLM_JUDGE,
            status=EvaluationStatus.PASSED,
            score=score,
            message=f"Scored {evaluation_case.id}.",
        )

    run = service.run_migration_eval(
        suite,
        source_config,
        target_config,
        executor=executor,
        evaluators={"judge": judge},
    )
    assert run.source_results[0].status is EvaluationStatus.PASSED
    assert run.target_results[0].status is EvaluationStatus.FAILED


def test_custom_evaluator_exception_is_sanitized(
    service: MigrationService, project_root: Path
) -> None:
    plan = _plan(service, project_root)
    case = EvaluationCase(
        id="broken-evaluator",
        input="Answer",
        validators=[
            EvaluationValidator(
                kind=EvaluatorKind.CUSTOM,
                name="broken",
                evaluator_name="broken",
            )
        ],
    )
    suite = service.generate_eval_suite(plan, [case])
    source_config, target_config = _configs()
    executor = FakeExecutor(
        {
            (source_config.model, case.id): _call("source"),
            (target_config.model, case.id): _call("target"),
        }
    )

    def broken(evaluation_case: EvaluationCase, output: str) -> EvaluatorResult:
        raise RuntimeError("fixture-secret-value")

    run = service.run_migration_eval(
        suite,
        source_config,
        target_config,
        executor=executor,
        evaluators={"broken": broken},
    )
    result = run.source_results[0].evaluator_results[0]
    assert result.status is EvaluationStatus.ERROR
    assert result.message == "Evaluator failed with RuntimeError."
    assert "fixture-secret-value" not in run.model_dump_json()


def test_http_executor_retries_rate_limit_and_does_not_surface_secret(monkeypatch) -> None:
    secret = "fixture-secret-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    statuses = iter(
        [
            (429, {"error": {"message": f"leaked {secret}"}}),
            (
                200,
                {
                    "output_text": "ok",
                    "usage": {"input_tokens": 4, "output_tokens": 1},
                },
            ),
        ]
    )
    sleeps: list[float] = []
    seen_headers: list[str] = []

    def transport(request, timeout: float) -> tuple[int, dict[str, Any]]:
        assert timeout == 30
        seen_headers.append(request.get_header("Authorization"))
        return next(statuses)

    ticks = iter([1.0, 1.1])
    executor = HttpEvaluationExecutor(
        transport=transport,
        sleeper=sleeps.append,
        monotonic=ticks.__next__,
    )
    config = _configs()[1]
    result = executor.execute(config, EvaluationCase(id="retry", input="hello"))
    assert result.status is EvaluationStatus.PASSED
    assert result.attempts == 2
    assert result.output == "ok"
    assert sleeps == [1]
    assert all(secret in header for header in seen_headers)
    assert secret not in result.model_dump_json()


def test_run_configuration_rejects_inline_secret_parameters() -> None:
    with pytest.raises(ValidationError, match="credential_env"):
        EvaluationRunConfig(
            provider="openai",
            platform="openai-api",
            model="gpt-5.6-sol",
            model_id="gpt-5.6-sol",
            parameters={"api_key": "sk-test-secret"},
        )
    with pytest.raises(ValidationError, match="user information"):
        EvaluationRunConfig(
            provider="openai",
            platform="openai-api",
            model="gpt-5.6-sol",
            model_id="gpt-5.6-sol",
            base_url="https://user:secret@example.com",
        )
    with pytest.raises(ValidationError, match="reserved request fields: model"):
        EvaluationRunConfig(
            provider="openai",
            platform="openai-api",
            model="gpt-5.6-sol",
            model_id="gpt-5.6-sol",
            parameters={"model": "wrong-model"},
        )
    with pytest.raises(ValidationError, match=r"parameters\.headers\.Authorization"):
        EvaluationRunConfig(
            provider="openai",
            platform="openai-api",
            model="gpt-5.6-sol",
            model_id="gpt-5.6-sol",
            parameters={"headers": {"Authorization": "Bearer secret"}},
        )
    nested = EvaluationRunConfig(
        provider="openai",
        platform="openai-api",
        model="gpt-5.6-sol",
        model_id="gpt-5.6-sol",
        parameters={
            "max_output_tokens": 100,
            "tools": [{"type": "function", "name": "lookup"}],
        },
    )
    assert nested.parameters["tools"] == [{"type": "function", "name": "lookup"}]
    with pytest.raises(ValidationError, match="exact_match validators require expected_output"):
        EvaluationCase(
            id="missing-golden",
            input="hello",
            validators=[EvaluationValidator(kind=EvaluatorKind.EXACT_MATCH, name="missing-golden")],
        )


def test_run_rejects_configuration_for_a_different_manifest_endpoint(
    service: MigrationService, project_root: Path
) -> None:
    suite = service.generate_eval_suite(
        _plan(service, project_root),
        [EvaluationCase(id="mismatch", input="hello", expected_output="ok")],
    )
    source_config, target_config = _configs()
    wrong_target = target_config.model_copy(update={"model_id": "a-different-deployment"})
    with pytest.raises(ValueError, match="target evaluation config"):
        service.run_migration_eval(
            suite,
            source_config,
            wrong_target,
            executor=FakeExecutor({}),
        )


def test_regex_and_invalid_schema_evaluators_are_deterministic(
    service: MigrationService, project_root: Path
) -> None:
    plan = _plan(service, project_root)
    cases = [
        EvaluationCase(
            id="regex",
            input="Answer",
            validators=[
                EvaluationValidator(
                    kind=EvaluatorKind.REGEX,
                    name="answer-pattern",
                    pattern=r"^answer: [0-9]+$",
                )
            ],
        ),
        EvaluationCase(
            id="bad-schema",
            input="Answer JSON",
            validators=[
                EvaluationValidator(
                    kind=EvaluatorKind.JSON_SCHEMA,
                    name="invalid-schema",
                    schema_definition={"type": "not-a-json-schema-type"},
                )
            ],
        ),
    ]
    suite = service.generate_eval_suite(plan, cases)
    source_config, target_config = _configs()
    executor = FakeExecutor(
        {
            (config.model, case.id): _call("answer: 42" if case.id == "regex" else "{}")
            for config in (source_config, target_config)
            for case in cases
        }
    )
    run = service.run_migration_eval(suite, source_config, target_config, executor=executor)
    assert run.source_results[0].status is EvaluationStatus.PASSED
    assert run.source_results[1].status is EvaluationStatus.ERROR
    assert "Invalid evaluator JSON Schema" in run.source_results[1].evaluator_results[0].message


def test_builtin_executor_rejects_bedrock_before_reading_credentials() -> None:
    calls = 0

    def transport(request, timeout: float):
        nonlocal calls
        calls += 1
        return 200, {}

    executor = HttpEvaluationExecutor(transport=transport)
    result = executor.execute(
        EvaluationRunConfig(
            provider="anthropic",
            platform="amazon-bedrock",
            model="claude-sonnet-4-6",
            model_id="anthropic.claude-sonnet-4-6",
        ),
        EvaluationCase(id="bedrock", input="hello"),
    )
    assert result.status is EvaluationStatus.ERROR
    assert result.error_kind == "unsupported"
    assert "inject an EvaluationExecutor" in (result.error_message or "")
    assert calls == 0


def test_builtin_executor_preserves_nested_runtime_parameters(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    payloads: list[dict[str, Any]] = []

    def transport(request, timeout: float):
        payloads.append(json.loads(request.data))
        return 200, {"output_text": "ok"}

    config = _configs()[1].model_copy(
        update={
            "parameters": {
                "max_output_tokens": 100,
                "tools": [{"type": "function", "name": "lookup"}],
            }
        }
    )
    result = HttpEvaluationExecutor(transport=transport).execute(
        config, EvaluationCase(id="nested", input="hello")
    )
    assert result.status is EvaluationStatus.PASSED
    assert payloads[0]["model"] == "gpt-5.6-sol"
    assert payloads[0]["tools"] == [{"type": "function", "name": "lookup"}]


def test_provider_tool_calls_normalize_to_one_cross_provider_contract() -> None:
    expected = [{"name": "lookup", "arguments": {"id": 1}}]
    openai_text, openai_tools, _ = HttpEvaluationExecutor._parse_openai(
        {
            "output_text": "done",
            "output": [
                {
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": '{"id": 1}',
                }
            ],
        }
    )
    anthropic_text, anthropic_tools, _ = HttpEvaluationExecutor._parse_anthropic(
        {
            "content": [
                {"type": "text", "text": "done"},
                {"type": "tool_use", "name": "lookup", "input": {"id": 1}},
            ]
        }
    )
    assert openai_text == anthropic_text == "done"
    assert openai_tools == anthropic_tools == expected


def test_builtin_executor_returns_explicit_invalid_output(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    executor = HttpEvaluationExecutor(transport=lambda request, timeout: (200, {}))
    result = executor.execute(_configs()[1], EvaluationCase(id="empty", input="hello"))
    assert result.status is EvaluationStatus.ERROR
    assert result.error_kind == "invalid_output"
    assert "neither text nor tool calls" in (result.error_message or "")
    assert "sk-test-secret" not in result.model_dump_json()


def test_multiple_detected_output_schemas_require_case_level_selection(
    service: MigrationService, project_root: Path
) -> None:
    plan = service.generate_migration_plan(
        project_root / "tests" / "fixtures" / "applications" / "openai_app",
        "gpt-5.6-sol",
        "claude-sonnet-5",
        source_platform="openai-api",
        target_platform="anthropic-api",
    )
    invocation = plan.invocation_changes[0]
    contract = invocation.source_analysis.structured_outputs[0]
    analysis = invocation.source_analysis.model_copy(
        update={
            "structured_outputs": [
                contract,
                contract.model_copy(update={"name": "second-contract"}),
            ]
        }
    )
    plan = plan.model_copy(
        update={"invocation_changes": [invocation.model_copy(update={"source_analysis": analysis})]}
    )
    suite = service.generate_eval_suite(
        plan,
        [EvaluationCase(id="multiple-schemas", input="hello", expected_output="ok")],
    )
    assert [validator.kind for validator in suite.cases[0].validators] == [
        EvaluatorKind.EXACT_MATCH
    ]
    assert "Multiple structured-output schemas were detected" in suite.warnings[0]


def test_tool_call_and_refusal_regressions_are_first_class_metrics(
    service: MigrationService, project_root: Path
) -> None:
    plan = _plan(service, project_root)
    expected_tools = [{"name": "lookup", "arguments": {"id": 1}}]
    case = EvaluationCase(
        id="tools",
        input="Lookup one",
        expected_output="done",
        expected_tool_calls=expected_tools,
    )
    suite = service.generate_eval_suite(plan, [case])
    source_config, target_config = _configs()
    executor = FakeExecutor(
        {
            (source_config.model, case.id): _call("done", tool_calls=expected_tools),
            (target_config.model, case.id): _call("cannot comply", refusal=True),
        }
    )
    run = service.run_migration_eval(suite, source_config, target_config, executor=executor)
    report = service.analyze_regressions(run)
    assert report.summary.source_refusal_rate == 0
    assert report.summary.target_refusal_rate == 1
    assert {finding.category.value for finding in report.regressions} >= {
        "tool_use_regression",
        "refusal_or_safety_change",
    }


def test_materially_shorter_output_is_reported_as_verbosity_change(
    service: MigrationService, project_root: Path
) -> None:
    plan = _plan(service, project_root)
    case = EvaluationCase(
        id="verbosity",
        input="Explain",
        validators=[
            EvaluationValidator(
                kind=EvaluatorKind.REGEX,
                name="answer-present",
                pattern="answer",
            )
        ],
    )
    suite = service.generate_eval_suite(plan, [case])
    source_config, target_config = _configs()
    executor = FakeExecutor(
        {
            (source_config.model, case.id): _call("answer " + "detail " * 30),
            (target_config.model, case.id): _call("answer"),
        }
    )
    run = service.run_migration_eval(suite, source_config, target_config, executor=executor)
    report = service.analyze_regressions(run)
    assert "output_length_decreased_over_50_percent" in report.comparisons[0].signals
    assert {finding.category for finding in report.regressions} >= {RegressionCategory.VERBOSITY}


def test_v05_structured_artifacts_match_goldens(project_root: Path) -> None:
    suite, report = build_artifacts()
    golden = project_root / "tests" / "golden" / "v05"
    assert suite == (golden / "evaluation-suite.yaml").read_text()
    assert report == (golden / "regression-report.yaml").read_text()
