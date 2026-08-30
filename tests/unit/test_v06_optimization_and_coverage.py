from __future__ import annotations

import json
import os
from base64 import b64encode
from datetime import UTC, datetime
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from llm_migrate.adapters.evaluation import (
    BedrockEvaluationExecutor,
    BuiltinEvaluationExecutor,
    _anthropic_messages,
    _openai_input,
)
from llm_migrate.cli.main import app
from llm_migrate.core.evaluation_models import (
    EvaluationCase,
    EvaluationMessage,
    EvaluationRunConfig,
    EvaluationStatus,
    EvaluationSuite,
    EvaluationTextContent,
    EvaluationValidator,
    EvaluatorKind,
    EvaluatorResult,
    OptimizationLimits,
    ProviderCallResult,
)
from llm_migrate.core.evaluator_registry import (
    configure_evaluators,
    register_evaluator,
    unregister_evaluator,
)
from llm_migrate.mcp.server import optimize_migration as mcp_optimize_migration
from llm_migrate.service import MigrationService

runner = CliRunner()


class FakeExecutor:
    def __init__(self, outputs: dict[tuple[str, str], str]) -> None:
        self.outputs = outputs

    def execute(self, config: EvaluationRunConfig, case: EvaluationCase) -> ProviderCallResult:
        output = self.outputs[(config.model, case.id)]
        return ProviderCallResult(
            status=EvaluationStatus.PASSED,
            output=output,
            latency_ms=50,
            input_tokens=5,
            output_tokens=2,
        )


def _plan(service: MigrationService, project_root: Path):
    return service.generate_migration_plan(
        project_root / "tests" / "fixtures" / "applications" / "anthropic_app",
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )


def _configs() -> tuple[EvaluationRunConfig, EvaluationRunConfig]:
    source = EvaluationRunConfig(
        provider="anthropic",
        platform="anthropic-api",
        model="claude-sonnet-5",
        model_id="claude-sonnet-5",
        input_price_per_million="1",
        output_price_per_million="1",
    )
    target = EvaluationRunConfig(
        provider="openai",
        platform="openai-api",
        model="gpt-5.6-sol",
        model_id="gpt-5.6-sol",
        input_price_per_million="1",
        output_price_per_million="1",
    )
    return source, target


def test_rich_cases_round_trip_and_map_to_reviewed_provider_payloads() -> None:
    image = b64encode(b"image").decode()
    document = b64encode(b"document").decode()
    case = EvaluationCase.model_validate(
        {
            "id": "rich",
            "system_prompt": "Be exact.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Remember Acme"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Remembered"}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "media_type": "image/png", "data": image},
                        {
                            "type": "document",
                            "media_type": "application/pdf",
                            "data": document,
                            "title": "evidence.pdf",
                        },
                        {"type": "text", "text": "Answer from these inputs"},
                    ],
                },
            ],
        }
    )
    assert EvaluationCase.model_validate_json(case.model_dump_json()) == case
    openai = _openai_input(case)
    anthropic = _anthropic_messages(case)
    assert isinstance(openai, list)
    assert openai[2]["content"][0]["image_url"].startswith("data:image/png;base64,")
    assert openai[2]["content"][1]["filename"] == "evidence.pdf"
    assert anthropic[2]["content"][0]["source"]["data"] == image
    assert anthropic[2]["content"][1]["type"] == "document"

    legacy = EvaluationSuite.model_validate(
        {
            "schema_version": "1",
            "name": "legacy-text-only",
            "migration_manifest_sha256": "0" * 64,
            "source": {
                "model": "source",
                "provider": "anthropic",
                "platform": "anthropic-api",
                "model_id": "source-id",
            },
            "target": {
                "model": "target",
                "provider": "openai",
                "platform": "openai-api",
                "model_id": "target-id",
            },
            "cases": [{"id": "legacy", "input": "text-only input"}],
        }
    )
    assert legacy.schema_version == "1"
    assert legacy.cases[0].input == "text-only input"

    with pytest.raises(ValidationError, match="input or messages"):
        EvaluationCase(id="empty")
    with pytest.raises(ValidationError, match="either input or messages"):
        EvaluationCase(
            id="both",
            input="text",
            messages=[
                EvaluationMessage(role="user", content=[EvaluationTextContent(text="duplicate")])
            ],
        )
    with pytest.raises(ValidationError, match="valid base64"):
        EvaluationCase.model_validate(
            {
                "id": "bad-media",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "media_type": "image/png", "data": "%%%"}],
                    }
                ],
            }
        )
    with pytest.raises(ValidationError, match="decode as UTF-8"):
        EvaluationCase.model_validate(
            {
                "id": "bad-text-document",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "media_type": "text/plain",
                                "data": b64encode(b"\xff").decode(),
                            },
                            {"type": "text", "text": "Read this"},
                        ],
                    }
                ],
            }
        )


class FakeBedrockClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return {
            "output": {
                "message": {
                    "content": [
                        {"text": "ok"},
                        {"toolUse": {"name": "lookup", "input": {"id": 1}}},
                    ]
                }
            },
            "usage": {"inputTokens": 9, "outputTokens": 3},
            "stopReason": "end_turn",
        }


def test_bedrock_executor_uses_converse_and_normalizes_outputs() -> None:
    client = FakeBedrockClient()
    ticks = iter([1.0, 1.1])
    executor = BedrockEvaluationExecutor(client, monotonic=ticks.__next__)
    config = EvaluationRunConfig(
        provider="anthropic",
        platform="amazon-bedrock",
        model="claude-sonnet-5",
        model_id="us.anthropic.claude-sonnet-5-v1:0",
        parameters={"maxTokens": 128, "region_name": "us-west-2"},
    )
    result = executor.execute(config, EvaluationCase(id="bedrock", input="hello"))
    assert result.status is EvaluationStatus.PASSED
    assert result.output == "ok"
    assert result.tool_calls == [{"name": "lookup", "arguments": {"id": 1}}]
    assert result.input_tokens == 9
    assert client.requests == [
        {
            "modelId": "us.anthropic.claude-sonnet-5-v1:0",
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
            "inferenceConfig": {"maxTokens": 128},
        }
    ]

    class SecretFailureClient:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("AKIA-secret-value")

    failure = BedrockEvaluationExecutor(
        SecretFailureClient(), monotonic=iter([2.0, 2.1]).__next__
    ).execute(config, EvaluationCase(id="safe-error", input="hello"))
    assert failure.status is EvaluationStatus.ERROR
    assert failure.error_message == "Bedrock request failed with RuntimeError."
    assert "AKIA-secret-value" not in failure.model_dump_json()


def test_bedrock_retry_classification_is_bounded_and_other_providers_fail_closed() -> None:
    class Throttled(Exception):
        def __init__(self) -> None:
            self.response = {
                "Error": {"Code": "ThrottlingException", "Message": "AKIA-secret-value"}
            }

    client = FakeBedrockClient()
    successful_converse = client.converse
    calls = 0

    def converse(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Throttled()
        return successful_converse(**kwargs)

    client.converse = converse  # type: ignore[method-assign]
    sleeps: list[float] = []
    executor = BedrockEvaluationExecutor(
        client,
        sleeper=sleeps.append,
        monotonic=iter([1.0, 1.2]).__next__,
    )
    config = EvaluationRunConfig(
        provider="anthropic",
        platform="amazon-bedrock",
        model="claude-sonnet-5",
        model_id="anthropic.claude-sonnet-5",
        max_retries=1,
    )
    result = executor.execute(config, EvaluationCase(id="retry", input="hello"))
    assert result.status is EvaluationStatus.PASSED
    assert result.attempts == 2
    assert sleeps == [1]
    assert "AKIA-secret-value" not in result.model_dump_json()

    unsupported = BuiltinEvaluationExecutor(bedrock=executor).execute(
        config.model_copy(update={"provider": "openai"}),
        EvaluationCase(id="unsupported", input="hello"),
    )
    assert unsupported.error_kind == "unsupported"


@pytest.mark.skipif(
    not os.environ.get("LLM_MIGRATE_BEDROCK_TEST_MODEL_ID"),
    reason="set LLM_MIGRATE_BEDROCK_TEST_MODEL_ID for the optional live contract test",
)
def test_optional_bedrock_live_contract_uses_local_aws_credentials() -> None:
    parameters: dict[str, Any] = {"maxTokens": 16}
    if region := os.environ.get("AWS_REGION"):
        parameters["region_name"] = region
    config = EvaluationRunConfig(
        provider="anthropic",
        platform="amazon-bedrock",
        model="credential-gated-contract",
        model_id=os.environ["LLM_MIGRATE_BEDROCK_TEST_MODEL_ID"],
        parameters=parameters,
        timeout_seconds=30,
        max_retries=1,
    )
    result = BedrockEvaluationExecutor().execute(
        config, EvaluationCase(id="live-contract", input="Reply with the word ok.")
    )
    assert result.status is EvaluationStatus.PASSED, result.error_message
    assert "credential" not in result.model_dump_json().lower()


def test_registered_evaluator_is_available_to_shared_cli_mcp_service_path(
    service: MigrationService, project_root: Path
) -> None:
    case = EvaluationCase(
        id="allowlisted",
        input="answer",
        validators=[
            EvaluationValidator(
                kind=EvaluatorKind.CUSTOM,
                name="trusted",
                evaluator_name="trusted.local",
            )
        ],
    )

    def trusted(evaluation_case: EvaluationCase, output: str) -> EvaluatorResult:
        return EvaluatorResult(
            name="trusted",
            kind=EvaluatorKind.CUSTOM,
            status=EvaluationStatus.PASSED,
            score=1,
            message=f"Checked {evaluation_case.id}: {len(output)} characters.",
        )

    register_evaluator("trusted.local", trusted)
    try:
        suite = service.generate_eval_suite(_plan(service, project_root), [case])
        source, target = _configs()
        run = service.run_migration_eval(
            suite,
            source,
            target,
            executor=FakeExecutor(
                {(source.model, case.id): "source", (target.model, case.id): "target"}
            ),
        )
    finally:
        unregister_evaluator("trusted.local")
    assert run.source_results[0].status is EvaluationStatus.PASSED
    assert run.warnings == []
    assert "import" not in json.dumps(run.model_dump(mode="json"))

    invalid_case = EvaluationCase(
        id="invalid-evaluator-result",
        input="answer",
        validators=[
            EvaluationValidator(
                kind=EvaluatorKind.CUSTOM,
                name="invalid",
                evaluator_name="invalid.local",
            )
        ],
    )

    def invalid_result(evaluation_case: EvaluationCase, output: str) -> Any:
        return {"secret": "must-not-leak"}

    register_evaluator("invalid.local", invalid_result)
    try:
        invalid_suite = service.generate_eval_suite(_plan(service, project_root), [invalid_case])
        invalid_run = service.run_migration_eval(
            invalid_suite,
            source,
            target,
            executor=FakeExecutor(
                {
                    (source.model, invalid_case.id): "source",
                    (target.model, invalid_case.id): "target",
                }
            ),
        )
    finally:
        unregister_evaluator("invalid.local")
    invalid_evaluation = invalid_run.source_results[0].evaluator_results[0]
    assert invalid_evaluation.status is EvaluationStatus.ERROR
    assert invalid_evaluation.message == "Evaluator failed with ValidationError."
    assert "must-not-leak" not in invalid_run.model_dump_json()


def test_installed_evaluator_entry_points_require_an_explicit_local_allowlist() -> None:
    loaded: list[str] = []

    def trusted(evaluation_case: EvaluationCase, output: str) -> EvaluatorResult:
        return EvaluatorResult(
            name="trusted",
            kind=EvaluatorKind.CUSTOM,
            status=EvaluationStatus.PASSED,
            score=1,
            message=f"Checked {evaluation_case.id}: {len(output)} characters.",
        )

    class FakeEntryPoint:
        name = "trusted.entrypoint"

        def load(self) -> Any:
            loaded.append(self.name)
            return trusted

    point = cast(EntryPoint, FakeEntryPoint())
    configure_evaluators(["trusted.entrypoint"], entry_point_provider=lambda: [point])
    try:
        assert loaded == ["trusted.entrypoint"]
        with pytest.raises(ValueError, match="were not found"):
            configure_evaluators(["os.system"], entry_point_provider=lambda: [point])
        assert loaded == ["trusted.entrypoint"]
    finally:
        unregister_evaluator("trusted.entrypoint")


def test_optimizer_is_bounded_reproducible_and_accepts_only_no_regression_candidates(
    service: MigrationService, project_root: Path
) -> None:
    case = EvaluationCase(id="quality", input="answer", expected_output="good")
    suite = service.generate_eval_suite(_plan(service, project_root), [case])
    source, target = _configs()

    def times() -> datetime:
        return datetime(2026, 8, 23, tzinfo=UTC)

    baseline = service.run_migration_eval(
        suite,
        source,
        target,
        executor=FakeExecutor({(source.model, case.id): "good", (target.model, case.id): "bad"}),
        clock=times,
    )
    candidate = service.run_migration_eval(
        suite,
        source,
        target.model_copy(update={"parameters": {"temperature": 0}}),
        executor=FakeExecutor({(source.model, case.id): "good", (target.model, case.id): "good"}),
        clock=times,
    )
    report = service.analyze_regressions(baseline)
    limits = OptimizationLimits(
        max_candidates=2,
        max_evaluation_runs=2,
        max_estimated_cost_usd="0.001",
        minimum_target_pass_rate=1,
    )
    first = service.optimize_migration(report, {"temperature-zero": candidate}, limits=limits)
    second = service.optimize_migration(report, {"temperature-zero": candidate}, limits=limits)
    assert first == second
    assert first.selected_candidate_id == "temperature-zero"
    assert first.candidates[0].accepted
    assert first.evaluated_run_count == 1
    assert first.recommendations
    with pytest.raises(ValueError, match="optimization limits"):
        service.optimize_migration(
            report,
            {"one": candidate, "two": candidate},
            limits=OptimizationLimits(max_candidates=1, max_evaluation_runs=2),
        )

    other_case = EvaluationCase(id="other", input="different corpus", expected_output="good")
    other_suite = service.generate_eval_suite(_plan(service, project_root), [other_case])
    other_run = service.run_migration_eval(
        other_suite,
        source,
        target,
        executor=FakeExecutor(
            {
                (source.model, other_case.id): "good",
                (target.model, other_case.id): "good",
            }
        ),
        clock=times,
    )
    with pytest.raises(ValueError, match="baseline corpus"):
        service.optimize_migration(report, {"unrelated": other_run}, limits=limits)


def test_optimizer_cli_and_mcp_use_the_shared_bounded_contract(
    service: MigrationService, project_root: Path, tmp_path: Path
) -> None:
    case = EvaluationCase(id="interface", input="answer", expected_output="good")
    suite = service.generate_eval_suite(_plan(service, project_root), [case])
    source, target = _configs()

    def clock() -> datetime:
        return datetime(2026, 8, 23, tzinfo=UTC)

    baseline = service.run_migration_eval(
        suite,
        source,
        target,
        executor=FakeExecutor({(source.model, case.id): "good", (target.model, case.id): "bad"}),
        clock=clock,
    )
    candidate = service.run_migration_eval(
        suite,
        source,
        target.model_copy(update={"parameters": {"temperature": 0}}),
        executor=FakeExecutor({(source.model, case.id): "good", (target.model, case.id): "good"}),
        clock=clock,
    )
    report = service.analyze_regressions(baseline)
    report_path = tmp_path / "report.yaml"
    candidate_path = tmp_path / "temperature-zero.yaml"
    report_path.write_text(service.evaluation_artifact_as_yaml(report), encoding="utf-8")
    candidate_path.write_text(service.evaluation_artifact_as_yaml(candidate), encoding="utf-8")

    cli_result = runner.invoke(
        app,
        [
            "eval",
            "optimize",
            str(report_path),
            "--candidate-run",
            str(candidate_path),
            "--objective",
            "quality",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert yaml.safe_load(cli_result.output)["selected_candidate_id"] == "temperature-zero"

    mcp_result = mcp_optimize_migration(
        report.model_dump(mode="json"),
        {"temperature-zero": candidate.model_dump(mode="json")},
        {"objective": "quality", "max_candidates": 1, "max_evaluation_runs": 1},
    )
    assert mcp_result["selected_candidate_id"] == "temperature-zero"
