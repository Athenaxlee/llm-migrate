"""Deterministic V0.5 evaluation orchestration and regression analysis."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from llm_migrate.core.evaluation_models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationRunConfig,
    EvaluationStatus,
    EvaluationSuite,
    EvaluationValidator,
    EvaluatorKind,
    EvaluatorResult,
    MigrationEvalRun,
    OptimizationCandidateComparison,
    OptimizationLimits,
    OptimizationRecommendation,
    OptimizationResult,
    OutputComparison,
    ProviderCallResult,
    RegressionCategory,
    RegressionFinding,
    RegressionReport,
    RegressionSummary,
)
from llm_migrate.core.models import MigrationPlan


class EvaluationExecutor(Protocol):
    """Provider boundary used by runtime evaluation and deterministic fakes."""

    def execute(self, config: EvaluationRunConfig, case: EvaluationCase) -> ProviderCallResult: ...


CustomEvaluator = Callable[[EvaluationCase, str], EvaluatorResult]


def manifest_sha256(plan: MigrationPlan) -> str:
    """The hash an evaluation suite binds to (`migration_manifest_sha256`)."""
    return _manifest_hash(plan)


def _manifest_hash(plan: MigrationPlan) -> str:
    """Hash of the plan's DELIVERABLE-relevant content.

    An unknown's `action` carries the run directory (moving a run must not
    invalidate its evaluation) and an observation only changes an unknown's
    `status`/`closed_reason` (recording one must not mark a BYOK evaluation
    stale): those fields are excluded, so the binding tracks what the
    migration changes, not how it is being reviewed.
    """
    stable = plan.model_copy(
        update={
            "unknowns": [
                item.model_copy(update={"action": "", "status": "open", "closed_reason": None})
                for item in plan.unknowns
            ]
        }
    )
    payload = stable.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _suite_hash(suite: EvaluationSuite) -> str:
    return _stable_hash(suite.model_dump(mode="json"))


def generate_eval_suite(
    plan: MigrationPlan,
    cases: list[EvaluationCase],
    *,
    name: str = "migration-evaluation",
) -> EvaluationSuite:
    """Bind a user corpus to the canonical migration manifest."""
    normalized: list[EvaluationCase] = []
    warnings: list[str] = []
    output_schemas = [
        contract.json_schema
        for invocation in plan.invocation_changes
        for contract in invocation.source_analysis.structured_outputs
        if contract.json_schema is not None
    ]
    if len(output_schemas) > 1:
        warnings.append(
            "Multiple structured-output schemas were detected; add an explicit JSON Schema "
            "validator to each applicable case."
        )
    for case in cases:
        validators = list(case.validators)
        if case.expected_output is not None and not any(
            validator.kind is EvaluatorKind.EXACT_MATCH for validator in validators
        ):
            validators.append(
                EvaluationValidator(kind=EvaluatorKind.EXACT_MATCH, name="golden-output")
            )
        if case.expected_tool_calls is not None and not any(
            validator.kind is EvaluatorKind.TOOL_CALLS for validator in validators
        ):
            validators.append(
                EvaluationValidator(kind=EvaluatorKind.TOOL_CALLS, name="expected-tool-calls")
            )
        if len(output_schemas) == 1 and not any(
            validator.kind is EvaluatorKind.JSON_SCHEMA for validator in validators
        ):
            validators.append(
                EvaluationValidator(
                    kind=EvaluatorKind.JSON_SCHEMA,
                    name="detected-output-schema",
                    schema_definition=output_schemas[0],
                )
            )
        if not validators:
            warnings.append(
                f"Case {case.id!r} has no deterministic validator and will be marked skipped "
                "unless a custom evaluator is added."
            )
        normalized.append(case.model_copy(update={"validators": validators}))
    return EvaluationSuite(
        name=name,
        migration_manifest_sha256=_manifest_hash(plan),
        source=plan.source,
        target=plan.target,
        cases=normalized,
        warnings=warnings,
    )


def _run_validator(
    validator: EvaluationValidator,
    case: EvaluationCase,
    output: str,
    tool_calls: list[dict[str, Any]],
    evaluators: Mapping[str, CustomEvaluator],
) -> EvaluatorResult:
    if validator.kind is EvaluatorKind.EXACT_MATCH:
        expected = case.expected_output
        actual: Any = output
        if not isinstance(expected, str):
            try:
                actual = json.loads(output)
            except json.JSONDecodeError:
                actual = output
        passed = actual == expected
        return EvaluatorResult(
            name=validator.name,
            kind=validator.kind,
            status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
            score=1.0 if passed else 0.0,
            message="Output exactly matched the golden value." if passed else "Golden mismatch.",
        )
    if validator.kind is EvaluatorKind.REGEX:
        assert validator.pattern is not None
        try:
            passed = re.search(validator.pattern, output) is not None
        except re.error as exc:
            return EvaluatorResult(
                name=validator.name,
                kind=validator.kind,
                status=EvaluationStatus.ERROR,
                message=f"Invalid regular expression: {exc}",
            )
        return EvaluatorResult(
            name=validator.name,
            kind=validator.kind,
            status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
            score=1.0 if passed else 0.0,
            message="Pattern matched." if passed else "Pattern did not match.",
        )
    if validator.kind is EvaluatorKind.JSON_SCHEMA:
        assert validator.schema_definition is not None
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            errors = [f"Output is not valid JSON: {exc.msg}"]
        else:
            try:
                Draft202012Validator.check_schema(validator.schema_definition)
            except SchemaError as exc:
                return EvaluatorResult(
                    name=validator.name,
                    kind=validator.kind,
                    status=EvaluationStatus.ERROR,
                    message=f"Invalid evaluator JSON Schema: {exc.message}",
                )
            schema_validator = Draft202012Validator(validator.schema_definition)
            errors = [error.message for error in schema_validator.iter_errors(value)]
        passed = not errors
        return EvaluatorResult(
            name=validator.name,
            kind=validator.kind,
            status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
            score=1.0 if passed else 0.0,
            message="JSON output satisfies the schema." if passed else "; ".join(errors[:5]),
        )
    if validator.kind is EvaluatorKind.TOOL_CALLS:
        passed = tool_calls == (case.expected_tool_calls or [])
        return EvaluatorResult(
            name=validator.name,
            kind=validator.kind,
            status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
            score=1.0 if passed else 0.0,
            message="Tool calls matched." if passed else "Tool calls did not match.",
        )
    evaluator_name = validator.evaluator_name
    assert evaluator_name is not None
    evaluator = evaluators.get(evaluator_name)
    if evaluator is None:
        return EvaluatorResult(
            name=validator.name,
            kind=validator.kind,
            status=EvaluationStatus.SKIPPED,
            message=f"Evaluator {evaluator_name!r} was not configured.",
        )
    try:
        result = EvaluatorResult.model_validate(evaluator(case, output))
    except Exception as exc:  # evaluator plugins are an explicit trust boundary
        return EvaluatorResult(
            name=validator.name,
            kind=validator.kind,
            status=EvaluationStatus.ERROR,
            message=f"Evaluator failed with {type(exc).__name__}.",
        )
    if result.kind is not validator.kind:
        return EvaluatorResult(
            name=validator.name,
            kind=validator.kind,
            status=EvaluationStatus.ERROR,
            message="Evaluator returned an incompatible evaluator kind.",
        )
    if result.score is not None and result.score < validator.minimum_score:
        return result.model_copy(update={"status": EvaluationStatus.FAILED})
    return result


def _estimated_cost(config: EvaluationRunConfig, call: ProviderCallResult) -> Decimal | None:
    parts: list[Decimal] = []
    if call.input_tokens is not None and config.input_price_per_million is not None:
        parts.append(
            Decimal(call.input_tokens) * config.input_price_per_million / Decimal(1_000_000)
        )
    if call.output_tokens is not None and config.output_price_per_million is not None:
        parts.append(
            Decimal(call.output_tokens) * config.output_price_per_million / Decimal(1_000_000)
        )
    return sum(parts, Decimal(0)) if parts else None


def _evaluate_case(
    case: EvaluationCase,
    configuration: Literal["source", "target"],
    config: EvaluationRunConfig,
    executor: EvaluationExecutor,
    evaluators: Mapping[str, CustomEvaluator],
) -> EvaluationCaseResult:
    call = executor.execute(config, case)
    if call.status is EvaluationStatus.ERROR or call.output is None:
        return EvaluationCaseResult(
            case_id=case.id,
            configuration=configuration,
            status=EvaluationStatus.ERROR,
            latency_ms=call.latency_ms,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            estimated_cost_usd=_estimated_cost(config, call),
            attempts=call.attempts,
            error_kind=call.error_kind or "provider_error",
            error_message=call.error_message or "Provider execution failed.",
        )
    results = [
        _run_validator(validator, case, call.output, call.tool_calls, evaluators)
        for validator in case.validators
    ]
    scored = [result.score for result in results if result.score is not None]
    score = sum(scored) / len(scored) if scored else None
    statuses = {result.status for result in results}
    if EvaluationStatus.ERROR in statuses:
        status = EvaluationStatus.ERROR
    elif EvaluationStatus.FAILED in statuses:
        status = EvaluationStatus.FAILED
    elif EvaluationStatus.PASSED in statuses:
        status = EvaluationStatus.PASSED
    else:
        status = EvaluationStatus.SKIPPED
    return EvaluationCaseResult(
        case_id=case.id,
        configuration=configuration,
        status=status,
        output=call.output,
        tool_calls=call.tool_calls,
        refusal_detected=call.refusal_detected,
        evaluator_results=results,
        task_score=score,
        latency_ms=call.latency_ms,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        estimated_cost_usd=_estimated_cost(config, call),
        attempts=call.attempts,
    )


def run_migration_eval(
    suite: EvaluationSuite,
    source_config: EvaluationRunConfig,
    target_config: EvaluationRunConfig,
    executor: EvaluationExecutor,
    *,
    evaluators: Mapping[str, CustomEvaluator] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> MigrationEvalRun:
    """Run the same immutable corpus against source and target configurations."""
    suite = EvaluationSuite.model_validate(suite.model_dump())
    source_config = EvaluationRunConfig.model_validate(source_config.model_dump())
    target_config = EvaluationRunConfig.model_validate(target_config.model_dump())
    if (
        source_config.model,
        source_config.provider,
        source_config.platform,
        source_config.model_id,
    ) != (
        suite.source.model,
        suite.source.provider,
        suite.source.platform,
        suite.source.model_id,
    ):
        raise ValueError("source evaluation config does not match the migration suite endpoint")
    if (
        target_config.model,
        target_config.provider,
        target_config.platform,
        target_config.model_id,
    ) != (
        suite.target.model,
        suite.target.provider,
        suite.target.platform,
        suite.target.model_id,
    ):
        raise ValueError("target evaluation config does not match the migration suite endpoint")
    now = clock or (lambda: datetime.now(UTC))
    started = now()
    plugins = evaluators or {}
    source_results: list[EvaluationCaseResult] = []
    target_results: list[EvaluationCaseResult] = []
    for case in suite.cases:
        source_results.append(_evaluate_case(case, "source", source_config, executor, plugins))
        target_results.append(_evaluate_case(case, "target", target_config, executor, plugins))
    missing = sorted(
        {
            validator.evaluator_name
            for case in suite.cases
            for validator in case.validators
            if validator.evaluator_name and validator.evaluator_name not in plugins
        }
    )
    warnings = list(suite.warnings)
    if missing:
        warnings.append("Unconfigured evaluators were skipped: " + ", ".join(missing) + ".")
    return MigrationEvalRun(
        suite=suite,
        source_config=source_config,
        target_config=target_config,
        started_at=started,
        completed_at=now(),
        source_results=source_results,
        target_results=target_results,
        warnings=warnings,
    )


_OPTIMIZATION_GUIDANCE: dict[
    RegressionCategory,
    tuple[Literal["prompt", "parameter", "model", "tool_schema", "output_contract"], str],
] = {
    RegressionCategory.FORMAT_SCHEMA_PARSER: (
        "output_contract",
        "Align the target output schema and parser contract.",
    ),
    RegressionCategory.HALLUCINATION_GROUNDING: (
        "prompt",
        "Strengthen grounding instructions and required evidence boundaries.",
    ),
    RegressionCategory.INSTRUCTION_ADHERENCE: (
        "prompt",
        "Make the regressed instruction explicit and move it to the target system channel.",
    ),
    RegressionCategory.REASONING: (
        "parameter",
        "Review target reasoning controls and token budget.",
    ),
    RegressionCategory.TOOL_USE: (
        "tool_schema",
        "Review target tool schema, choice policy, and argument constraints.",
    ),
    RegressionCategory.REFUSAL_SAFETY: (
        "prompt",
        "Review safety-sensitive wording and preserve the intended policy boundary.",
    ),
    RegressionCategory.VERBOSITY: (
        "parameter",
        "Constrain response length in the prompt or target token limit.",
    ),
    RegressionCategory.CONTEXT_LOSS: (
        "model",
        "Use a target with sufficient context or reduce the supplied context deterministically.",
    ),
    RegressionCategory.PARAMETER_CAPABILITY: (
        "model",
        "Choose a compatible target or remove the unsupported control.",
    ),
    RegressionCategory.LATENCY_COST: (
        "parameter",
        "Reduce output/token budgets or compare a lower-cost target model.",
    ),
    RegressionCategory.UNCLEAR: (
        "prompt",
        "Add a deterministic evaluator before changing the migration.",
    ),
}


def optimize_migration(
    report: RegressionReport,
    candidate_runs: Mapping[str, MigrationEvalRun] | None = None,
    *,
    limits: OptimizationLimits | None = None,
) -> OptimizationResult:
    """Produce bounded, review-only changes and compare explicitly supplied candidate runs."""
    bounded = limits or OptimizationLimits()
    runs = dict(candidate_runs or {})
    if len(runs) > bounded.max_candidates or len(runs) > bounded.max_evaluation_runs:
        raise ValueError("candidate runs exceed configured optimization limits")
    recommendations: list[OptimizationRecommendation] = []
    grouped: dict[RegressionCategory, list[str]] = {}
    for finding in report.regressions:
        grouped.setdefault(finding.category, []).append(finding.case_id)
    for category, case_ids in grouped.items():
        action, summary = _OPTIMIZATION_GUIDANCE[category]
        recommendations.append(
            OptimizationRecommendation(
                category=category,
                action=action,
                summary=summary,
                rationale=f"Observed {category.value} in {len(set(case_ids))} case(s).",
                affected_cases=sorted(set(case_ids)),
            )
        )
    candidates: list[OptimizationCandidateComparison] = []
    for candidate_id, run in sorted(runs.items()):
        if run.suite.source != report.source or run.suite.target != report.target:
            raise ValueError(f"candidate {candidate_id!r} does not match regression endpoints")
        if report.evaluation_suite_sha256 is None:
            raise ValueError(
                "candidate comparison requires a regression report with an evaluation suite hash"
            )
        if _suite_hash(run.suite) != report.evaluation_suite_sha256:
            raise ValueError(f"candidate {candidate_id!r} does not use the baseline corpus")
        if (
            report.migration_manifest_sha256 is not None
            and run.suite.migration_manifest_sha256 != report.migration_manifest_sha256
        ):
            raise ValueError(f"candidate {candidate_id!r} does not use the baseline manifest")
        candidate_report = analyze_regressions(run)
        target_results = run.target_results
        mean_latency = (
            sum(result.latency_ms for result in target_results) / len(target_results)
            if target_results
            else 0
        )
        costs = [result.estimated_cost_usd for result in target_results]
        total_cost = (
            sum((cost for cost in costs if cost is not None), Decimal(0))
            if costs and all(cost is not None for cost in costs)
            else None
        )
        reasons: list[str] = []
        if candidate_report.summary.target_pass_rate < bounded.minimum_target_pass_rate:
            reasons.append("target pass rate is below the configured acceptance threshold")
        if not bounded.allow_quality_regression and candidate_report.summary.regression_count:
            reasons.append("candidate contains regression signals")
        if bounded.max_estimated_cost_usd is not None:
            if total_cost is None:
                reasons.append("candidate cost is unknown")
            elif total_cost > bounded.max_estimated_cost_usd:
                reasons.append("candidate exceeds the configured cost limit")
        candidates.append(
            OptimizationCandidateComparison(
                candidate_id=candidate_id,
                config=run.target_config,
                target_pass_rate=candidate_report.summary.target_pass_rate,
                pass_rate_delta_from_baseline=(
                    candidate_report.summary.target_pass_rate - report.summary.target_pass_rate
                ),
                regression_count=candidate_report.summary.regression_count,
                mean_target_latency_ms=mean_latency,
                estimated_target_cost_usd=total_cost,
                accepted=not reasons,
                rejection_reasons=reasons,
            )
        )
    accepted = [candidate for candidate in candidates if candidate.accepted]

    def candidate_rank(item: OptimizationCandidateComparison) -> tuple[Any, ...]:
        quality = (-item.target_pass_rate, item.regression_count)
        cost = (
            item.estimated_target_cost_usd is None,
            item.estimated_target_cost_usd or Decimal(0),
        )
        latency = (item.mean_target_latency_ms,)
        if bounded.objective == "quality":
            return (*quality, *cost, *latency, item.candidate_id)
        if bounded.objective == "cost":
            return (*cost, *quality, *latency, item.candidate_id)
        if bounded.objective == "latency":
            return (*latency, *quality, *cost, item.candidate_id)
        return (*quality, *cost, *latency, item.candidate_id)

    selected = min(
        accepted,
        key=candidate_rank,
        default=None,
    )
    report_data = report.model_dump(mode="json")
    input_data = {
        "report": report_data,
        "limits": bounded.model_dump(mode="json"),
        "candidate_runs": {name: run.model_dump(mode="json") for name, run in sorted(runs.items())},
    }
    return OptimizationResult(
        regression_report_sha256=_stable_hash(report_data),
        reproducibility_sha256=_stable_hash(input_data),
        limits=bounded,
        recommendations=recommendations,
        candidates=candidates,
        selected_candidate_id=selected.candidate_id if selected else None,
        evaluated_run_count=len(runs),
        warnings=[]
        if recommendations or candidates
        else ["No regressions or candidate runs were supplied."],
    )


def compare_outputs(source: EvaluationCaseResult, target: EvaluationCaseResult) -> OutputComparison:
    if source.case_id != target.case_id:
        raise ValueError("source and target results must refer to the same case")
    score_delta = (
        target.task_score - source.task_score
        if source.task_score is not None and target.task_score is not None
        else None
    )
    source_tokens = (
        source.input_tokens + source.output_tokens
        if source.input_tokens is not None and source.output_tokens is not None
        else None
    )
    target_tokens = (
        target.input_tokens + target.output_tokens
        if target.input_tokens is not None and target.output_tokens is not None
        else None
    )
    token_delta = (
        target_tokens - source_tokens
        if source_tokens is not None and target_tokens is not None
        else None
    )
    cost_delta = (
        target.estimated_cost_usd - source.estimated_cost_usd
        if source.estimated_cost_usd is not None and target.estimated_cost_usd is not None
        else None
    )
    signals: list[str] = []
    quality_regression = source.status is EvaluationStatus.PASSED and target.status in {
        EvaluationStatus.FAILED,
        EvaluationStatus.ERROR,
    }
    if quality_regression:
        signals.append("source_passed_target_failed")
    if score_delta is not None and score_delta < 0:
        signals.append("task_score_decreased")
        quality_regression = True
    if source.tool_calls != target.tool_calls:
        signals.append("tool_calls_changed")
        quality_regression = True
    if target.refusal_detected and not source.refusal_detected:
        signals.append("target_only_refusal")
        quality_regression = True
    excessive_verbosity = (
        source.output is not None
        and target.output is not None
        and len(target.output) > max(100, int(len(source.output) * 1.5))
    )
    reduced_verbosity = (
        source.output is not None
        and target.output is not None
        and len(source.output) > 100
        and len(target.output) < len(source.output) * 0.5
    )
    verbosity_regression = excessive_verbosity or reduced_verbosity
    if excessive_verbosity:
        signals.append("output_length_increased_over_50_percent")
    if reduced_verbosity:
        signals.append("output_length_decreased_over_50_percent")
    latency_regression = (
        target.latency_ms > source.latency_ms * 1.5 and target.latency_ms - source.latency_ms > 100
    )
    if latency_regression:
        signals.append("latency_increased_over_50_percent")
    token_regression = (
        source_tokens is not None
        and target_tokens is not None
        and target_tokens > source_tokens * 1.25
    )
    if token_regression:
        signals.append("token_usage_increased_over_25_percent")
    cost_regression = (
        source.estimated_cost_usd is not None
        and target.estimated_cost_usd is not None
        and target.estimated_cost_usd > source.estimated_cost_usd * Decimal("1.25")
    )
    if cost_regression:
        signals.append("estimated_cost_increased_over_25_percent")
    return OutputComparison(
        case_id=source.case_id,
        source_status=source.status,
        target_status=target.status,
        source_score=source.task_score,
        target_score=target.task_score,
        score_delta=score_delta,
        outputs_equal=(source.output == target.output)
        if source.output is not None and target.output is not None
        else None,
        source_latency_ms=source.latency_ms,
        target_latency_ms=target.latency_ms,
        latency_delta_ms=target.latency_ms - source.latency_ms,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        token_delta=token_delta,
        source_cost_usd=source.estimated_cost_usd,
        target_cost_usd=target.estimated_cost_usd,
        cost_delta_usd=cost_delta,
        regression=(
            quality_regression
            or verbosity_regression
            or latency_regression
            or token_regression
            or cost_regression
        ),
        signals=signals,
    )


def _pass_rate(results: list[EvaluationCaseResult]) -> float:
    return sum(result.status is EvaluationStatus.PASSED for result in results) / len(results)


def _validator_categories(target: EvaluationCaseResult) -> set[RegressionCategory]:
    categories: set[RegressionCategory] = set()
    for result in target.evaluator_results:
        if result.status not in {EvaluationStatus.FAILED, EvaluationStatus.ERROR}:
            continue
        lowered = f"{result.name} {result.message}".lower()
        if result.kind in {EvaluatorKind.JSON_SCHEMA, EvaluatorKind.REGEX}:
            categories.add(RegressionCategory.FORMAT_SCHEMA_PARSER)
        elif "ground" in lowered or "hallucin" in lowered:
            categories.add(RegressionCategory.HALLUCINATION_GROUNDING)
        elif "instruction" in lowered:
            categories.add(RegressionCategory.INSTRUCTION_ADHERENCE)
        elif "reason" in lowered:
            categories.add(RegressionCategory.REASONING)
        elif "tool" in lowered:
            categories.add(RegressionCategory.TOOL_USE)
        elif "refusal" in lowered or "safety" in lowered:
            categories.add(RegressionCategory.REFUSAL_SAFETY)
        elif "context" in lowered or "truncat" in lowered:
            categories.add(RegressionCategory.CONTEXT_LOSS)
    return categories


def analyze_regressions(run: MigrationEvalRun) -> RegressionReport:
    """Compare paired results and categorize observed target regressions."""
    if len(run.source_results) != len(run.target_results):
        raise ValueError("evaluation run has unpaired source and target results")
    comparisons = [
        compare_outputs(source, target)
        for source, target in zip(run.source_results, run.target_results, strict=True)
    ]
    findings: list[RegressionFinding] = []
    for source, target, comparison in zip(
        run.source_results, run.target_results, comparisons, strict=True
    ):
        if not comparison.regression:
            continue
        categories = _validator_categories(target)
        if source.tool_calls != target.tool_calls and source.status is EvaluationStatus.PASSED:
            categories.add(RegressionCategory.TOOL_USE)
        if target.refusal_detected and not source.refusal_detected:
            categories.add(RegressionCategory.REFUSAL_SAFETY)
        if any(
            signal in comparison.signals
            for signal in {
                "output_length_increased_over_50_percent",
                "output_length_decreased_over_50_percent",
            }
        ):
            categories.add(RegressionCategory.VERBOSITY)
        if target.status is EvaluationStatus.ERROR:
            if target.error_kind in {"invalid_request", "unsupported"}:
                category = RegressionCategory.PARAMETER_CAPABILITY
            elif target.error_kind == "invalid_output":
                category = RegressionCategory.FORMAT_SCHEMA_PARSER
            else:
                category = RegressionCategory.UNCLEAR
            categories.add(category)
        if any(
            signal in comparison.signals
            for signal in {
                "latency_increased_over_50_percent",
                "token_usage_increased_over_25_percent",
                "estimated_cost_increased_over_25_percent",
            }
        ):
            categories.add(RegressionCategory.LATENCY_COST)
        if not categories:
            categories.add(RegressionCategory.UNCLEAR)
        evidence = list(comparison.signals)
        evidence.extend(
            result.message
            for result in target.evaluator_results
            if result.status in {EvaluationStatus.FAILED, EvaluationStatus.ERROR}
        )
        for category in sorted(categories, key=str):
            findings.append(
                RegressionFinding(
                    case_id=source.case_id,
                    category=category,
                    severity="blocker" if target.status is EvaluationStatus.ERROR else "regression",
                    summary=f"Target regression categorized as {category.value}.",
                    evidence=evidence,
                )
            )
    source_pass_rate = _pass_rate(run.source_results)
    target_pass_rate = _pass_rate(run.target_results)
    return RegressionReport(
        suite_name=run.suite.name,
        evaluation_suite_sha256=_suite_hash(run.suite),
        migration_manifest_sha256=run.suite.migration_manifest_sha256,
        source=run.suite.source,
        target=run.suite.target,
        summary=RegressionSummary(
            case_count=len(run.suite.cases),
            source_pass_rate=source_pass_rate,
            target_pass_rate=target_pass_rate,
            pass_rate_delta=target_pass_rate - source_pass_rate,
            regression_count=sum(comparison.regression for comparison in comparisons),
            source_error_count=sum(
                result.status is EvaluationStatus.ERROR for result in run.source_results
            ),
            target_error_count=sum(
                result.status is EvaluationStatus.ERROR for result in run.target_results
            ),
            source_refusal_rate=sum(result.refusal_detected for result in run.source_results)
            / len(run.source_results),
            target_refusal_rate=sum(result.refusal_detected for result in run.target_results)
            / len(run.target_results),
        ),
        comparisons=comparisons,
        regressions=findings,
        warnings=run.warnings,
    )


def evaluation_artifact_as_yaml(
    value: EvaluationSuite | MigrationEvalRun | RegressionReport | OptimizationResult,
) -> str:
    return yaml.safe_dump(value.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
