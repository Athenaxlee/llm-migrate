# Changelog

All notable changes are documented here. The project follows semantic
versioning.

## Unreleased

- Added the V1.2 guided migration run workspace: `start_migration` creates a
  per-run workspace (default `<application>/.llm-migrate/runs/<run-id>/`),
  decides whether research is needed with explicit reasons, and returns
  ordered next steps.
- Added lenient registry-first model matching: Bedrock cross-region
  inference-profile prefixes, version suffixes, and vague platform names are
  normalized deterministically, with ranked confirmation candidates instead of
  hard failures (`match_model`, `llm-migrate models match`, MCP
  `resolve_model`).
- Added deterministic, scope-isolated researcher/reviewer prompt rendering
  from the research request (`get_research_prompts`, `llm-migrate research
  prompts`).
- Added adaptation deliverables: a per-file worklist, validated submissions of
  adapted prompts (`output/prompts/`) and complete adapted application files
  (`output/files/`), and a finalize step writing the manifest, change log, and
  a report with per-file changes, rationale, and coverage gaps.
- Endpoint context now flows through planning, invocation preparation, and
  prompt validation, fixing ambiguity failures for models with several
  same-platform representations; ambiguous detected in-app model IDs no longer
  crash the invocation consistency check.
- The application scanner ignores `.llm-migrate` run workspaces.
- Documented an agent-driven installation path in the README.

## 1.1.0 — 2026-08-30

- Added bounded, user-hosted agent research for missing or stale model knowledge.
- Added independent claim-level evidence review and deterministic consensus.
- Added immutable, expiring session registry overlays.
- Added CLI and MCP research-stage operations and session-based planning.
- Added cited-source refetching without storing full webpage content.
- Reworked the README around the agent-hosted MCP workflow.

## 1.0.0 — 2026-08-29

- Stabilized the end-to-end local migration toolkit.
- Added application scanning, model intelligence, prompt and invocation
  preparation, planning, reporting, evaluation, regression analysis, bounded
  optimization, lifecycle checks, and workload cost estimation.
