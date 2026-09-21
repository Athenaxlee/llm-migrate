# Changelog

All notable changes are documented here. The project follows semantic
versioning.

## Unreleased

- Evidence-backed adaptation review: every changed submission documents each
  edit as an anchored, evidence-linked annotated change (operation, exact
  original/adapted spans, a one-sentence why, and evidence entries — a
  `mechanical` kind covers typo-level fixes), reconciled fail-closed against
  the real diff of the decoded runtime values: undocumented edits, phantom
  claims, duplicate annotations, unresolved anchors, and missing evidence are
  rejected, and evidence URLs are checked against the plan's known sources.
  Guidance accountability: every prompt-task guidance item and file-task
  required change carries a stable content-derived id and must be disposed
  per submission (applied / not applicable / declined with a note);
  `deterministic_candidate` is renamed `verbatim_source` and every surface
  states it is the unmodified original, never a proposed adaptation.
  Deliverables reviewed with `unchanged=true` render under an explicit
  "no change needed" heading with their reasoning, and finalization counts
  `reviewed_unchanged` separately.
- Per-change review decisions: new `get_change_review` /
  `record_change_decision` MCP tools and `llm-migrate run review` /
  `run decide-change` CLI commands present, per deliverable, the decoded
  unified diff plus every annotated change with its why, evidence,
  before/after spans, and status; the user accepts or rejects each change
  individually and the agent presents verbatim, never decides. Decisions are
  durable in the run's `change-decisions.yaml`, keyed to the submission
  fingerprints (a resubmission makes them stale — reported, never silently
  applied — and an identical resubmission re-applies the still-live
  rejections); every decision deterministically regenerates the deliverable
  from the original content, the preserved as-submitted copy, and the live
  rejections, refusing non-deterministic cases instead of recording them.
  Finalization reports undecided changes as their own bucket and the report
  renders each change's decision with a per-file review outcome.
  Schema versions: `changes.yaml` 2 (version-1 logs still load),
  `AdaptationTaskList` 3, `MigrationRunFinalization` 4.

- Interactive blocker resolution: blockers are structured `MigrationBlocker`
  records (stable deterministic id, code, category, message, registry evidence
  URLs, source locations) — `MigrationPlan` is schema version 4 and
  `InvocationMigrationSpec` schema version 3. New `get_blocker_resolutions` /
  `record_blocker_decision` MCP tools and `llm-migrate run blockers` /
  `run decide` CLI commands derive, per blocker, the question to ask the user
  plus 2–5 registry-fact-backed options (retarget to a capable endpoint or
  model via the recommendation engine, an evidence-linked redesign task, an
  exact source correction, or an explicit accept that requires the user's own
  rationale and is never a default). The tool never chooses; the host agent
  presents questions, options, and evidence verbatim, one blocker at a time.
- Durable decisions: choices persist in the run's `decisions.yaml` and
  re-apply on every plan regeneration — retarget/correction decisions update
  the run identity registry-first, redesign decisions inject the required
  evidence-linked adaptation task into the plan and worklist, accept decisions
  downgrade the blocker to a prominently reported accepted decision (honored
  end-to-end, including at the prompt-submission gate), and a decision whose
  blocker no longer exists is reported stale, never silently applied.
  Superseded retargets are recorded history, foreign decision logs are
  refused, and template rationales are rejected.
- Honest reporting: the manifest carries the applied decisions, the report
  gains a Decisions section (accepted risk and stale decisions highlighted)
  plus a post-decision rationale line, and `finalize_migration` distinguishes
  unresolved blockers, decision-resolved blockers, and stale decisions.
  Complexity stays `blocked` only for unresolved blockers.
- Registry: the AWS model card is attached as a platform-level source on
  Claude Sonnet 5's Bedrock entries (recorded through its proposal bundle), so
  override-derived capability blockers cite their own evidence.

## 1.3.0 — 2026-09-18

- Added prompt provenance discovery: the scanner now parses YAML/JSON/TOML
  configuration structurally alongside Python and reports structured
  `PromptSource` records (components with roles, provenance chains,
  evidence-based confidence) plus a prompt-discovery coverage summary, so
  configuration-driven prompt loading (`config -> prompt path -> yaml.safe_load
  -> sys_prompt`) is resolved instead of reported as "no prompt changes".
  Bounded Python loader/path recognition (`open`, `Path` joins,
  `Path(__file__).parent`, `yaml/json/tomllib` loads, nested config
  subscripts, loader-style helpers) without executing application code.
  `ApplicationAnalysis` is schema version 3.
- Added explicit `prompt_sources` overrides on scan/plan/report/run start
  (service, CLI `--prompt-source`, MCP), persisted in the run's
  `migration.yaml`.
- The migration report's model differences are now a prioritized table with
  per-side claims hyperlinked to the registry sources that back them, and
  hyperlinked file:line locations of the code each difference affects.
  `MigrationPlan` is schema version 3.
- Prompt adaptation is minimal and evidence-based: the deterministic candidate
  is the source prompt verbatim, guidance advice carries evidence URLs, prompt
  tasks list protected structural sections, in-prompt findings, and the
  evidenced migration-knowledge differences, and submissions fail closed on
  structural drops unless `allow_restructure` records a justification.
- Runtime-aware submission integrity: validation and adaptation checks operate
  on decoded runtime prompt values, so serialization tricks (unicode escapes,
  quoting, whitespace or case-only edits) can neither hide content from
  validation nor pass as adaptations; the context-window budget applies to the
  joined decoded components; `unchanged=true` records reviewed no-change
  prompts and files (refused when the content still references the source
  model); the sanctioned source-to-target model-id swap inside prompt
  documents is allowed and noted; dynamically built requests (`**kwargs`)
  surface an explicit unknown-compatibility assessment.
- Prompt text mentioning JSON or tools now warns instead of blocking: only
  invocation evidence (a request that actually configures the native feature)
  produces blockers, so prompt-enforced JSON parsed by the application never
  falsely blocks a plan.
- Guided-workflow efficiency for host agents: shared prompt guidance is
  hoisted once per worklist (`shared_prompt_guidance`, roughly halving the
  task payload on multi-prompt applications), the workflow instructs a single
  task listing with no re-listing between submissions, and finalization
  coverage derives from the same worklist the agent received.

## 1.2.0 — 2026-09-15

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
