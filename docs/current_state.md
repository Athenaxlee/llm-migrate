# Current Project State

## Current milestone

V1: Stable End-to-End LLM Migration Toolkit — complete and released as
`v1.0.0` on 2026-08-29.

V1.1: On-Demand Agent-Researched Migration — complete and released as
`v1.1.0` on 2026-08-30. Typed contracts,
deterministic review/consensus rules, the session registry overlay, the
fake-agent end-to-end workflow, CLI/MCP stage operations, the agent-host
workflow, and bounded cited-source refetching are complete and tested; see
`docs/project_phases.md` for the milestone detail and deliberate deferrals.

V1.2: Guided Migration Run Workspace — complete and released as `v1.2.0` on
2026-09-15, built in
direct response to observed coding-agent host friction (vague identifiers
triggering unnecessary research, self-invented colliding research prompts,
scattered outputs, and no adapted-prompt/file deliverables). See the V1.2
section below and `docs/project_phases.md` §12.1.

V1.3: Prompt Provenance Discovery — implemented on 2026-09-16 in response to a
real V1.2 migration run where configuration-driven prompt loading
(`model_profiles.yaml` → prompt-path values → `yaml.safe_load` →
`sys_prompt`/`user_prompt`) was invisible to the scanner and the report showed
"Prompt changes: None" despite real prompt migration work. See the V1.3 section
below and `docs/project_phases.md` §12.2.

## Released foundation

- Git tag `v1.2.0` is the current stable release; `v1.1.0`, `v1.0.0`, and
  `v0.1.0` remain prior recorded release tags.
- The V0.1 reviewed registry, proposal workflow, provenance, freshness, model
  profiles, and directional Claude migration knowledge remain intact.
- Every non-fixture provider profile has exactly one checked-in proposal
  bundle. The three OpenAI proposals were maintainer-reviewed on 2026-08-29
  against refetched official sources and approved; their candidate profiles are
  promoted into the canonical registry.
- The checked-in model-profile JSON Schema is generated from the authoritative
  Pydantic model and protected by an exact parity test.

## V0.2 completed capabilities

### Model intelligence

- Deterministic matching of canonical names, aliases, display names, and exact
  platform model IDs.
- Resolution returns the canonical profile, match type, relevant platform
  representation, and effective capabilities after platform overrides.
- Platform-aware comparison accepts source/target platform and endpoint context.
- The required comparison surface reports explicit `same`, `different`,
  `unsupported`, and `unknown` states plus migration severity and action.
- Recommendations consume normalized application requirements and report ranked
  candidates, excluded candidates, blockers, tradeoffs, unknowns, stale topics,
  and evidence state.
- `query_live_pricing` is isolated behind an OpenRouter adapter with explicit ID
  mappings, bounded timeouts, exact matching, `Decimal` unit normalization,
  provenance, discrepancy reporting, and closed offline/schema-failure behavior.
- Live pricing remains OpenRouter-scoped evidence and never mutates the canonical
  registry.
- Comparisons and recommendations can attach an explicitly requested live-pricing
  overlay from one shared, time-stamped OpenRouter catalog snapshot.

### Application discovery

- A versioned provider-neutral `ApplicationAnalysis` schema captures findings,
  source locations, warnings, and derived `ApplicationRequirements`.
- The Python AST scanner detects Anthropic, OpenAI, and Bedrock SDK/configuration,
  model IDs, invocation sites, inline and file-backed prompts, parameters, tools,
  structured output, response parsers, streaming, multimodal inputs, and
  recognizable error handling.
- The scanner does not import or execute the application being analyzed.
- Anthropic, OpenAI, and Bedrock fixture repositories have deterministic golden
  scan outputs.
- Scan-to-requirements recommendation and scan-to-resolve-to-compare integration
  paths work without inference or network access.

### Interfaces

- The shared Python service owns V0.2 behavior.
- CLI commands expose `scan-application`, platform-aware `models resolve` and
  `compare`, application-driven `recommend`, and explicit `query-live-pricing`.
- MCP exposes `resolve_model`, `get_model_profile`, `compare_models`,
  `recommend_models`, `query_live_pricing`, and `scan_application` as thin tools.

## V0.3 completed capabilities

### Prompt migration

- Prompt analysis emits a versioned, provider-neutral intent representation for
  objectives, contracts, instructions, examples, grounding, reasoning, tools,
  and verbosity policies.
- Prompt preparation emits a source hash/path/role, review candidate, explicit
  semantic diff states, rationale, risks, and validation recommendations.
- Source/target provider and platform context drives explicit system-message
  transport mapping, and all registry prompt-guidance categories participate in
  transferable-assumption analysis.
- Static prompt validation checks empty input, context capacity, tool and
  structured-output capability, reasoning instructions, and target platform
  compatibility without making a model call.

### Invocation and adjacent contracts

- `ApplicationAnalysis` schema version 2 preserves normalized tool definitions,
  JSON Schemas, structured-output contracts, and multimodal payload formats in
  addition to source-located findings.
- Invocation analysis normalizes detected SDKs, operations, model IDs, prompt
  fields, typed parameters, streaming, reasoning controls, multimodal payloads,
  tool contracts, output schemas, parser assumptions, strictness, and locations.
- Invocation preparation produces a target SDK/client/operation representation,
  exact platform model ID, multimodal payload mappings, deterministic parameter
  mappings, required changes, warnings, and blockers.
- Supported mappings cover direct Anthropic to OpenAI, direct Anthropic to
  Amazon Bedrock Claude, and Bedrock Claude to direct Anthropic.
- Tool and structured-output analysis uses explicit compatibility states;
  schema/choice/strictness differences require review, while unsupported
  capabilities and missing platform adapters are blockers.
- Declared source model/provider/platform context is checked against scan results.
  Unknown parameter facts do not emit false target names; unsupported parameters
  block, while reasoning/deprecated controls preserve their distinct semantics.

### Interfaces and safety

- Shared services, CLI commands, and MCP tools expose `analyze_invocation`,
  `prepare_invocation_migration`, and `validate_prompt`.
- Preparation is review-only. Tests verify application files are unchanged,
  unknown facts remain visible, and deterministic mappings require no inference.

## V0.4 completed capabilities

### Integrated planner

- `MigrationPlan` schema version 2 is the canonical application-level contract
  and serializes beneath the `migration` key in `migration-manifest.yaml`.
- One service workflow composes application scanning, platform-aware source and
  target resolution, model comparison, prompt preparation/validation, invocation
  preparation, and tool/output/configuration analysis.
- The manifest includes source and target provider/platform/model identifiers,
  application summary, affected files and source locations, required and optional
  changes, blockers, warnings, unknowns, prompt and invocation preparations,
  tool/output/configuration changes, validation results, required tests, rollout
  recommendations, complexity, and behavioral risk.
- Detected file-backed prompts beneath the application root are loaded and
  prepared. Inline, dynamic, unreadable, or escaping prompt inputs remain
  explicit unknowns.
- Cross-concern validation blocks clearly invalid resolved JSON Schemas and
  declared context-window regressions in addition to preserving prompt and
  invocation validation results.

### Reporting and interfaces

- `generate_migration_report` renders a human-readable Markdown report from the
  canonical manifest, covering target rationale, model differences, complexity,
  risk, unresolved items, tests, and rollout.
- CLI `plan` emits YAML by default (or JSON explicitly), CLI `report` emits
  Markdown, and both write only to an explicitly requested output path.
- MCP exposes both `generate_migration_plan` and `generate_migration_report` as
  thin wrappers over the shared service.
- Four end-to-end fixture routes and exact YAML golden manifests cover Anthropic
  upgrade, Anthropic-to-Bedrock, Anthropic-to-OpenAI, and OpenAI-to-Anthropic.
- Planning and reporting do not rewrite analyzed source files. CLI artifact
  writes require an explicit path and reject affected source-file targets.

## V0.5 completed capabilities

### Evaluation contracts and execution

- Versioned `EvaluationSuite`, `MigrationEvalRun`, `OutputComparison`, and
  `RegressionReport` contracts preserve the V0.4 manifest hash and exact
  source/target endpoint identity.
- Suites accept golden outputs, exact and regex checks, Draft 2020-12 JSON
  Schemas, expected tool calls, named custom evaluators, and optional named
  LLM-judge hooks.
- One immutable corpus runs against both endpoint configurations. Config/model/
  provider/platform/model-ID mismatches fail closed.
- Direct Anthropic and OpenAI HTTP execution uses BYOK environment-variable
  references, bounded timeouts, and bounded retries for the exact direct-API
  platform pairs. Bedrock and other platforms fail closed unless an
  `EvaluationExecutor` is injected.
- Nested provider parameters support tool and output configuration while
  reserved endpoint fields and recursively detected secret-bearing fields are
  rejected. Provider tool calls normalize to `{name, arguments}`.
- Run artifacts contain outputs, tool calls, evaluator results, pass/task score,
  refusal/error state, latency, tokens, attempts, and estimated cost, but never
  credential values.

### Regression analysis and interfaces

- Paired comparison reports score, output, latency, token, and estimated-cost
  deltas with explicit regression signals.
- Regression reports cover format/schema/parser, grounding, instruction,
  reasoning, tool, refusal/safety, verbosity, context, capability, latency/cost,
  and unclear categories without inventing unsupported diagnoses.
- CLI `eval generate-suite`, `eval run`, `eval compare-outputs`, and
  `eval analyze-regressions` persist reviewable YAML only to explicit output
  paths and refuse to overwrite evaluation inputs.
- MCP exposes the same four operations as thin shared-service wrappers.
- Stable suite/report goldens and fake-provider tests cover persistence,
  deterministic evaluation, partial failure, retry, invalid output/schema,
  custom evaluator isolation, tool/refusal metrics, and secret safety.

## V0.6 completed capabilities

### Bounded optimization

- `optimize_migration` consumes a structured regression report and only
  explicitly supplied candidate evaluation runs; it does not create an
  autonomous rerun loop or mutate application code.
- Typed limits bind selection to a balanced, quality, cost, or latency objective
  and bound candidates, evaluation runs, estimated target cost, minimum target
  pass rate, and whether quality regressions may be accepted.
- Results contain targeted review recommendations, candidate pass/regression/
  latency/cost comparisons, deterministic selection, algorithm version, exact
  input hashes, and reproducibility metadata.
- Python, CLI `eval optimize`, and MCP `optimize_migration` are thin interfaces
  over the same deterministic service operation.

### Broader evaluation coverage

- Evaluation-suite schema version 2 adds provider-neutral multi-turn messages,
  text/image/document content blocks, validated base64 media, deterministic
  serialization, and direct Anthropic/OpenAI/Bedrock mappings while retaining
  schema-version-1 and text-only input compatibility.
- The reviewed built-in Bedrock executor uses the Converse API, accepts model or
  inference-profile identifiers, normal AWS SDK credential resolution, bounded
  retries, normalized tool calls, and credential-safe failures. `boto3` is an
  optional `aws` package extra.
- A built-in router dispatches only direct Anthropic, direct OpenAI, and Amazon
  Bedrock platform pairs; every other platform remains fail-closed.
- Trusted host code can explicitly register custom or judge evaluators by name.
  Standalone CLI/MCP processes load only installed evaluator entry points named
  in the local `LLM_MIGRATE_EVALUATORS` allowlist. Request artifacts may
  reference names but cannot provide module paths or executable code; unknown
  names stay skipped.

## Verification status

Verified on 2026-08-23:

- `pytest`: 116 tests passed, one optional credential-gated Bedrock contract
  test skipped because no live model ID was configured
- `ruff check src tests scripts`: passed
- `ruff format --check src tests scripts`: passed
- `mypy`: passed for all source files
- wheel build and isolated install smoke: passed; the installed `1.0.0` package
  discovered its bundled registry and generated a migration manifest from
  outside the checkout
- registry validation: nine models, three migrations, zero observation records;
  one intentional fixture alias warning
- model-profile JSON Schema parity: passed
- live-pricing tests use injected responses and make no network calls

## V1 stabilization capabilities

- `check_model_lifecycle` interprets reviewed registry status and end-of-life
  dates at a caller-selected date, reports stale lifecycle evidence, and gives a
  conservative suitability decision without live research.
- `estimate_migration_cost` compares one normalized workload using canonical
  input/output pricing, preserves pricing provenance, reports unknown price
  components, and keeps recurring token cost distinct from engineering cost and
  observed evaluation cost.
- Python, CLI, and MCP expose both contracts through the shared service.
- The stable V1 naming scheme retains the established review-oriented
  `prepare_*` and `generate_migration_plan` operations rather than adding
  mutation-suggesting parallel aliases.
- Invocation-preparation schema version 2 emits typed provider-specific tool
  wrapper candidates for Anthropic, OpenAI, and Bedrock plus Anthropic/OpenAI
  structured-output candidates. It preserves target fields, mapped tool choice,
  compatibility state, source location, and review notes; unsupported or
  unresolved mappings emit no candidate.
- The same preparation emits a typed target platform configuration contract for
  Anthropic/OpenAI environment-variable credentials or the AWS default
  credential chain, including SDK/client and region-variable names but never
  credential values.
- One offline acceptance workflow covers scan through bounded optimization and
  verifies that analyzed source files remain byte-for-byte unchanged.

## Known debt that does not block V1

- The scanner deliberately supports Python only and uses conservative static AST
  analysis; broad multi-language parsing is outside V0.2.
- Canonical pricing and lifecycle freshness reflect their recorded check dates;
  stale status remains visible and live OpenRouter evidence does not refresh it.
- Static scanning does not retain arbitrary inline or dynamic prompt text;
  integrated planning records this as an unknown rather than inventing analysis.
- Candidate prompt text, semantic diffs, provider-specific target
  invocation/tool/output payloads, and planned changes are reviewable;
  source-code patch synthesis and automatic application mutation remain
  intentionally outside V1.
- Bedrock runtime support is synchronous Converse only; streaming and providers
  beyond direct Anthropic, direct OpenAI, and Amazon Bedrock remain future work.
- The trusted evaluator registry is process-local; installed entry points load
  only when the local operator explicitly allowlists their names, never from
  artifact-controlled module paths.

## Planned V1.1: on-demand agent-researched migration

V1.1 will preserve the small reviewed built-in registry while adding an
explicitly requested, user-scoped research path for a source model/platform and
target model/platform that are missing, stale, or insufficient for the scanned
application.

The intended workflow is:

```text
local application scan
  -> targeted source/target/platform research
  -> typed ResearchResult artifacts
  -> deterministic proposals
  -> independent claim-level agent review
  -> deterministic validation and conflict policy
  -> expiring session registry overlay
  -> migration planning/evaluation
```

The agent host supplies its own models, search capability, credentials, and
token budget. Core Python, CLI, and MCP services remain provider-neutral and do
not require project-owned inference credentials or hosted infrastructure.

Session knowledge must be visibly classified as `session_agent_reviewed` or
`session_unreviewed`; it never silently overwrites `canonical_verified`
knowledge. Agent review may authorize use inside the current user-scoped run,
but promotion into the shared canonical registry still requires an exported
proposal bundle and explicit maintainer approval.

V1.1 requires new typed contracts for bounded research requests, claim-level
evidence review, deterministic consensus, session-registry manifests, and
orchestration run metadata. It also requires a proposal/review path for
source-target `MigrationKnowledge`, not only individual model profiles.

## V1.1 completed capabilities

- Typed `MigrationResearchRequest`, `EvidenceReview`, `ResearchConsensus`,
  `SessionRegistryManifest`, and `OrchestrationRun` contracts with strict
  serialization, stable content hashes, and unknown-field rejection.
- Request creation scans locally, hashes normalized requirements, and bounds
  research to the scopes and topics that are actually missing or stale;
  fresh canonical coverage refuses research entirely.
- Deterministic claim-level consensus: reviewer independence and source
  refetch are enforced, agent agreement is never evidence, contradictions
  stay unresolved, and high-impact conflicts block session use. A dispute
  arbiter can resolve a conflict only by adding independently sourced
  qualifying evidence.
- Pair-specific `MigrationKnowledge` compiles only from accepted
  `migration_behavior.*` claims; validated profiles never imply it.
- The session registry overlay is immutable, expiring, hash-verified,
  visibly non-canonical, refuses `session_unreviewed` knowledge for
  consequential use, and shadows canonical profiles only via explicit
  manifest selection. It never writes to `registry/`.
- The orchestration driver is resumable and fail-closed across attempts,
  token budgets, and stage failures; hosts supply agents via the
  `AgentRunner` protocol or drive stages through YAML artifacts in the run
  workspace. A fake-agent host exercises the full path end to end, including
  prompt-injected artifacts and fabricated citations, with an exact golden
  run workspace under `tests/golden/v11/`.
- CLI stage operations (`llm-migrate research create-request /
  validate-result / validate-review / consensus / build-session /
  fetch-sources / status` and `plan/report --session`) and matching MCP
  tools stay thin over the shared service. Session-planned manifests and
  reports always surface the run, trust level, expiry, shadowing, and
  unresolved issues.
- `docs/agent-research-workflow.md` documents the agent-neutral host
  protocol. The opt-in evidence adapter refetches only cited HTTPS sources,
  records reachability and content hashes, and never stores page content.
- A release-readiness review on 2026-08-29 passed the full test, lint, format,
  type-check, wheel-build, isolated-install, and installed planning/reporting
  smoke workflows. Package/runtime metadata now identifies this branch as
  `1.1.0`, with a parity regression test, and the README is organized around
  user prerequisites, source installation, common workflows, evaluation, MCP,
  and the explicit V1.1 agent-host boundary.

## V1.2 completed capabilities

- Lenient, registry-first model matching (`match_model`, CLI `models match`,
  MCP `resolve_model`): deterministic stripping of Bedrock cross-region
  inference-profile prefixes (`us.`, `eu.`, ...) and version suffixes
  (`-v1:0`), vague platform normalization ("bedrock" → `amazon-bedrock`), and
  ranked similarity candidates with explicit `resolved` /
  `needs_confirmation` / `not_found` states. Confirmation is returned to the
  user instead of guessing or hard-failing; genuinely unknown models are
  directed to the V1.1 research workflow.
- Endpoint context flows through planning: `generate_migration_plan/report`
  and `prepare_invocation_migration` accept source/target endpoints, so models
  with several same-platform representations (e.g. `bedrock-runtime` vs
  `bedrock-mantle`) plan deterministically. Detected in-app model IDs that are
  ambiguous across representations no longer crash the consistency check.
- Guided run workspace: `start_migration` (MCP) / `llm-migrate run start`
  matches both models first (writing nothing until both resolve), scans the
  application, decides whether research is needed and records the exact
  reasons (missing model, stale topics, missing pair knowledge), writes
  `migration.yaml` plus a bounded `request.yaml` when applicable, and returns
  ordered `next_steps`. The default workspace is
  `<application>/.llm-migrate/runs/<run-id>/` with all deliverables under
  `output/`; an explicit `output_dir` overrides the location. The scanner
  ignores `.llm-migrate` so run artifacts never contaminate scans.
- Deterministic research prompts: `get_research_prompts` / `llm-migrate
  research prompts` renders one scope-isolated researcher and reviewer prompt
  pair per remaining scope from `request.yaml`, embedding exact identities,
  topic/field-path bounds, the source policy, output schemas and paths, the
  untrusted-data rule, and per-scope status so completed stages are never
  re-run.
- Adaptation deliverables: `list_adaptation_tasks` derives a per-file worklist
  from the plan; `submit_adapted_prompt` statically validates and stores
  refined prompts under `output/prompts/`; `submit_adapted_file` stores
  complete adapted application files under `output/files/` behind fail-closed
  checks (path containment, non-empty, actually changed, Python syntax,
  model-id consistency warnings); `finalize_migration` writes
  `migration-manifest.yaml`, `changes.yaml`, and `migration-report.md` with a
  per-file "what changed and why" section plus coverage gaps. The analyzed
  application tree is never modified.

## V1.3 completed capabilities

- Prompt provenance discovery: `scan_application` now scans YAML/JSON/TOML
  configuration structurally alongside Python and reports `PromptSource`
  records (path, format, structured `PromptComponent`s, provenance chain,
  evidence-based `high`/`medium`/`low` confidence, discovered/override origin)
  plus a `PromptDiscoverySummary`. `ApplicationAnalysis` is schema version 3;
  `MigrationPlan` is schema version 3 and carries the discovery summary.
- Bounded Python loader/path recognition without executing or importing the
  application: `open(...)`, `Path(...).open()`, `read_text()`,
  `yaml.safe_load`/`yaml.load`/`json.load(s)`/`tomllib.load(s)`, `Path` joins,
  `Path(__file__).parent` composition, simple constants, nested config
  subscripts (`profile["prompts"]["multi"]` resolved against the parsed config
  document), and loader-style helper calls. Every prompt consumer is
  classified `inline`, `source`, or `dynamic`.
- Structured prompt documents stay structured: only recognized prompt keys
  become components, per-component `PromptMigrationSpec`s carry
  `source_component` and real roles, workspace prompt tasks group components
  per file with a reconstructed full-document candidate, and adapted-prompt
  submissions fail closed on syntax errors or changes to non-prompt values.
- Coverage is explicit: plans and reports state
  `resolved`/`partial`/`unresolved` prompt coverage, warn when coverage is
  incomplete, and never render dynamic prompt consumers as "no prompt
  changes". Low-confidence prompt-like files are reported but never become
  tasks; dynamic chat-history prompts are warnings, not blockers.
- Explicit `prompt_sources` overrides on `scan_application`,
  `generate_migration_plan/report`, and `start_migration` across service, CLI
  (`--prompt-source`), and MCP, persisted in the run's `migration.yaml` so
  every later run stage sees them.
- Runtime-aware submission integrity: prompt-lexical capability mismatches
  (a prompt mentioning JSON or tools) are warnings that name the mechanism —
  prompt-enforced output formats need no native structured-output support —
  and can never block a plan; only invocation evidence blocks. Submission
  validation and adaptation checks run on the DECODED runtime prompt values,
  so serialization tricks (unicode escapes, quoting, whitespace style) can
  neither hide content from validation nor make an unchanged prompt look
  adapted: a submission whose decoded values equal the original is rejected
  with a pointer to `unchanged=true` (now supported for prompts as well as
  files), and whitespace/case-only edits are flagged. Prompt tasks surface
  in-prompt curation findings (duplicated requirements, negative wording,
  chain-of-thought requests, prefill and JSON-only workarounds) so agents
  adapt the actual prompt rather than restating model-level guidance, and
  finalization coverage is computed from the same derived worklist the agent
  received, so the two can never disagree.
- Host-agent token efficiency: guidance shared by every prompt task is hoisted
  once into `AdaptationTaskList.shared_prompt_guidance` instead of repeating
  per task (roughly halving the worklist payload on multi-prompt apps), the
  workflow instructions say to list tasks once and never re-list between
  submissions, and `submit_adapted_file(..., unchanged=true)` records a
  reviewed no-change file without resending its content — closing the
  coverage-gap loop where identical content was otherwise rejected. An
  unchanged claim is refused when the file still references the source model
  id, and task guidance says to surface plan blockers to the user rather than
  retry submissions.
- Regression fixture `configured_prompt_app` (config-driven Bedrock prompt
  loading) with a golden scan, plus discovery/coverage/override/format-
  preservation unit tests.
- Evidence-linked model differences: every comparison claim carries the
  registry source URL that backs it (`ModelDifference.source_evidence_url` /
  `target_evidence_url`, resolved field-level sources first, then the profile
  source whose `supports` covers the topic; migration-knowledge rows cite
  their supporting source). The report's differences table hyperlinks each
  claim so it is verifiable in one click.
- Evidence-based prompt adaptation: the deterministic candidate is the source
  prompt verbatim (the unevidenced chain-of-thought auto-rewrite was removed);
  guidance advice carries `evidence_urls`; prompt adaptation tasks list the
  prompt's protected structural sections and inject the plan's
  migration-knowledge differences with their evidence links; and submissions
  that drop XML-like sections or prompt components are rejected unless
  `allow_restructure` is set (CLI `--allow-restructure`, MCP parameter) with
  the justification recorded in `changes.yaml` and the final report.

## Next work

1. Exercise the V1.2 guided workflow with real coding-agent hosts (Claude Code,
   Copilot) and fold observed friction back into the tool guidance.
2. Add broader scanners/model families through the existing normalized scanner,
   reviewed canonical registry, and V1.1 session-overlay boundaries.
3. Optional V1.1 follow-ups: a persistent content-addressed research cache,
   concurrent agent execution within recorded limits, and an orchestrated
   arbiter stage.

## Not yet in scope

- Automatic source-code rewriting
- Continuous autonomous model discovery
- Hosted inference or project-owned provider credentials
- Automatic promotion of agent-reviewed session facts into the shared canonical
  registry

## Durable decisions

- The registry remains reviewed canonical knowledge; research only produces
  proposals.
- Model identity, provider, and platform representation remain separate.
- Basic scanning and compatibility analysis require neither inference nor network
  access.
- Live OpenRouter pricing is scoped evidence and never canonical knowledge.
- Application and registry mutation remain review-first.
- The V0.4 machine-readable artifact is `migration-manifest.yaml`.
- The Markdown migration report is derived from the manifest, not separately
  reasoned.
- Runtime evaluation is explicit BYOK work; credentials stay local and outside
  all persisted contracts.
- Optimization is bounded, evidence-driven, and review-only; it never silently
  mutates migrations or applications.
- The V1 public naming surface favors `prepare_*` review artifacts over
  mutation-suggesting aliases.
- V1.1 research is user-initiated, bounded, application-targeted, and local-run;
  it is not a background crawler or continuously self-updating registry.
- A compact reviewed registry remains the offline and reproducible baseline;
  ephemeral session overlays cover the long tail without becoming canonical.
- Agent agreement is not proof. Claims are accepted through source authority,
  independent refetch/review, deterministic validation, and explicit conflict
  handling rather than majority vote.
