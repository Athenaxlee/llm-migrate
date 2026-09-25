# Changelog

All notable changes are documented here. The project follows semantic
versioning.

## Unreleased

- Documentation: the toolkit never calls an LLM API to research, adapt, or
  review — the coding-agent host does that work with its own model. README
  and the MCP playbook recommend the host's most capable reasoning model with
  extended thinking for the guided workflow.

## 1.6.1 — 2026-09-24

Run economics, evidence refresh, and plain-language output, driven by a
v1.6.0 guided run that was slow and token-hungry: every Anthropic profile's
freshness had lapsed, so every run recommended research on every topic for
both models, and the host stopped to ask, then drove four web-researching
agents one at a time through a visible browser. No contract shape changes;
ids are stable.

### Hands-free default research

- Research stays `recommended` whenever knowledge is missing or past its
  freshness window, and now runs by default: `start_migration.next_steps`,
  the `research_pending` status action, the server playbook, and the
  generated researcher/reviewer prompts tell the host to run the stages with
  separate non-interactive background agents (web search and page-fetch
  tools, never a visible browser), scopes in parallel, asking the user first
  only if they asked to be consulted. The README starting instruction no
  longer says "ask me before running research".
- Freshness reasons name the stale topics and their last-checked date.

### Per-call economics

- The application walk prunes ignored directories (`.venv`, `node_modules`,
  the `.llm-migrate` workspace, ...) instead of listing everything beneath
  them; snapshot-key file digests are cached by size, mtime, and ctime; the
  MCP process keeps one service per registry content digest, and the digest
  is computed once per service. Steady-state guided calls dropped from
  0.7–2.3 s to about 36 ms on an 18k-file repository with a virtual
  environment inside it.

### Maintainer evidence refresh

- `llm-migrate registry refresh-evidence [--model …] [--rebaseline]
  [--no-record] [--output DIR]` refetches only the source URLs already
  recorded in canonical profiles, hashes each page's visible text against
  the baseline in `.registry-proposals/<bundle>/refresh-evidence.yaml`, and
  reports: unchanged pages that still name the model propose moving
  `checked_at` for the categories they support; a page silent on the model
  vouches for nothing (`no_mention`); changed pages are held until
  re-verified (`--rebaseline`); a URL without a baseline gets one recorded
  and proposes nothing; a category the bundle's review put on hold stays
  held with its rationale. Fetch failures, including truncated responses, are
  recorded per URL. Registry files are never written.

### Registry

- `claude-sonnet-5` (all five categories), `claude-sonnet-4-6` (pricing,
  lifecycle, availability), and `claude-sonnet-4-5-20250929` (pricing,
  lifecycle, availability, capabilities) re-verified on 2026-09-24 against
  the refetched official pages and promoted through their bundles. Legacy
  pricing is now sourced from the pricing page (the models overview lists
  only current models). Held: Sonnet 4.6 capabilities (the AWS model card
  states a 64K maximum output against the recorded 128K) and prompt
  guidance (the migration-guide URL now serves an index page).

### Plain-language output

- Blocker questions ("What do you want to do: … (retarget); … (redesign);
  … (accept)?"), option summaries and consequences, unknown statements,
  the report's "Action required" list, status warnings, and difference
  guidance (`[high] impact → action (evidence: url)`) were rewritten to be
  short and direct. The report shows the run directory once and `<run_dir>`
  in every action, and empty change sections collapse into one "Not
  affected" line.

## 1.6.0 — 2026-09-23

Discovery that survives real repositories, and directions instead of dead
ends: the second, third, and fourth tranches of the roadmap driven by a real
v1.5.1 guided run of a production Bedrock application whose config-driven
prompt library produced zero prompt tasks. On that layout the run now
prepares six prompt migrations with zero configuration. `MigrationPlan`
moves to schema 5; every other contract change is additive.

### Prompt discovery on real repository layouts

- Config paths written relative to the repository root resolve when a
  subdirectory is scanned: bounded leading-segment stripping (at most three
  segments that must equal the application root's trailing path, always into
  the scanned file set), recorded in provenance and never ranked high.
  Windows-style backslash values resolve, and case folding applies only on a
  filesystem detected case-insensitive. Config values may climb with `..`
  relative to their config file; symlinks resolving outside the application
  are never scanned.
- Chained single-file loaders (`open(p).read()`, `handle.read()`,
  `Path(p).read_text()`) are traced; dynamic consumers record the keys they
  read, and the unique candidate document holding them is promoted one level.
- `start_migration` returns `prompt_candidates` for confirmation when prompt
  consumers exist but no prompt source resolved, writing nothing;
  `defer_prompt_candidates` proceeds and the run reports
  `discovery_incomplete` until the live-run tools `add_prompt_sources`
  (sources, or rationale-bearing dismissals) and `confirm_prompt_consumer`
  close it. Consumer addresses are `path:line:keyword`.
- Regression fixtures for the field layout and the probe loader forms; a
  Windows CI leg.

### Unknowns that give directions

- Plan unknowns are typed records (schema 5): subject, why it matters for
  this application, an exact next action (a ready-to-run tool call with the
  run directory filled in, or a probe command), the closing condition, and
  how each was closed (by the scan, or by a recorded action).
- Contested registry facts (for example whether adaptive thinking can be
  disabled on Bedrock, where the provider docs and the AWS model card
  disagree) emit a BYOK probe script under `output/probes/`;
  `record_observation` stores the user's result as a run-scoped
  observation that closes the unknown and never touches the registry.

### Review integrity and validation

- Adapted pricing values are compared with the registry unit-aware (per
  token / 1K / 1M, exact decimals): a departure from the target's recorded
  price is a `pricing_contradicts_registry` finding with its implied factor
  unless the change that set the value cites registry-recorded or
  plan-carried evidence; leftover source pricing is `pricing_stale_source`;
  an unrecognizable scale is reported, never passed. Strict mode blocks.
- A change that exercises a contested registry setting is marked CONTESTED
  in change review until an observation resolves it.
- After the first finalize the run is `validation_pending`;
  `scaffold_evaluation` drafts DRAFT evaluation cases from the application's
  own sample-input folders into a suite bound to the current plan.
- Pair knowledge gains the `prompt_guidance` topic; the
  claude-sonnet-4-6 → claude-sonnet-5 pair carries two documented items
  (sampling behavior moves to system-prompt instructions; recount prompt
  tokens), landed through a reviewed proposal bundle.
- Registry advice identical across a structured prompt document's components
  is owed once per prompt task.

### Registry

- `claude-sonnet-5` pricing is recorded as the base price of $2 / $10 per
  million tokens (three official pages refetched 2026-09-23 state no
  introductory period).

The guided toolset is 23 tools (`add_prompt_sources`,
`confirm_prompt_consumer`, `record_observation`, `scaffold_evaluation`).

## 1.5.2 — 2026-09-23

Precision and honesty patch, driven by a real v1.5.1 guided run of a
production Bedrock application whose config-driven prompt library produced
zero prompt tasks and whose report raised problems without directions. No
schema version changed; every contract change is an additive field.

- Prompt consumers are recorded only on model invocations or provider-client
  calls; helpers taking `input=` no longer inflate the consumer count.
- Out-of-scope prompt files: prompts referenced only by another model's
  configuration profile, and unreferenced candidates beside selected
  siblings, move from unknowns to the plan's new `out_of_scope` list and are
  never prepared.
- Unknowns the scan already answers (for example `parallel_tool_use` in an
  app with no tools) are not raised; the XML-formatting finding is dropped on
  same-provider moves.
- Known evidence includes every source URL recorded in the run's resolved
  registry profiles and pair knowledge; change review shows each citation's
  verification state (registry-recorded, plan-carried, or UNKNOWN).
- Evidence-backed validation: `generated_tests` needs the user's passing
  outcome and summary line, and `byok_evaluation` needs the evaluation-run
  artifact bound to the finalized manifest. Finalize re-checks the binding
  and reports NOT VALIDATED otherwise (a strict violation).
- Whole-application source-reference sweep: finalize reports any scanned
  file still naming the source model that nothing accounts for
  (`source_reference_uncovered`; strict blocks). Intentional mentions close
  through `confirm_unaffected(acknowledge_source_references=true)` with the
  user's rationale.
- Registry guidance about structured output, tool use, or reasoning is
  pre-disposed not applicable, with the reason, when the application never
  exhibits the trigger under fully resolved prompt coverage.
- The run report opens with "Action required" and gives unknowns an Action
  column; run status warns while prompt coverage is incomplete and states
  the two-pass validation flow; server instructions order review before
  finalize; MCP `start_migration` returns paths relative to the run root.
- README documents per-host MCP registration and verification.

## 1.5.1 — 2026-09-22

Review-fix patch: a same-day multi-axis review of v1.5.0 confirmed three
high-severity defects by reproduction; this release closes them and brings
the documentation current.

- Alias-aware reference detection: the run identity additively records
  source/target reference spellings — the invocable platform spellings plus
  the reviewed canonical name and registry aliases. Unchanged-claim guards,
  lingering-source warnings, the cross-surface consistency gate, and the
  generated contract test now detect registry aliases of the source model;
  invocable enforcement (bare-id rejection, sanctioned-swap targets,
  selector lists) still uses only platform spellings, so an alias can never
  masquerade as an invocation id. Detection excludes source spellings
  contained in a target spelling, so a cross-platform migration of the same
  model is never falsely flagged.
- New files face review: a `new_file` submission must document its content
  with at least one evidence-linked annotated change (an anchor-less
  insert/restructure claims the whole file). Invented content can no longer
  enter the deliverable set with nothing for change review to decide; the
  rejection uses the submission-format tone.
- Read-only run status: `get_run_status` no longer infers "moved past
  research" from the worklist snapshot its own call persists as a cache.
  The signal is an explicit marker written only by `list_adaptation_tasks`,
  a recorded blocker decision, or a submission — a second status call
  cannot silently steer the host past the research question.
- Documentation currency pass: the architecture document gains the
  adaptation-review and v1.5 sections and an updated guided-run flow, the
  project context documents the 19-tool guided surface with the guided path
  as the expected agent workflow, milestone records and schema-version notes
  are corrected, and the README covers strict mode, invocation-selector
  confirmation, batching, run status, the validation disposition, and the
  consistency-gated finalize.

## 1.5.0 — 2026-09-22

Deployment fidelity and workflow economics, driven by the audit of a real
v1.4.0 production Bedrock migration run.

- Semantic configuration coupling (v1.5.0-b): anchored structured-config
  documents — a registry-matched model-id spelling appears in them, or their
  values are traced into a detected invocation call chain — now contribute
  model-id, sampling, token-budget, region/routing, and pricing couplings
  (pricing-shaped values without a model anchor are never couplings), so the
  configuration file that carries the model setup becomes a real worklist
  task. One difference-propagation mechanism replaces the prompt-guidance
  special case: every material model difference attaches a required change to
  exactly the files whose detected couplings it governs, and reaches prompt
  guidance only when the governing surface is a prompt. Files whose only
  coupling is an incidental SDK import move to one `unaffected_files` bucket,
  closed by a single `confirm_unaffected(run_dir, paths, rationale)` call
  (MCP + CLI) recording per-file reviewed no-change entries. Finalization
  runs a deterministic cross-surface consistency gate over the whole
  effective deliverable set: no active source-model configuration remains
  (catching review-rejection reverts), no forbidden bare target references,
  one selector-qualified target across every surface, model-coupled files
  still name the target, and scanner-recognized coupling markers cannot
  vanish without an explicit disposition.
- Workflow economics (v1.5.0-c): the derived worklist and its plan evidence
  persist as a run-workspace snapshot keyed by a complete staleness hash
  (application content, `migration.yaml`, the blocker decision log, the
  session manifest, and the registry content); submissions and status checks
  reuse it until the key changes, task statuses are recomputed from the
  adaptation log at read time, and the v1.4.0 "one scan per submission"
  trade-off is retired. New `submit_adaptations` validates every item
  independently (per-item accept/reject, never all-or-nothing) and applies
  the accepted subset in one locked atomic `changes.yaml` write;
  `record_change_decisions` batches review decisions; canonical-shadow
  selections surface together in one error. Shared prompt guidance is
  disposable once per run, and `default_disposition` covers unlisted
  guidance ids — expanded into per-item records marked `defaulted` and
  flagged DEFAULTED in the report. Researcher/reviewer prompts embed the
  exact JSON Schema of their artifacts, and `get_run_status` (MCP + CLI
  `run status`) reports the run's state machine with the single next action.
- Strict production mode (v1.5.0-d): an opt-in `strict` flag on
  `start_migration` (CLI `run start --strict`) rejects unknown evidence URLs
  at submission, turns missing invocation facts on the target profile and
  incomplete prompt coverage into blockers (normal resolution flow, accept
  always offered), and refuses to finalize cleanly while coverage gaps,
  unconfirmed unaffected files, consistency findings, undecided changes, or
  a missing validation disposition remain. `record_validation_disposition`
  (MCP + CLI `run record-validation`) durably records how the migration was
  validated — BYOK evaluation, user-executed generated tests, or an explicit
  accept requiring the user's own rationale — and finalization emits a
  deterministic request-shape contract test under `output/validation/` that
  the user wires to their application and runs themselves; the toolkit never
  executes application code.
- Invocation identity (v1.5.0-a): the registry now separates canonical
  identity (which model this is) from the invocation selector (which id the
  platform accepts for an on-demand call). `PlatformAvailability` gains an
  optional reviewed `invocation` block (`bare_on_demand_supported`, named
  evidence-linked selectors carrying full inference-profile ids) — fail-open
  when absent, fail-closed schema validation when stated. Runs record the
  selector-qualified invocation id (seeded from the user's original spelling,
  confirmed with selector candidates at start, or driven through a new
  `invocation_selector_required` blocker with one evidence-linked option per
  selector), selector ids resolve as exact registry identifiers, worklist
  guidance instructs the invocation id, prompt and file submissions reject
  bare-id references the reviewed profile forbids, identity checks match on
  the full reviewed spelling set (unchanged-claim guards, source/target
  reference warnings, the sanctioned structured-config model-id swap), and
  the report records the invocation identity or warns when invocation facts
  are unknown. The canonical Anthropic Bedrock representations record their
  cross-region inference profiles; the direct Anthropic/OpenAI APIs record
  bare invocability; all proposal bundles carry the evidence.

## 1.4.1 — 2026-09-22

Immediate relief from a real v1.4.0 production migration run. No schema or
registry changes.

- Durable run-state writes: every run-workspace write is atomic (temp file +
  `os.replace`, portable to Windows), and every read-modify-write over
  `changes.yaml`, the blocker and change decision logs, deliverables, and
  submission copies is serialized by an OS-level per-run advisory lock
  (`fcntl`/`msvcrt` on `<run>/.llm-migrate.lock`, released automatically on
  process exit). Parallel host submissions no longer lose updates, and a
  crashed holder never wedges the run.
- UTC normalization at every MCP/CLI boundary: naive host timestamps are
  interpreted as UTC and aware ones converted, so session-overlay expiry and
  freshness comparisons never raise naive/aware `TypeError`s.
- Tool-surface diet: the guided-workflow tool descriptions were cut ~40%
  (server instructions −31%) with the safety invariants kept, and the opt-in
  `LLM_MIGRATE_TOOLSET=guided` environment variable exposes only the 14
  guided-workflow tools for hosts with tight inline-tool budgets.
- Rejection-message tone: disposition/annotation rejections state explicitly
  that they are submission-format requirements of the tool, not judgements
  on the adaptation content, so host agents repair the submission payload
  instead of asking the user to "refine the prompt".
- Research artifact validation by path: the new `validate_research_artifact`
  MCP tool and `llm-migrate research validate-artifact` CLI command read the
  researcher and reviewer YAML directly from the run workspace and run the
  existing deterministic gates; the generated researcher/reviewer prompts
  steer agents to them, ending full-artifact resends through MCP.

## 1.4.0 — 2026-09-21

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
