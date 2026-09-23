# LLM_Migration Development Phases

## 1. Roadmap Philosophy

The project should grow in layers.

Each phase must:

- produce a coherent usable increment
- establish contracts needed by later phases
- include tests
- avoid implementing adjacent future features prematurely
- preserve the local-first, provider-neutral architecture

The intended progression is:

```text
V0
foundation
  ↓
V0.1
verified model knowledge + provenance
  ↓
V0.2
model intelligence + application understanding
  ↓
V0.3
migration analysis and preparation
  ↓
V0.4
integrated migration planning + validation
  ↓
V0.5
runtime evaluation + regression analysis
  ↓
V0.6
optimization + broader provider support
  ↓
V1
stable end-to-end migration product
  ↓
V1.1
on-demand agent-researched migration
```

Version boundaries are capability milestones, not marketing promises.

---

# 2. V0: Project Foundation

## Goal

Create a clean repository and stable domain foundation so later capabilities do not grow as one-off scripts.

## Scope

### Project structure

Establish:

```text
src/
registry/
tests/
docs/
```

with clear boundaries for:

- domain models
- registry
- research
- scanners
- migrations
- validation
- MCP
- CLI

### Core schemas

Define initial typed models for concepts such as:

- Provider
- Platform
- ModelIdentity
- ModelReference
- ModelProfile
- SourceReference
- Provenance
- Capability
- MigrationIssue
- MigrationChange
- MigrationPlan

### Configuration

- local configuration
- paths
- logging
- version metadata

### CLI/MCP scaffolding

Create thin interfaces with no duplicated domain logic.

### Engineering baseline

- formatting
- linting
- type checking
- unit test setup
- CI
- fixtures
- example registry records

## Tools

Only enough MCP/CLI wiring to prove architecture.

No need for the full migration tool set.

## Tests

- schema serialization/deserialization
- invalid schema rejection
- provider/platform enum/value behavior
- registry file loading
- CLI smoke test
- MCP server smoke test
- package import test

## Exit criteria

- repository structure is stable
- domain objects can be loaded/serialized
- tests run in CI
- MCP and CLI can call shared service code
- no feature logic is trapped inside interface adapters

---

# 3. V0.1: Verified Knowledge Foundation

## Goal

Build the model knowledge layer that later migration reasoning can trust.

## Core design

```text
registry/
    ↓
source provenance
    ↓
research result schema
    ↓
manual / agent-assisted population
    ↓
reviewable registry proposals
```

## Scope

### Canonical model registry

Create version-controlled YAML records.

Initial real-model batch should include:

- Claude Sonnet 4.5
- Claude Sonnet 4.6
- Claude Sonnet 5
- current major GPT model families relevant to migration
- AWS Bedrock representations of those models where applicable

Do not populate these with invented placeholder facts.

Use reviewed research.

### Provenance

Important factual fields should support evidence.

Represent:

- source
- source type
- publication/retrieval date when available
- claim relationship
- confidence
- freshness/review state

### Research schema

Create a `ResearchResult` contract.

It should capture:

- researched subject
- claims
- evidence
- conflicts
- missing information
- confidence
- research timestamp

### Registry proposal workflow

Implement:

```text
propose_registry_update(...)
```

This compares research against current registry knowledge.

Outputs:

```text
proposal.yaml
proposal.md
evidence.yaml
candidate-model.yaml
```

`proposal.yaml` is the structured machine-readable diff.

`proposal.md` is the human-readable review report.

### Manual/agent-assisted population

Research is deliberately initiated by a developer or coding agent.

No continuous internet discovery service.

## Explicitly out of scope

Do not build:

```text
auto_update_everything_on_the_internet()
```

Also exclude:

- autonomous continuous crawling
- scheduled model-news ingestion
- automatic canonical registry mutation
- automatic merging of conflicting sources
- migration execution
- runtime benchmarking

## Tools

Primary new tool:

```text
propose_registry_update
```

Supporting internal services may include registry load/query/validate helpers.

## Tests

### Registry tests

- valid model YAML loads
- invalid fields fail
- aliases resolve without ambiguity
- provider and platform representations remain distinct
- duplicate canonical IDs fail
- provenance structure validates

### Research tests

- `ResearchResult` serialization
- conflicting claims preserved
- missing fields preserved
- evidence references preserved

### Proposal tests

- no-change research creates empty/no-op diff
- added claim produces add operation
- changed claim produces before/after diff
- removed claim is explicit
- conflicts are surfaced, not silently selected
- candidate YAML is deterministic
- canonical registry remains unchanged

### Golden tests

Use fixed registry/research fixtures and snapshot expected:

- `proposal.yaml`
- `proposal.md`
- `evidence.yaml`
- `candidate-model.yaml`

## Exit criteria

- real model records exist
- important facts are provenance-backed
- research can be normalized
- proposed registry changes are deterministic and reviewable
- registry never mutates implicitly

## Current project status

V0.1 was released as `v0.1.0` and is treated as complete for roadmap sequencing. Remaining conformance debt and registry freshness are tracked in `docs/current_state.md`; completion does not imply that later contracts or every aspirational test shape already exists.

### Remaining known conformance debt

None. The checked-in model-profile schema is generated from the authoritative
Pydantic model and parity-tested. Proposal-bundle coverage was completed on
2026-08-29: every non-fixture provider profile has exactly one checked-in
bundle, with coverage enforced by CI. The three OpenAI proposals were
maintainer-approved on 2026-08-29 and their candidates promoted. Proposal
artifacts are protected by exact golden snapshots regenerated from a fixture
`ResearchResult` (`tests/golden/proposals/`).

---

# 4. V0.2: Model Intelligence + Application Discovery

## Goal

Make the registry useful to a coding agent and teach the system to understand a real LLM application.

This is the first phase where LLM_Migration starts behaving like a migration product rather than only a knowledge repository.

## Scope

## A. Model resolution

Implement:

```text
resolve_model
get_model_profile
```

Requirements:

- normalize aliases
- distinguish model identity from platform ID
- return canonical model
- return relevant provider/platform representation
- handle ambiguous identifiers explicitly

## B. Comparison

Implement:

```text
compare_models
```

Compare migration-relevant fields, not every registry field.

Examples:

- modalities
- context/output limits
- tools
- structured outputs
- streaming
- parameters
- reasoning controls
- lifecycle
- platform differences

Output should include:

- same
- different
- unsupported
- unknown
- migration significance

## C. Recommendation

Implement:

```text
recommend_models
```

Inputs:

- application requirements
- source model
- allowed target provider/platform
- priorities

Outputs:

- ranked candidates
- rationale
- blockers
- tradeoffs
- unknowns

Recommendation logic should be explainable.

## D. On-demand pricing inquiry

Implement:

```text
query_live_pricing
```

Initial source:

```text
GET https://openrouter.ai/api/v1/models
```

Requirements:

- explicit opt-in network call with a bounded timeout
- exact, configured mapping between registry identities and OpenRouter model IDs
- normalize `pricing.prompt` and `pricing.completion` from USD per token to the registry's USD-per-million-token unit
- return source, retrieval time, match status, raw values, normalized values, and discrepancies
- identify results as OpenRouter-scoped pricing
- allow a time-stamped live overlay for a stale-pricing comparison or recommendation without marking the registry record fresh
- never mutate canonical registry records
- route proposed canonical changes through `ResearchResult` and `propose_registry_update(...)`
- preserve offline behavior when the API is unavailable

Tests must use recorded fixtures or mocked HTTP responses and must not require network access.

## E. Application scanner

Implement:

```text
scan_application
```

Initial language focus may be narrow, but architecture must be extensible.

At minimum detect:

- model identifiers
- provider SDK
- invocation locations
- prompt locations
- generation parameters
- tool/function definitions
- structured-output configuration
- obvious response parsing
- environment/config references

Generate a normalized application analysis.

## F. Coupling classification

Classify findings such as:

```text
model_id
provider_sdk
invocation_format
parameter
prompt
tool_schema
structured_output
response_parser
streaming
multimodal
platform_config
```

## MCP tools in this phase

```text
resolve_model
get_model_profile
compare_models
recommend_models
query_live_pricing
scan_application
```

## Tests

### Model intelligence

- alias resolution
- ambiguous alias behavior
- direct API vs Bedrock resolution
- model comparisons from fixtures
- unknown field behavior
- deterministic recommendation with fixed inputs

### Scanner

Create small fixture repositories for:

- Anthropic Python app
- OpenAI Python app
- Bedrock Python app

Test detection of:

- provider SDK
- configured model
- prompt file
- invocation function
- parameters
- tools
- structured output

### Integration

```text
fixture repo
 ↓
scan_application
 ↓
resolve source model
 ↓
compare target
```

must work end-to-end without an external LLM call for basic cases.

### Live pricing

- exact identifier mapping and not-found behavior
- decimal and per-million-token normalization
- malformed or partial response handling
- timeout and offline fallback
- discrepancy output without canonical mutation

## Out of scope

- automatic source-code rewriting
- runtime evaluation
- optimization
- continuous research
- broad multi-language parsing

## Exit criteria

A coding agent can point LLM_Migration at a small real repository and obtain:

- current LLM integration summary
- normalized model identity
- migration-relevant application couplings
- target model comparisons
- reasonable candidate recommendations

## Completion status

V0.2 was completed in the `v0.2.0` working tree on 2026-08-16. The shared service,
CLI, and MCP surfaces expose platform-aware resolution and comparison,
requirements-driven recommendation, opt-in OpenRouter pricing evidence, and a
deterministic Python application scanner. Live pricing can be attached to
comparisons and recommendations from one time-stamped catalog snapshot.
Anthropic, OpenAI, and Bedrock fixture repositories have checked-in golden scan
outputs, including multimodal couplings and an end-to-end
scan-to-resolve-to-compare test. V0.3 remains the owner of prompt and invocation
transformation contracts.

---

# 5. V0.3: Migration Analysis and Preparation

## Goal

Move from "understand and compare" to "tell me what must change."

## Scope

## A. Prompt analysis

Implement:

```text
analyze_prompt
prepare_prompt_migration
validate_prompt
```

Analyze:

- provider/model-specific prompt conventions
- system prompt behavior
- structured-output instructions
- tool-use instructions
- deprecated workarounds
- prompt assumptions that may not transfer

`prepare_prompt_migration` should produce a candidate plus rationale/diff rather than silently overwrite files.

## B. Invocation analysis

Implement:

```text
analyze_invocation
prepare_invocation_migration
```

Analyze:

- SDK/client
- request shape
- messages
- parameters
- unsupported parameter behavior
- streaming
- reasoning controls
- platform identifiers
- multimodal payload structure

Produce:

- target invocation representation
- parameter mappings
- required changes
- warnings

## C. Tool compatibility foundation

Analyze tool definitions and identify whether a migration is:

- directly compatible
- mechanically convertible
- semantically different
- unsupported
- unknown

Full automatic conversion can remain later.

## D. Structured-output compatibility

Analyze:

- schema
- provider configuration
- parser assumptions
- strictness expectations

## MCP tools

```text
analyze_prompt
prepare_prompt_migration
analyze_invocation
prepare_invocation_migration
validate_prompt
```

Existing V0.2 tools remain available.

## Tests

### Prompt fixtures

Test migrations involving:

- system-message differences
- JSON-only prompting
- tool instructions
- provider-specific formatting

Expected outputs should be snapshot/golden tested where possible.

### Invocation fixtures

Test:

- parameter mapping
- unsupported parameters
- model ID replacement
- direct Anthropic → OpenAI
- direct Anthropic → Bedrock Claude
- Bedrock → direct provider

### Safety/behavior tests

- no source file is silently modified
- unknown registry facts are reported
- incompatible behavior produces blocker/warning
- deterministic mappings do not require an LLM

## Exit criteria

For a supported fixture application, the system can produce reviewable prompt and invocation migration recommendations with explicit compatibility warnings.

## Completion status

V0.3 was completed in the `v0.3.0` working tree on 2026-08-17. Versioned prompt
contracts include a provider-neutral intent representation, traceable review
candidate, provider/platform-aware system-role mapping, semantic diff, and static
target validation. Application-analysis schema version 2 preserves normalized
tool definitions, JSON Schemas, structured-output contracts, parser assumptions,
and multimodal payload formats. Versioned invocation contracts prepare target
SDK/operation/model-ID representations, parameter mappings, required changes,
warnings, and blockers, and reject declared source context that contradicts the
scan.
Anthropic-to-OpenAI, Anthropic-to-Bedrock-Claude, and Bedrock-Claude-to-Anthropic
fixture paths are tested, including all five tool/output compatibility states,
schema strictness, parser behavior, unsupported and reasoning parameters,
unknown facts, source immutability, CLI/MCP exposure, and checked-in prompt and
invocation golden review surfaces. Integrated manifests, affected-file planning,
and reports were deferred to V0.4 and are now implemented there; optional patches
remain unimplemented.

---

# 6. V0.4: Integrated Migration Planner

## Goal

Combine model intelligence, application scanning, and migration preparation into one coherent workflow.

## Scope

Implement:

```text
generate_migration_plan
generate_migration_report
```

Add fuller migration transformations as contracts stabilize:

```text
migrate_invocation
migrate_tool_schema
migrate_output_contract
```

## Migration plan requirements

The structured plan is emitted as `migration-manifest.yaml` and should include:

- source model/provider/platform
- target model/provider/platform
- application summary
- affected files
- required changes
- optional changes
- blockers
- warnings
- unknowns
- prompt changes
- invocation changes
- tool changes
- output-contract changes
- configuration changes
- validation results
- required tests
- rollout recommendations

## Human report

Explain:

- why target was selected
- important model differences
- migration complexity
- behavioral risk
- unresolved unknowns
- what should be tested before deployment

## Optional patch generation

Candidate patches may be generated, but:

- plan first
- reviewable diff
- no silent writes
- preserve separation between analysis and code mutation

## MCP tools

By the end of this phase, the lean local-first surface should be coherent:

```text
resolve_model
get_model_profile
compare_models
recommend_models

scan_application

analyze_prompt
prepare_prompt_migration

analyze_invocation
prepare_invocation_migration

validate_prompt

generate_migration_plan
generate_migration_report

propose_registry_update
```

Additional conversion tools may be exposed if stable.

## Tests

### End-to-end fixture migrations

Examples:

1. Anthropic direct → newer Anthropic model
2. Anthropic direct → Bedrock Claude
3. Anthropic → OpenAI
4. OpenAI → supported alternative

For each:

```text
scan
 → resolve
 → compare
 → analyze
 → prepare
 → validate
 → plan
 → report
```

### Golden migration plans

Store expected `migration-manifest.yaml` outputs.

### Error cases

- unknown model
- ambiguous model
- unsupported target
- missing registry data
- incompatible tools
- invalid structured-output schema
- context-window regression

## Exit criteria

A coding agent can obtain an actionable migration plan for a supported application without manually stitching together individual tool results.

## Completion status

V0.4 was completed in the `v0.4.0` source tree on 2026-08-23. Schema version 2
of `MigrationPlan` is the canonical application-level contract and serializes as
`migration-manifest.yaml`. The shared service composes scan, resolve, compare,
prompt preparation/validation, invocation preparation, tool/output compatibility,
configuration mapping, cross-concern validation, affected files, tests, and
rollout advice. A Markdown report is rendered from the same manifest.

The four documented fixture routes run end to end and have checked-in golden
manifests. Error coverage includes unknown and ambiguous models, missing platform
context, unsupported targets, incompatible tools, unresolved facts, invalid
structured-output schemas, and context-window regressions. CLI and MCP expose
both integrated operations; tests confirm planning does not rewrite application
source files and explicit artifact outputs reject affected source-file targets.
Optional patch generation and standalone full schema-conversion tools were not
required for the exit criterion and remain unimplemented.

---

# 7. V0.5: Evaluation and Regression Framework

## Goal

Answer the question:

> "The migration is mechanically valid, but did it actually preserve application quality?"

## Scope

Implement:

```text
generate_eval_suite
run_migration_eval
compare_outputs
analyze_regressions
```

Potential supporting capability:

```text
estimate_migration_cost
```

## Credential model

Use:

- user-provided credentials
- local environment configuration
- BYOK

Do not require project-owned inference credentials or hosted proxy infrastructure.

## Evaluation inputs

Support:

- existing test/eval datasets
- golden input/output examples
- schema validators
- exact-match/regex checks
- user-defined evaluators
- optional LLM-as-judge evaluators

## Metrics

Where applicable:

- task score
- pass rate
- structured-output validity
- tool-use correctness
- refusal/error rate
- latency
- token usage
- cost

## Regression analysis

Categorize failures:

- format, schema, or parser incompatibility
- hallucination or grounding regression
- instruction-adherence regression
- over-reasoning or under-reasoning
- tool overuse, underuse, or argument errors
- refusal or safety-behavior change
- verbosity change
- context loss or truncation
- parameter or model-capability mismatch
- latency or token-cost regression
- non-deterministic or unclear regression

## Tests

- fake/mock provider clients
- deterministic evaluator fixtures
- result persistence
- comparison math
- invalid output handling
- partial provider failure
- rate-limit/retry behavior
- no credentials leaked into artifacts/logs

## Exit criteria

A user can run the same evaluation corpus against source and target configurations and receive a structured regression report.

## Completion status

V0.5 was completed in the `v0.5.0` source tree on 2026-08-23. Versioned,
credential-free suite, run, comparison, and regression contracts preserve the
V0.4 manifest as their input boundary. The shared service supports deterministic
exact, regex, Draft 2020-12 JSON Schema, and expected-tool-call checks plus
injected custom or optional LLM-judge evaluators. Direct Anthropic/OpenAI runtime
calls use environment-variable credential references, bounded timeouts, and
bounded retries; provider execution remains injectable for fakes and additional
platforms. The built-in adapter accepts only the exact direct-API platform pairs
and rejects Bedrock rather than routing it to Anthropic direct. Tool calls
normalize across provider envelopes, and nested provider parameters cannot
override endpoint identity or secret-bearing fields.

CLI and MCP expose all four V0.5 operations. Tests cover fake providers, stable
goldens, persistence round-trips, endpoint mismatch, comparison and cost math,
invalid output/schema handling, tool and refusal regressions, custom evaluator
isolation, partial provider failure, rate-limit retry, missing credentials, and
secret-safe artifacts/errors. The completion audit also covers Bedrock
fail-closed behavior, nested tool parameters, reserved-field protection,
multi-schema ambiguity, and provider-neutral tool-call normalization.
Standalone `estimate_migration_cost` was deferred because per-run token-cost
estimation was already represented; V1 subsequently adds the canonical workload
estimate while V0.6 owns optimization and broader provider adapters.

---

# 8. V0.6: Optimization and Broader Coverage

## Goal

Use evaluation evidence to improve migrations and expand provider coverage without destabilizing the core architecture.

## Scope

Implement:

```text
optimize_migration
```

Potential behaviors:

- suggest prompt adjustments
- suggest parameter changes
- suggest alternative target model
- identify cost/quality tradeoffs
- rerun selected evaluations
- compare candidate migrations

Expand:

- provider adapters
- model families
- language scanners
- tool schema converters
- structured-output converters

### V0.5 limitation follow-ups

Treat these as required V0.6 broader-coverage work without reopening the
completed V0.5 milestone:

1. **Bedrock runtime evaluation adapter**
   - Add an `amazon-bedrock` `EvaluationExecutor` using user-owned local AWS
     credentials and normal SDK credential resolution.
   - Support the Bedrock model/profile identifiers already represented by the
     migration manifest and preserve the same credential-free run artifacts.
   - Keep unsupported providers/platforms fail-closed; do not route Bedrock
     through the Anthropic direct adapter.
2. **Richer evaluation corpus inputs**
   - Extend `EvaluationCase` with provider-neutral multi-turn messages and
     multimodal image/document inputs while retaining the V0.5 single-text form
     for backward compatibility.
   - Define deterministic serialization, hashing, validation, and provider
     mapping for the richer payloads.
3. **Safe evaluator registration for CLI and MCP**
   - Add explicit local registration/allowlisting for custom and LLM-judge
     evaluator implementations so CLI and MCP callers can select configured
     evaluator names.
   - Never import or execute code paths supplied by an evaluation YAML file or
     MCP request. Unknown or unregistered evaluator names must remain explicit
     skipped/error states.

## Constraints

V0.6 work must remain bounded by:

- user migration objectives
- explicit evaluation metrics
- reviewable recommendations
- configurable run/cost limits
- local BYOK credential boundaries
- explicit evaluator allowlists

Do not create an unbounded autonomous optimization loop.
Do not dynamically load executable evaluator code from untrusted corpus,
configuration artifact, or MCP inputs.

## Tests

- bounded iteration
- reproducibility metadata
- cost/run limits
- no-regression acceptance criteria
- optimization result comparison
- provider-specific integration suites
- fake Bedrock execution plus optional credential-gated Bedrock contract tests
- multi-turn and multimodal corpus round-trip/provider-mapping tests
- backward compatibility for V0.5 text-only suites
- evaluator allowlist, unknown-name, exception-sanitization, and no-arbitrary-import tests
- no AWS/provider credential values in artifacts or logs

## Exit criteria

The system can use observed migration regressions to recommend targeted
improvements and compare a small set of candidate configurations. The supported
evaluation path also covers a reviewed Bedrock route, provider-neutral
multi-turn/multimodal cases, and explicitly registered custom/judge evaluators
through CLI and MCP without loading executable code from request artifacts.

## Completion status

V0.6 was completed in the `v0.6.0` source tree on 2026-08-23. The bounded
`optimize_migration` contract consumes regression evidence and explicitly
supplied candidate runs, enforces candidate/run/cost/quality limits, emits
targeted review recommendations, compares candidate pass rate, regression,
latency, and cost under an explicit user objective, requires the exact baseline
suite hash, and records deterministic reproducibility hashes. The shared
Python service, CLI, and MCP expose the same operation.

Evaluation-suite schema version 2 adds provider-neutral multi-turn messages and
validated text/image/document blocks while accepting V0.5 text-only suites. The
direct Anthropic, direct OpenAI, and Bedrock Converse adapters map that contract
without placing credentials in artifacts. Bedrock uses normal local AWS SDK
credential resolution and an optional `boto3` package extra. Trusted host code
can explicitly register custom and judge evaluators in a process-local name
allowlist. Standalone CLI/MCP processes load only installed evaluator entry
points explicitly named by the local operator; request artifacts cannot supply
imports or executable paths. Unsupported platform pairs and unknown evaluator
names remain fail-closed or explicitly skipped.

---

# 9. V1: Stable End-to-End LLM Migration Toolkit

## Goal

Deliver a stable open-source product that a developer or coding agent can rely on for real migration work.

## Required V1 capabilities

### Knowledge

- versioned model registry
- provenance
- lifecycle knowledge
- platform representations
- migration rules
- reviewed registry proposal workflow

### Application understanding

- repository scan
- normalized application representation
- migration coupling detection

### Model intelligence

- model resolution
- model profiles
- lifecycle checks
- comparisons
- recommendations

### Migration

- prompt migration
- invocation migration
- parameter migration
- tool schema migration
- structured-output migration
- platform configuration migration

### Validation

- static compatibility validation
- semantic migration checks
- explicit blockers/warnings/unknowns

### Planning/reporting

- structured migration plan
- human-readable report
- reviewable candidate diffs/patches

### Evaluation

- evaluation-suite generation/support
- source-target evaluation
- output comparison
- regression analysis
- migration cost analysis

### Optimization

- bounded optimization recommendations

### Interfaces

- stable MCP server
- stable CLI
- importable Python domain/services API

## Completion status

V1 is complete and tagged `v1.0.0` as of 2026-08-29. The stable public
naming scheme preserves the review-oriented V0.x operations
`prepare_prompt_migration`, `prepare_invocation_migration`, and
`generate_migration_plan`; their typed artifacts cover prompt candidates and
semantic diffs plus invocation, parameter, multimodal, tool, output-contract,
and configuration migration without implying application mutation.
Resolved supported tool definitions produce provider-specific Anthropic,
OpenAI, or Bedrock wrapper candidates and tool-choice mappings. Resolved
structured-output schemas produce Anthropic or OpenAI configuration candidates;
unsupported target mappings produce no candidate and an explicit blocker.
Every prepared invocation also carries a credential-safe target platform
configuration contract with SDK/client, credential strategy, environment or
region variable names, source locations, and required changes.

V1 adds explicit point-in-time `check_model_lifecycle` and deterministic
`estimate_migration_cost` contracts to the shared Python, CLI, and MCP surfaces.
Cost estimates use checked-in canonical input/output pricing and retain source
provenance; they fail visibly when a required price is unknown and remain
separate from observed evaluation-run costs.

An offline V1 acceptance test exercises scan, resolution, comparison,
prompt/invocation preparation, validation, integrated plan generation,
evaluation-suite binding, source/target execution through an injected executor,
regression analysis, and bounded optimization. The test also proves that the
analyzed application is unchanged. Direct Anthropic, direct OpenAI, and Amazon
Bedrock remain the supported runtime platform baseline; unsupported platform
pairs fail closed.

---

# 10. Stable V1 MCP Capability Set

V1 exposes one coherent review-oriented naming scheme:

```text
resolve_model
get_model_profile
check_model_lifecycle

scan_application

compare_models
recommend_models
query_live_pricing

analyze_prompt
prepare_prompt_migration
validate_prompt

analyze_invocation
prepare_invocation_migration

generate_migration_plan

estimate_migration_cost

generate_eval_suite
run_migration_eval
compare_outputs
analyze_regressions
optimize_migration

generate_migration_report

propose_registry_update
```

The prepared invocation and integrated plan contracts carry parameter, tool,
structured-output, multimodal, and platform-configuration migration details.
V1 does not add parallel `adapt_*` or `migrate_*` aliases that could imply source
mutation; the established V0.x names therefore remain backwards compatible.

---

# 11. V1 Supported Platform Baseline

At minimum, V1 should provide strong migration knowledge for:

- Anthropic direct API
- OpenAI direct API
- AWS Bedrock representations of supported models

Initial registry development specifically targets:

- Claude Sonnet 4.5
- Claude Sonnet 4.6
- Claude Sonnet 5
- current major GPT models
- Bedrock representations of those models where available/applicable

V1 does not need to support every model provider to be complete.

Depth and reliability of migration support are more important than a long provider list.

---

# 12. V1.1: On-Demand Agent-Researched Migration

## Goal

Allow a user or coding-agent host to supply an application repository plus exact
source and target provider/platform/model identifiers, research only missing or
stale migration-critical knowledge with the user's own agents and tools, and
produce a reviewable migration plan from a validated user-scoped registry
overlay.

V1.1 extends the verified built-in registry; it does not replace it. Common
supported migrations remain offline and reproducible. Long-tail and newly
released models use expiring session knowledge rather than requiring a permanent
checked-in proposal bundle for every model discovered in the market.

## Product decisions

- Research is explicit, bounded, and initiated by the active migration request.
- The local application scan happens before research and determines the relevant
  fact topics.
- The user's agent host supplies models, search tools, credentials, and token
  budget; core workflows require no project-owned inference credentials.
- Agent review can authorize `session_agent_reviewed` knowledge for the current
  user-scoped run.
- Only a maintainer can promote knowledge into the shared
  `canonical_verified` registry.
- Agent agreement is not evidence. Independent source refetch, authority/scope
  checks, deterministic validation, and explicit conflict handling are required.
- V1 stable tools and offline behavior remain backwards compatible.

## Required contracts

### MigrationResearchRequest

Capture:

- exact source and target provider/platform/model/endpoint identifiers
- normalized application-requirements hash
- requested research topics and as-of date
- authoritative-source policy
- agent, source, concurrency, retry, token/cost, and stopping limits

### EvidenceReview

Return one independent verdict per claim:

- `supported`
- `contradicted`
- `insufficient_evidence`
- `stale`
- `incorrectly_scoped`

Each verdict records independently checked source IDs, provider/platform scope,
rationale, warnings, and blockers. A researcher cannot review its own result.

### ResearchConsensus

Deterministically combine research and review artifacts into accepted/rejected
claims, unresolved conflicts, topic coverage, and a session-suitability result.
Majority vote must not resolve contradictory factual evidence. A conditional
arbiter may add separately sourced evidence but cannot waive validation.

### SessionRegistryManifest

Record:

- immutable run ID and base canonical-registry hash
- candidate model and migration-knowledge hashes
- research/review/consensus hashes
- `session_agent_reviewed` or `session_unreviewed` trust level
- creation, expiration, and freshness metadata
- explicit overlay/shadow selections
- unresolved conflicts and blocked topics

### OrchestrationRun

Record resumable stage status, artifact hashes, agent/tool identifiers, usage,
retries, failures, and stopping reason without credentials, secret values, full
webpage copies, or private reasoning traces.

### Pair-specific migration proposal

Extend the research/proposal/review boundary to `MigrationKnowledge`. Validated
source and target profiles do not by themselves establish pair-specific prompt,
parameter, tool, structured-output, reasoning, or platform behavior.

## Orchestration workflow

```text
scan repository locally
  -> normalize application requirements
  -> resolve source/target against canonical registry
  -> identify missing/stale required topics
  -> research source and target model/platform facts in parallel
  -> research pair-specific migration behavior
  -> compile deterministic proposals
  -> independently refetch and review every consequential claim
  -> arbitrate material disputes only when needed
  -> run deterministic schema/evidence/scope/conflict gates
  -> build immutable expiring session registry overlay
  -> generate migration plan/report
  -> optionally evaluate and optimize through existing V1 contracts
```

Research agents receive exact identifiers and normalized requirements rather
than the full repository whenever possible. Repository and webpage content are
untrusted data and cannot redefine orchestration instructions, tool access,
approval policy, or output schemas.

## Agent roles

1. Source model/platform researcher
2. Target model/platform researcher
3. Source-target migration researcher
4. Independent claim-level evidence reviewer
5. Conditional dispute arbiter

Source and target research may run concurrently. Pair-specific research follows
stable identity resolution. Proposal compilation, review aggregation, candidate
validation, overlay construction, and migration planning remain deterministic
service logic rather than agent judgment.

## Interfaces

Expose shared Python services with thin CLI and MCP adapters for operations
conceptually equivalent to:

```text
create_migration_research_request
validate_research_result
validate_evidence_review
build_research_consensus
build_session_registry
generate_migration_plan
```

The final public surface may consolidate names, but stage artifacts must remain
typed, hash-linked, resumable, and independently testable. Ship an agent-neutral
workflow/skill for coding-agent hosts. An optional injected `AgentRunner` may
support headless automation without making any provider runtime mandatory.

## Local storage and lifecycle

Ordinary artifacts live under a user-local `.llm-migrate/runs/<run-id>/`
workspace and are ignored by default. A content-addressed cache deduplicates
normalized evidence and stores retrieval/freshness metadata and content hashes,
not full page archives. Entries expire by local policy. Users can explicitly
persist a run for audit or export a conventional proposal bundle for upstream
maintainer review.

Session overlays never edit `registry/`, never silently replace canonical
profiles, and must expose their selected trust level and provenance in migration
plans and reports.

## Explicitly out of scope

- continuous or scheduled autonomous web crawling
- automatic canonical registry mutation or promotion
- project-owned agent credentials or inference spending
- mandatory hosted orchestration
- majority-vote truth selection
- application source rewriting
- sending repository secrets or credential values to research agents
- treating cached or live third-party pricing as direct-provider pricing without
  source-specific evidence

## Tests

### Contract and deterministic tests

- strict serialization and unknown-field rejection for every new contract
- stable hashes and deterministic consensus ordering
- candidate model and `MigrationKnowledge` validation
- canonical/session duplicate and explicit-shadow behavior
- expiration and stale-evidence handling
- source-reference integrity and topic coverage gates
- high-impact unresolved conflicts block session approval
- session overlays never mutate canonical registry files

### Agent-workflow fixtures

- accurate official evidence
- missing model in the built-in registry
- stale canonical profile
- ambiguous provider/platform identity
- contradictory official sources
- fabricated or unreachable citations
- incorrectly platform-scoped claims
- prompt-injected repository and webpage content
- correlated researcher/reviewer errors
- partial/failed/interrupted agent runs
- cache reuse and expiration

### Integration/golden tests

- exact research request, review, consensus, session manifest, plan, and report
  artifacts
- offline canonical-only path remains byte-for-byte stable
- researched missing-model path reaches planning through a fake agent host
- resumable runs do not repeat completed stages
- budgets, retries, and stopping limits fail closed
- exported upstream bundles still require maintainer approval

## Exit criteria

- A coding-agent host can take a local Python application plus exact source and
  target provider/platform/model identifiers and complete the researched
  planning workflow using only user-configured agents and tools.
- Missing or stale required facts become typed, source-backed, independently
  reviewed session knowledge or explicit blockers/unknowns.
- Every accepted claim is traceable through research, review, consensus, and
  session-manifest hashes.
- Session overlays are immutable for the run, expiring, visibly non-canonical,
  and cannot mutate the built-in registry.
- Canonical promotion remains a separate exported proposal and maintainer-review
  action.
- Existing V1 offline, CLI, MCP, planning, evaluation, and source-immutability
  tests remain green.
- Security fixtures prove that untrusted repository/web content cannot redefine
  instructions, obtain write authority, or leak credential values.

## Current status

Accepted on 2026-08-29 and implemented on the `codex/v1.1-agent-research`
branch the same day, on top of the `v1.0.0` baseline.

Implemented: the typed `MigrationResearchRequest`, `EvidenceReview`,
`ResearchConsensus`, `SessionRegistryManifest`, and `OrchestrationRun`
contracts; deterministic claim-level consensus with a conditional
evidence-adding dispute arbiter; pair-specific `MigrationKnowledge`
compilation from accepted `migration_behavior.*` claims; the immutable,
expiring, hash-verified session registry overlay with explicit shadow
selection; the resumable fail-closed orchestration driver with an injectable
`AgentRunner`; the fake-agent end-to-end workflow, security, budget, and
resumability tests with an exact golden run workspace (`tests/golden/v11/`);
CLI (`llm-migrate research …`, `plan/report --session`) and MCP stage
operations; the agent-neutral host workflow in
`docs/agent-research-workflow.md`; and the bounded opt-in cited-source
refetch adapter.

Deliberately deferred within the documented boundaries: a persistent
content-addressed research cache (the schema placeholder exists), concurrent
agent execution (limits are recorded and enforced as bounds; stages currently
run sequentially), and an orchestrated arbiter stage (arbitration is a typed,
gated manual operation).

---

# 12.1. V1.2: Guided Migration Run Workspace

Motivated by observed coding-agent host behavior on real migrations: hosts
struggled to map vague user model identifiers onto registry knowledge, invented
their own research prompts (colliding across scopes and over-iterating), spread
outputs across ad-hoc files, and never produced the adapted prompt/file
deliverables users actually wanted.

## Scope

- Lenient, registry-first identifier matching (`match_model`): deterministic
  normalization of regional inference-profile prefixes, version suffixes, and
  vague platform spellings; ranked candidates for explicit user confirmation;
  never a silent guess.
- One guided entry point (`start_migration`) that resolves models, scans the
  application, reports whether research is needed and why, and creates the run
  workspace at the default `.llm-migrate/runs/<run-id>/` (user-overridable).
- Deterministic, scope-isolated researcher/reviewer prompt rendering from the
  typed research request (`get_research_prompts` / `research prompts`).
- Adaptation deliverables: a per-file worklist derived from the migration plan,
  validated submissions of complete adapted prompts (`output/prompts/`) and
  adapted application files (`output/files/`), and a finalize step writing the
  manifest plus a report with per-file changes and rationale.

## Boundary

The host agent performs all semantic rewriting; the toolkit derives worklists,
validates submissions deterministically, stores deliverables, and reports. The
application tree is never modified: adaptation outputs are review candidates in
the run workspace, preserving the review-first invariant.

## Current status

Implemented and released as `v1.2.0` on 2026-09-15, with unit coverage for
matching, workspace lifecycle, fail-closed submissions, and report composition.

---

# 12.2. V1.3: Prompt Provenance Discovery

Motivated by a real V1.2 migration run: the application loaded its prompts
through a configuration file (`model_profiles.yaml` → prompt-path values →
loader helper → `yaml.safe_load` → `sys_prompt`/`user_prompt` keys), the
scanner saw the LLM consumers but resolved no prompt file, and the report's
"Prompt changes: None" hid real prompt migration work.

## Scope

- Structured prompt source models: `PromptSource` (path, format, components,
  provenance chain, confidence, origin) and `PromptComponent` (role, key,
  content), carried on `ApplicationAnalysis` (schema version 3) with a
  `PromptDiscoverySummary` coverage report.
- A config scanner for YAML/JSON/TOML that parses documents structurally and
  finds prompt-scoped path references; `.txt`/`.md` remain whole-file prompt
  sources.
- Bounded Python loader/path recognition: `open(...)`, `Path(...).open()`,
  `read_text()`, `yaml.safe_load`/`yaml.load`/`json.load(s)`/`tomllib.load(s)`,
  `Path` joins, `Path(__file__).parent` composition, simple constants, nested
  config subscripts (`profile["prompts"]["multi"]`), and loader-style helper
  calls — never execution, imports, or interprocedural analysis.
- Evidence-based confidence (`high`/`medium`/`low`); low-confidence prompt-like
  files are reported but never become migration tasks.
- Explicit `prompt_sources` overrides (scan/plan/report/run start across
  service, CLI, MCP; persisted in the run's `migration.yaml`) as the escape
  hatch when discovery is incomplete.
- Per-component prompt preparation (`PromptMigrationSpec.source_component`,
  real roles) with structured documents kept structured: candidate
  reconstruction preserves non-prompt values, and adapted-prompt submissions
  fail closed on syntax errors or non-prompt value drift.
- Coverage reporting: every prompt consumer classified `inline`/`source`/
  `dynamic`; migration plans (schema version 3) and reports carry a
  `resolved`/`partial`/`unresolved` coverage state and warn when coverage is
  incomplete instead of implying no prompt work exists. Dynamic prompt content
  (e.g. a built chat history) is a warning-level unknown, not a blocker.
- Evidence-linked claims: model differences carry per-side registry source
  URLs and the report's differences table hyperlinks every claim; migration
  advice carries evidence URLs.
- Minimal, evidence-based prompt adaptation: the deterministic candidate is
  the source prompt verbatim; recommended changes are advice, never applied
  automatically. Prompt tasks list protected structural sections and the
  evidenced migration-knowledge differences; submissions that drop XML-like
  sections or components fail closed unless `allow_restructure` is set with a
  recorded justification ("no evidence, no rewrite").
- Runtime-aware integrity: prompt-lexical capability mismatches warn instead
  of blocking (prompt-enforced JSON needs no native structured-output API);
  validation and adaptation checks compare DECODED runtime prompt values, so
  serialization-only edits are rejected rather than recorded as adaptations;
  prompt tasks carry in-prompt curation findings; and finalization coverage
  derives from the same worklist the agent received.
- Host-agent efficiency: shared prompt guidance is hoisted once per worklist
  (`shared_prompt_guidance`), the guided flow instructs one `list` call with
  no re-listing between submissions, and reviewed no-change files are recorded
  with `submit_adapted_file(unchanged=true)` (content not resent; refused when
  the file still references the source model id) so coverage always converges.

## Boundary

Deterministic static analysis only: the analyzed application is never executed
or imported, and there is no general data-flow engine. Unsupported dynamic
patterns deliberately stay `dynamic`/unresolved and are reported as such.

## Current status

Complete and released as `v1.3.0` on 2026-09-18: scanner/config/provenance
modules, per-component
planning and workspace adaptation, overrides across all surfaces, the
`configured_prompt_app` regression fixture with golden scan output, and unit
coverage for discovery, coverage states, false-positive protection, overrides,
and format-preserving submissions.

---

# 12.3. V1.4: Interactive Blocker Resolution

Motivated by V1.3's honest-but-dead-end blocked runs: the tool correctly
detected migration blockers (e.g. native structured output configured while
the Bedrock target declares it unsupported), labeled the plan `blocked`, and
stopped — with no guided path from `blocked` to a shippable state.

## Scope

- Structured blockers: `MigrationBlocker` (stable deterministic id, code,
  category, message, registry evidence URLs, source locations, machine-readable
  data) replaces `plan.blockers: list[str]`; `MigrationPlan` is schema
  version 4 and `InvocationMigrationSpec` schema version 3, with a rendered
  string view kept for reports and worklists.
- Deterministic resolution: `get_blocker_resolutions` derives, per blocker,
  the question to ask the user and 2–5 evidence-backed options (retarget to a
  capable endpoint/platform of the same model, an alternative model via the
  existing recommendation engine, an evidence-linked redesign task, an exact
  source correction, and always an accept-with-rationale that is never a
  default). Options come only from registry facts; when none exist the
  resolution says so explicitly.
- Durable decisions: `record_blocker_decision` writes `decisions.yaml` in the
  run workspace (blocker id, chosen option, rationale, date). Retarget and
  correction decisions update the run's `migration.yaml` identity
  registry-first; redesign decisions suppress exactly their blocker and inject
  the required, evidence-linked adaptation task on every regeneration; accept
  decisions downgrade the blocker to a prominently reported accepted decision.
  A decision that no longer matches a live blocker is reported stale, never
  silently applied.
- Honest finalization: the manifest carries `decisions`, the report gains a
  Decisions section (accepted risk highlighted, stale decisions loud), and
  `finalize_migration` distinguishes unresolved blockers, decision-resolved
  blockers, and stale decisions. Complexity stays `blocked` only for
  unresolved blockers.
- Workflow surfaces: MCP `get_blocker_resolutions`/`record_blocker_decision`,
  CLI `llm-migrate run blockers`/`run decide`, and start/worklist guidance
  that directs the host agent to present questions, options, and evidence
  verbatim, one blocker at a time, and never to decide for the user or retry
  submissions to clear a blocker.

## Boundary

The tool stays deterministic and never calls an LLM: the host agent is the
interactive party and the user is the decision maker. Fail closed everywhere —
no decision, no suppression; accepts require a rationale; unknown blocker or
option ids are refused. The evaluation subsystem is untouched.

## Current status

Implemented on 2026-09-18: structured blockers across the analyzers, planner,
and consistency checks; the resolver and decision lifecycle in
`core/blockers.py`; service, MCP, and CLI surfaces; golden regeneration for
the schema bump; and unit coverage for capability retarget/redesign/accept,
source correction, stale decisions, fail-closed ids, and the guided
blocked-to-finalized end-to-end run. Hardened the same day against its own
review findings (accept decisions honored at the submission gate, concrete
correction endpoints, location-discriminated blocker ids with merge-on-dedupe,
prompt-task routing for injected redesign tasks, foreign decision logs
refused, placeholder rationales rejected, superseded-decision history, honest
post-decision rationale, and single-scan blocker calls).

---

# 12.4. V1.4.1: Deployment-Run Immediate Relief (patch)

Motivated by a real v1.4.0 migration run of a production Bedrock application,
audited on 2026-09-22: the mechanical, low-risk failures (lost run-state
updates under parallel submissions on a Windows host, naive/aware timestamp
crashes, a tool surface past host inline budgets, rejection messages read as
content judgements, research artifacts resent in full through MCP) got a
patch release ahead of the v1.5 schema work on invocation identity and
configuration coupling.

## Scope

- Durable run-state writes, portable: every run-workspace write goes through
  `core/runstate.py` — atomic writes (temp file + `os.replace`, atomic on
  POSIX and Windows) and an OS-level per-run advisory lock
  (`fcntl.flock`/`msvcrt.locking` on `<run>/.llm-migrate.lock`, released by
  the OS on process exit) around every read-modify-write of `changes.yaml`,
  both decision logs, deliverables, and submission copies. Parallel
  submissions from a concurrent host no longer lose updates. (F5)
- UTC normalization at every MCP/CLI boundary (`core/moments.py`): naive
  host timestamps are interpreted as UTC, aware ones converted, and internal
  comparisons stay aware — no more naive/aware `TypeError` at the
  session-overlay expiry check. (F6)
- Tool-surface diet: the long guided-workflow tool descriptions were cut
  ~40% (submit_adapted_prompt alone from 2.4k to 1.0k chars, server
  instructions -31%) without losing the safety invariants, and the opt-in
  `LLM_MIGRATE_TOOLSET=guided` exposes only the guided-workflow tools for
  hosts with tight inline-tool budgets (14 at v1.4.1; 19 since the
  v1.5.0 additions). (F10)
- Rejection-message tone: disposition/annotation rejections state explicitly
  that they are submission-format requirements of the tool, not judgements
  on the adaptation content, so host agents fix the payload instead of
  bouncing "please refine your prompt" to the user.
- Research artifact validation by path: `validate_research_artifact
  (run_dir, scope)` (MCP + `llm-migrate research validate-artifact`) reads
  the researcher and reviewer YAML from the run workspace and runs the
  existing deterministic gates; the researcher/reviewer prompts now steer
  agents to it, ending full-artifact resends through MCP. (F8, partial)

## Boundary

No schema versions change, no registry changes, no new workflow stages; the
object-based validators remain for hosts that pass artifacts inline.

## Current status

Implemented on 2026-09-22 and released as `v1.4.1` the same day: the
`runstate`/`moments` core modules, locked submission/decision persistence
across workspace, blockers, and change review, boundary normalization across
MCP and CLI, the trimmed tool surface with the guided toolset, and unit
coverage for atomicity, cross-thread lock serialization, timeout, tone,
path-based validation, naive-timestamp boundaries, and the guided toolset.

---

# 12.5. V1.5.0-a: Invocation Identity

Motivated by the highest-priority finding of the audited v1.4.0 production
Bedrock run: canonical identity (which model this is) was conflated with the
invocation selector (which id the platform accepts for an on-demand call).
Matching correctly normalized regional inference-profile prefixes away, but
the run identity then recorded the bare id, and every downstream surface —
worklist required changes, submission model-id checks, the sanctioned swap,
unchanged-claim guards, the report — enforced referencing an id the platform
cannot invoke on demand: toolkit-induced hallucination.

## Scope

- Registry schema: `PlatformAvailability` gains an optional `invocation`
  block — `bare_on_demand_supported`, named `selectors` (full invocable ids,
  e.g. regional inference profiles), per-fact sources. Fail-open: no block
  means today's behavior plus a report warning; only a reviewed
  `bare_on_demand_supported: false` hard-requires a selector (the schema
  refuses `false` without selectors, duplicate names, duplicate ids).
- Registry machinery: checked-in JSON Schema regenerated (parity test),
  evidence-backed proposal-bundle updates for every canonical profile that
  gained invocation facts (AWS inference-profile documentation for the
  Bedrock representations; Anthropic/OpenAI model pages for the direct
  APIs), fixture coverage (selector-requiring, bare-supported, and
  fact-free fixtures), and regenerated goldens.
- Run identity: `migration.yaml` records the invocation model id, the
  chosen selector, whether the target requires one, and the reviewed
  spelling set per side (all additive; pre-v1.5 configs still load). The
  user's original spelling seeds the selector (typing `us.anthropic...`
  selects the `us` profile); a selector-requiring target with several
  selectors and no derivable choice returns `needs_confirmation` at start
  with one candidate per selector, and a legacy run rides an
  `invocation_selector_required` blocker whose options (one correction per
  selector, evidence-linked) record the choice durably.
- Resolution: selector ids are exact registry identifiers
  (`IdentifierMatchType.INVOCATION_SELECTOR`), so `resolve_model` and
  `match_model` accept the full inference-profile spelling and pin its
  platform representation.
- Downstream enforcement: the worklist carries the selector-qualified
  invocation id (and rewrites plan-derived guidance that mentions the bare
  id), submissions reject adapted content that references a bare id the
  reviewed profile forbids (selector-qualified references stay untouched),
  identity matching everywhere works on the spelling set (bare id plus every
  selector form) for both sides — unchanged-claim guards, source-remains
  warnings, target-reference warnings, and the sanctioned structured-config
  model-id swap all accept or refuse by reviewed spellings — and the report
  records the invocation identity (or warns when facts are unknown).

## Boundary

Honestly bounded: only reviewed selectors count (no invented prefixes), an
absent invocation block never blocks (warning only; strict mode arrives in
v1.5.0-d), and the source side never requires a selector — its spellings only
widen the "no source configuration remains" checks.

## Current status

Implemented on 2026-09-22 and released as `v1.5.0` the same day: schema
and validators, the
`core/invocation_identity.py` derivation/spelling/bare-reference machinery,
resolver selector matching, run-identity seeding and the confirmation flow,
the `invocation_selector_required` blocker with selector-correction options,
worklist/submission/report enforcement, canonical registry facts with
updated proposal bundles, regenerated schema and goldens, and unit coverage
for the schema, derivation rules, bare-reference detection, resolution,
seeding, enforcement, the legacy-run blocker path, and the sanctioned
selector-qualified swap.

---

# 12.6. V1.5.0-b: Semantic Configuration Coupling and Consistency

Motivated by F2/F3/F4/F7 of the audited v1.4.0 production run: the config file
that carried the model id, sampling, budgets, and pricing was invisible to the
scanner; breaking model differences reached only prompt guidance; nothing
checked the deliverables against each other; and incidental SDK imports cost a
full submission round-trip each.

## Scope

- Config couplings with precision guardrails: anchored structured-config
  documents (a registry-matched model-id spelling appears, or values traced
  into a detected invocation call chain through the same loader recognition
  prompt discovery uses) contribute model-id, sampling, token-budget,
  region/routing, and pricing couplings; pricing-shaped values without a
  model anchor are never couplings. Anchored config files become worklist
  tasks and feed the source-model requirements.
- One difference-propagation mechanism: every material model difference
  attaches a required change to exactly the files whose detected couplings
  it governs (the location mapping the plan already computes); only
  prompt-governing differences travel as shared prompt guidance; a
  difference with no mapped coupling stays a report-level fact, never a
  phantom task.
- Worklist diet: files whose only coupling is an incidental SDK import are
  listed once under `unaffected_files` (worklist schema 4) and closed with
  one `confirm_unaffected(run_dir, paths, rationale)` call (MCP + CLI) that
  records per-file reviewed no-change entries through the existing unchanged
  guards; a full adaptation may still be submitted for any of them, and
  coverage accounting flows through the same derived worklist.
- Cross-surface consistency gate at finalize (`core/consistency.py`):
  deterministic checks over the whole effective deliverable set — no active
  source-model configuration remains (spelling-set matching, catching
  review-rejection reverts), no forbidden bare target references, one
  selector-qualified target across every surface, model-coupled files still
  name the target, and scanner-recognized coupling markers cannot vanish
  without an explicit disposition. Findings land in the report's consistency
  section and the finalization payload; strict mode (V1.5.0-d) turns them
  into violations.

## Current status

Implemented on 2026-09-22 and released as `v1.5.0` the same day, with
scan/v04/v05/v11 goldens regenerated and unit
coverage for anchoring rules, usage anchoring, the pricing guardrail, the
unaffected bucket and its confirmation flow, difference propagation, and the
consistency gate's revert/mixed-selector/dropped-marker findings.

---

# 12.7. V1.5.0-c: Workflow Economics

Motivated by F8/F9: the audited run paid one full plan derivation per
submission, serial one-at-a-time decision round-trips, schema-shape research
retries, and per-item disposition costs.

## Scope

- Worklist snapshot with a complete staleness key (`core/snapshot.py`):
  `list_adaptation_tasks` persists the derived worklist and plan evidence,
  keyed by a content hash over the scanned application files, the run's
  `migration.yaml`, the blocker decision log, the session manifest, and the
  registry content. Submissions and status checks reuse it until the key
  changes; task statuses are recomputed from the adaptation log at read time,
  never baked in; unreadable or foreign snapshots are discarded. Supersedes
  the v1.4.0 "one scan per submission" accepted trade-off.
- Batching with explicit semantics: `submit_adaptations` validates every
  item independently (per-item accept/reject, never all-or-nothing) and
  applies the accepted subset in ONE locked, atomic write to `changes.yaml`
  (the workspace submissions split into prepare + apply);
  `record_change_decisions` batches review decisions sequentially with
  per-decision results; canonical-shadow selections surface together in one
  error instead of one per retry.
- Disposition economics with accountability preserved: shared prompt
  guidance is disposable once per run, and `default_disposition` covers
  every unlisted guidance id — expanded into full per-item records marked
  `defaulted` in `changes.yaml` and flagged DEFAULTED in the report.
- Research prompts embed the exact JSON Schema of the expected artifact,
  generated from the same Pydantic models the toolkit validates with.
- `get_run_status` (MCP + CLI `run status`): the state machine (research →
  blockers → tasks → review → ready_to_finalize) with the single next
  action, derived from the snapshot so it is cheap.

## Current status

Implemented on 2026-09-22 and released as `v1.5.0` the same day, with unit
coverage for snapshot reuse and every
staleness input, read-time statuses, batch per-item semantics, default
expansion and report marking, shared-guidance once-per-run, and the state
machine walk.

---

# 12.8. V1.5.0-d: Strict Production Mode

The policy layer over -a/-b/-c: a production run must not ship on warnings.

## Scope

- An opt-in `strict` flag recorded in `migration.yaml` (additive; MCP and
  CLI `run start --strict`). In strict mode: unknown evidence URLs are
  rejected at submission; missing invocation facts on the target profile
  and incomplete prompt coverage become blockers (riding the normal
  resolution flow, accept always offered); and finalization lists every
  unmet requirement — coverage gaps, unconfirmed unaffected files,
  consistency findings, undecided changes, and a missing validation
  disposition — as loud strict violations.
- Validation disposition: `record_validation_disposition` (MCP + CLI `run
  record-validation`) durably records how the migration was validated —
  BYOK evaluation run, user-executed generated contract tests, or an
  explicit accept that REQUIRES the user's own rationale. The report gains
  a Validation section.
- Contract-test deliverable: finalize deterministically emits a mocked
  request-shape test under `output/validation/` from the prepared
  invocation representation (supported invocation mappings only): expected
  invocation id and operation, forbidden source/bare ids, statically mapped
  parameters. The user wires `build_request()` to their adapted application
  and runs it themselves; the toolkit never executes application code.
- The guided workflow surfaces the evaluation stage as an explicit
  post-review step (workflow instructions step 7, run-status next actions).

## Boundary

The toolkit stays deterministic, local-first, and review-only: no
application execution, no provider calls, no host-identity attestation.

## Current status

Implemented on 2026-09-22 and released as `v1.5.0` the same day, with unit
coverage for the strict flag, the
missing-invocation-facts and incomplete-coverage blockers, strict evidence
rejection, the validation disposition lifecycle, strict finalization
violations, and the generated contract test's shape.

---

# 12.9. V1.5.1: Review-Fix Patch

## Purpose

Close the three high-severity defects a same-day multi-axis review of the
released v1.5.0 confirmed by reproduction, before hosts build on the broken
behavior.

## Scope

- Alias-aware reference detection: `migration.yaml` additively records
  source/target REFERENCE spellings (invocable platform spellings plus the
  reviewed canonical name and registry aliases). Source-reference detection —
  unchanged-claim guards, lingering-source warnings, the cross-surface
  consistency gate, the generated contract test's forbidden ids — uses
  reference spellings; invocable enforcement (bare-id rejection,
  sanctioned-swap targets, selector candidate lists) keeps using only
  platform spellings. Detection excludes source spellings contained in a
  target spelling so cross-platform same-model migrations are never falsely
  flagged.
- Reviewed new files: a `new_file` submission requires at least one
  evidence-linked annotated change (an anchor-less insert/restructure claims
  the whole file), so invented content cannot enter the deliverable set with
  nothing for the user's change review to decide. The rejection is a
  submission-format requirement.
- Read-only run status: "moved past research" is an explicit signal — a
  marker written only by `list_adaptation_tasks`, a recorded blocker
  decision, or a submission — never the existence of the worklist snapshot,
  which `get_run_status` itself persists as a cache.

## Exit criteria

- A file naming only a registry alias of the source model cannot be recorded
  `unchanged=true`, and the consistency gate reports the alias.
- Aliases never appear in invocable spelling sets, selector candidate lists,
  or sanctioned-swap targets.
- A `new_file` submission without annotated changes is rejected with the
  format-requirement tone; one with an evidence-linked annotation is
  accepted and reviewable.
- Two consecutive `get_run_status` calls on a research-pending run both
  report `research_pending`; `list_adaptation_tasks` (or a submission or a
  blocker decision) moves the state on.

## Current status

Implemented on 2026-09-22 and released as `v1.5.1` the same day, with unit
coverage for every exit criterion (`tests/unit/test_v151_review_fixes.py`).

---

# 12.10. V1.5.2: Precision and Honesty Patch

## Purpose

First tranche of the next roadmap, driven by a real v1.5.1 guided run of a
production Bedrock application: stop the noise and dishonesty that made the
run's report untrustworthy, and put a deterministic net under prompt
discovery, without schema-version bumps.

## Scope

- Prompt consumers only on model invocations or provider-client calls.
- Out-of-scope classification for prompts referenced only by another model's
  profile, and for unreferenced candidates beside selected siblings.
- Scan-answered unknowns not emitted; XML formatting finding dropped on
  same-provider moves.
- Known evidence widened to registry-recorded URLs; per-citation
  verification state in change review.
- Evidence-backed validation dispositions and finalize-time re-check.
- Whole-application source-reference sweep with an explicit acknowledgement
  path for intentional mentions.
- Guidance applicability pre-disposition under resolved coverage.
- "Action required" report section, unknowns Action column, honest run
  status, two-pass validation legibility, relative MCP run paths, per-host
  registration docs.

## Exit criteria

- On a fixture whose helpers take `input=`, only invocation-gated sites are
  prompt consumers.
- Foreign-profile and sibling-candidate prompt files are out of scope; a
  candidate with no selected sibling stays an unknown.
- An app with no tools emits no `parallel_tool_use` unknown; a same-provider
  run carries no XML formatting finding.
- A registry-recorded citation is not unknown, and review shows each
  citation's state.
- `generated_tests` is refused without `run_passed`; `byok_evaluation` is
  refused without, or with a mismatched, evaluation run; a legacy record
  renders NOT VALIDATED.
- An untraced application file naming the source model is a
  `source_reference_uncovered` finding that strict mode blocks on and that
  closes only through an explicit acknowledgement with a rationale; worklist
  tasks cannot be acknowledged away.
- Inapplicable registry guidance is pre-disposed only under resolved
  coverage.
- Status warns on incomplete prompt coverage and states the two-pass flow.

## Current status

Implemented on 2026-09-23 and released as `v1.5.2` the same day, with unit
coverage for every exit criterion (`tests/unit/test_v152_precision_honesty.py`).
No schema version changed; every contract change is an additive field.

---

# 12.11. V1.6.0-a: Prompt Discovery That Survives Real Repositories

## Purpose

Second tranche of the roadmap driven by the 2026-09-23 field run, whose
config-driven prompt library (repo-root-relative paths, a loader in one
module, consumers in another) produced zero prompt tasks and said so only
at finalize. Make that pattern resolve with zero configuration, make every
remaining discovery gap a start-time or state-machine decision, and let a
live run fix discovery in place.

## Scope

- Bounded multi-base path resolution: config-directory and application-root
  bases, then leading-segment stripping (at most three segments, only when
  they equal the application root's trailing components, only into the
  scanned file set); the resolving rule recorded in provenance; stripped
  resolutions rank medium, never high.
- Separator- and case-robust values: backslashes normalized; case folding
  only on a filesystem detected case-insensitive, always resolving to the
  scanned spelling.
- Chained single-file loaders (`open(p).read()`, `handle.read()`,
  `Path(p).read_text()`), assigned or passed directly to a model call.
- Consumer access keys in PROMPT coupling metadata and unique key-match
  promotion (low -> medium, recorded evidence; ambiguous matches promote
  nothing).
- Start-time prompt-source confirmation (`prompt_candidates`,
  `defer_prompt_candidates`), writing nothing until confirmed.
- Live-run discovery: `add_prompt_sources` (sources and rationale-bearing
  dismissals) and `confirm_prompt_consumer`, recorded in `migration.yaml`
  and re-deriving the worklist in place; CLI `run add-prompt-source` and
  `run confirm-consumer`. The guided toolset grows to 21 tools.
- The `discovery_incomplete` run state, ordered after research and before
  blockers.
- A Windows CI leg for the resolution, scanner, discovery, and run-state
  suites.

## Boundary

Still no execution and no general dataflow: anything beyond the enumerated
forms stays dynamic and remains under the v1.5.2 whole-application
source-reference sweep. Typed actionable unknowns remain v1.6.0-b.

## Exit criteria

- The field-pattern fixture yields six prompt migrations (three prompt
  files, two components each) with zero configuration, three prompt
  consumers, the four foreign or sibling llama files out of scope, and the
  stripping rule recorded in provenance.
- Stripping never resolves outside the scanned file set or past three
  segments, and requires the dropped prefix to match the application root.
- Backslash config values and the chained-loader probe forms resolve.
- A start with consumers, zero resolved sources, and candidates returns
  `needs_confirmation` with the candidates and writes nothing; the retry with
  `prompt_sources` proceeds.
- `add_prompt_sources` on a live run re-derives the worklist, keeps prior
  entries, and adds pending prompt tasks; dismissals require a rationale,
  refuse unknown targets, and are reported out of scope with the rationale.
- `discovery_incomplete` is entered only under its condition (a non-strict
  dynamic chat-history application with no candidates never is) and closes
  through an addition, a consumer confirmation, or a dismissal.
- The Windows CI leg runs green.

## Current status

Implemented on 2026-09-23 (ships with the v1.6.0 release), with unit
coverage for every exit criterion (`tests/unit/test_v160a_prompt_discovery.py`)
and golden scans for the two new fixtures (`field_pattern_repo`,
`probe_forms_app`). No schema version changed except the worklist snapshot
cache (v2, so pre-v1.6.0-a snapshots re-derive); every contract change is an
additive field (`PromptDiscoverySummary.dismissed_consumers`,
`MigrationRunConfig.prompt_consumer_confirmations` /
`prompt_discovery_dismissals`, `MigrationRunStart.prompt_candidates`,
`AdaptationTaskList.dynamic_prompt_consumers`, the `RunStatus` discovery
fields and `discovery_incomplete` state, and PROMPT coupling
`access_keys` metadata).

---

# 12.12. V1.6.0-b: Unknowns That Give Directions

## Purpose

The field run's unknowns were dead ends: one-line strings with no subject,
no reason grounded in the application, no action, and no closing condition;
and an empirically testable conflict (can adaptive thinking be disabled on
Bedrock, where the provider docs and the AWS model card disagree) handed the
user nothing to run and nowhere to record the answer.

## Scope

- `MigrationUnknown` (plan schema 4 → 5): stable id, subject,
  why it matters, action, closing condition, status (`open` /
  `closed_by_scan` / `closed_by_action`) with its reason, evidence URLs;
  `message` keeps the pre-v5 text as the rendered view.
- Emitters for unreferenced candidate prompt files, per-consumer dynamic
  prompt consumers, unknown invocation compatibility, unknown model
  differences (scan-answered ones kept `closed_by_scan`), and registry
  evidence conflicts scoped to the target platform.
- Actions as exact MCP tool calls (or a CLI command for probes), with the
  run directory substituted in guided runs.
- BYOK probe scripts under `output/probes/` for testable contested facts;
  `record_observation` (MCP, CLI `run record-observation`) storing
  run-scoped `observations.yaml`. The guided toolset grows to 22 tools.
- Closing loop: worklist `unknowns`, `RunStatus.open_unknowns`, finalize
  lists open / closed-by-action / closed-by-scan, and the report's
  "Unresolved unknowns" table (Why it matters / Action / Closes when),
  "Closed unknowns", and "Observations (run-scoped)" sections.

## Boundary

Observations never mutate the registry; promotion is an explicit
`propose_registry_update`. Probes are emitted, never executed, and only for
a contested setting and target call `core/probes.py` knows how to exercise.

## Exit criteria

- Every emitted unknown has all four fields populated, asserted over every
  emitter on the fixture set.
- Every action in a run is a syntactically valid call to a real MCP tool
  with valid parameters and the run's directory, or a CLI command naming an
  existing run file; executing a candidate's action closes it.
- A probe is emitted only for contested-evidence unknowns (none for a
  platform the conflict does not govern).
- A recorded observation closes its unknown, renders in the report, and
  leaves `registry/` byte-identical.

## Current status

Implemented on 2026-09-23 (ships with the v1.6.0 release), with unit
coverage for every exit criterion
(`tests/unit/test_v160b_actionable_unknowns.py`). `MigrationPlan` moved to
schema 5 (v04/v05 goldens regenerated); everything else is additive
(`AdaptationTaskList.unknowns`, `RunStatus.open_unknowns`, the
`MigrationRunFinalization` unknown lists and `probe_paths`,
`WorklistSnapshot.probe_paths`) plus the new run-scoped observation
artifact (versioned "1").

---

# 12.13. V1.6.0-c: Review Integrity and Validation Teeth

## Purpose

The field run accepted a pricing edit that contradicted the run's own
registry facts while citing unknown evidence, adapted code around a
contested fact without warning, recorded validation that never ran, and gave
prompt tasks thin pair-specific guidance.

## Scope

- Unit-aware registry-contradiction gate over adapted PRICING values
  (consistent / stale / contradiction with factor / unit unrecognized),
  scoped to governing profiles, cleared only by registry-recorded or
  plan-carried evidence, with registry price expiry noted; new consistency
  codes `pricing_contradicts_registry`, `pricing_stale_source`,
  `pricing_unit_unrecognized` (strict blocks).
- CONTESTED marks on `ChangeReviewItem.contested` and in the report's
  "Action required", cleared by a recorded observation.
- `scaffold_evaluation` (MCP; CLI `run scaffold-eval`) and the
  `validation_pending` run state after the first finalize. The guided
  toolset grows to 23 tools.
- The `prompt_guidance` pair-knowledge topic with `applies_when`, first
  content for claude-sonnet-4-6 → claude-sonnet-5 through a reviewed pair
  bundle (`.registry-proposals/migrations/`).

## Boundary

Only values the scanner's PRICING couplings identified are compared; the
gate never guesses a unit. Prompt-guidance content is limited to what the
official pages state; the scaffold drafts cases, it never runs them.

## Exit criteria

- The field pricing edit (0.0022/0.011 per 1K against $2/$10 per 1M) is a
  contradiction reporting 1.1x; the source price under any scale is stale;
  an unrecognized scale is reported; registry-recorded evidence clears the
  finding and unknown evidence does not; strict mode blocks.
- A change exercising a contested setting is marked CONTESTED (citing the
  conflict's source alone is not), and a recorded observation clears it.
- The 4.6 → 5 `prompt_guidance` knowledge lands through an approved bundle
  whose candidate equals the canonical file, reaches prompt tasks with
  evidence, and is pre-disposed only when its trigger is absent.
- After the first finalize, status is `validation_pending`; the scaffold
  drafts cases from sample folders only, bound to the current plan hash.

## Current status

Implemented on 2026-09-23 (ships with the v1.6.0 release), with unit
coverage for every exit criterion (`tests/unit/test_v160c_review_integrity.py`)
and the extended bundle test (`tests/unit/test_checked_in_proposals.py`).
Additive only: `ModelDifference.applies_when`,
`MigrationKnowledgeItem.applies_when`, `ChangeReviewItem.contested`,
`MigrationRunFinalization.contested_changes`,
`WorklistSnapshot.contested_facts`, the `validation_pending` state, the new
consistency codes, and the `EvaluationScaffold` result; v04/v05/v11 goldens
regenerated for the new optional field.

---

# 13. Cross-Phase Testing Strategy

## Unit tests

For:

- schemas
- loaders
- resolvers
- adapters
- diffing
- validators
- parameter mappings
- converters
- scoring

## Golden/snapshot tests

For:

- registry proposals
- application scan outputs
- comparisons
- migration analyses
- migration plans
- reports

## Fixture repositories

Maintain intentionally small sample apps representing supported integrations.

Examples:

```text
fixtures/apps/
├── anthropic_python_basic/
├── anthropic_tools_structured/
├── openai_python_basic/
├── openai_tools_structured/
├── bedrock_claude_basic/
└── multimodal_example/
```

## Integration tests

Test workflows across components, not only functions.

## Provider contract tests

When external credentials are available in CI or developer environments, run optional provider integration tests separately from the default offline suite.

## Regression tests

Every real bug found during a migration should become:

- a fixture
- a unit test
- a golden test
- or a provider contract test

as appropriate.

---

# 14. Phase Discipline for Codex

When working from this roadmap, Codex should:

1. implement only the current milestone plus minimal prerequisites,
2. reuse existing domain contracts rather than create parallel schemas,
3. add tests with every capability,
4. prefer fixture-driven development,
5. avoid provider-specific conditionals leaking across the codebase,
6. avoid adding hosted services unless the roadmap is explicitly changed,
7. avoid adding continuous model research/update automation in V0.x,
8. keep canonical registry updates reviewable,
9. avoid implementing evaluation before migration plans are structurally stable,
10. avoid building a frontend before the MCP/CLI workflow is useful,
11. document architectural changes when they alter these phase boundaries,
12. treat `architecture.md`, `project_context.md`, and `project_phases.md` as the product contract unless a newer explicit project decision supersedes them,
13. keep V1.1 agent research explicit, bounded, user-funded, and separated from
    deterministic validation and canonical promotion,
14. preserve the offline canonical-only path while adding session overlays.

---

# 15. Practical Milestone Summary

| Version | Primary outcome |
|---|---|
| V0 | Clean foundation and domain contracts |
| V0.1 | Verified model registry, provenance, research proposal workflow |
| V0.2 | Model intelligence and repository scanning |
| V0.3 | Prompt/invocation migration analysis and preparation |
| V0.4 | Integrated migration plan, validation, and reporting |
| V0.5 | Runtime evaluation and regression analysis |
| V0.6 | Bounded optimization and broader coverage |
| V1 | Stable end-to-end local-first migration toolkit |
| V1.1 | Bounded user-side agent research with independently reviewed session registry overlays |
| V1.2 | Guided migration run workspace: lenient matching, generated research prompts, adaptation deliverables |
| V1.3 | Prompt provenance discovery: config-driven prompt sources, structured prompt documents, coverage reporting |
| V1.4 | Interactive blocker resolution and evidence-backed per-change adaptation review |
| V1.4.1 | Deployment-run immediate relief: durable run state, UTC boundaries, tool-surface diet |
| V1.5 | Invocation identity, semantic config couplings, worklist snapshot, consistency gate, strict production mode |
| V1.5.1 | Review-fix patch: alias-aware reference detection, reviewed new files, read-only run status |
| V1.5.2 | Precision and honesty patch: consumer precision, prompt scoping, evidence-backed validation, whole-app source-reference sweep |
| V1.6.0-a | Prompt discovery on real repo layouts: bounded multi-base resolution, chained loaders, start-time confirmation, live-run discovery tools, `discovery_incomplete` |
| V1.6.0-b | Unknowns that give directions: typed actionable unknowns (plan v5), contested-evidence probes, run-scoped observations |
| V1.6.0-c | Review integrity: unit-aware pricing contradiction gate, CONTESTED review marks, evaluation scaffold and `validation_pending`, pair `prompt_guidance` knowledge |

The critical sequencing rule is:

> **Understand the application → use verified model knowledge or research what is missing → prepare the migration → validate the plan → evaluate behavior → optimize.**

Do not reverse that order.
