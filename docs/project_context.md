# LLM_Migration Project Context

## 1. Project Summary

**LLM_Migration** is an open-source, local-first, provider-neutral toolkit for migrating LLM applications across models and platforms.

It is best thought of as an:

> **LLM application migration compiler + migration intelligence layer**

rather than a model leaderboard, prompt rewriter, API wrapper, or autonomous web-research agent.

The project helps developers and coding agents answer:

- What in this application is coupled to the current model or provider?
- Which target models are genuinely compatible with the application's requirements?
- What needs to change in prompts, invocation code, parameters, tools, and structured outputs?
- What behavior cannot be migrated one-to-one?
- What should be tested before switching?
- What are the likely cost, quality, latency, and lifecycle tradeoffs?
- How can the migration be represented as a reviewable plan instead of ad hoc edits?

---

## 2. Problem Being Solved

Migrating an application from one LLM to another is not equivalent to changing:

```python
model = "old-model"
```

to:

```python
model = "new-model"
```

Real applications accumulate model/provider coupling in:

- prompts
- system-message conventions
- generation parameters
- SDKs
- request/response formats
- reasoning controls
- tool/function schemas
- tool-choice semantics
- structured outputs
- streaming
- multimodal input formats
- context-window assumptions
- parser behavior
- retries
- error handling
- pricing assumptions
- deployment platforms
- model lifecycle and deprecations

Provider documentation describes individual APIs. It rarely answers the application-level question:

> "Given this existing application, what exactly must change to migrate it safely to this target?"

LLM_Migration fills that gap.

---

## 3. Product Positioning

### It is

- an LLM migration compiler
- a model compatibility knowledge base
- an application scanner
- a migration planning tool
- an MCP toolkit for coding agents
- a CLI for developers
- a provider/platform translation layer
- an evidence-backed model registry
- an evaluation and regression framework

### It is not

- a generic chat UI
- a hosted inference gateway
- a replacement for LiteLLM
- a generic model leaderboard
- an automatic "pick the best LLM" product
- an unrestricted autonomous browsing agent
- a continuously self-mutating model database
- a system that assumes prompts are the only migration concern
- a system that automatically edits production applications without review

---

## 4. Intended Users

### Primary users

#### AI/ML engineers and software engineers

Developers maintaining applications that use:

- Anthropic
- OpenAI
- AWS Bedrock
- later additional LLM providers/platforms

#### Teams performing model migrations

Examples:

- upgrading from an older model generation
- switching providers
- moving from direct provider API to Bedrock
- moving from Bedrock to direct provider API
- replacing a deprecated model
- reducing inference cost
- reducing latency
- adopting better tool use or structured outputs
- standardizing multiple applications on a target model/platform

#### Coding agents

Primary agent clients include:

- OpenAI Codex
- Claude Code
- GitHub Copilot or other MCP-compatible development agents

The long-term interaction model is that a coding agent can inspect the repository and call LLM_Migration tools as specialized migration expertise.

### Secondary users

- AI platform teams
- architecture teams
- developer enablement teams
- technical leads reviewing migration proposals

---

## 5. Core User Scenarios

### Scenario A: model upgrade

```text
Claude Sonnet version N
        ↓
Claude Sonnet version N+1
```

The tool identifies behavioral/API changes and prepares the migration.

### Scenario B: cross-provider migration

```text
Anthropic Claude
        ↓
OpenAI GPT
```

The tool analyzes prompts, SDK calls, parameters, tools, outputs, and compatibility.

### Scenario C: platform migration

```text
Anthropic direct API
        ↓
AWS Bedrock representation of Claude
```

Underlying model lineage may be similar, but identifiers, platform capabilities, request paths, authentication, regional availability, and feature support may differ.

### Scenario D: model selection before migration

The user knows the source application but not the target model.

The tool analyzes requirements and recommends realistic candidates.

### Scenario E: deprecation/lifecycle migration

A model is being retired or replaced.

The tool identifies the lifecycle issue and prepares successor options.

### Scenario F: evaluate a prepared migration

The toolkit runs source/target comparisons using a user-provided evaluation
corpus and credentials referenced from the local environment.

---

## 6. Supported Platforms and Model Families

The registry must distinguish **model**, **provider**, and **platform representation**.

### Initial platform scope

#### Anthropic direct API

Initial knowledge should include major Claude generations needed for realistic migrations, including:

- Claude Sonnet 4.5
- Claude Sonnet 4.6
- Claude Sonnet 5
- other current major Claude models when added through reviewed registry updates

#### OpenAI direct API

Include current major GPT model families relevant to application migration.

Do not hard-code the project architecture around one naming generation. Model identities and aliases belong in the registry.

#### AWS Bedrock

Include Bedrock representations of supported model families, especially:

- Anthropic Claude models available through Bedrock
- corresponding Bedrock model/profile identifiers
- platform-specific capability differences
- later other relevant Bedrock-native or third-party models

### Later providers/platforms

The architecture must permit expansion to:

- Google / Gemini
- Azure-hosted OpenAI models
- other cloud model platforms
- open-weight/self-hosted models
- additional Bedrock model families

These are extension targets, not requirements for the earliest milestone.

---

## 7. Canonical Knowledge Strategy

### Core rule

> **Registry = verified knowledge. Research = discovery mechanism.**

### Canonical registry

Contains reviewed, normalized knowledge used by migration logic.

Examples:

- model identity
- aliases
- model family
- provider/platform
- lifecycle
- capabilities
- limits
- parameters
- tool support
- structured-output support
- platform identifiers
- migration rules
- provenance

### Research layer

Used to discover missing or stale facts.

Research may be:

- manual
- coding-agent assisted
- scripted
- web-assisted

Research does not directly become truth.

Beginning with V1.1, research may also be initiated on demand for the exact
source and target model/platform pair in a user-requested migration. The local
application scan determines which facts matter, so agents receive normalized
requirements and identifiers rather than the full repository whenever
possible. The user's agent host supplies inference/search capability and pays
its own token or tool costs.

### Proposal layer

`propose_registry_update(...)` converts research into a reviewable diff.

Expected artifacts:

- `proposal.yaml`
- `proposal.md`
- `evidence.yaml`
- `candidate-model.yaml`

Only reviewed changes enter the canonical registry.

### Session registry overlay

V1.1 adds a second, explicitly non-canonical destination for bounded research.
A validated and independently reviewed candidate may be assembled into an
expiring registry overlay for the current user's migration run without being
committed to this repository.

Session knowledge has one of these trust levels:

- `session_agent_reviewed` — claim-level independent review and deterministic
  validation passed for the declared provider/platform/model scope
- `session_unreviewed` — research is present but review, coverage, or conflict
  requirements did not pass

The existing checked-in registry is `canonical_verified`. A session overlay may
supplement or explicitly shadow it only for the current run; the selected source
and trust level must be visible in migration artifacts. It never silently edits
the canonical registry. Upstream promotion still follows the proposal bundle
and maintainer-review workflow.

This hybrid keeps common supported migrations offline and reproducible while
allowing long-tail or newly released models to be researched on the user's side
without committing a permanent proposal bundle for every discovered model.

---

## 8. Inputs

LLM_Migration should support progressively richer inputs.

### Application inputs

- repository path
- local source directory
- individual source files
- prompt files
- YAML/JSON configuration
- Python/TypeScript application code
- model/provider configuration
- tool definitions
- structured-output schemas
- tests/evaluation data references

### Migration inputs

- current model
- current provider
- current platform
- desired target model, if known
- desired target provider/platform
- migration priorities

### Migration priorities

Examples:

```yaml
priorities:
  quality: high
  cost: medium
  latency: medium
  minimize_code_change: high
  preserve_tool_behavior: required
  preserve_structured_output: required
```

### Optional evaluation inputs

- golden input/output pairs
- evaluation datasets
- user-defined graders
- target thresholds
- locally configured provider credentials

V0.5 binds these inputs to the migration-manifest hash in a versioned
`EvaluationSuite`. Credential values never enter the suite, run, or regression
artifacts.

---

## 9. Outputs

The system should generate structured outputs that both humans and coding agents can consume.

### Application analysis

Contains:

- detected provider/model usage
- relevant files
- prompts
- invocation patterns
- parameters
- tools
- output contracts
- migration couplings
- risks
- unknowns

### Model comparison

Contains:

- canonical model identities
- relevant capabilities
- compatibility
- differences
- migration implications
- provenance

### Model recommendations

Contains:

- ranked candidates
- rationale
- hard blockers
- soft tradeoffs
- unknowns
- relevant source evidence

### Migration preparation artifacts

May include:

- proposed prompt changes
- proposed invocation changes
- parameter mappings
- tool-schema changes
- output-schema changes
- file-level change plan
- warnings
- validation results

### Migration plan

The canonical structured machine-readable plan is `migration-manifest.yaml`.

Conceptually:

```yaml
migration:
  source:
    provider: anthropic
    model: claude-sonnet-4-6

  target:
    provider: openai
    model: ...

  required_changes:
    - category: invocation
      file: src/llm.py
      severity: required

    - category: prompt
      file: prompts/extract.yaml
      severity: review

  blockers: []
  warnings: []
  required_tests: []
```

The V0.4 implementation uses schema version 2 of `MigrationPlan` as this
application-level contract. It composes the normalized scan, resolved platform
representations, model comparison, prompt preparations, invocation preparation,
tool/output compatibility, configuration locations, and static validation. YAML
serialization adds the top-level `migration` key shown above. Manifest generation
does not write to the analyzed application.

### Migration report

Human-readable explanation of:

- current architecture
- proposed target
- compatibility
- required code/prompt/config changes
- risks
- unknowns
- evaluation recommendations

The report is deterministically rendered from the manifest so it cannot silently
disagree with the machine-readable plan.

### Later migration package

Later phases may package the manifest and report with reviewable candidate prompts, code patches, evaluation results, and a rollback plan. These are optional companions to `migration-manifest.yaml`; generating the manifest must not imply that analyzed source files were modified.

### Registry proposal artifacts

Research workflow outputs:

- `proposal.yaml`
- `proposal.md`
- `evidence.yaml`
- `candidate-model.yaml`

---

## 10. MCP Tool Surface

The project developed from a smaller early tool set and now exposes the stable V1 surface below.

### Model intelligence

```text
resolve_model
get_model_profile
check_model_lifecycle
compare_models
recommend_models
query_live_pricing
```

`query_live_pricing` is an explicit, on-demand V0.2 network operation backed initially by the public OpenRouter Models API. It reports current OpenRouter-listed pricing with source and retrieval metadata; it does not silently refresh or replace canonical direct-provider registry facts.

### Application understanding

```text
scan_application
```

`scan_application` is a major product capability, not a thin text search utility.

### Prompt migration

```text
analyze_prompt
prepare_prompt_migration
validate_prompt
```

`prepare_prompt_migration` is the stable V1 operation. It
returns a source-traceable review candidate and semantic diff, with explicit
provider/platform and prompt-role context; it never writes the prompt file.

### Invocation migration

```text
analyze_invocation
prepare_invocation_migration
```

V0.3 invocation analysis consumes normalized scan contracts for tool and output
JSON Schemas, parser and strictness assumptions, and multimodal payload formats.
Preparation verifies declared source context and returns a review-only target
representation, mappings, warnings, and blockers. The integrated plan carries
provider-specific target tool and output payload candidates plus explicit
compatibility states. It also carries a typed credential-safe target
configuration contract for direct Anthropic, direct OpenAI, or the AWS default
credential chain, without serializing secret values. Source mutation remains
outside the product.

### Tool and structured-output migration

Tool schemas and structured-output contracts are normalized and assessed by
`analyze_invocation`. `prepare_invocation_migration` emits typed target payload
candidates for supported resolved Anthropic/OpenAI tool and output wrappers and
Bedrock tool wrappers, including target fields and tool-choice mappings.
Unsupported or unresolved cases emit no candidate and remain explicit in
`generate_migration_plan` and its report.

### Cost analysis

```text
estimate_migration_cost
```

### Migration planning/reporting

```text
generate_migration_plan
generate_migration_report
```

### Evaluation and regression

```text
generate_eval_suite
run_migration_eval
compare_outputs
analyze_regressions
optimize_migration
```

V0.5 implements the first four operations and V0.6 implements
`optimize_migration`. `generate_eval_suite` accepts golden
outputs, exact/regex/JSON Schema checks, expected tool calls, and named custom or
LLM-judge evaluator hooks. `run_migration_eval` executes one immutable corpus
against both manifest endpoints using local BYOK configuration. It persists
outputs, validator results, pass/task scores, refusal/error state, latency,
tokens, and estimated cost without serializing credentials. `compare_outputs`
computes paired deltas, and `analyze_regressions` produces the structured report.
`optimize_migration` produces bounded, review-only recommendations and compares
explicitly supplied candidate runs under run, cost, and quality limits plus an
explicit balanced, quality, cost, or latency objective. Candidate runs must
match the baseline evaluation-suite hash.

V1 adds deterministic, provenance-aware canonical workload estimation through
`estimate_migration_cost`; observed per-run cost remains part of evaluation.

Direct OpenAI and Anthropic tool calls normalize to one `{name, arguments}`
shape. The built-in executor supports their direct-API platform pairs and Amazon
Bedrock Converse using normal local AWS credential resolution; other platforms
require an injected executor and fail closed. Nested provider parameters support
tools and output configuration but cannot override manifest endpoint fields.

Custom and judge evaluator code is never selected by an artifact path. Embedded
hosts register trusted callables directly; standalone CLI/MCP processes load
only installed `llm_migrate.evaluators` entry points named in the local
`LLM_MIGRATE_EVALUATORS` allowlist.

### Registry research/review

```text
propose_registry_update
```

This remains the only tool that proposes writes to the shared canonical
registry, and it writes review artifacts rather than canonical registry
records. V1.1 session-overlay construction writes only user-local run artifacts.

### Planned V1.1 researched migration

V1.1 adds bounded stage operations rather than embedding a provider-specific
agent runtime in the core package. The intended conceptual operations are:

```text
create_migration_research_request
validate_research_result
validate_evidence_review
build_research_consensus
build_session_registry
generate_migration_plan
```

The final public names may be consolidated during implementation, but each
stage must remain callable through shared Python services and thin CLI/MCP
interfaces. Codex, Claude Code, or another agent host orchestrates the stages
using the user's configured models and tools. An optional injected `AgentRunner`
may support headless workflows, but no agent provider is mandatory for core
operation.

Research roles are deliberately separated:

1. source model/platform researcher,
2. target model/platform researcher,
3. source-target migration researcher,
4. independent claim-level evidence reviewer,
5. conditional dispute arbiter.

The proposal compiler, schema checks, consensus policy, overlay construction,
and migration planning remain deterministic code. A researcher cannot approve
its own claims, agent majority vote does not resolve a factual conflict, and an
arbiter is invoked only for material disagreement.

---

## 11. Stable V1 MCP Surface

The local-first V1 surface is:

```text
resolve_model
get_model_profile
check_model_lifecycle
compare_models
recommend_models
query_live_pricing

scan_application

analyze_prompt
prepare_prompt_migration

analyze_invocation
prepare_invocation_migration

validate_prompt

generate_migration_plan
generate_migration_report

estimate_migration_cost

generate_eval_suite
run_migration_eval
compare_outputs
analyze_regressions
optimize_migration

propose_registry_update
```

### Planned V1.1 agent-facing surface

V1.1 extends rather than renames the stable V1 operations. It will expose typed
research/review/session stages suitable for an agent host and one integrated
user workflow equivalent to:

```text
repository + source provider/platform/model + target provider/platform/model
  -> local scan
  -> research missing-or-stale required facts
  -> independently review evidence
  -> build user-scoped session registry
  -> generate migration plan/report
  -> optionally generate and run an evaluation suite
```

Ordinary run artifacts belong beneath a user-local `.llm-migrate/` workspace,
which should be ignored by default. Content-addressed cache entries deduplicate
research, carry retrieval/freshness metadata, and expire according to local
policy. Users may explicitly persist a run for audit or export its proposal
bundle for upstream maintainer review.

---

## 12. Example End-to-End User Experience

### User prompt to Codex

```text
Analyze this repository. It currently uses Claude Sonnet through the Anthropic API.
I want to migrate it to an appropriate current GPT model.

Preserve structured output and tool behavior.
Minimize code churn.
Do not change files yet.
```

### Expected agent workflow

```text
scan_application
      ↓
resolve_model
      ↓
if required knowledge is missing or stale:
  create bounded research requests
      ↓
  independent research and evidence review
      ↓
  validated session registry overlay
      ↓
get_model_profile
      ↓
recommend_models
      ↓
compare_models
      ↓
analyze_prompt
      ↓
analyze_invocation
      ↓
prepare_prompt_migration
      ↓
prepare_invocation_migration
      ↓
generate_migration_plan
      ↓
generate_migration_report
```

### Expected result

The user receives a concrete migration plan describing:

- what the repository currently does
- target candidates
- selected target rationale
- required changes
- affected files
- incompatible or changed behavior
- tests required before rollout

The coding agent can then implement the reviewed plan.

---

## 13. Repository Concepts

A reasonable long-term repository layout is:

```text
llm-migration/
├── src/
│   └── llm_migrate/
│       ├── domain/
│       ├── registry/
│       ├── research/
│       ├── scanners/
│       ├── analyzers/
│       ├── migrations/
│       ├── validation/
│       ├── evaluation/
│       ├── reporting/
│       ├── mcp/
│       └── cli/
│
├── registry/
│   ├── models/
│   ├── migrations/
│   ├── schemas/
│   └── sources/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── golden/
│
├── docs/
│   ├── current_state.md
│   ├── architecture.md
│   ├── project_context.md
│   ├── project_phases.md
│   └── research-policy.md
│
└── pyproject.toml
```

Exact folder names may change, but boundaries should remain clear.

---

## 14. Product Non-Goals

Unless the roadmap explicitly changes, LLM_Migration should not become:

- a SaaS product
- a hosted API gateway
- a centralized provider credential store
- a production inference router
- an automated trading-style model optimizer
- a generic RAG framework
- a generic agent framework
- a model benchmarking website
- a continuous crawler of all model news
- an automatically self-updating registry
- an IDE extension requiring a custom frontend
- a migration system for every AI platform on day one

---

## 15. Quality Bar

A feature belongs in LLM_Migration when it improves one or more of:

1. understanding the current LLM application,
2. understanding source/target model differences,
3. selecting a compatible migration target,
4. translating application behavior across models/providers,
5. validating the proposed migration,
6. evaluating whether the migration preserved required behavior,
7. maintaining reliable migration knowledge.

Features that do not materially support these outcomes should be treated as scope expansion.

---

## 16. Long-Term Product Outcome

A mature LLM_Migration should allow a developer to point an AI coding agent at an existing LLM application and say:

> "Migrate this application from model/platform A to model/platform B, preserve these behaviors, and show me the risks before changing anything."

The coding agent should then be able to use LLM_Migration as its specialized migration knowledge and tooling layer rather than inventing provider behavior from memory.
