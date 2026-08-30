# Registry research policy

The governing boundary is:

```text
Registry = verified knowledge
Research = discovery mechanism
```

Normal V1 resolution, comparison, recommendation, and migration operations use
the local, version-controlled registry. Research—whether performed by a person
or an agent—produces a typed `ResearchResult`. That evidence is normalized into
a review-only `RegistryUpdateProposal`; canonical YAML changes require
maintainer review.

V1.1 may use explicitly requested research in an immutable, expiring,
user-scoped registry overlay. Session use does not make a claim canonical. The
plan and report must identify whether knowledge is `canonical_verified`,
`session_agent_reviewed`, or `session_unreviewed`.

The project does not continuously scrape the internet, perform background
discovery, or silently refresh stale facts. V1.1 live research is bounded to an
active user request, exact source/target identities, and application-relevant
topics. Retrieval dates and freshness metadata make uncertainty visible, and
the canonical-only path remains available without a network dependency.

## Source hierarchy

1. Prefer authoritative structured provider or platform metadata.
2. Prefer official documentation for model facts.
3. Use official release notes, migration guides, and engineering announcements for behavioral or migration changes.
4. Use official SDK repositories and issue trackers for implementation-specific behavior and known bugs.
5. Use credible independent research when official sources are insufficient.
6. Use community reports primarily to discover issues, not to establish facts.

Never promote an unverified community claim into a canonical fact automatically. Record contradictions rather than hiding or averaging them. Keep provider capabilities separate from what each platform or endpoint actually exposes. Record retrieval dates and the field paths each source supports. Do not store full copies of webpages.

## Topic-specific priorities

### Pricing

1. Official provider pricing or model page.
2. Official hosting-platform pricing.
3. The OpenRouter Models API may be queried on demand for current OpenRouter-listed pricing and discrepancy discovery.

Platform pricing is not assumed to equal direct-provider pricing.
OpenRouter pricing is scoped to OpenRouter and is not automatically promoted as a direct-provider canonical price. Its `pricing.prompt` and `pricing.completion` values are USD per token and must be normalized explicitly before comparison with registry values.

### Lifecycle

1. Official structured model metadata.
2. Official deprecation and lifecycle documentation.
3. Official release announcements.

“Not sooner than” dates are recorded as earliest possible retirement dates, not exact end-of-life dates.

### Platform and region availability

1. Official platform API.
2. Official platform model card or regional availability documentation.

Endpoint-specific restrictions must be represented as explicit platform overrides.

### Prompting behavior

1. Model-specific prompting documentation.
2. Model-specific migration documentation.
3. Official release notes.
4. Official examples.

Avoid provider-wide folklore and inferred universal formatting rules.

### Known issues

1. Official documentation.
2. Official SDK repositories and issue trackers.
3. Credible independent engineering reports.
4. Community reports.

Known issues that do not establish model facts belong in `registry/observations/`.

## Agent research and review policy

Agent research is permitted only through typed, bounded assignments that name
the provider, platform, model, endpoint when known, relevant fact topics, as-of
date, source policy, and execution limits. Research agents should receive
normalized application requirements rather than full repository contents when
those requirements are sufficient.

Source and target model/platform research may run independently. Pair-specific
migration behavior must be researched separately because two valid model
profiles do not establish prompt, parameter, tool, output, reasoning, or
platform migration behavior between them.

An agent cannot review its own `ResearchResult`. The independent reviewer must
refetch cited sources and issue a verdict for every consequential claim:

- `supported`
- `contradicted`
- `insufficient_evidence`
- `stale`
- `incorrectly_scoped`

The review records checked source IDs, provider/platform scope, rationale,
warnings, and blockers. A dispute arbiter is allowed only when the researcher
and reviewer materially disagree. The arbiter may add independently sourced
evidence but cannot waive schema, source, scope, coverage, freshness, or conflict
requirements.

Agent majority vote is not a truth mechanism. Acceptance is determined by
source authority, directness, independent verification, correct scope,
freshness, deterministic validation, and explicit conflict policy. High-impact
unresolved conflicts remain unknown or block use.

Repository files and retrieved pages are untrusted data. They cannot redefine
orchestration instructions, tools, approval rules, output schemas, token/cost
limits, or write authority. Research stages receive no canonical-registry or
application-source write permission. Credential values, secrets, and private
reasoning traces must not be included in requests or persisted artifacts.

## Session research workflow

```text
local scan and normalized requirements
  -> bounded source/target/platform research
  -> ResearchResult
  -> deterministic RegistryUpdateProposal
  -> independent claim-level EvidenceReview
  -> deterministic ResearchConsensus
  -> immutable SessionRegistryManifest
  -> user-scoped migration planning/evaluation
```

The stage-by-stage host protocol for this workflow is documented in
`docs/agent-research-workflow.md`.

Session artifacts live in the user's `.llm-migrate/` workspace rather than this
repository's `.registry-proposals/` tree. A content-addressed cache may retain
normalized claims, bounded excerpts or source locations, retrieval metadata,
and content hashes. It must not retain full webpage archives by default, and
entries expire according to explicit local policy.

`session_agent_reviewed` knowledge may be used only within the run and scope
recorded by its session manifest. `session_unreviewed` knowledge remains visible
but cannot support consequential compatibility, lifecycle, or cost claims. A
session candidate may shadow canonical knowledge only through an explicit run
selection recorded in the manifest and final plan.

To contribute session research upstream, export a normal four-file proposal
bundle. Agent review may accompany the bundle, but only explicit maintainer
approval can promote the candidate into the shared canonical registry.

## Contributor workflow

```text
research
  -> normalized sources, claims, observations, and conflicts
  -> RegistryUpdateProposal
  -> human review
  -> canonical registry change
```

Use `llm-migrate registry propose-update research-result.yaml` to inspect a proposal. Pass `--output .registry-proposals` only when review artifacts should be written. Proposal generation never edits `registry/`, commits changes, opens pull requests, browses the web, calls a provider, or performs inference.

For research that may change canonical knowledge, writing the review artifacts is
required before editing `registry/`. The canonical review artifact is
`.registry-proposals/<model>/proposal.yaml`; `candidate-model.yaml`,
`evidence.yaml`, and `proposal.md` accompany it. Keep these files versioned so a
maintainer or CI workflow can review the evidence and structured diff without
repeating the research or parsing prose.

Every non-fixture canonical provider profile should have a checked-in review bundle, or an explicit maintainer-approved record explaining why it is grandfathered. Coverage tests should enforce whichever policy is selected.

Checked-in proposal bundles are required only for models intentionally included
in the shared canonical registry. Long-tail models used through a user-scoped
session overlay do not create repository storage or maintenance obligations
unless a contributor explicitly proposes them for canonical support.
