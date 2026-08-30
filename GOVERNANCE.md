# Governance

`llm-migrate` is currently maintained by
[@Athenaxlee](https://github.com/Athenaxlee).

## Decision making

- Changes to `main` go through pull requests and required CI.
- Maintainers make final decisions on scope, architecture, security, releases,
  and canonical registry knowledge.
- Significant public-contract or architecture changes should be discussed
  before implementation.
- Decisions prioritize user safety, reproducibility, provider neutrality, and
  reviewability over feature volume.

## Registry governance

Canonical registry changes require evidence artifacts and maintainer review.
Agent-reviewed session knowledge is useful for one user-scoped run but cannot
promote itself into the shared registry.

CODEOWNERS review is required for registry, security, workflow, and governance
changes.

## Releases

Releases are cut from protected `main`, use semantic version tags, pass the full
quality workflow, and publish reviewable release notes and artifacts.

## Maintainer growth

New maintainers may be invited based on sustained, high-quality contributions,
sound review judgment, respect for the evidence boundary, and constructive
community participation.
