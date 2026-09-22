# Claude Sonnet 4.5 Registry Proposal

## Summary

0 facts added
1 fact changed
0 conflicts detected
3 fields remain unknown

The candidate preserves the currently registered facts and advances the evidence check date to August 9, 2026. The repository has no commit history, so an earlier pre-research baseline could not be reconstructed.

## High-impact changes

No high-impact registry changes were found in the refreshed official evidence.

## Unknown fields

- `parameters.temperature`
- `parameters.top_p`
- `parameters.top_k`

## Sources

- [Anthropic models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Anthropic model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations)
- [AWS Claude Sonnet 4.5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-5.html)

The proposal was generated without modifying canonical registry files.

## Review decision

Approved on August 15, 2026. The evidence refresh was promoted. Sampling parameter fields remain unknown pending authoritative cross-platform evidence.

## Invocation identity (2026-09-22)

- Amazon Bedrock: on-demand invocation requires a cross-region inference-profile id; the bare model id is not invocable on demand. Reviewed selectors recorded per the AWS inference-profile documentation.
- Anthropic API: the model id is directly invocable (`bare_on_demand_supported: true`).
