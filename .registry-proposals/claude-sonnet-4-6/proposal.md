# Claude Sonnet 4.6 Registry Proposal

## Summary

0 facts added
2 facts changed
0 conflicts detected
1 field remains unknown

## High-impact changes

### Maximum output tokens

Current: 64000
Proposed: 128000
Risk: HIGH

Reason:
Anthropic's migration guide says Claude Sonnet 5 supports 128K output tokens, unchanged from Claude Sonnet 4.6. The current canonical entry says 64K.

Sources:
- Anthropic migration guide

## Unknown fields

- `parameters.adaptive_thinking.default`

## Sources

- [Anthropic models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Anthropic migration guide](https://platform.claude.com/docs/en/about-claude/models/migration-guide)
- [Anthropic model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations)
- [AWS Claude Sonnet 4.6 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html)

The proposal was generated without modifying canonical registry files.

## Review decision

Approved on August 15, 2026. The 128K maximum-output correction and evidence refresh were promoted. The adaptive-thinking default remains unresolved.

## Invocation identity (2026-09-22)

- Amazon Bedrock: on-demand invocation requires a cross-region inference-profile id; the bare model id is not invocable on demand. Reviewed selectors recorded per the AWS inference-profile documentation.
- Anthropic API: the model id is directly invocable (`bare_on_demand_supported: true`).
