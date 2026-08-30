# Claude Sonnet 5 Registry Proposal

## Summary

0 facts added
1 fact changed
1 conflict detected
1 field remains unknown

Temperature, `top_p`, `top_k`, and manual-thinking restrictions were already present in the current canonical file and were reverified. The repository has no commit history, so the earlier pre-research baseline cannot be reconstructed reliably.

## High-impact changes

### Adaptive thinking disable behavior

Current: unknown
Proposed: platform-scoped behavior
Risk: HIGH

Reason:
Anthropic documents `thinking: {type: "disabled"}` for Claude Sonnet 5, while AWS says adaptive thinking is always on and cannot be disabled. The candidate remains conservative and does not invent a provider-wide answer.

Sources:
- Anthropic Sonnet 5 launch guide
- AWS Claude Sonnet 5 model card

## Verified high-impact facts already in the canonical entry

- Non-default `temperature`, `top_p`, and `top_k` values return HTTP 400.
- Manual extended thinking is unsupported and returns HTTP 400.
- Adaptive thinking is the supported reasoning mode; `effort` defaults to `high`.

## Unknown fields

- `platforms.amazon-bedrock.parameters.adaptive_thinking`

## Sources

- [Anthropic models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [What's new in Claude Sonnet 5](https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5)
- [Anthropic model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations)
- [AWS Claude Sonnet 5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html)

The proposal was generated without modifying canonical registry files.

## Review decision

Approved with an unresolved conflict on August 15, 2026. The conflict record and evidence refresh were promoted, but no universal value for disabling adaptive thinking was added.
