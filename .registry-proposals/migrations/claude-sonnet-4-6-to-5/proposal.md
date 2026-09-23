# Claude Sonnet 4.6 → Claude Sonnet 5 Migration Knowledge Proposal

## Summary

2 prompt-guidance items added
1 freshness date updated
0 conflicts detected

First content of the `prompt_guidance` knowledge topic (v1.6.0-c): pair-specific prompt advice
that travels the same reviewed evidence path as parameter facts and reaches every prompt task as
evidence-linked shared guidance.

## Added

### prompt_guidance.sampling_instructions (applies when the application sets sampling parameters)

Sonnet 5 rejects non-default `temperature`, `top_p`, and `top_k`; the documented replacement is
system-prompt instructions. Pre-disposed not applicable when the scan (under resolved coverage)
shows no sampling parameter.

### prompt_guidance.token_counts

The new tokenizer produces approximately 30% more tokens for the same text; recount prompts
against Sonnet 5 and revisit budgets sized close to their limits.

## Considered and not proposed

Advice such as deduplicating repeated rules because of the tokenizer, or collapsing
no-reasoning boilerplate because thinking is on by default, is not stated by either official page
and was not proposed.

## Out of scope, flagged

Both pages state $2/$10 per million tokens with no introductory expiry; the canonical
`claude-sonnet-5` profile records `valid_until: 2026-08-31`. That needs a separate profile
proposal.

Sources:
- What's new in Claude Sonnet 5 (refetched 2026-09-23)
- Migrating to Claude Sonnet 5 (retrieved 2026-09-23)
