"""Conservative, deterministic prompt characteristics analysis."""

from __future__ import annotations

import re
from collections import Counter

from llm_migrate.core.models import FindingSeverity, PromptAnalysis, PromptFinding, PromptIntent

_NEGATIVE = re.compile(r"\b(?:do not|don't|never|must not|avoid|under no circumstances)\b", re.I)
_COT = re.compile(
    r"\b(?:chain[- ]of[- ]thought|think step by step|show your reasoning|internal reasoning)\b",
    re.I,
)
_JSON = re.compile(r"\bjson\b|json schema|output format", re.I)
_EXAMPLE = re.compile(r"(?:^|\n)\s*(?:example|input|output)\s*\d*\s*:", re.I)
_TOOLS = re.compile(r"\b(?:tool|function call|function calling)\b", re.I)
_PREFILL = re.compile(
    r"\b(?:assistant prefill|prefill the assistant|begin (?:your|the) response with)\b", re.I
)
_JSON_ONLY = re.compile(r"\b(?:json only|only (?:valid )?json)\b", re.I)
_TAG = re.compile(r"<(/?)([A-Za-z][\w.-]*)\b[^>]*>")


def structural_sections(prompt: str) -> dict[str, int]:
    """Counts of paired XML-like sections, keyed by tag name.

    Only names that appear as BOTH an opening and a closing tag count as
    prompt structure; unpaired angle-bracketed tokens (placeholders such as
    `<CUSTOMER_NAME>`, emails, autolinks) are not sections. Counting pairs
    rather than deduplicated names lets callers detect the loss of one of
    several same-named sections (e.g. one of two `<example>` blocks).
    """
    opening: Counter[str] = Counter()
    closing: Counter[str] = Counter()
    for slash, name in _TAG.findall(prompt):
        (closing if slash else opening)[name] += 1
    return {
        name: min(opening[name], closing[name]) for name in sorted(opening.keys() & closing.keys())
    }


def analyze_prompt(prompt: str) -> PromptAnalysis:
    headers = re.findall(r"(?m)^#{1,6}\s+(.+?)\s*$", prompt)
    tags = sorted(set(re.findall(r"</?([A-Za-z][\w.-]*)\b[^>]*>", prompt)))
    findings: list[PromptFinding] = []
    lines = [
        re.sub(r"\s+", " ", line.strip()).casefold()
        for line in prompt.splitlines()
        if len(line.strip()) >= 12
    ]
    repeats = sorted(line for line, count in Counter(lines).items() if count > 1)
    if repeats:
        findings.append(
            PromptFinding(
                category="duplicated_requirements",
                severity=FindingSeverity.WARNING,
                message="Potentially duplicated instruction lines were found.",
                evidence=repeats[:5],
            )
        )
    negatives = _NEGATIVE.findall(prompt)
    if len(negatives) >= 4:
        findings.append(
            PromptFinding(
                category="negative_wording",
                severity=FindingSeverity.WARNING,
                message="The prompt contains many negative instruction phrases.",
                evidence=[f"count={len(negatives)}"],
            )
        )
    detectors = (
        ("explicit_chain_of_thought", _COT, "Explicit reasoning-process instructions are present."),
        ("structured_output", _JSON, "JSON or explicit output-format instructions are present."),
        ("few_shot_examples", _EXAMPLE, "Example-like input/output blocks are present."),
        ("tool_instructions", _TOOLS, "Tool or function-calling instructions are present."),
        ("assistant_prefill", _PREFILL, "An assistant-prefill workaround is present."),
        ("json_only_prompting", _JSON_ONLY, "A JSON-only prompting workaround is present."),
    )
    for category, pattern, message in detectors:
        matches = pattern.findall(prompt)
        if matches:
            findings.append(
                PromptFinding(
                    category=category,
                    severity=FindingSeverity.INFO,
                    message=message,
                    evidence=[str(item) for item in matches[:5]],
                )
            )
    if headers or tags:
        findings.append(
            PromptFinding(
                category="section_structure",
                severity=FindingSeverity.INFO,
                message="Explicit Markdown or XML-like structure is present.",
                evidence=[*headers[:3], *tags[:3]],
            )
        )
    if tags:
        findings.append(
            PromptFinding(
                category="provider_specific_formatting",
                severity=FindingSeverity.WARNING,
                message="XML-like formatting may encode model-specific prompting assumptions.",
                evidence=tags[:5],
            )
        )
    meaningful_lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    examples = [line for line in meaningful_lines if _EXAMPLE.search(line)]
    contracts = [line for line in meaningful_lines if _JSON.search(line)]
    grounding = [
        line
        for line in meaningful_lines
        if re.search(r"\b(?:ground|source|cite|fact|invent|uncertain)\w*\b", line, re.I)
    ]
    reasoning = [line for line in meaningful_lines if _COT.search(line)]
    tools = [line for line in meaningful_lines if _TOOLS.search(line)]
    verbosity = [
        line
        for line in meaningful_lines
        if re.search(r"\b(?:concise|brief|verbose|prose|word|sentence)\w*\b", line, re.I)
    ]
    objective = [
        line
        for line in meaningful_lines
        if not line.startswith(("#", "<"))
        and line not in examples
        and line not in contracts
        and line not in reasoning
        and line not in tools
    ][:3]
    return PromptAnalysis(
        character_count=len(prompt),
        word_count=len(re.findall(r"\S+", prompt)),
        approximate_token_count=(len(prompt) + 3) // 4,
        markdown_headers=headers,
        xml_like_tags=tags,
        findings=findings,
        intent=PromptIntent(
            objective=objective,
            input_output_contracts=contracts[:10],
            instructions=meaningful_lines[:20],
            examples=examples[:10],
            grounding_policies=grounding[:10],
            reasoning_policies=reasoning[:10],
            tool_policies=tools[:10],
            verbosity_policies=verbosity[:10],
        ),
    )
