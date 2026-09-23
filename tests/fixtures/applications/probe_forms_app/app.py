from pathlib import Path

import anthropic

SYSTEM = open("prompts/system.txt").read()  # noqa: SIM115 - the probed loader form

with open("prompts/guidelines.txt", encoding="utf-8") as handle:
    GUIDELINES = handle.read()

client = anthropic.Anthropic()


def answer(question):
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM + "\n\n" + GUIDELINES,
        messages=[
            {"role": "user", "content": Path("prompts/user.txt").read_text(encoding="utf-8")},
            {"role": "user", "content": question},
        ],
    )
