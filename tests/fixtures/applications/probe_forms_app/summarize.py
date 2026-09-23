import anthropic
import yaml

with open("settings.yaml", encoding="utf-8") as handle:
    SETTINGS = yaml.safe_load(handle)

prompt_file = SETTINGS["prompts"]["summary"]

with open(prompt_file, encoding="utf-8") as handle:
    SUMMARY = yaml.safe_load(handle)

client = anthropic.Anthropic()


def summarize(text):
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SUMMARY["system"],
        messages=[{"role": "user", "content": text}],
    )
