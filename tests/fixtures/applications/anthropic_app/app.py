import os
from pathlib import Path

from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
SYSTEM_PROMPT = Path("prompts/system.txt").read_text()
STREAM = True
response = client.messages.create(
    model="claude-sonnet-5",
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": "Weather for Seattle"}],
    max_tokens=500,
    temperature=0,
    tools=[
        {
            "name": "get_weather",
            "description": "Return current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ],
    stream=STREAM,
)
