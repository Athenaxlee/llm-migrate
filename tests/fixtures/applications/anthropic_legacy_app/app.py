from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    system="Return concise JSON grounded in the user input.",
    messages=[{"role": "user", "content": "Summarize this record."}],
    max_tokens=500,
)
