import json

from openai import OpenAI

MODEL = "gpt-5.6-sol"
client = OpenAI()
response = client.responses.create(
    model=MODEL,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Extract the invoice as JSON."},
                {"type": "input_image", "image_url": "https://example.test/invoice.png"},
            ],
        }
    ],
    max_output_tokens=1000,
    reasoning_effort="medium",
    tools=[
        {
            "type": "function",
            "name": "lookup_vendor",
            "description": "Look up a vendor by ID.",
            "parameters": {
                "type": "object",
                "properties": {"vendor_id": {"type": "string"}},
                "required": ["vendor_id"],
            },
        }
    ],
    text={
        "format": {
            "type": "json_schema",
            "name": "invoice",
            "schema": {
                "type": "object",
                "properties": {"vendor_id": {"type": "string"}},
                "required": ["vendor_id"],
            },
        }
    },
)
invoice = json.loads(response.output_text)
