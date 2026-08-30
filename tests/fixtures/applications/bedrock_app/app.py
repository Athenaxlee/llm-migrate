import os

import boto3

client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])
response = client.converse(
    modelId="anthropic.claude-sonnet-4-6",
    system=[{"text": "Answer using the supplied tool."}],
    messages=[
        {
            "role": "user",
            "content": [
                {"text": "Find order 123"},
                {
                    "document": {
                        "format": "txt",
                        "name": "order",
                        "source": {"bytes": b"order 123"},
                    }
                },
            ],
        }
    ],
    toolConfig={
        "tools": [
            {
                "toolSpec": {
                    "name": "find_order",
                    "description": "Find an order by ID.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"order_id": {"type": "string"}},
                            "required": ["order_id"],
                        }
                    },
                }
            }
        ]
    },
)
