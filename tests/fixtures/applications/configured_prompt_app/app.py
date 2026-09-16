from pathlib import Path

import boto3
import yaml

BASE_DIR = Path(__file__).parent
PROFILE_PATH = BASE_DIR / "model_profiles.yaml"

with open(PROFILE_PATH) as handle:
    PROFILES = yaml.safe_load(handle)

profile = PROFILES["models"]["claude"]
prompt_path = profile["prompts"]["multi"]


def _load_prompt(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


prompts = _load_prompt(prompt_path)

client = boto3.client("bedrock-runtime")
response = client.converse(
    modelId="anthropic.claude-sonnet-4-6",
    system=[{"text": prompts["sys_prompt"]}],
    messages=[{"role": "user", "content": [{"text": prompts["user_prompt"]}]}],
    inferenceConfig={"maxTokens": 800},
)
