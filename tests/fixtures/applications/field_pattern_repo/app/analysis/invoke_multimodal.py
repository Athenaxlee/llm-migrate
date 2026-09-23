import os

import boto3

from analysis.config_loader import load_profiles
from analysis.preprocessing import normalize
from analysis.prompt_loader import load_prompt_file

REPO_ROOT = os.environ.get("REPO_ROOT", "..")


class MultimodalInvoker:
    def __init__(self, profile_name="claude"):
        self.profile = load_profiles()["models"][profile_name]
        prompt_path = os.path.join(REPO_ROOT, self.profile["prompts"]["multi"])
        self.multi_prompts = load_prompt_file(prompt_path)
        self.client = boto3.client("bedrock-runtime", region_name=self.profile["region"])

    def invoke(self, document_text):
        response = self.client.converse(
            modelId=self.profile["model_id"],
            system=[{"text": self.multi_prompts["sys_prompt"]}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": self.multi_prompts["user_prompt"] + normalize(input=document_text)}
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 2048, "temperature": 0.0},
        )
        return response["output"]["message"]["content"][0]["text"]
