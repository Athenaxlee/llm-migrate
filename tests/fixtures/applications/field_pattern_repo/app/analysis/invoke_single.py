import os

import boto3

from analysis.config_loader import load_profiles
from analysis.prompt_loader import load_prompt_file

REPO_ROOT = os.environ.get("REPO_ROOT", "..")


class SingleInvoker:
    def __init__(self, profile_name="claude"):
        self.profile = load_profiles()["models"][profile_name]
        prompt_path = os.path.join(REPO_ROOT, self.profile["prompts"]["single"])
        self.single_prompts = load_prompt_file(prompt_path)
        self.client = boto3.client("bedrock-runtime", region_name=self.profile["region"])

    def invoke(self, text):
        prompt = self.single_prompts["sys_prompt"] + "\n\n" + self.single_prompts["user_prompt"]
        response = self.client.converse(
            modelId=self.profile["model_id"],
            messages=[{"role": "user", "content": [{"text": prompt + text}]}],
            inferenceConfig={"maxTokens": 1024},
        )
        return response["output"]["message"]["content"][0]["text"]
