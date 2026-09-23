import yaml


def load_profiles():
    with open("config/model_profiles.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
