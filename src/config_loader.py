import os, yaml

def load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
