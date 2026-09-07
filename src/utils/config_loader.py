from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)