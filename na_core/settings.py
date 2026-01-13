import json
from pathlib import Path

from na_core.constants import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / ".na_project.json"


def load_settings():
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(settings):
    CONFIG_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
