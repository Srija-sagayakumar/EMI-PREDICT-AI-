import json
from .config import MODEL_DIR


def load_meta() -> dict:
    p = MODEL_DIR / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def category_options() -> dict:
    return load_meta().get("categories", {})
