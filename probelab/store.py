"""JSON persistence for resumable investigations."""

import json
from pathlib import Path
from .models import Investigation

STORE = Path("data")


def save(item: Investigation) -> Investigation:
    STORE.mkdir(exist_ok=True)
    (STORE / f"{item.id}.json").write_text(item.model_dump_json(indent=2), encoding="utf-8")
    return item


def load(item_id: str) -> Investigation | None:
    path = STORE / f"{item_id}.json"
    if not path.exists():
        return None
    item = Investigation.model_validate(json.loads(path.read_text(encoding="utf-8")))
    residual = item.parameters.get("residual_rms_deg")
    adequacy = item.parameters.get("adequacy")
    if adequacy != "good" or not isinstance(residual, (int, float)) or residual > 10.0:
        return item.model_copy(update={"status": "needs better evidence", "recommendation": {"angle_deg": 0.0, "duration_s": 0.0, "eig_nats": 0.0, "rationale": "No experiment recommendation: the persisted observation does not support a scientifically adequate model fit."}})
    return item
