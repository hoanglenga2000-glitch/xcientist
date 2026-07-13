from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..adapters.storage_adapter import StorageAdapter
from ..schemas.experiment import ExperimentRecord


class ExperimentService:
    def __init__(self, storage: StorageAdapter) -> None:
        self.storage = storage

    def record_run(self, record: ExperimentRecord) -> Path:
        if record.output_dir is None:
            raise ValueError("ExperimentRecord.output_dir is required.")
        return self.storage.write_json(record.output_dir / "experiment_record.json", record)

    def latest_experiment_dir(self, task_id: str) -> Path | None:
        root = Path("experiments") / task_id
        if not root.exists():
            return None
        candidates = [path for path in root.iterdir() if path.is_dir()]
        return sorted(candidates)[-1] if candidates else None

    def summarize_existing_run(self, output_dir: Path) -> dict[str, Any]:
        model_path = output_dir / "model_results.json"
        log_path = output_dir / "experiment_log.json"
        validation_path = output_dir / "validation_gate.json"
        model = json.loads(model_path.read_text(encoding="utf-8")) if model_path.exists() else {}
        log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
        validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
        best_model = model.get("best_model")
        metric = model.get("model_results", {}).get(best_model, {}) if best_model else {}
        return {
            "output_dir": str(output_dir),
            "best_model": best_model,
            "best_metrics": metric,
            "accepted": bool(log.get("accepted") or validation.get("passed")),
            "validation_gate": validation,
        }

