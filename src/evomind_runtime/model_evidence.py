"""Independent safe model reload adapter; no pickle or arbitrary inference code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def verify_managed_tensor_bundle(runtime, session: dict, candidate: str) -> dict[str, Any]:
    """Verify execution-journal hashes and recompute scores from protected data."""
    import numpy as np
    from .tensor_inference import predict
    from .training_control import digest, load_protocol, read_json

    evidence = None
    for call in reversed(runtime.store.list_tool_calls(session["id"])):
        result = call.get("result") or {}
        item = (result.get("content") or {}).get("managed_training")
        if call.get("tool_name") == "managed_tensor_train" and call.get("status") == "completed" and result.get("ok") is True and isinstance(item, dict) and item.get("candidate") == candidate:
            evidence = item
            break
    if evidence is None:
        return {"verified": False, "independent": True, "reason": "managed_training_execution_evidence_required"}
    try:
        required = {"model.safetensors", "model-config.json", "predictions.npz", "training-receipt.json", "environment.lock.json", "training-log.jsonl", "predict.py", "data-protocol.json"}
        if (evidence.get("schema") != "evomind.managed_training_evidence.v1" or not isinstance(evidence.get("files"), dict)
                or not required.issubset(evidence["files"]) or type(evidence.get("fit_steps")) is not int or evidence["fit_steps"] < 1):
            raise ValueError("managed_training_evidence_invalid")
        workspace = Path(session["workspace_root"]).resolve()
        root = workspace / "outputs/hpc" / candidate
        root.resolve(strict=True).relative_to(workspace)
        if root.is_symlink() or evidence.get("relative_root") != root.relative_to(workspace).as_posix():
            raise ValueError("model_bundle_location_mismatch")
        for name, expected in evidence["files"].items():
            if Path(name).name != name or (root / name).is_symlink() or digest(root / name) != expected:
                raise ValueError("model_execution_artifact_hash_mismatch")
        protocol = load_protocol(runtime.runtime_root, session.get("metadata") or {}, evidence["protocol_id"])
        config = read_json(root / "model-config.json")
        if config.get("protocol_id") != protocol["id"] or config.get("task") != protocol["contract"]["task"]:
            raise ValueError("model_protocol_mismatch")
        recomputed = predict(root, protocol["evaluation"]["x"])
        path = root / "predictions.npz"
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("prediction_file_too_large")
        with np.load(path, allow_pickle=False) as values:
            submitted = values["predictions"]
        if submitted.shape != recomputed.shape or not np.isfinite(submitted).all() or not np.allclose(submitted, recomputed, rtol=1e-4, atol=1e-5):
            raise ValueError("independent_prediction_mismatch")
        y = protocol["evaluation"]["y"]
        metric = protocol["contract"]["metric"]
        if metric == "rmse":
            value = float(np.sqrt(np.mean((recomputed.astype(np.float64).reshape(-1) - y.astype(np.float64).reshape(-1)) ** 2)))
        else:
            value = float(np.mean(np.argmax(recomputed, axis=1) == y.reshape(-1)))
        if not np.isfinite(value):
            raise ValueError("independent_metric_nonfinite")
        gate = protocol["contract"].get("acceptance") or {}
        accepted = ("max_value" not in gate or value <= float(gate["max_value"])) and ("min_value" not in gate or value >= float(gate["min_value"]))
        return {"verified": bool(accepted), "independent": True, "reason": "" if accepted else "model_quality_gate_failed", "metric": metric, "value": value, "scope": protocol["contract"]["scope"], "protocol_id": protocol["id"], "source_hashes": evidence["files"], "adapter_sha256": evidence["adapter_sha256"], "fit_steps": evidence["fit_steps"], "paper_baseline_beat": False, "official_score": False}
    except (OSError, ValueError, KeyError, TypeError) as error:
        code = str(error)
        return {"verified": False, "independent": True, "reason": code if code.replace("_", "").isalnum() and len(code) <= 120 else type(error).__name__}


def verify_linear_bundle(root: Path) -> dict[str, Any]:
    """Independently load a non-pickle linear checkpoint and recompute predictions.

    Other architectures require their own registered verifier; they never pass
    merely because a training script wrote an 'independent-verification' JSON.
    """
    import numpy as np

    root = root.resolve(strict=True)
    required = ("model.npz", "evaluation.npz", "predictions.npz", "evaluation-contract.json")
    paths = [root / name for name in required]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        return {"verified": False, "independent": True, "reason": "model_reload_adapter_inputs_missing"}
    contract = json.loads(paths[-1].read_text(encoding="utf-8"))
    if contract.get("schema") != "evomind.linear_evaluation.v1" or contract.get("scope") != "public_internal":
        return {"verified": False, "independent": True, "reason": "evaluator_contract_required"}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if not all(contract.get("sha256", {}).get(name) == hashes[name] for name in required[:3]):
        return {"verified": False, "independent": True, "reason": "model_evidence_hash_mismatch"}
    with np.load(paths[0], allow_pickle=False) as model, np.load(paths[1], allow_pickle=False) as data, np.load(paths[2], allow_pickle=False) as submitted:
        x, y = data["x"], data["y"]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return {"verified": False, "independent": True, "reason": "nonfinite_evaluation"}
        predictions = x @ model["coef"] + model["intercept"]
        if predictions.shape != submitted["predictions"].shape or not np.allclose(predictions, submitted["predictions"], rtol=1e-6, atol=1e-8):
            return {"verified": False, "independent": True, "reason": "reloaded_prediction_mismatch"}
        if len(set(data["train_ids"].tolist()) & set(data["evaluation_ids"].tolist())):
            return {"verified": False, "independent": True, "reason": "split_overlap"}
        metric = contract.get("metric")
        if metric == "rmse":
            if predictions.shape != y.shape:
                return {"verified": False, "independent": True, "reason": "target_shape_mismatch"}
            value = float(np.sqrt(np.mean((predictions - y) ** 2)))
        elif metric == "accuracy":
            labels = np.argmax(predictions, axis=1) if predictions.ndim == 2 and predictions.shape[1] > 1 else (predictions.reshape(-1) >= contract.get("threshold", 0.5)).astype(int)
            value = float(np.mean(labels == y.reshape(-1)))
        else:
            return {"verified": False, "independent": True, "reason": "unsupported_metric"}
    return {"verified": True, "independent": True, "metric": metric, "value": value, "source_hashes": hashes, "scope": "public_internal", "paper_baseline_beat": False}
