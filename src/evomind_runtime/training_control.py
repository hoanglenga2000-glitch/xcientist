"""Administrator-owned rollout and immutable evaluation-data contracts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import zipfile


SHA = re.compile(r"[a-f0-9]{64}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ValueError("training_control_file_invalid")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_training_control_key")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("training_control_object_required")
    return value


def policy_path() -> Path | None:
    configured = os.getenv("EVOMIND_AIBUILD_POLICY_FILE", "")
    if configured:
        return Path(configured)
    bundle = os.getenv("WORKSTATION_BYOA_RUNTIME_ROOT", "")
    return Path(bundle).parent.parent / "config/research-control/policy.json" if bundle else None


def load_policy(runtime_root: Path) -> dict | None:
    path = policy_path()
    if path is None or not path.exists():
        return None
    value = read_json(path)
    if (value.get("schema") != "evomind.training_control.v1"
            or Path(str(value.get("runtime_root", ""))).resolve() != Path(runtime_root).resolve()
            or type(value.get("enabled")) is not bool
            or not isinstance(value.get("tenants"), list) or not value["tenants"]
            or not isinstance(value.get("owners"), list) or not value["owners"]
            or not isinstance(value.get("projects"), list) or not value["projects"]
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(value.get("study_id", "")))):
        raise ValueError("training_control_binding_invalid")
    return {**value, "policy_sha256": digest(path), "protocol_root": path.parent / "protocols"}


def identity(metadata: dict) -> tuple[str, str, str]:
    scope = metadata.get("http_access_scope") or {}
    hpc = metadata.get("managed_hpc_identity") or {}
    return (str(scope.get("tenant_id") or metadata.get("tenant_id") or hpc.get("tenant_id") or ""),
            str(scope.get("owner_principal_id") or metadata.get("owner_principal_id") or hpc.get("owner_principal_id") or ""),
            str(metadata.get("project_id") or ""))


def controls(policy: dict | None, metadata: dict, *, require_project: bool = False) -> bool:
    if policy is None:
        return False
    tenant, owner, project = identity(metadata)
    if tenant not in policy["tenants"]:
        return False
    if owner not in policy["owners"]:
        raise ValueError("training_control_owner_mismatch")
    return not require_project or project in policy["projects"]


def safe_data_file(root: Path, row: dict) -> Path:
    name = row.get("path")
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or ".." in Path(name).parts:
        raise ValueError("protocol_data_path_invalid")
    path = root / name
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("protocol_data_file_invalid")
    if not SHA.fullmatch(str(row.get("sha256", ""))) or digest(path) != row["sha256"]:
        raise ValueError("protocol_data_hash_mismatch")
    return path


def load_npz(path: Path) -> dict:
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 16 or sum(item.file_size for item in entries) > 256 * 1024 * 1024 or len({item.filename for item in entries}) != len(entries):
            raise ValueError("dataset_archive_bounds")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if set(result) != {"x", "y", "ids"} or any(value.dtype.hasobject for value in result.values()):
        raise ValueError("dataset_arrays_invalid")
    x, y, ids = result["x"], result["y"], result["ids"]
    if (x.ndim != 2 or not 1 <= len(x) <= 100000 or not 1 <= x.shape[1] <= 2048
            or x.size > 10_000_000 or y.ndim not in {1, 2} or len(y) != len(x)
            or ids.ndim != 1 or len(ids) != len(x) or len(np.unique(ids)) != len(ids)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("dataset_shape_or_values_invalid")
    return result


def load_protocol(runtime_root: Path, metadata: dict, protocol_id: str) -> dict:
    import numpy as np
    if not isinstance(protocol_id, str) or not SHA.fullmatch(protocol_id):
        raise ValueError("protocol_id_invalid")
    policy = load_policy(runtime_root)
    if not controls(policy, metadata, require_project=True):
        raise ValueError("training_project_not_enabled")
    root = policy["protocol_root"]
    path = root / (protocol_id + ".json")
    protocol = read_json(path)
    tenant, owner, project = identity(metadata)
    if (digest(path) != protocol_id or protocol.get("schema") != "evomind.tensor_protocol.v1"
            or (protocol.get("tenant_id"), protocol.get("owner_id"), protocol.get("project_id")) != (tenant, owner, project)
            or protocol.get("scope") not in {"engineering", "public_validation"}
            or protocol.get("task") not in {"regression", "classification"}
            or protocol.get("metric") != ("rmse" if protocol.get("task") == "regression" else "accuracy")
            or not all(isinstance(protocol.get("source", {}).get(key), str) and protocol["source"][key] for key in ("uri", "version", "license"))):
        raise ValueError("protocol_identity_or_scope_invalid")
    train_path = safe_data_file(root, protocol.get("training") or {})
    eval_path = safe_data_file(root, protocol.get("evaluation") or {})
    train, evaluation = load_npz(train_path), load_npz(eval_path)
    if train["x"].shape[1] != evaluation["x"].shape[1] or np.intersect1d(train["ids"], evaluation["ids"]).size:
        raise ValueError("protocol_split_overlap_or_schema_mismatch")
    if protocol["task"] == "classification":
        classes = protocol.get("classes")
        if type(classes) is not int or not 2 <= classes <= 1000:
            raise ValueError("protocol_classes_invalid")
        for data in (train, evaluation):
            y = data["y"].reshape(-1)
            if len(y) != len(data["x"]) or np.any(y != y.astype(np.int64)) or np.any(y < 0) or np.any(y >= classes):
                raise ValueError("protocol_labels_invalid")
    elif train["y"].size != len(train["x"]) or evaluation["y"].size != len(evaluation["x"]):
        raise ValueError("protocol_regression_target_invalid")
    return {"id": protocol_id, "contract": protocol, "training_path": train_path, "training": train, "evaluation": evaluation, "policy": policy}


def list_protocols(runtime_root: Path, metadata: dict) -> list[dict]:
    policy = load_policy(runtime_root)
    if not controls(policy, metadata, require_project=True):
        return []
    tenant, owner, project = identity(metadata)
    result = []
    for path in sorted(policy["protocol_root"].glob("*.json"))[:100]:
        value = read_json(path)
        if (value.get("tenant_id"), value.get("owner_id"), value.get("project_id")) != (tenant, owner, project):
            continue
        checked = load_protocol(runtime_root, metadata, path.stem)
        result.append({"protocol_id": path.stem, "scope": value["scope"], "task": value["task"], "metric": value["metric"], "source": value["source"], "training_rows": len(checked["training"]["x"]), "evaluation_rows": len(checked["evaluation"]["x"]), "features": checked["training"]["x"].shape[1], "official_score": False})
    return result


def training_config(value: dict, protocol: dict) -> dict:
    import math
    if not isinstance(value, dict) or set(value) - {"hidden_sizes", "epochs", "batch_size", "learning_rate", "seed"}:
        raise ValueError("training_configuration_fields_invalid")
    hidden = value.get("hidden_sizes", [32, 16])
    if not isinstance(hidden, list) or len(hidden) > 6 or any(type(item) is not int or not 1 <= item <= 1024 for item in hidden):
        raise ValueError("training_architecture_invalid")
    epochs, batch, seed, rate = value.get("epochs", 15), value.get("batch_size", 64), value.get("seed", 0), value.get("learning_rate", 0.01)
    if (type(epochs) is not int or not 1 <= epochs <= 200 or type(batch) is not int or not 1 <= batch <= 4096
            or type(seed) is not int or not 0 <= seed <= 2**31-1 or type(rate) not in {int, float} or not math.isfinite(rate) or not 1e-6 <= rate <= 0.1):
        raise ValueError("training_hyperparameters_invalid")
    contract = protocol["contract"]
    inputs, outputs = protocol["training"]["x"].shape[1], contract.get("classes", 1) if contract["task"] == "classification" else 1
    widths = [inputs, *hidden, outputs]
    if sum((left + 1) * right for left, right in zip(widths, widths[1:])) > 5_000_000:
        raise ValueError("training_parameter_budget_exceeded")
    return {"schema": "evomind.tensor_training.v1", "protocol_id": protocol["id"], "task": contract["task"], "input_features": inputs, "outputs": outputs, "hidden_sizes": hidden, "epochs": epochs, "batch_size": batch, "learning_rate": float(rate), "seed": seed}
