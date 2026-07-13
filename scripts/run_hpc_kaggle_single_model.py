"""HPC SSH Launcher for S6E6 Single Model (LGB/XGB/CAT) 5fold×3seed CV."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko
from hpc_connect import secure_ssh_client
from hpc_runtime_contract import add_hpc_runtime_arguments, validate_hpc_runtime_arguments

ROOT = Path(__file__).resolve().parents[1]


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError(f"SOCKS5 response ended early while reading {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def socks5_connect_once(proxy_host: str, proxy_port: int, dest_host: str, dest_port: int, timeout: float) -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout)
    sock.settimeout(timeout)
    sock.sendall(b"\x05\x01\x00")
    if recv_exact(sock, 2) != b"\x05\x00":
        sock.close()
        raise RuntimeError("SOCKS5 method negotiation failed")
    try:
        ipv4 = ipaddress.IPv4Address(dest_host)
        request = b"\x05\x01\x00\x01" + ipv4.packed + struct.pack("!H", dest_port)
    except ipaddress.AddressValueError:
        host_bytes = dest_host.encode("ascii")
        request = b"\x05\x01\x00\x03" + bytes([len(host_bytes)]) + host_bytes + struct.pack("!H", dest_port)
    sock.sendall(request)
    header = recv_exact(sock, 4)
    if header[0] != 5 or header[1] != 0:
        sock.close()
        raise RuntimeError(f"SOCKS5 connect failed with response {header!r}")
    if header[3] == 1:
        recv_exact(sock, 4)
    elif header[3] == 3:
        recv_exact(sock, recv_exact(sock, 1)[0])
    elif header[3] == 4:
        recv_exact(sock, 16)
    else:
        sock.close()
        raise RuntimeError(f"Unsupported SOCKS5 address type {header[3]}")
    recv_exact(sock, 2)
    sock.settimeout(None)
    return sock


def socks5_connect(proxy_host: str, proxy_port: int, dest_host: str, dest_port: int, timeout: float = 30.0) -> socket.socket:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            return socks5_connect_once(proxy_host, proxy_port, dest_host, dest_port, timeout)
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"SOCKS5 connect failed after retries: {last_error}")


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get(args.password_env, "")
    if not password:
        raise RuntimeError(f"{args.password_env} is not configured.")
    sock = socks5_connect(args.proxy_host, args.proxy_port, args.host, args.port) if args.proxy_host else None
    client = secure_ssh_client()
    client.connect(
        args.host, port=args.port, username=args.user, password=password, sock=sock,
        allow_agent=False, look_for_keys=False, timeout=30, banner_timeout=30, auth_timeout=30,
    )
    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(30)
    return client


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    current = ""
    for part in remote_path.strip("/").split("/"):
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_file(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    sftp_mkdirs(sftp, str(Path(remote).parent).replace("\\", "/"))
    sftp.put(str(local), remote)


def download_file(sftp: paramiko.SFTPClient, remote: str, local: Path) -> bool:
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, str(local))
        return True
    except FileNotFoundError:
        return False


def close_sftp_quietly(sftp: paramiko.SFTPClient | None) -> None:
    if sftp is None:
        return
    try:
        sftp.close()
    except Exception:
        pass


def download_outputs_with_retries(
    args: argparse.Namespace,
    remote_dir: str,
    local_artifact_dir: Path,
    names: list[str],
    attempts: int = 3,
) -> dict[str, bool]:
    downloaded: dict[str, bool] = {}
    for name in names:
        remote_path = f"{remote_dir}/outputs/{name}"
        local_path = local_artifact_dir / name
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            download_client: paramiko.SSHClient | None = None
            sftp: paramiko.SFTPClient | None = None
            try:
                download_client = connect(args)
                sftp = download_client.open_sftp()
                downloaded[name] = download_file(sftp, remote_path, local_path)
                last_error = None
                break
            except FileNotFoundError:
                downloaded[name] = False
                last_error = None
                break
            except (RuntimeError, OSError, EOFError, paramiko.SSHException, socket.error) as exc:
                downloaded[name] = False
                last_error = exc
                time.sleep(min(2 * attempt, 6))
            finally:
                close_sftp_quietly(sftp)
                if download_client is not None:
                    download_client.close()
        if last_error is not None:
            raise RuntimeError(f"Failed to download {name} after {attempts} attempts: {last_error}") from last_error
    return downloaded


def remote_single_model_script(model_name: str) -> str:
    """Generate remote training script for a single model."""
    model_configs = {
        "lightgbm": {
            "imports": "import lightgbm as lgb",
            "build_fn": """def build_model(seed, n_classes, accelerator, profile):
    params = {
        "objective": "multiclass" if n_classes > 2 else "binary",
        "metric": "multi_logloss" if n_classes > 2 else "binary_logloss",
        "num_class": n_classes if n_classes > 2 else 1,
        "boosting_type": "gbdt",
        "n_estimators": 800,
        "learning_rate": 0.04,
        "max_depth": 7,
        "num_leaves": 63,
        "min_child_samples": 32,
        "subsample": 0.78,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.02,
        "reg_lambda": 0.04,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if profile == "high_capacity":
        params.update({
            "n_estimators": 1200,
            "learning_rate": 0.028,
            "max_depth": 8,
            "num_leaves": 95,
            "min_child_samples": 24,
            "subsample": 0.84,
            "colsample_bytree": 0.78,
            "reg_alpha": 0.01,
            "reg_lambda": 0.05,
        })
    elif profile == "conservative":
        params.update({
            "n_estimators": 1000,
            "learning_rate": 0.032,
            "max_depth": 6,
            "num_leaves": 47,
            "min_child_samples": 48,
            "subsample": 0.72,
            "colsample_bytree": 0.58,
            "reg_alpha": 0.06,
            "reg_lambda": 0.12,
        })
    elif profile == "minority_recall":
        params.update({
            "n_estimators": 1100,
            "learning_rate": 0.03,
            "max_depth": 7,
            "num_leaves": 79,
            "min_child_samples": 20,
            "subsample": 0.86,
            "colsample_bytree": 0.72,
            "reg_alpha": 0.015,
            "reg_lambda": 0.04,
        })
    if accelerator == "gpu":
        params["device_type"] = "gpu"
    kwargs = {k: v for k, v in params.items() if k != "num_class" or n_classes > 2}
    return lgb.LGBMClassifier(**kwargs)""",
        },
        "xgboost": {
            "imports": "import xgboost as xgb",
            "build_fn": """def build_model(seed, n_classes, accelerator, profile):
    params = {
        "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
        "num_class": n_classes if n_classes > 2 else 1,
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "n_estimators": 700,
        "learning_rate": 0.04,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.75,
        "colsample_bytree": 0.60,
        "colsample_bylevel": 0.75,
        "reg_alpha": 0.04,
        "reg_lambda": 0.06,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    }
    if profile == "high_capacity":
        params.update({
            "n_estimators": 1100,
            "learning_rate": 0.026,
            "max_depth": 8,
            "min_child_weight": 3,
            "subsample": 0.84,
            "colsample_bytree": 0.74,
            "colsample_bylevel": 0.82,
            "reg_alpha": 0.015,
            "reg_lambda": 0.08,
        })
    elif profile == "conservative":
        params.update({
            "n_estimators": 900,
            "learning_rate": 0.032,
            "max_depth": 6,
            "min_child_weight": 8,
            "subsample": 0.70,
            "colsample_bytree": 0.55,
            "colsample_bylevel": 0.70,
            "reg_alpha": 0.08,
            "reg_lambda": 0.16,
        })
    elif profile == "minority_recall":
        params.update({
            "n_estimators": 1000,
            "learning_rate": 0.028,
            "max_depth": 7,
            "min_child_weight": 2,
            "subsample": 0.88,
            "colsample_bytree": 0.78,
            "colsample_bylevel": 0.84,
            "reg_alpha": 0.01,
            "reg_lambda": 0.05,
        })
    if accelerator == "gpu":
        params["device"] = "cuda"
    if n_classes <= 2:
        params.pop("num_class", None)
    return xgb.XGBClassifier(**{k: v for k, v in params.items() if k != "num_class" or n_classes > 2})""",
        },
        "catboost": {
            "imports": "import catboost as cb",
            "build_fn": """def build_model(seed, n_classes, accelerator, profile):
    params = {
        "iterations": 600,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 4,
        "border_count": 128,
        "random_seed": seed,
        "loss_function": "MultiClass" if n_classes > 2 else "Logloss",
        "eval_metric": "MultiClass" if n_classes > 2 else "Logloss",
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if profile == "high_capacity":
        params.update({
            "iterations": 1000,
            "learning_rate": 0.032,
            "depth": 8,
            "l2_leaf_reg": 5,
            "border_count": 254,
        })
    elif profile == "conservative":
        params.update({
            "iterations": 850,
            "learning_rate": 0.035,
            "depth": 6,
            "l2_leaf_reg": 8,
            "border_count": 128,
        })
    elif profile == "minority_recall":
        params.update({
            "iterations": 900,
            "learning_rate": 0.036,
            "depth": 8,
            "l2_leaf_reg": 3,
            "border_count": 254,
        })
    if accelerator == "gpu":
        params["task_type"] = "GPU"
        params["devices"] = "0"
    return cb.CatBoostClassifier(**params)""",
        },
    }
    cfg = model_configs[model_name]
    return f'''
"""S6E6 Single Model CV — {model_name.upper()} 5fold×3seed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

{cfg["imports"]}


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    pairs = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "r"), ("g", "i"), ("r", "z")]
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f"{{a}}_minus_{{b}}"] = df[a] - df[b]
    if "redshift" in df.columns:
        df["redshift_log1p"] = np.log1p(np.clip(df["redshift"].astype(float), 0, None))
    return df


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(part.strip()) for part in seed_text.split(",") if part.strip()]


def stratified_sample(frame: pd.DataFrame, y: np.ndarray, sample_rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if sample_rows <= 0 or sample_rows >= len(frame):
        return frame.reset_index(drop=True), y
    rng = np.random.default_rng(seed)
    keep_indices = []
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        take = max(1, int(round(sample_rows * len(label_indices) / len(y))))
        take = min(take, len(label_indices))
        keep_indices.extend(rng.choice(label_indices, size=take, replace=False).tolist())
    keep_indices = np.array(sorted(keep_indices[:sample_rows]), dtype=int)
    return frame.iloc[keep_indices].reset_index(drop=True), y[keep_indices]


def normalize_probabilities(probs: np.ndarray, n_classes: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    if probs.shape[1] != n_classes:
        raise ValueError("predict_proba output class count does not match the label encoder.")
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    bad_rows = (~np.isfinite(row_sums[:, 0])) | (row_sums[:, 0] <= 0.0)
    if np.any(bad_rows):
        probs[bad_rows, :] = 1.0 / n_classes
        row_sums = probs.sum(axis=1, keepdims=True)
    return probs / row_sums


def make_sample_weights(y: np.ndarray, n_classes: int, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_weights = len(y) / (n_classes * counts)
    if mode == "half_balanced":
        class_weights = 1.0 + 0.5 * (class_weights - 1.0)
    elif mode == "sqrt_balanced":
        class_weights = np.sqrt(class_weights)
    elif mode == "strong_balanced":
        class_weights = np.power(class_weights, 1.25)
    elif mode != "balanced":
        raise ValueError(f"Unsupported class weight mode: {{mode}}")
    class_weights = np.clip(class_weights, 0.25, 6.0)
    return class_weights[y].astype(np.float64)


{cfg["build_fn"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a governed S6E6 single-model challenger.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="42,3407,12345")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260612)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--class-weight", choices=["none", "half_balanced", "sqrt_balanced", "balanced", "strong_balanced"], default="none")
    parser.add_argument("--profile", choices=["default", "high_capacity", "conservative", "minority_recall"], default="default")
    args = parser.parse_args()

    started = time.time()
    workdir = Path.cwd()
    data = workdir / "data"
    out = workdir / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    sample = pd.read_csv(data / "sample_submission.csv")

    target = "class"
    id_col = "id"
    class_names = sorted(train[target].dropna().unique().tolist())
    le = LabelEncoder()
    full_y = le.fit_transform(train[target].astype(str)).astype("int64")
    train, y = stratified_sample(train, full_y, args.sample_rows, args.seed)

    train_x = add_features(train.drop(columns=[target]))
    test_x = add_features(test)
    feature_cols = [c for c in train_x.columns if c != id_col]
    categorical = [c for c in feature_cols if train_x[c].dtype == "object"]
    numeric = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric), ("cat", make_encoder(), categorical)],
        remainder="drop",
    )
    x_all = preprocessor.fit_transform(train_x[feature_cols]).astype(np.float32)
    x_test = preprocessor.transform(test_x[feature_cols]).astype(np.float32)

    n_folds = args.folds
    n_classes = len(class_names)
    seeds = parse_seed_list(args.seeds)

    oof_preds = np.zeros((len(y), n_classes), dtype=np.float64)
    test_preds = np.zeros((len(x_test), n_classes), dtype=np.float64)
    cv_accuracies = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for seed in seeds:
        for fold, (train_idx, val_idx) in enumerate(skf.split(x_all, y)):
            X_tr, X_va = x_all[train_idx], x_all[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            sample_weight = make_sample_weights(y_tr, n_classes, args.class_weight)

            model = build_model(seed, n_classes, args.accelerator, args.profile)
            if sample_weight is None:
                model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr, sample_weight=sample_weight)

            p_val = normalize_probabilities(model.predict_proba(X_va), n_classes)
            p_test = normalize_probabilities(model.predict_proba(x_test), n_classes)

            oof_preds[val_idx] += p_val / len(seeds)
            test_preds += p_test / (n_folds * len(seeds))
            acc = float(accuracy_score(y_va, p_val.argmax(axis=1)))
            cv_accuracies.append(acc)

    oof_preds = normalize_probabilities(oof_preds, n_classes)
    test_preds = normalize_probabilities(test_preds, n_classes)
    oof_row_sum_error = float(np.max(np.abs(oof_preds.sum(axis=1) - 1.0)))
    test_row_sum_error = float(np.max(np.abs(test_preds.sum(axis=1) - 1.0)))

    oof_acc = float(accuracy_score(y, oof_preds.argmax(axis=1)))
    oof_bal_acc = float(balanced_accuracy_score(y, oof_preds.argmax(axis=1)))
    oof_ll = float(log_loss(y, oof_preds, labels=list(range(n_classes))))

    pred_labels = [class_names[int(i)] for i in test_preds.argmax(axis=1)]
    submission = pd.DataFrame({{id_col: sample[id_col].to_numpy(), target: pred_labels}})
    submission.to_csv(out / "submission.csv", index=False)
    np.savez_compressed(out / "oof_and_test_probabilities.npz", oof_mean=oof_preds, test_mean=test_preds, classes=np.array(class_names))

    metrics = {{
        "schema": "academic_research_os.hpc_single_model_metrics.v1",
        "status": "passed",
        "competition": "playground-series-s6e6",
        "task_id": "playground_series_s6e6",
        "model": "{model_name}",
        "runner": "hpc_single_model_{model_name}",
        "accelerator": args.accelerator,
        "class_weight": args.class_weight,
        "profile": args.profile,
        "n_folds": n_folds,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "mode": "dry_run" if args.sample_rows else "full_training",
        "sample_rows": int(args.sample_rows),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features_after_encoding": int(x_all.shape[1]),
        "classes": class_names,
        "cv_accuracy_mean": float(np.mean(cv_accuracies)),
        "cv_accuracy_std": float(np.std(cv_accuracies)),
        "oof_accuracy": oof_acc,
        "oof_balanced_accuracy": oof_bal_acc,
        "oof_log_loss": oof_ll,
        "probability_checks": {{
            "oof_max_abs_row_sum_error": oof_row_sum_error,
            "test_max_abs_row_sum_error": test_row_sum_error,
        }},
        "outputs": {{
            "oof_and_test_probabilities": "oof_and_test_probabilities.npz",
            "submission": "submission.csv",
            "metrics": "metrics.json",
            "report": "report.md",
        }},
        "seconds": round(time.time() - started, 3),
        "submission_rows": int(len(submission)),
        "prediction_distribution": submission[target].value_counts().to_dict(),
        "human_gate_required_for_official_submission": True,
    }}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    (out / "report.md").write_text("\\n".join([
        "# HPC Single Model Run: {model_name.upper()}",
        "",
        f"- task: `{{metrics['task_id']}}`",
        f"- model: `{{metrics['model']}}`",
        f"- folds: `{{n_folds}}` x seeds: `{{len(seeds)}}`",
        f"- CV accuracy: {{metrics['cv_accuracy_mean']:.6f}} +/- {{metrics['cv_accuracy_std']:.6f}}",
        f"- OOF accuracy: {{oof_acc:.6f}}",
        f"- OOF balanced accuracy: {{oof_bal_acc:.6f}}",
        f"- OOF log loss: {{oof_ll:.6f}}",
        f"- submission rows: `{{metrics['submission_rows']}}`",
        "- official Kaggle submission remains behind Human Gate.",
    ]), encoding="utf-8")

    print(json.dumps({{"event": "single_model_completed", "model": "{model_name}", "oof_accuracy": oof_acc, "oof_balanced_accuracy": oof_bal_acc, "seconds": metrics["seconds"]}}))


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S6E6 single model through HPC SSH.")
    parser.add_argument("--model", required=True, choices=["lightgbm", "xgboost", "catboost"])
    add_hpc_runtime_arguments(parser)
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--local-artifact-dir", default="")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="42,3407,12345")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260612)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--class-weight", choices=["none", "half_balanced", "sqrt_balanced", "balanced", "strong_balanced"], default="none")
    parser.add_argument("--profile", choices=["default", "high_capacity", "conservative", "minority_recall"], default="default")
    args = parser.parse_args()
    validate_hpc_runtime_arguments(parser, args)

    model_name = args.model
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_dir = f"{args.remote_root.rstrip('/')}/playground_series_s6e6/{run_id}_{model_name}"
    local_artifact_dir = Path(args.local_artifact_dir) if args.local_artifact_dir else ROOT / "workspace" / "gpu" / "playground_series_s6e6" / f"{run_id}_{model_name}"
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    client = connect(args)
    sftp = client.open_sftp()
    try:
        sftp_mkdirs(sftp, f"{remote_dir}/data")
        for name in ["train.csv", "test.csv", "sample_submission.csv"]:
            upload_file(sftp, ROOT / "tasks" / "playground_series_s6e6" / "data" / name, f"{remote_dir}/data/{name}")
        script_local = local_artifact_dir / f"remote_{model_name}.py"
        script_local.write_text(remote_single_model_script(model_name), encoding="utf-8")
        upload_file(sftp, script_local, f"{remote_dir}/remote_{model_name}.py")

        remote_script_args = f"--folds {args.folds} --seeds '{args.seeds}' --sample-rows {args.sample_rows} --seed {args.seed} --accelerator {args.accelerator} --class-weight {args.class_weight} --profile {args.profile}"
        if args.python_executable.strip():
            command = f"cd '{remote_dir}' && '{args.python_executable.strip()}' remote_{model_name}.py {remote_script_args}"
        else:
            command = (
                f"cd '{remote_dir}' && (command -v python3 >/dev/null 2>&1 && "
                f"python3 remote_{model_name}.py {remote_script_args} "
                f"|| python remote_{model_name}.py {remote_script_args})"
            )
        started = time.time()
        _, stdout, stderr = client.exec_command(command, timeout=args.timeout_seconds)
        exit_status = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode("utf-8", "replace")
        stderr_text = stderr.read().decode("utf-8", "replace")
        (local_artifact_dir / "remote_stdout.log").write_text(stdout_text, encoding="utf-8")
        (local_artifact_dir / "remote_stderr.log").write_text(stderr_text, encoding="utf-8")
        close_sftp_quietly(sftp)
        sftp = None

        downloaded = download_outputs_with_retries(
            args,
            remote_dir,
            local_artifact_dir,
            ["metrics.json", "submission.csv", "report.md", "oof_and_test_probabilities.npz"],
        )

        metrics: dict[str, Any] | None = None
        metrics_path = local_artifact_dir / "metrics.json"
        if metrics_path.is_file() and metrics_path.stat().st_size > 0:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        manifest = {
            "schema": "academic_research_os.hpc_single_model_manifest.v1",
            "status": "passed" if (
                exit_status == 0
                and downloaded.get("metrics.json")
                and downloaded.get("submission.csv")
                and downloaded.get("oof_and_test_probabilities.npz")
            ) else "failed",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "model": model_name,
            "runner": f"hpc_single_model_{model_name}",
            "accelerator": args.accelerator,
            "class_weight": args.class_weight,
            "profile": args.profile,
            "remote_dir": remote_dir,
            "local_artifact_dir": str(local_artifact_dir.relative_to(ROOT)).replace("\\", "/"),
            "exit_status": exit_status,
            "seconds": round(time.time() - started, 3),
            "downloaded": downloaded,
            "metrics": metrics,
            "stdout_tail": stdout_text[-4000:],
            "stderr_tail": stderr_text[-4000:],
            "human_gate_required_for_official_submission": True,
        }
        (local_artifact_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        if manifest["status"] != "passed":
            raise SystemExit(1)
    finally:
        close_sftp_quietly(sftp)
        client.close()


if __name__ == "__main__":
    main()
