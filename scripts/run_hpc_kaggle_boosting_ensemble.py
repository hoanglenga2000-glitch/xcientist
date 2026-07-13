"""HPC SSH Launcher for S6E6 Boosting Ensemble (LGB+XGB+CAT only).

Detects required packages on the remote HPC server and blocks with an
auditable dependency artifact if LightGBM, XGBoost, or CatBoost is missing.
This score-improvement template must not fall back to weaker sklearn models.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shlex
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


def socks5_connect(proxy_host: str, proxy_port: int, dest_host: str, dest_port: int, timeout: float = 30.0) -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout)
    sock.settimeout(timeout)
    sock.sendall(b"\x05\x01\x00")
    if recv_exact(sock, 2) != b"\x05\x00":
        raise RuntimeError("SOCKS5 method negotiation failed")
    try:
        ipv4 = ipaddress.IPv4Address(dest_host)
        request = b"\x05\x01\x00\x01" + ipv4.packed + struct.pack("!H", dest_port)
    except ipaddress.AddressValueError:
        host_bytes = dest_host.encode("ascii")
        request = b"\x05\x01\x00\x03" + bytes([len(host_bytes)]) + host_bytes + struct.pack("!H", dest_port)
    sock.sendall(request)
    header = recv_exact(sock, 4)
    if len(header) != 4 or header[0] != 5 or header[1] != 0:
        raise RuntimeError(f"SOCKS5 connect failed with response {header!r}")
    if header[3] == 1:
        recv_exact(sock, 4)
    elif header[3] == 3:
        recv_exact(sock, recv_exact(sock, 1)[0])
    elif header[3] == 4:
        recv_exact(sock, 16)
    else:
        raise RuntimeError(f"Unsupported SOCKS5 address type {header[3]}")
    recv_exact(sock, 2)
    return sock


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
    if sock is not None:
        sock.settimeout(None)
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
            except (OSError, EOFError, paramiko.SSHException, socket.error) as exc:
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


def wait_for_remote_outputs(
    args: argparse.Namespace,
    remote_dir: str,
    local_artifact_dir: Path,
    names: list[str],
    wait_seconds: int,
    poll_seconds: int = 30,
) -> dict[str, bool]:
    deadline = time.time() + max(0, wait_seconds)
    downloaded = {name: (local_artifact_dir / name).is_file() and (local_artifact_dir / name).stat().st_size > 0 for name in names}
    while not all(downloaded.values()) and time.time() <= deadline:
        try:
            downloaded = download_outputs_with_retries(args, remote_dir, local_artifact_dir, names, attempts=2)
        except Exception:
            time.sleep(min(poll_seconds, max(1, int(deadline - time.time()))))
            continue
        if all(downloaded.values()):
            break
        time.sleep(min(poll_seconds, max(1, int(deadline - time.time()))))
    return downloaded


def remote_boosting_ensemble_script() -> str:
    return r'''
"""S6E6 Boosting Ensemble — LGB+XGB+CAT only.

Package detection: tries lightgbm, xgboost, catboost first.
Blocks with dependency evidence if any boosting package is missing.
Produces OOF blend + logistic stacker + calibration diagnostics.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# ── Package detection ────────────────────────────────────────────────────────

def load_run_config():
    path = Path("run_config.json")
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


RUN_CONFIG = load_run_config()


def cfg_gpu_device_id() -> str:
    requested = str(RUN_CONFIG.get("gpu_device_id", "auto")).strip()
    if requested and requested.lower() not in {"auto", "none", "null"}:
        return requested
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        )
        candidates = []
        for line in raw.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2:
                candidates.append((int(parts[1]), parts[0]))
        if candidates:
            return max(candidates)[1]
    except Exception:
        pass
    return ""


SELECTED_GPU_DEVICE = cfg_gpu_device_id()
if SELECTED_GPU_DEVICE:
    os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU_DEVICE


# Package detection runs after CUDA_VISIBLE_DEVICES is set so GPU libraries see the selected card.
AVAILABLE = {"lightgbm": False, "xgboost": False, "catboost": False}

try:
    import lightgbm as lgb
    AVAILABLE["lightgbm"] = True
except ImportError:
    pass

try:
    import xgboost as xgb
    AVAILABLE["xgboost"] = True
except ImportError:
    pass

try:
    import catboost as cb
    AVAILABLE["catboost"] = True
except ImportError:
    pass

USE_BOOSTING = AVAILABLE["lightgbm"] and AVAILABLE["xgboost"] and AVAILABLE["catboost"]


def cfg_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(RUN_CONFIG.get(name, default))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def cfg_choice(name: str, default: str, allowed: set[str]) -> str:
    value = str(RUN_CONFIG.get(name, default)).strip()
    return value if value in allowed else default


def cfg_seeds() -> list[int]:
    raw = RUN_CONFIG.get("seeds", [42, 3407, 12345])
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        parts = raw
    else:
        parts = []
    seeds = []
    for part in parts:
        try:
            seeds.append(int(part))
        except Exception:
            pass
    return seeds or [42, 3407, 12345]


def write_progress(out: Path, event: str, **payload) -> None:
    row = {"event": event, "at": time.time(), **payload}
    try:
        with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass
    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


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
            left = df[a].astype(float)
            right = df[b].astype(float)
            df[f"{a}_minus_{b}"] = left - right
            df[f"{a}_over_{b}"] = (left / right.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    if "redshift" in df.columns:
        df["redshift_log1p"] = np.log1p(np.clip(df["redshift"].astype(float), 0, None))
    if "alpha" in df.columns:
        alpha_rad = np.deg2rad(df["alpha"].astype(float))
        df["alpha_sin"] = np.sin(alpha_rad)
        df["alpha_cos"] = np.cos(alpha_rad)
    if "delta" in df.columns:
        delta_rad = np.deg2rad(df["delta"].astype(float))
        df["delta_sin"] = np.sin(delta_rad)
        df["delta_cos"] = np.cos(delta_rad)
    return df


def build_lgb_model(seed: int, n_classes: int):
    params = {
        "objective": "multiclass" if n_classes > 2 else "binary",
        "metric": "multi_logloss" if n_classes > 2 else "binary_logloss",
        "num_class": n_classes if n_classes > 2 else 1,
        "boosting_type": "gbdt",
        "n_estimators": cfg_int("lgb_estimators", 1500, 80, 4000),
        "learning_rate": 0.03,
        "max_depth": -1,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    return lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != "num_class" or n_classes > 2})


def build_xgb_model(seed: int, n_classes: int):
    xgb_device = cfg_choice("xgb_device", "cuda", {"cpu", "cuda", "auto"})
    params = {
        "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
        "num_class": n_classes if n_classes > 2 else 1,
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "n_estimators": cfg_int("xgb_estimators", 1800, 80, 4000),
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "colsample_bylevel": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    }
    if xgb_device in {"cuda", "auto"}:
        params["device"] = "cuda"
    if n_classes <= 2:
        params.pop("num_class", None)
    return xgb.XGBClassifier(**{k: v for k, v in params.items() if k != "num_class" or n_classes > 2})


def build_cat_model(seed: int, n_classes: int):
    cat_task_type = cfg_choice("cat_task_type", "GPU", {"CPU", "GPU", "auto"})
    if cat_task_type == "auto":
        cat_task_type = "GPU"
    params = {
        "iterations": cfg_int("cat_iterations", 2000, 80, 4000),
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 3,
        "border_count": 128,
        "random_seed": seed,
        "loss_function": "MultiClass" if n_classes > 2 else "Logloss",
        "eval_metric": "MultiClass" if n_classes > 2 else "Logloss",
        "auto_class_weights": "Balanced",
        "task_type": cat_task_type,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if cat_task_type == "GPU":
        params["devices"] = "0"
    return cb.CatBoostClassifier(**params)


def build_sklearn_rf(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=400, max_depth=18, min_samples_leaf=32,
        max_features="sqrt", n_jobs=-1, random_state=seed,
    )


def build_sklearn_hgb(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=8, max_leaf_nodes=63,
        min_samples_leaf=32, l2_regularization=0.5, early_stopping=True,
        validation_fraction=0.12, n_iter_no_change=30, random_state=seed,
    )


def build_sklearn_et(seed: int):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(
        n_estimators=400, max_depth=18, min_samples_leaf=32,
        max_features="sqrt", n_jobs=-1, random_state=seed,
    )


def main() -> None:
    started = time.time()
    workdir = Path.cwd()
    data = workdir / "data"
    out = workdir / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    write_progress(
        out,
        "boosting_run_started",
        config=RUN_CONFIG,
        packages_available=AVAILABLE,
        selected_gpu_device=SELECTED_GPU_DEVICE or None,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    if not USE_BOOSTING:
        missing = [name for name, available in AVAILABLE.items() if not available]
        metrics = {
            "schema": "academic_research_os.hpc_boosting_dependency_gate.v1",
            "status": "blocked_dependency",
            "competition": "playground-series-s6e6",
            "task_id": "playground_series_s6e6",
            "runner": "hpc_boosting_ensemble_lgb_xgb_cat",
            "packages_available": AVAILABLE,
            "missing_packages": missing,
            "using_boosting": False,
            "submission_created": False,
            "reason": "This score-improvement template requires LightGBM, XGBoost, and CatBoost. It must not fall back to sklearn for official submission candidates.",
            "human_gate_required_for_official_submission": True,
        }
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "report.md").write_text("\n".join([
            "# HPC Boosting Dependency Gate",
            "",
            "- status: `blocked_dependency`",
            f"- missing packages: `{', '.join(missing)}`",
            "- no submission was created",
            "- next action: repair the remote Python environment, then rerun this same workstation template.",
        ]), encoding="utf-8")
        write_progress(out, "boosting_dependency_blocked", missing_packages=missing)
        raise SystemExit(2)

    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    write_progress(out, "data_loaded", train_rows=int(len(train)), test_rows=int(len(test)))

    target = "class"
    id_col = "id"
    sample_rows = cfg_int("sample_rows", 0, 0, 600000)
    if sample_rows > 0 and sample_rows < len(train):
        train = (
            train.groupby(target, group_keys=False)
            .apply(lambda part: part.sample(max(1, int(round(sample_rows * len(part) / len(train)))), random_state=42))
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
    class_names = sorted(train[target].dropna().unique().tolist())
    le = LabelEncoder()
    y = le.fit_transform(train[target].astype(str)).astype("int64")

    train_x = add_features(train.drop(columns=[target]))
    test_x = add_features(test)
    original_feature_cols = [c for c in train.drop(columns=[target]).columns if c != id_col]
    feature_cols = [c for c in train_x.columns if c != id_col]
    engineered_feature_cols = [c for c in feature_cols if c not in original_feature_cols]
    categorical = [c for c in feature_cols if train_x[c].dtype == "object"]
    numeric = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric), ("cat", make_encoder(), categorical)],
        remainder="drop",
    )
    x_all = preprocessor.fit_transform(train_x[feature_cols]).astype(np.float32)
    x_test = preprocessor.transform(test_x[feature_cols]).astype(np.float32)
    write_progress(
        out,
        "features_encoded",
        features_after_encoding=int(x_all.shape[1]),
        engineered_feature_count=int(len(engineered_feature_cols)),
        categorical_feature_count=int(len(categorical)),
    )

    min_class_count = int(pd.Series(y).value_counts().min())
    n_folds = min(cfg_int("folds", 5, 2, 5), min_class_count)
    n_classes = len(class_names)
    seeds = cfg_seeds()
    write_progress(out, "cv_configured", folds=int(n_folds), seeds=seeds, classes=class_names)

    # ── Select model builders based on available packages ────────────────────
    model_builders = {
        "lgb": lambda s: build_lgb_model(s, n_classes),
        "xgb": lambda s: build_xgb_model(s, n_classes),
        "cat": lambda s: build_cat_model(s, n_classes),
    }
    runner_label = "hpc_boosting_ensemble_lgb_xgb_cat"

    model_names = list(model_builders.keys())
    oof = {name: np.zeros((len(y), n_classes), dtype=np.float64) for name in model_names}
    test_preds = {name: np.zeros((len(x_test), n_classes), dtype=np.float64) for name in model_names}
    cv_scores = {name: [] for name in model_names}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    event_log = []

    for seed in seeds:
        for fold, (train_idx, val_idx) in enumerate(skf.split(x_all, y)):
            X_tr, X_va = x_all[train_idx], x_all[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            for name, build_fn in model_builders.items():
                write_progress(out, "boosting_model_fit_started", seed=seed, fold=fold + 1, model=name)
                model = build_fn(seed)
                model.fit(X_tr, y_tr)
                p_val = model.predict_proba(X_va)
                if p_val.ndim == 1:
                    p_val = np.column_stack([1 - p_val, p_val])
                if p_val.shape[1] != n_classes:
                    full = np.zeros((len(p_val), n_classes), dtype=np.float64)
                    full[:, :p_val.shape[1]] = p_val
                    p_val = full
                p_test = model.predict_proba(x_test)
                if p_test.ndim == 1:
                    p_test = np.column_stack([1 - p_test, p_test])
                if p_test.shape[1] != n_classes:
                    full = np.zeros((len(p_test), n_classes), dtype=np.float64)
                    full[:, :p_test.shape[1]] = p_test
                    p_test = full

                oof[name][val_idx] += p_val / len(seeds)
                test_preds[name] += p_test / (n_folds * len(seeds))
                acc = float(accuracy_score(y_va, p_val.argmax(axis=1)))
                cv_scores[name].append(acc)
                event_log.append({"model": name, "seed": seed, "fold": fold + 1, "accuracy": acc})
                write_progress(out, "boosting_model_fit_completed", seed=seed, fold=fold + 1, model=name, accuracy=acc)

    # ── OOF scores ───────────────────────────────────────────────────────────
    write_progress(out, "oof_scoring_started")
    oof_scores = {}
    for name in model_names:
        oof_scores[name] = {
            "accuracy": float(accuracy_score(y, oof[name].argmax(axis=1))),
            "balanced_accuracy": float(balanced_accuracy_score(y, oof[name].argmax(axis=1))),
            "log_loss": float(log_loss(y, oof[name], labels=list(range(n_classes)))),
        }

    # ── Grid search blend weights ────────────────────────────────────────────
    write_progress(out, "oof_scoring_completed", oof_balanced_accuracy={name: oof_scores[name]["balanced_accuracy"] for name in model_names})
    write_progress(out, "blend_grid_search_started")
    best_blend_ll = float("inf")
    best_weights = tuple([1.0 / len(model_names)] * len(model_names))
    best_blend_acc = -1.0
    for w1 in range(0, 101, 1):
        for w2 in range(0, 101 - w1, 1):
            w3 = 100 - w1 - w2
            w = (w1 / 100.0, w2 / 100.0, w3 / 100.0)
            blend = w[0] * oof[model_names[0]] + w[1] * oof[model_names[1]] + w[2] * oof[model_names[2]]
            ll = float(log_loss(y, blend, labels=list(range(n_classes))))
            acc = float(balanced_accuracy_score(y, blend.argmax(axis=1)))
            if acc > best_blend_acc or (abs(acc - best_blend_acc) < 1e-9 and ll < best_blend_ll):
                best_blend_ll = ll
                best_weights = w
                best_blend_acc = acc

    blend_oof = best_weights[0] * oof[model_names[0]] + best_weights[1] * oof[model_names[1]] + best_weights[2] * oof[model_names[2]]
    blend_bal_acc = float(balanced_accuracy_score(y, blend_oof.argmax(axis=1)))
    best_single_name = max(model_names, key=lambda name: oof_scores[name]["balanced_accuracy"])
    best_single_bal_acc = float(oof_scores[best_single_name]["balanced_accuracy"])
    blend_delta_vs_best_single = blend_bal_acc - best_single_bal_acc
    if blend_delta_vs_best_single < 0:
        best_weights = tuple(1.0 if name == best_single_name else 0.0 for name in model_names)
        blend_oof = oof[best_single_name]
        blend_bal_acc = best_single_bal_acc
        best_blend_ll = float(oof_scores[best_single_name]["log_loss"])
        blend_delta_vs_best_single = 0.0

    # ── Logistic Regression Stacking ─────────────────────────────────────────
    write_progress(
        out,
        "blend_grid_search_completed",
        weights={model_names[i]: round(best_weights[i], 4) for i in range(len(model_names))},
        balanced_accuracy=float(blend_bal_acc),
        log_loss=float(best_blend_ll),
    )
    write_progress(out, "stacker_fit_started")
    stack_features = np.hstack([oof[name] for name in model_names])
    stack_test = np.hstack([test_preds[name] for name in model_names])
    stacker = LogisticRegression(multi_class="multinomial", max_iter=5000, C=1.0, random_state=42)
    stacker.fit(stack_features, y)
    stack_oof = stacker.predict_proba(stack_features)
    stack_bal_acc = float(balanced_accuracy_score(y, stack_oof.argmax(axis=1)))
    stack_log_loss = float(log_loss(y, stack_oof, labels=list(range(n_classes))))
    write_progress(out, "stacker_fit_completed", balanced_accuracy=stack_bal_acc, log_loss=stack_log_loss)

    # ── Choose best method ───────────────────────────────────────────────────
    stack_margin = stack_bal_acc - blend_bal_acc
    if stack_margin >= 0.0005 and stack_log_loss <= max(best_blend_ll + 0.004, 0.1015):
        final_oof = stack_oof
        final_test_proba = stacker.predict_proba(stack_test)
        final_pred_test = final_test_proba.argmax(axis=1)
        best_method = "stack"
        best_oof_score = stack_bal_acc
    else:
        blend_test = best_weights[0] * test_preds[model_names[0]] + best_weights[1] * test_preds[model_names[1]] + best_weights[2] * test_preds[model_names[2]]
        final_oof = blend_oof
        final_test_proba = blend_test
        final_pred_test = final_test_proba.argmax(axis=1)
        best_method = "blend"
        best_oof_score = blend_bal_acc
    write_progress(
        out,
        "ensemble_selected",
        best_method=best_method,
        best_oof_balanced_accuracy=float(best_oof_score),
        blend_balanced_accuracy=float(blend_bal_acc),
        stack_balanced_accuracy=float(stack_bal_acc),
    )

    pred_labels = [class_names[int(i)] for i in final_pred_test]
    submission = pd.DataFrame({id_col: sample[id_col].to_numpy(), target: pred_labels})
    submission.to_csv(out / "submission.csv", index=False)
    write_progress(out, "submission_artifact_written", rows=int(len(submission)))

    np.savez_compressed(
        out / "oof_and_test_probabilities.npz",
        y=y,
        class_names=np.array(class_names),
        model_names=np.array(model_names),
        final_oof=final_oof.astype(np.float32),
        final_test=final_test_proba.astype(np.float32),
        **{f"oof_{name}": oof[name].astype(np.float32) for name in model_names},
        **{f"test_{name}": test_preds[name].astype(np.float32) for name in model_names},
    )

    # ── Calibration diagnostic ───────────────────────────────────────────────
    true_proba = np.array([final_oof[i, y[i]] for i in range(len(y))], dtype=np.float32)
    conf_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    calibration_rows = []
    for b in range(len(conf_bins) - 1):
        mask = (true_proba >= conf_bins[b]) & (true_proba < conf_bins[b + 1])
        if mask.sum() == 0:
            continue
        calibration_rows.append({
            "confidence_bin": f"[{conf_bins[b]:.2f}, {conf_bins[b+1]:.2f})",
            "count": int(mask.sum()),
            "accuracy": float((final_oof[mask].argmax(axis=1) == y[mask]).mean()),
            "mean_confidence": float(final_oof[mask].max(axis=1).mean()),
        })

    # ── Metrics ──────────────────────────────────────────────────────────────
    cv_summary = {}
    for name in model_names:
        scores = cv_scores[name]
        cv_summary[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}

    metrics = {
        "schema": "academic_research_os.hpc_boosting_ensemble_metrics.v1",
        "status": "passed",
        "competition": "playground-series-s6e6",
        "task_id": "playground_series_s6e6",
        "runner": runner_label,
        "packages_available": AVAILABLE,
        "using_boosting": USE_BOOSTING,
        "n_folds": n_folds,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "resource_config": RUN_CONFIG,
        "selected_gpu_device": SELECTED_GPU_DEVICE or None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features_after_encoding": int(x_all.shape[1]),
        "feature_columns": feature_cols,
        "engineered_feature_columns": engineered_feature_cols,
        "engineered_feature_count": int(len(engineered_feature_cols)),
        "categorical_features": categorical,
        "classes": class_names,
        "model_params": {
            "lgb": build_lgb_model(seeds[0], n_classes).get_params(),
            "xgb": build_xgb_model(seeds[0], n_classes).get_params(),
            "cat": {
                "iterations": cfg_int("cat_iterations", 2000, 80, 4000),
                "learning_rate": 0.03,
                "depth": 8,
                "l2_leaf_reg": 3,
                "auto_class_weights": "Balanced",
                "task_type": cfg_choice("cat_task_type", "GPU", {"CPU", "GPU", "auto"}),
            },
        },
        "cv_fold_accuracy": cv_summary,
        "oof_accuracy": {name: oof_scores[name]["accuracy"] for name in model_names},
        "oof_balanced_accuracy": {name: oof_scores[name]["balanced_accuracy"] for name in model_names},
        "oof_log_loss": {name: oof_scores[name]["log_loss"] for name in model_names},
        "ensemble": {
            "blend": {
                "weights": {model_names[i]: round(best_weights[i], 4) for i in range(len(model_names))},
                "balanced_accuracy": blend_bal_acc,
                "log_loss": float(log_loss(y, blend_oof, labels=list(range(n_classes)))),
                "best_single_model": best_single_name,
                "best_single_balanced_accuracy": best_single_bal_acc,
                "blend_delta_vs_best_single": blend_delta_vs_best_single,
            },
            "stack": {
                "balanced_accuracy": stack_bal_acc,
                "log_loss": stack_log_loss,
                "margin_vs_blend": stack_margin,
                "promotion_policy": "requires >=0.0005 balanced_accuracy lift and bounded log_loss risk",
            },
            "best_method": best_method,
            "best_validation_score": float(best_oof_score),
            "risk_recommendation": "score_gate_review_required",
        },
        "calibration": calibration_rows,
        "seconds": round(time.time() - started, 3),
        "submission_rows": int(len(submission)),
        "prediction_distribution": submission[target].value_counts().to_dict(),
        "human_gate_required_for_official_submission": True,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress(out, "metrics_artifact_written", best_method=best_method, best_validation_score=float(best_oof_score))

    # ── Report ───────────────────────────────────────────────────────────────
    report_lines = [
        f"# HPC {'Boosting' if USE_BOOSTING else 'Sklearn'} Ensemble Run",
        "",
        f"- task: `{metrics['task_id']}`",
        f"- runner: `{runner_label}`",
        f"- packages: LGB={AVAILABLE['lightgbm']}, XGB={AVAILABLE['xgboost']}, CAT={AVAILABLE['catboost']}",
        f"- folds: `{n_folds}` x seeds: `{len(seeds)}`",
        f"- best method: `{best_method}` balanced_accuracy: `{best_oof_score:.6f}`",
        "",
        "## CV Accuracy (per-fold, mean +/- std)",
        *[f"- {n.upper()}: {cv_summary[n]['mean']:.6f} +/- {cv_summary[n]['std']:.6f}" for n in model_names],
        "",
        "## OOF Ensemble",
        *[f"- {n.upper()}: bal_acc={oof_scores[n]['balanced_accuracy']:.6f} log_loss={oof_scores[n]['log_loss']:.6f}" for n in model_names],
    ]
    if len(model_names) == 3:
        report_lines += [
            f"- Blend: bal_acc={blend_bal_acc:.6f} (weights: {best_weights[0]:.2f}/{best_weights[1]:.2f}/{best_weights[2]:.2f})",
            f"- Stack (LogisticRegression): bal_acc={stack_bal_acc:.6f}",
        ]
    report_lines += [
        "",
        f"- submission rows: `{metrics['submission_rows']}`",
        "- official Kaggle submission remains behind Human Gate.",
    ]
    (out / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    write_progress(
        out,
        "boosting_ensemble_completed",
        runner=runner_label,
        best_method=best_method,
        best_oof_balanced_accuracy=float(best_oof_score),
        packages=AVAILABLE,
        seconds=metrics["seconds"],
    )


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S6E6 boosting ensemble through HPC SSH.")
    add_hpc_runtime_arguments(parser)
    parser.add_argument("--remote-python", default="")
    parser.add_argument("--local-artifact-dir", default="")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="42,3407,12345")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--xgb-device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--cat-task-type", choices=["CPU", "GPU", "auto"], default="GPU")
    parser.add_argument("--gpu-device-id", default="auto")
    parser.add_argument("--lgb-estimators", type=int, default=1500)
    parser.add_argument("--xgb-estimators", type=int, default=1800)
    parser.add_argument("--cat-iterations", type=int, default=2000)
    parser.add_argument("--recovery-wait-seconds", type=int, default=1800)
    args = parser.parse_args()
    validate_hpc_runtime_arguments(parser, args)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_dir = f"{args.remote_root.rstrip('/')}/playground_series_s6e6/{run_id}_boosting_ensemble"
    local_artifact_dir = Path(args.local_artifact_dir) if args.local_artifact_dir else ROOT / "workspace" / "gpu" / "playground_series_s6e6" / f"{run_id}_boosting_ensemble"
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    client = connect(args)
    sftp = client.open_sftp()
    try:
        sftp_mkdirs(sftp, f"{remote_dir}/data")
        for name in ["train.csv", "test.csv", "sample_submission.csv"]:
            upload_file(sftp, ROOT / "tasks" / "playground_series_s6e6" / "data" / name, f"{remote_dir}/data/{name}")
        script_local = local_artifact_dir / "remote_boosting_ensemble.py"
        script_local.write_text(remote_boosting_ensemble_script(), encoding="utf-8")
        upload_file(sftp, script_local, f"{remote_dir}/remote_boosting_ensemble.py")
        run_config = {
            "folds": args.folds,
            "seeds": args.seeds,
            "sample_rows": args.sample_rows,
            "xgb_device": args.xgb_device,
            "cat_task_type": args.cat_task_type,
            "gpu_device_id": args.gpu_device_id,
            "lgb_estimators": args.lgb_estimators,
            "xgb_estimators": args.xgb_estimators,
            "cat_iterations": args.cat_iterations,
        }
        config_local = local_artifact_dir / "run_config.json"
        config_local.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        upload_file(sftp, config_local, f"{remote_dir}/run_config.json")

        if args.remote_python.strip():
            command = f"cd {shlex.quote(remote_dir)} && PYTHONUNBUFFERED=1 {shlex.quote(args.remote_python.strip())} remote_boosting_ensemble.py"
        else:
            command = f"cd {shlex.quote(remote_dir)} && (command -v python3 >/dev/null 2>&1 && PYTHONUNBUFFERED=1 python3 remote_boosting_ensemble.py || PYTHONUNBUFFERED=1 python remote_boosting_ensemble.py)"
        started = time.time()
        _, stdout, _stderr = client.exec_command(command, timeout=args.timeout_seconds)
        channel = stdout.channel
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        deadline = started + args.timeout_seconds
        while not channel.exit_status_ready():
            if channel.recv_ready():
                stdout_chunks.append(channel.recv(65536).decode("utf-8", "replace"))
            if channel.recv_stderr_ready():
                stderr_chunks.append(channel.recv_stderr(65536).decode("utf-8", "replace"))
            if time.time() > deadline:
                channel.close()
                raise TimeoutError(f"Remote boosting command exceeded timeout_seconds={args.timeout_seconds}")
            time.sleep(1)
        while channel.recv_ready():
            stdout_chunks.append(channel.recv(65536).decode("utf-8", "replace"))
        while channel.recv_stderr_ready():
            stderr_chunks.append(channel.recv_stderr(65536).decode("utf-8", "replace"))
        exit_status = channel.recv_exit_status()
        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)
        (local_artifact_dir / "remote_stdout.log").write_text(stdout_text, encoding="utf-8")
        (local_artifact_dir / "remote_stderr.log").write_text(stderr_text, encoding="utf-8")
        close_sftp_quietly(sftp)
        sftp = None

        output_names = ["metrics.json", "submission.csv", "report.md", "oof_and_test_probabilities.npz", "progress.jsonl"]
        downloaded = download_outputs_with_retries(args, remote_dir, local_artifact_dir, output_names)
        recovered_after_control_channel_loss = False
        recovery_attempted = not all(downloaded.values())
        if not all(downloaded.values()):
            recovered = wait_for_remote_outputs(
                args,
                remote_dir,
                local_artifact_dir,
                output_names,
                wait_seconds=args.recovery_wait_seconds,
            )
            recovered_after_control_channel_loss = all(recovered.values()) and exit_status != 0
            downloaded = recovered

        metrics: dict[str, Any] | None = None
        metrics_path = local_artifact_dir / "metrics.json"
        if metrics_path.is_file() and metrics_path.stat().st_size > 0:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        passed = bool(downloaded.get("metrics.json") and downloaded.get("submission.csv") and downloaded.get("oof_and_test_probabilities.npz"))
        manifest_status = "passed" if exit_status == 0 and passed else "recovered_passed" if recovered_after_control_channel_loss and passed else "failed"
        manifest = {
            "schema": "academic_research_os.hpc_boosting_ensemble_manifest.v1",
            "status": manifest_status,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "runner": metrics.get("runner", "unknown") if metrics else "unknown",
            "remote_dir": remote_dir,
            "remote_python": args.remote_python.strip() or "system_python_auto",
            "run_config": run_config,
            "local_artifact_dir": str(local_artifact_dir.relative_to(ROOT)).replace("\\", "/"),
            "exit_status": exit_status,
            "seconds": round(time.time() - started, 3),
            "downloaded": downloaded,
            "recovery": {
                "attempted": recovery_attempted,
                "recovered_after_control_channel_loss": recovered_after_control_channel_loss,
                "wait_seconds": args.recovery_wait_seconds,
            },
            "metrics": metrics,
            "stdout_tail": stdout_text[-4000:],
            "stderr_tail": stderr_text[-4000:],
            "human_gate_required_for_official_submission": True,
        }
        (local_artifact_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        if manifest["status"] not in {"passed", "recovered_passed"}:
            raise SystemExit(1)
    finally:
        close_sftp_quietly(sftp)
        client.close()


if __name__ == "__main__":
    main()
