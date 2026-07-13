"""HPC SSH Launcher for S6E6 Ensemble via scikit-learn tree models."""

from __future__ import annotations

import argparse
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


def socks5_connect(proxy_host: str, proxy_port: int, dest_host: str, dest_port: int, timeout: float = 30.0) -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout)
    sock.settimeout(timeout)
    sock.sendall(b"\x05\x01\x00")
    if sock.recv(2) != b"\x05\x00":
        raise RuntimeError("SOCKS5 method negotiation failed")
    host_bytes = dest_host.encode("ascii")
    sock.sendall(b"\x05\x01\x00\x03" + bytes([len(host_bytes)]) + host_bytes + struct.pack("!H", dest_port))
    header = sock.recv(4)
    if len(header) != 4 or header[0] != 5 or header[1] != 0:
        raise RuntimeError(f"SOCKS5 connect failed with response {header!r}")
    if header[3] == 1:
        sock.recv(4)
    elif header[3] == 3:
        sock.recv(sock.recv(1)[0])
    elif header[3] == 4:
        sock.recv(16)
    sock.recv(2)
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
    client: paramiko.SSHClient,
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
            sftp: paramiko.SFTPClient | None = None
            try:
                sftp = client.open_sftp()
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
        if last_error is not None:
            raise RuntimeError(f"Failed to download {name} after {attempts} attempts: {last_error}") from last_error
    return downloaded


def remote_ensemble_script() -> str:
    return r'''
"""S6E6 Ensemble via scikit-learn — RandomForest + ExtraTrees + HistGradientBoosting + LogisticRegression stacking.

Uses only packages known to be installed on the HPC server.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
            df[f"{a}_minus_{b}"] = df[a] - df[b]
    if "redshift" in df.columns:
        df["redshift_log1p"] = np.log1p(np.clip(df["redshift"].astype(float), 0, None))
    return df


def main() -> None:
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
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = train[target].map(class_to_idx).astype("int64").to_numpy()

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

    n_folds = 5
    n_classes = len(class_names)
    seeds = [42, 3407, 12345]

    # ── Base models (scikit-learn, guaranteed available) ─────────────────
    base_models = {
        "rf": lambda rseed: RandomForestClassifier(
            n_estimators=400, max_depth=18, min_samples_leaf=32,
            max_features="sqrt", n_jobs=-1, random_state=rseed, verbose=0,
        ),
        "hgb": lambda rseed: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=8, max_leaf_nodes=63,
            min_samples_leaf=32, l2_regularization=0.5, early_stopping=True,
            validation_fraction=0.12, n_iter_no_change=30, random_state=rseed, verbose=0,
        ),
        "et": lambda rseed: ExtraTreesClassifier(
            n_estimators=400, max_depth=18, min_samples_leaf=32,
            max_features="sqrt", n_jobs=-1, random_state=rseed, verbose=0,
        ),
    }

    # ── OOF storage ──────────────────────────────────────────────────────
    model_names = list(base_models.keys())
    oof = {name: np.zeros((len(y), n_classes), dtype=np.float64) for name in model_names}
    test_preds = {name: np.zeros((len(x_test), n_classes), dtype=np.float64) for name in model_names}
    cv_scores = {name: [] for name in model_names}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    event_log = []

    for seed in seeds:
        for fold, (train_idx, val_idx) in enumerate(skf.split(x_all, y)):
            X_tr, X_va = x_all[train_idx], x_all[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            tag = f"seed{seed}_fold{fold+1}"

            for name, build_fn in base_models.items():
                model = build_fn(seed)
                model.fit(X_tr, y_tr)
                p_val = model.predict_proba(X_va)
                # Handle case where predict_proba returns 1D array for binary
                if p_val.ndim == 1:
                    p_val = np.column_stack([1 - p_val, p_val])
                p_test = model.predict_proba(x_test)
                if p_test.ndim == 1:
                    p_test = np.column_stack([1 - p_test, p_test])

                oof[name][val_idx] += p_val / len(seeds)
                test_preds[name] += p_test / (n_folds * len(seeds))
                acc = float(accuracy_score(y_va, p_val.argmax(axis=1)))
                cv_scores[name].append(acc)
                event_log.append({"model": name, "seed": seed, "fold": fold + 1, "accuracy": acc})

    # ── OOF scores ───────────────────────────────────────────────────────
    oof_scores = {}
    for name in model_names:
        oof_scores[name] = {
            "accuracy": float(accuracy_score(y, oof[name].argmax(axis=1))),
            "log_loss": float(log_loss(y, oof[name], labels=list(range(n_classes)))),
        }

    # ── Logistic Regression Stacking ─────────────────────────────────────
    stack_features = np.hstack([oof[name] for name in model_names])
    stack_test = np.hstack([test_preds[name] for name in model_names])

    stacker = LogisticRegression(multi_class="multinomial", max_iter=5000, C=1.0, random_state=42)
    stacker.fit(stack_features, y)
    stack_oof = stacker.predict_proba(stack_features)
    stack_acc = float(accuracy_score(y, stack_oof.argmax(axis=1)))
    stack_ll = float(log_loss(y, stack_oof, labels=list(range(n_classes))))

    # ── Simple blend ─────────────────────────────────────────────────────
    # Grid search blend weights
    best_blend_ll = float("inf")
    best_weights = (0.4, 0.35, 0.25)
    for w1 in range(10, 71, 3):
        for w2 in range(10, 71, 3):
            w3 = 100 - w1 - w2
            if w3 < 5:
                continue
            w = (w1 / 100.0, w2 / 100.0, w3 / 100.0)
            blend = w[0] * oof["rf"] + w[1] * oof["hgb"] + w[2] * oof["et"]
            ll = float(log_loss(y, blend, labels=list(range(n_classes))))
            if ll < best_blend_ll:
                best_blend_ll = ll
                best_weights = w

    blend_oof = best_weights[0] * oof["rf"] + best_weights[1] * oof["hgb"] + best_weights[2] * oof["et"]
    blend_acc = float(accuracy_score(y, blend_oof.argmax(axis=1)))
    blend_ll = float(log_loss(y, blend_oof, labels=list(range(n_classes))))

    # ── Test predictions (use blend as primary, stacker as alternative) ──
    blend_test = best_weights[0] * test_preds["rf"] + best_weights[1] * test_preds["hgb"] + best_weights[2] * test_preds["et"]
    stack_test_pred = stacker.predict_proba(stack_test)

    # Choose best OOF method for final submission
    if stack_acc > blend_acc:
        final_pred = stack_test_pred.argmax(axis=1)
        best_method = "stack"
        best_oof_acc = stack_acc
    else:
        final_pred = blend_test.argmax(axis=1)
        best_method = "blend"
        best_oof_acc = blend_acc

    pred_labels = [class_names[int(i)] for i in final_pred]
    submission = pd.DataFrame({id_col: sample[id_col].to_numpy(), target: pred_labels})
    submission.to_csv(out / "submission.csv", index=False)

    # ── Metrics ──────────────────────────────────────────────────────────
    cv_summary = {}
    for name in model_names:
        scores = cv_scores[name]
        cv_summary[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}

    metrics = {
        "schema": "academic_research_os.hpc_ensemble_metrics.v2",
        "status": "passed",
        "competition": "playground-series-s6e6",
        "task_id": "playground_series_s6e6",
        "runner": "hpc_sklearn_ensemble_rf_hgb_et_stack",
        "n_folds": n_folds,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features_after_encoding": int(x_all.shape[1]),
        "classes": class_names,
        "cv_fold_accuracy": cv_summary,
        "oof_accuracy": {
            **{name: oof_scores[name]["accuracy"] for name in model_names},
            "blend": blend_acc,
            "stack": stack_acc,
        },
        "oof_log_loss": {
            **{name: oof_scores[name]["log_loss"] for name in model_names},
            "blend": blend_ll,
            "stack": stack_ll,
        },
        "blend_weights": {
            "rf": round(best_weights[0], 4),
            "hgb": round(best_weights[1], 4),
            "et": round(best_weights[2], 4),
        },
        "best_method": best_method,
        "best_validation_score": float(best_oof_acc),
        "seconds": round(time.time() - started, 3),
        "submission_rows": int(len(submission)),
        "submission_columns": submission.columns.tolist(),
        "prediction_distribution": submission[target].value_counts().to_dict(),
        "packages": "sklearn_only",
        "human_gate_required_for_official_submission": True,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Report ───────────────────────────────────────────────────────────
    (out / "report.md").write_text("\n".join([
        "# HPC Ensemble Run (sklearn RF+HGB+ET + Stacking)",
        "",
        f"- task: `{metrics['task_id']}`",
        f"- runner: `{metrics['runner']}`",
        f"- folds: `{n_folds}` x seeds: `{len(seeds)}`",
        f"- best method: `{best_method}` score: `{best_oof_acc:.6f}`",
        "",
        "## CV Accuracy (per-fold, mean +/- std)",
        *[f"- {n.upper()}: {cv_summary[n]['mean']:.6f} +/- {cv_summary[n]['std']:.6f}" for n in model_names],
        "",
        "## OOF Ensemble",
        *[f"- {n.upper()}: {oof_scores[n]['accuracy']:.6f}" for n in model_names],
        f"- Blend: {blend_acc:.6f} (weights: {best_weights[0]:.2f}/{best_weights[1]:.2f}/{best_weights[2]:.2f})",
        f"- Stack (LogisticRegression): {stack_acc:.6f}",
        "",
        f"- submission rows: `{metrics['submission_rows']}`",
        "- official Kaggle submission remains behind Human Gate.",
    ]), encoding="utf-8")

    print(json.dumps({"event": "ensemble_completed", "best_method": best_method, "best_oof_accuracy": best_oof_acc, "seconds": metrics["seconds"]}))


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S6E6 sklearn ensemble through HPC SSH.")
    add_hpc_runtime_arguments(parser)
    parser.add_argument("--local-artifact-dir", default="")
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    args = parser.parse_args()
    validate_hpc_runtime_arguments(parser, args)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_dir = f"{args.remote_root.rstrip('/')}/playground_series_s6e6/{run_id}_ensemble"
    local_artifact_dir = Path(args.local_artifact_dir) if args.local_artifact_dir else ROOT / "workspace" / "gpu" / "playground_series_s6e6" / f"{run_id}_ensemble"
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    client = connect(args)
    sftp = client.open_sftp()
    try:
        sftp_mkdirs(sftp, f"{remote_dir}/data")
        for name in ["train.csv", "test.csv", "sample_submission.csv"]:
            upload_file(sftp, ROOT / "tasks" / "playground_series_s6e6" / "data" / name, f"{remote_dir}/data/{name}")
        script_local = local_artifact_dir / "remote_ensemble.py"
        script_local.write_text(remote_ensemble_script(), encoding="utf-8")
        upload_file(sftp, script_local, f"{remote_dir}/remote_ensemble.py")

        command = f"cd '{remote_dir}' && (command -v python3 >/dev/null 2>&1 && python3 remote_ensemble.py || python remote_ensemble.py)"
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
            client,
            remote_dir,
            local_artifact_dir,
            ["metrics.json", "submission.csv", "report.md"],
        )

        metrics: dict[str, Any] | None = None
        metrics_path = local_artifact_dir / "metrics.json"
        if metrics_path.is_file() and metrics_path.stat().st_size > 0:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        manifest = {
            "schema": "academic_research_os.hpc_ensemble_manifest.v2",
            "status": "passed" if exit_status == 0 and downloaded.get("metrics.json") and downloaded.get("submission.csv") else "failed",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "runner": "hpc_sklearn_ensemble_rf_hgb_et_stack",
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
