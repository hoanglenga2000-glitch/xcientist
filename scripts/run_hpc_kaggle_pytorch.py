from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko


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
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        sock=sock,
        allow_agent=False,
        look_for_keys=False,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
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


def remote_train_script() -> str:
    return r'''
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
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


class MLP(torch.nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.12),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.08),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    started = time.time()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True

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
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", make_encoder(), categorical),
        ],
        remainder="drop",
    )
    x_all = preprocessor.fit_transform(train_x[feature_cols]).astype(np.float32)
    x_test = preprocessor.transform(test_x[feature_cols]).astype(np.float32)
    x_train, x_val, y_train, y_val = train_test_split(x_all, y, test_size=0.18, random_state=42, stratify=y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(x_train.shape[1], len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    counts = np.bincount(y_train, minlength=len(class_names)).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    train_tensor = torch.tensor(x_train)
    target_tensor = torch.tensor(y_train, dtype=torch.long)
    val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
    batch_size = 8192
    epochs = 18
    best = {"accuracy": -1.0, "epoch": 0, "log_loss": None}
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_tensor))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = train_tensor[idx].to(device, non_blocking=True)
            yb = target_tensor[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            logits = model(val_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)
        acc = float(accuracy_score(y_val, pred))
        ll = float(log_loss(y_val, probs, labels=list(range(len(class_names)))))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": acc, "val_log_loss": ll}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if acc > best["accuracy"]:
            best = {"accuracy": acc, "epoch": epoch, "log_loss": ll}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_tensor = torch.tensor(x_test, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_tensor), batch_size):
            logits = model(test_tensor[start:start + batch_size].to(device, non_blocking=True))
            preds.append(torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy())
    pred_idx = np.concatenate(preds)
    pred_labels = [class_names[int(i)] for i in pred_idx]
    submission = pd.DataFrame({id_col: sample[id_col].to_numpy(), target: pred_labels})
    submission.to_csv(out / "submission.csv", index=False)

    metrics = {
        "status": "passed",
        "competition": "playground-series-s6e6",
        "task_id": "playground_series_s6e6",
        "runner": "hpc_pytorch_mlp",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "device0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features_after_encoding": int(x_all.shape[1]),
        "classes": class_names,
        "best": best,
        "history": history,
        "seconds": round(time.time() - started, 3),
        "submission_rows": int(len(submission)),
        "submission_columns": submission.columns.tolist(),
        "prediction_distribution": submission[target].value_counts().to_dict(),
        "human_gate_required_for_official_submission": True,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(
        "\n".join([
            "# HPC Kaggle PyTorch Run",
            "",
            f"- task: `{metrics['task_id']}`",
            f"- device: `{metrics['device']}` / `{metrics['device0']}`",
            f"- validation accuracy: `{best['accuracy']:.6f}` at epoch `{best['epoch']}`",
            f"- validation log_loss: `{best['log_loss']:.6f}`",
            f"- submission rows: `{metrics['submission_rows']}`",
            "- official Kaggle submission remains behind Human Gate.",
        ]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
'''


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run playground-series-s6e6 through the verified HPC SSH GPU path.")
    parser.add_argument("--host", default="100.85.169.63")
    parser.add_argument("--port", type=int, default=1235)
    parser.add_argument("--user", required=True)
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=7890)
    parser.add_argument("--password-env", default="GPU_SSH_PASSWORD")
    parser.add_argument("--remote-root", default="/hpc2ssd/JH_DATA/spooler/aimslab/research_agent_workstation")
    parser.add_argument("--local-artifact-dir", default="")
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_dir = f"{args.remote_root.rstrip('/')}/playground_series_s6e6/{run_id}"
    local_artifact_dir = Path(args.local_artifact_dir) if args.local_artifact_dir else ROOT / "workspace" / "gpu" / "playground_series_s6e6" / run_id
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    client = connect(args)
    sftp = client.open_sftp()
    try:
        sftp_mkdirs(sftp, f"{remote_dir}/data")
        for name in ["train.csv", "test.csv", "sample_submission.csv"]:
            upload_file(sftp, ROOT / "tasks" / "playground_series_s6e6" / "data" / name, f"{remote_dir}/data/{name}")
        script_local = local_artifact_dir / "remote_train.py"
        script_local.write_text(remote_train_script(), encoding="utf-8")
        upload_file(sftp, script_local, f"{remote_dir}/remote_train.py")

        command = f"cd '{remote_dir}' && (command -v python3 >/dev/null 2>&1 && python3 remote_train.py || python remote_train.py)"
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
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        manifest = {
            "status": "passed" if exit_status == 0 and downloaded.get("metrics.json") and downloaded.get("submission.csv") else "failed",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
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
