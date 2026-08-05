"""Truthful Titanic proof workflow for the AIBuildAI-2-equivalent DAG."""
from __future__ import annotations

import hashlib
import json
import os
import py_compile
import shutil
import textwrap
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score

from xsci.user_request import UserRequest

from ..hpc_runtime import HpcRuntime
from .aibuild_v1 import build_aibuild_run, run_directory, write_current_run_pointer
from .multi_agent import AgentResult, HandoffEnvelope, MultiAgentStore, MultiAgentSupervisor, SupervisorRun


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _combined_data_hash(data_dir: Path) -> str:
    required = [data_dir / "train.csv", data_dir / "test.csv", data_dir / "sample_submission.csv"]
    return hashlib.sha256("".join(_sha256(path) for path in required).encode("ascii")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _artifact(path: Path, run_dir: Path, *, kind: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)).replace("\\", "/"),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "kind": kind,
    }


_MUTABLE_RUN_STATE = {
    "control.json",
    "events.jsonl",
    "handoffs.jsonl",
    "messages.jsonl",
    "run.json",
    "task_graph.json",
}


def _manifest_entries(run_dir: Path) -> list[dict[str, Any]]:
    """Hash immutable research artifacts without creating ledger hash cycles."""
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if relative.as_posix() in _MUTABLE_RUN_STATE:
            continue
        if path.name == "artifact_manifest.json" or ".tmp" in path.name:
            continue
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        entries.append(_artifact(path, run_dir, kind=path.name))
    return entries


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_STRATEGIES = {
    1: {
        "name": "linear_baseline",
        "hypothesis": "A leakage-safe regularized logistic baseline establishes a stable lower bound.",
        "model": "LogisticRegression",
        "uses_gpu": False,
    },
    2: {
        "name": "tree_baseline",
        "hypothesis": "A nonlinear ExtraTrees model captures family and fare interactions missed by the linear baseline.",
        "model": "ExtraTreesClassifier",
        "uses_gpu": False,
    },
    3: {
        "name": "feature_mlp_gpu",
        "hypothesis": "Leakage-safe engineered features plus a small CUDA MLP improve cross-validated accuracy.",
        "model": "PyTorchMLP",
        "uses_gpu": True,
    },
}


def _solution_source(variant: str, *, run_id: str, data_hash: str) -> str:
    return textwrap.dedent(
        f'''\
        import argparse
        import hashlib
        import json
        import os
        import platform
        import random
        import sys
        import time

        import numpy as np
        import pandas as pd
        import sklearn
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        VARIANT = {variant!r}
        RUN_ID = {run_id!r}
        DATA_HASH = {data_hash!r}
        SEED = 20260721
        FEATURES_NUM = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
        FEATURES_CAT = ["Sex", "Embarked", "Title"]

        def feature_frame(frame):
            out = frame.copy()
            out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
            out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
            out["Title"] = out["Name"].fillna("").str.extract(r",\\s*([^.]*)\\.", expand=False).fillna("Unknown")
            rare = ~out["Title"].isin(["Mr", "Mrs", "Miss", "Master"])
            out.loc[rare, "Title"] = "Rare"
            return out[FEATURES_NUM + FEATURES_CAT]

        def encoder():
            try:
                return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                return OneHotEncoder(handle_unknown="ignore", sparse=False)

        def preprocessor():
            numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
            categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", encoder())])
            return ColumnTransformer([("num", numeric, FEATURES_NUM), ("cat", categorical, FEATURES_CAT)])

        def sklearn_model():
            if VARIANT == "linear_baseline":
                return LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            return ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=SEED)

        def torch_fold(x_train, y_train, x_valid, x_test, fold):
            import torch
            torch.manual_seed(SEED + fold)
            torch.cuda.manual_seed_all(SEED + fold)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA_REQUIRED: torch.cuda.is_available() is false")
            device = torch.device("cuda:0")
            train_x = torch.tensor(x_train, dtype=torch.float32, device=device)
            train_y = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
            valid_x = torch.tensor(x_valid, dtype=torch.float32, device=device)
            test_x = torch.tensor(x_test, dtype=torch.float32, device=device)
            model = torch.nn.Sequential(
                torch.nn.Linear(train_x.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.15),
                torch.nn.Linear(64, 24), torch.nn.ReLU(), torch.nn.Linear(24, 1),
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
            positives = float(train_y.sum().item())
            negatives = float(len(train_y) - positives)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / max(positives, 1.0)], device=device))
            model.train()
            for _epoch in range(70):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(train_x), train_y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                valid_prob = torch.sigmoid(model(valid_x)).cpu().numpy().reshape(-1)
                test_prob = torch.sigmoid(model(test_x)).cpu().numpy().reshape(-1)
            return valid_prob, test_prob, {{
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated(0)),
                "torch_version": torch.__version__,
            }}

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--data-dir", required=True)
            parser.add_argument("--out-dir", required=True)
            args = parser.parse_args()
            os.makedirs(args.out_dir, exist_ok=True)
            random.seed(SEED); np.random.seed(SEED)
            train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
            test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
            x = feature_frame(train)
            x_test = feature_frame(test)
            y = train["Survived"].astype(int).to_numpy()
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            oof_prob = np.zeros(len(train), dtype=float)
            oof_fold = np.full(len(train), -1, dtype=int)
            test_prob = np.zeros(len(test), dtype=float)
            fold_scores = []
            telemetry = []
            started = time.time()
            for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y)):
                prep = preprocessor()
                x_train = prep.fit_transform(x.iloc[train_idx])
                x_valid = prep.transform(x.iloc[valid_idx])
                x_holdout = prep.transform(x_test)
                if VARIANT == "feature_mlp_gpu":
                    valid_prob, fold_test_prob, fold_telemetry = torch_fold(
                        np.asarray(x_train, dtype=np.float32), y[train_idx],
                        np.asarray(x_valid, dtype=np.float32), np.asarray(x_holdout, dtype=np.float32), fold,
                    )
                    telemetry.append(fold_telemetry)
                else:
                    model = sklearn_model()
                    model.fit(x_train, y[train_idx])
                    valid_prob = model.predict_proba(x_valid)[:, 1]
                    fold_test_prob = model.predict_proba(x_holdout)[:, 1]
                prediction = (valid_prob >= 0.5).astype(int)
                score = float(accuracy_score(y[valid_idx], prediction))
                fold_scores.append(score)
                oof_prob[valid_idx] = valid_prob
                oof_fold[valid_idx] = fold
                test_prob += fold_test_prob / 5.0
                print(f"FOLD={{fold}} ACCURACY={{score:.8f}}", flush=True)
            oof_prediction = (oof_prob >= 0.5).astype(int)
            cv_score = float(accuracy_score(y, oof_prediction))
            submission = pd.DataFrame({{"PassengerId": test["PassengerId"].astype(int), "Survived": (test_prob >= 0.5).astype(int)}})
            oof = pd.DataFrame({{
                "PassengerId": train["PassengerId"].astype(int),
                "Survived": oof_prediction,
                "probability": oof_prob,
                "fold": oof_fold,
            }})
            submission.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False)
            oof.to_csv(os.path.join(args.out_dir, "oof_predictions.csv"), index=False)
            source_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
            metrics = {{
                "schema": "evomind.titanic.metrics.v1",
                "run_id": RUN_ID,
                "data_hash": DATA_HASH,
                "variant": VARIANT,
                "metric": "accuracy",
                "cv_strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=20260721)",
                "cv_score": cv_score,
                "fold_scores": fold_scores,
                "fold_mean": float(np.mean(fold_scores)),
                "fold_std": float(np.std(fold_scores)),
                "seed": SEED,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "oof_rows": int(len(oof)),
                "uses_cuda": VARIANT == "feature_mlp_gpu",
                "gpu_telemetry": telemetry,
                "source_sha256": source_hash,
                "elapsed_seconds": round(time.time() - started, 3),
                "official_kaggle_score": None,
            }}
            with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, ensure_ascii=False, indent=2)
            environment = {{
                "run_id": RUN_ID,
                "data_hash": DATA_HASH,
                "python": sys.version,
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "variant": VARIANT,
                "source_sha256": source_hash,
                "uses_cuda": VARIANT == "feature_mlp_gpu",
                "gpu_telemetry": telemetry,
            }}
            with open(os.path.join(args.out_dir, "environment.json"), "w", encoding="utf-8") as handle:
                json.dump(environment, handle, ensure_ascii=False, indent=2)
            print(f"CV_SCORE={{cv_score:.8f}}", flush=True)

        if __name__ == "__main__":
            main()
        '''
    )


def review_solution_evidence(
    *,
    run_dir: Path,
    data_dir: Path,
    run_id: str,
    solution_id: str,
    expected_variant: str,
    expected_cuda: bool,
    data_hash: str,
    remote_run_dir: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Recompute a candidate contract from raw files without trusting agent summaries."""
    output_dir = run_dir / "solutions" / solution_id / "output"
    solution_dir = run_dir / "solutions" / solution_id
    required_output = {"metrics.json", "oof_predictions.csv", "submission.csv", "environment.json", "training.log"}
    required_solution = {"solution.py", "code_manifest.json", "hpc_job.json"}
    present_output = {path.name for path in output_dir.iterdir()} if output_dir.is_dir() else set()
    present_solution = {path.name for path in solution_dir.iterdir()} if solution_dir.is_dir() else set()
    checks: dict[str, Any] = {
        "solution_id": solution_id,
        "required_artifacts": required_output.issubset(present_output) and required_solution.issubset(present_solution),
    }
    missing = sorted((required_output - present_output) | (required_solution - present_solution))
    if missing:
        checks.update({"missing": missing, "passed": False})
        return checks, None

    try:
        train = pd.read_csv(data_dir / "train.csv").sort_values("PassengerId").reset_index(drop=True)
        test = pd.read_csv(data_dir / "test.csv").reset_index(drop=True)
        metrics = _read_json(output_dir / "metrics.json")
        environment = _read_json(output_dir / "environment.json")
        code_manifest = _read_json(solution_dir / "code_manifest.json")
        hpc_job = _read_json(solution_dir / "hpc_job.json")
        oof = pd.read_csv(output_dir / "oof_predictions.csv").sort_values("PassengerId").reset_index(drop=True)
        submission = pd.read_csv(output_dir / "submission.csv")
        training_log = (output_dir / "training.log").read_text(encoding="utf-8", errors="replace")
        source_hash = _sha256(solution_dir / "solution.py")

        oof_predictions = pd.to_numeric(oof.get("Survived"), errors="coerce")
        oof_probabilities = pd.to_numeric(oof.get("probability"), errors="coerce")
        oof_folds = pd.to_numeric(oof.get("fold"), errors="coerce")
        recomputed = (
            float(accuracy_score(train["Survived"].to_numpy(), oof_predictions.to_numpy()))
            if len(oof) == len(train) and not bool(oof_predictions.isna().any())
            else -1.0
        )
        fold_scores = []
        if len(oof) == len(train) and not bool(oof_predictions.isna().any()) and not bool(oof_folds.isna().any()):
            for fold in range(5):
                mask = oof_folds == fold
                if not bool(mask.any()):
                    fold_scores = []
                    break
                fold_scores.append(float(accuracy_score(train.loc[mask, "Survived"], oof_predictions.loc[mask])))
        claimed_fold_scores = metrics.get("fold_scores") if isinstance(metrics.get("fold_scores"), list) else []
        fold_scores_match = len(fold_scores) == len(claimed_fold_scores) == 5 and all(
            abs(left - float(right)) < 1e-12 for left, right in zip(fold_scores, claimed_fold_scores)
        )

        telemetry = metrics.get("gpu_telemetry") if isinstance(metrics.get("gpu_telemetry"), list) else []
        environment_telemetry = environment.get("gpu_telemetry") if isinstance(environment.get("gpu_telemetry"), list) else []
        gpu_telemetry_ok = not expected_cuda or (
            len(telemetry) == 5
            and telemetry == environment_telemetry
            and all(
                isinstance(item, dict)
                and bool(item.get("cuda_device"))
                and int(item.get("cuda_max_memory_allocated") or 0) > 0
                for item in telemetry
            )
        )
        job_hashes = {
            Path(str(item.get("path") or "")).name: str(item.get("sha256") or "")
            for item in hpc_job.get("local_artifacts", [])
            if isinstance(item, dict)
        }
        downloaded_hashes_match = all(
            job_hashes.get(name) == _sha256(output_dir / name)
            for name in required_output
        )
        checks.update({
            "evidence_parseable": True,
            "oof_rows": len(oof) == len(train) == int(metrics.get("oof_rows", -1)),
            "oof_ids_exact": oof["PassengerId"].tolist() == train["PassengerId"].tolist(),
            "oof_unique_ids": bool(oof["PassengerId"].is_unique),
            "oof_no_missing": not bool(oof[["PassengerId", "Survived", "probability", "fold"]].isna().any().any()),
            "oof_prediction_binary": set(oof_predictions.dropna().astype(int).unique()).issubset({0, 1}),
            "oof_probability_range": bool(oof_probabilities.notna().all() and oof_probabilities.between(0.0, 1.0).all()),
            "folds": sorted(oof_folds.dropna().astype(int).unique().tolist()) == [0, 1, 2, 3, 4],
            "metric_recomputed": abs(recomputed - float(metrics.get("cv_score", -2))) < 1e-12,
            "fold_scores_recomputed": fold_scores_match,
            "row_contract": int(metrics.get("n_train", -1)) == len(train) and int(metrics.get("n_test", -1)) == len(test),
            "submission_rows": len(submission) == len(test),
            "submission_columns": list(submission.columns) == ["PassengerId", "Survived"],
            "submission_no_missing": not bool(submission.isna().any().any()),
            "submission_ids_exact": submission["PassengerId"].tolist() == test["PassengerId"].tolist(),
            "submission_prediction_binary": set(pd.to_numeric(submission["Survived"], errors="coerce").dropna().astype(int).unique()).issubset({0, 1}),
            "official_score_absent": metrics.get("official_kaggle_score") is None,
            "variant_contract": metrics.get("variant") == environment.get("variant") == expected_variant,
            "cuda_contract": bool(metrics.get("uses_cuda")) == bool(environment.get("uses_cuda")) == expected_cuda,
            "gpu_telemetry": gpu_telemetry_ok,
            "run_id_binding": metrics.get("run_id") == environment.get("run_id") == hpc_job.get("run_id") == run_id,
            "data_hash_binding": metrics.get("data_hash") == environment.get("data_hash") == data_hash,
            "source_hash_binding": (
                metrics.get("source_sha256")
                == environment.get("source_sha256")
                == code_manifest.get("source_sha256")
                == source_hash
            ),
            "hpc_job_binding": (
                hpc_job.get("status") == "completed"
                and int(hpc_job.get("exit_code", -1)) == 0
                and hpc_job.get("solution_id") == solution_id
                and str(hpc_job.get("remote_dir") or "").startswith(remote_run_dir + "/solutions/")
            ),
            "downloaded_artifact_hashes": downloaded_hashes_match,
            "training_log_evidence": training_log.count("FOLD=") == 5 and "CV_SCORE=" in training_log,
        })
    except Exception as exc:
        checks.update({"evidence_parseable": False, "error_type": type(exc).__name__, "passed": False})
        return checks, None

    checks["passed"] = all(
        value is True
        for key, value in checks.items()
        if key not in {"solution_id", "passed"}
    )
    candidate = None
    if checks["passed"]:
        candidate = {
            "solution_id": solution_id,
            "cv_score": float(metrics["cv_score"]),
            "uses_cuda": bool(metrics["uses_cuda"]),
            "source_sha256": source_hash,
            "data_hash": data_hash,
        }
    return checks, candidate


class TitanicExecutors:
    def __init__(self, *, workspace_root: Path, run_dir: Path, request: UserRequest, runtime: HpcRuntime) -> None:
        self.workspace_root = workspace_root
        self.run_dir = run_dir
        self.request = request
        self.runtime = runtime
        self.data_dir = workspace_root / "tasks" / "titanic" / "data"
        self.data_hash = _combined_data_hash(self.data_dir)

    def setup(self, task, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"
        sample_path = self.data_dir / "sample_submission.csv"
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        if len(train) != 891 or "Survived" not in train or len(test) != 418:
            raise RuntimeError("Titanic data contract mismatch")
        initial_probe = self.runtime.probe()
        initial_probe_path = self.run_dir / "hpc_probe_initial.json"
        _atomic_json(initial_probe_path, initial_probe.to_dict())
        runtime_environment = None
        if initial_probe.status != "passed" and initial_probe.gpu_inventory:
            runtime_environment = self.runtime.prepare_python_environment()
        probe = self.runtime.probe(use_run_environment=runtime_environment is not None)
        probe_path = self.run_dir / "hpc_probe.json"
        _atomic_json(probe_path, probe.to_dict())
        if probe.status != "passed":
            raise RuntimeError(f"HPC_{probe.failure_type.upper()}: {probe.error}")
        runtime_path = self.run_dir / "hpc_runtime_environment.json"
        _atomic_json(runtime_path, runtime_environment or {
            "schema": "evomind.hpc_runtime_environment.v1",
            "status": "system_environment_ready",
            "remote_python_deps": None,
        })
        stage = self.runtime.stage_data(self.data_dir)
        if stage["data_hash"] != self.data_hash:
            raise RuntimeError("HPC_EVIDENCE: staged data hash differs from the local data contract")
        contract = {
            "schema": "evomind.data_contract.v1",
            "task_id": "titanic",
            "target": "Survived",
            "metric": "accuracy",
            "train_rows": len(train),
            "test_rows": len(test),
            "train_columns": list(train.columns),
            "test_columns": list(test.columns),
            "files": {
                "train.csv": _sha256(train_path),
                "test.csv": _sha256(test_path),
                "sample_submission.csv": _sha256(sample_path),
            },
            "combined_data_hash": stage["data_hash"],
        }
        contract_path = self.run_dir / "data_contract.json"
        _atomic_json(contract_path, contract)
        setup_path = self.run_dir / "setup.json"
        _atomic_json(setup_path, {
            "schema": "evomind.setup.v1",
            "status": "passed",
            "local_gpu_used": False,
            "remote_root": self.runtime.remote_run_dir,
            "hpc_probe_ref": "hpc_probe.json",
            "data_contract_ref": "data_contract.json",
            "generated_at": _now(),
        })
        artifacts = [_artifact(path, self.run_dir, kind=kind) for path, kind in (
            (initial_probe_path, "hpc_probe_initial"), (probe_path, "hpc_probe"),
            (runtime_path, "hpc_runtime_environment"), (contract_path, "data_contract"), (setup_path, "setup"),
        )]
        return AgentResult(task.task_id, "HPC and Titanic data contract verified", [item["path"] for item in artifacts], artifacts=artifacts, confidence=1.0)

    def research(self, task, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        memory_path = self.workspace_root / "experiments" / "evolution" / "retrospective_memory.json"
        memory_count = 0
        if memory_path.is_file():
            try:
                payload = json.loads(memory_path.read_text(encoding="utf-8"))
                memory_count = len(payload) if isinstance(payload, list) else 0
            except json.JSONDecodeError:
                memory_count = 0
        output = self.run_dir / "research_context.json"
        _atomic_json(output, {
            "schema": "evomind.research_context.v1",
            "task": "Titanic binary classification",
            "validated_memory_records_available": memory_count,
            "candidate_hypotheses": [strategy["hypothesis"] for strategy in _STRATEGIES.values()],
            "evidence_boundary": "Candidate hypotheses are tested by fresh 5-fold CV; no Kaggle score is assumed.",
            "generated_at": _now(),
        })
        artifact = _artifact(output, self.run_dir, kind="research_context")
        return AgentResult(task.task_id, "Three falsifiable candidate families defined", [artifact["path"]], artifacts=[artifact], confidence=0.85)

    def data_audit(self, task, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        train = pd.read_csv(self.data_dir / "train.csv")
        test = pd.read_csv(self.data_dir / "test.csv")
        overlap = sorted(set(train.columns).intersection(test.columns))
        leakage_columns = [column for column in test.columns if column.lower() == "survived"]
        audit = {
            "schema": "evomind.data_audit.v1",
            "status": "passed" if not leakage_columns else "rejected",
            "train_rows": len(train),
            "test_rows": len(test),
            "target": "Survived",
            "target_distribution": {str(key): int(value) for key, value in train["Survived"].value_counts().sort_index().items()},
            "missing_train": {column: int(value) for column, value in train.isna().sum().items() if value},
            "missing_test": {column: int(value) for column, value in test.isna().sum().items() if value},
            "duplicate_train_rows": int(train.duplicated().sum()),
            "shared_feature_columns": overlap,
            "target_present_in_test": bool(leakage_columns),
            "passenger_id_unique_train": bool(train["PassengerId"].is_unique),
            "passenger_id_unique_test": bool(test["PassengerId"].is_unique),
            "generated_at": _now(),
        }
        output = self.run_dir / "data_audit.json"
        _atomic_json(output, audit)
        artifact = _artifact(output, self.run_dir, kind="data_audit")
        return AgentResult(task.task_id, "Data audit passed" if audit["status"] == "passed" else "Data audit rejected", [artifact["path"]], artifacts=[artifact], confidence=1.0, accepted=audit["status"] == "passed", failure_type="data_leakage" if leakage_columns else "")

    def design(self, task, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        index = int(task.payload["solution_index"])
        strategy = _STRATEGIES[index]
        solution_dir = self.run_dir / "solutions" / task.solution_id
        output = solution_dir / "design.json"
        _atomic_json(output, {
            "schema": "evomind.solution_design.v1",
            "solution_id": task.solution_id,
            **strategy,
            "cv": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 20260721},
            "metric": "accuracy",
            "acceptance": ["891 unique OOF rows", "418-row candidate submission", "no target in test features", "raw fold metrics"],
            "stop_conditions": ["20 minute wall budget", "CUDA unavailable for GPU candidate", "artifact contract failure"],
        })
        artifact = _artifact(output, self.run_dir, kind="solution_design")
        return AgentResult(task.task_id, strategy["hypothesis"], [artifact["path"]], artifacts=[artifact], confidence=0.8)

    def code(self, task, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        index = int(task.payload["solution_index"])
        strategy = _STRATEGIES[index]
        solution_dir = self.run_dir / "solutions" / task.solution_id
        solution_dir.mkdir(parents=True, exist_ok=True)
        source_path = solution_dir / "solution.py"
        source_path.write_text(
            _solution_source(strategy["name"], run_id=_run.run_id, data_hash=self.data_hash),
            encoding="utf-8",
            newline="\n",
        )
        py_compile.compile(str(source_path), doraise=True)
        manifest_path = solution_dir / "code_manifest.json"
        _atomic_json(manifest_path, {
            "schema": "evomind.code_manifest.v1",
            "solution_id": task.solution_id,
            "source_sha256": _sha256(source_path),
            "static_check": "py_compile_passed",
            "local_gpu_used": False,
            "generated_at": _now(),
        })
        artifacts = [_artifact(source_path, self.run_dir, kind="source"), _artifact(manifest_path, self.run_dir, kind="code_manifest")]
        return AgentResult(task.task_id, "Solution code generated and statically validated", [item["path"] for item in artifacts], artifacts=artifacts, confidence=0.95)

    def tune(self, task, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        source_path = self.run_dir / "solutions" / task.solution_id / "solution.py"
        job = self.runtime.execute_solution(solution_id=task.solution_id, script_path=source_path, data_dir=self.data_dir)
        job_path = self.run_dir / "solutions" / task.solution_id / "hpc_job.json"
        _atomic_json(job_path, asdict(job))
        artifacts = []
        for item in job.local_artifacts:
            path = Path(item["path"])
            if path.is_file():
                artifacts.append(_artifact(path, self.run_dir, kind=path.name))
        artifacts.append(_artifact(job_path, self.run_dir, kind="hpc_job"))
        if job.status != "completed":
            raise RuntimeError(f"HPC_{job.failure_type.upper()}: {job.error}")
        metrics_path = self.run_dir / "solutions" / task.solution_id / "output" / "metrics.json"
        metrics = _read_json(metrics_path)
        return AgentResult(
            task.task_id,
            f"Fresh HPC 5-fold CV completed: accuracy={metrics['cv_score']:.6f}",
            [item["path"] for item in artifacts],
            metrics={"cv_score": metrics["cv_score"], "fold_scores": metrics["fold_scores"], "uses_cuda": metrics["uses_cuda"]},
            artifacts=artifacts,
            confidence=0.95,
        )

    def review(self, task, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        checks: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        for index in range(1, self.request.budget.solution_repositories + 1):
            solution_id = f"solution_{index:02d}"
            strategy = _STRATEGIES[index]
            solution_checks, candidate = review_solution_evidence(
                run_dir=self.run_dir,
                data_dir=self.data_dir,
                run_id=_run.run_id,
                solution_id=solution_id,
                expected_variant=strategy["name"],
                expected_cuda=bool(strategy["uses_gpu"]),
                data_hash=self.data_hash,
                remote_run_dir=self.runtime.remote_run_dir,
            )
            checks.append(solution_checks)
            if candidate is not None:
                candidates.append(candidate)
        passed = len(candidates) == self.request.budget.solution_repositories and all(check.get("passed") for check in checks)
        review = {
            "schema": "evomind.independent_review.v1",
            "status": "passed" if passed else "rejected",
            "review_input": "raw_metrics_oof_submission_environment_logs",
            "parent_subjective_summary_received": False,
            "checks": checks,
            "accepted_candidates": candidates,
            "claim_audit": {
                "status": "passed" if passed else "rejected",
                "official_kaggle_score_claimed": False,
                "fresh_run_only": True,
            },
            "generated_at": _now(),
        }
        output = self.run_dir / "review.json"
        _atomic_json(output, review)
        artifact = _artifact(output, self.run_dir, kind="independent_review")
        return AgentResult(
            task.task_id,
            "All candidate evidence contracts passed" if passed else "Reviewer rejected incomplete or inconsistent evidence",
            [artifact["path"]],
            metrics={"accepted_candidates": len(candidates)},
            artifacts=[artifact],
            confidence=1.0,
            accepted=passed,
            failure_type="review_rejected" if not passed else "",
        )

    def aggregate(self, task, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        review = _read_json(self.run_dir / "review.json")
        if review.get("status") != "passed":
            raise RuntimeError("review gate is not passed")
        candidates = list(review["accepted_candidates"])
        selected = max(candidates, key=lambda item: item["cv_score"])
        selected_output = self.run_dir / "solutions" / selected["solution_id"] / "output"
        shutil.copy2(selected_output / "submission.csv", self.run_dir / "submission.csv")
        aggregate_metrics = {
            "schema": "evomind.aggregate_metrics.v1",
            "selected_solution": selected["solution_id"],
            "metric": "accuracy",
            "cv_score": selected["cv_score"],
            "official_kaggle_score": None,
            "candidates": candidates,
            "review_status": "passed",
        }
        _atomic_json(self.run_dir / "metrics.json", aggregate_metrics)
        report_path = self.run_dir / "research_report.md"
        rows = "\n".join(f"| {item['solution_id']} | {item['cv_score']:.6f} | {'yes' if item['uses_cuda'] else 'no'} |" for item in candidates)
        report_path.write_text(
            "# EvoMind Titanic 小型模型开发报告\n\n"
            f"- Run ID: `{_run.run_id}`\n"
            "- 数据：891 行训练集、418 行测试集\n"
            "- 验证：固定随机种子 20260721 的 5 折 Stratified CV\n"
            "- 本地 GPU：未使用\n"
            "- Kaggle 正式提交：未执行，Human Gate 保持阻断\n\n"
            "## 候选对比\n\n| Solution | CV accuracy | CUDA telemetry |\n|---|---:|---|\n"
            f"{rows}\n\n"
            f"## 结论\n\n独立 Reviewer 通过后选择 `{selected['solution_id']}`，本地 CV accuracy 为 "
            f"`{selected['cv_score']:.6f}`。该数值不是 Kaggle 官方成绩。候选 `submission.csv` 已生成。\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest_entries = _manifest_entries(self.run_dir)
        manifest_path = self.run_dir / "artifact_manifest.json"
        _atomic_json(manifest_path, {
            "schema": "evomind.artifact_manifest.v1",
            "run_id": _run.run_id,
            "selected_solution": selected["solution_id"],
            "artifacts": manifest_entries,
            "official_submission": "blocked",
            "generated_at": _now(),
        })
        artifacts = [_artifact(path, self.run_dir, kind=kind) for path, kind in (
            (self.run_dir / "submission.csv", "candidate_submission"),
            (self.run_dir / "metrics.json", "aggregate_metrics"),
            (self.run_dir / "review.json", "independent_review"),
            (manifest_path, "artifact_manifest"),
            (report_path, "research_report"),
        )]
        return AgentResult(task.task_id, f"Selected {selected['solution_id']} after independent review", [item["path"] for item in artifacts], metrics=aggregate_metrics, artifacts=artifacts, confidence=1.0)

    def mapping(self):
        return {
            "SetupAgent": self.setup,
            "ResearchLead": self.research,
            "DataAuditor": self.data_audit,
            "DesignerAgent": self.design,
            "CoderAgent": self.code,
            "TunerAgent": self.tune,
            "IndependentReviewer": self.review,
            "Aggregator": self.aggregate,
        }


def run_titanic_aibuild(workspace_root: str | Path, request: UserRequest, *, run_id: str | None = None) -> SupervisorRun:
    root = Path(workspace_root).resolve()
    if request.dataset not in {None, "titanic"}:
        raise ValueError("Titanic proof workflow received a different dataset")
    if not request.requests_execution:
        raise ValueError("Titanic proof workflow requires an explicit execution request")
    if request.compute_policy.local_gpu_allowed:
        raise ValueError("local GPU must remain disabled")
    data_dir = root / "tasks" / "titanic" / "data"
    data_hash = _combined_data_hash(data_dir)
    run = build_aibuild_run(request, task_id="titanic", run_id=run_id)
    request_hash = _sha256_text(json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True))
    for task in run.tasks.values():
        task.payload["data_hash"] = data_hash
        task.payload["config_hash"] = _sha256_text(f"{request_hash}\0{task.task_id}\0{task.goal}")
        if task.solution_id:
            strategy = _STRATEGIES[int(task.solution_id.rsplit("_", 1)[1])]
            task.payload["code_hash"] = _sha256_text(
                _solution_source(strategy["name"], run_id=run.run_id, data_hash=data_hash)
            )
    local_run_dir = run_directory(root, run.run_id)
    store = MultiAgentStore(local_run_dir)
    runtime = HpcRuntime(run_id=run.run_id, local_run_dir=local_run_dir, timeout_seconds=request.budget.max_minutes * 60)
    executors = TitanicExecutors(workspace_root=root, run_dir=local_run_dir, request=request, runtime=runtime)
    store.append_message(run, sender="user", receiver="ExecutiveSupervisor", content=request.objective)
    write_current_run_pointer(root, task_id="titanic", run=run, run_dir=local_run_dir)
    supervisor = MultiAgentSupervisor(run, store, executors.mapping())
    try:
        result = supervisor.run_until_blocked()
    finally:
        write_current_run_pointer(root, task_id="titanic", run=run, run_dir=local_run_dir)
    return result


def resume_titanic_aibuild(workspace_root: str | Path, run_id: str) -> SupervisorRun:
    from xsci.user_request import parse_user_request

    root = Path(workspace_root).resolve()
    local_run_dir = run_directory(root, run_id)
    store = MultiAgentStore(local_run_dir)
    run = store.load()
    request = parse_user_request(run.objective)
    runtime = HpcRuntime(run_id=run.run_id, local_run_dir=local_run_dir, timeout_seconds=request.budget.max_minutes * 60)
    executors = TitanicExecutors(workspace_root=root, run_dir=local_run_dir, request=request, runtime=runtime)
    supervisor = MultiAgentSupervisor(run, store, executors.mapping())
    supervisor.resume(retry_failed=run.status == "needs_continuation")
    try:
        result = supervisor.run_until_blocked()
    finally:
        write_current_run_pointer(root, task_id="titanic", run=run, run_dir=local_run_dir)
    return result


__all__ = [
    "TitanicExecutors",
    "resume_titanic_aibuild",
    "review_solution_evidence",
    "run_titanic_aibuild",
]
