"""Bounded reconstruction, independent model reload, and public Ariel OOF audit.

This is retrospective engineering reproduction, not a new holdout experiment.
The final checkpoint is loaded, never refitted. Five outer models are rebuilt
with their already frozen parameters; there is no search or promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# These limits affect this process and its own verifier children only.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "2"

ROOT = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
DATA = ROOT / "competition_data/ariel_2025"
DATA_SHA = "4437e130014a305112fb7e7b2ad825d453424f1eaec64eb86cd2cdd4e50aaf71"
SEED = 93207
EXPECTED_SCORE = 0.15922198685508845
ATOL = 1e-10
RTOL = 1e-7
FROZEN = {
    "ariel_2025_nested_gll.py": "fe3869a96267e3d12dfd36ebc99043cffedd42ef497c3795d8083662b79b6858",
    "model-bundle.joblib": "7d1380710260008bd42addf668b01336e6227ff047e2780ae8aff7470377803e",
    "oof-predictions.npz": "45d8f71a1d6cc5865881622512b53c956cdf4f645e7c7cdd47adeedf98102ead",
    "training-config.json": "79be053b4bd14724ec1ad286f3061cfec29ed23c807b1f2999dcfdba0dfa56c7",
    "metrics.json": "9a9dc4fb46e99f689ecc900fa27e7e7d71923ac3218e9f576a2b6431acc8ed7c",
    "artifact-manifest.json": "0b416e32be341b0d47596adc23e3821ade2c71048a4328dfaf5c7d6eaef5132e",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("symlink_rejected")
    resolved = path.resolve(strict=True)
    resolved.relative_to(ROOT.resolve(strict=True))
    return resolved


def ids_sha(values) -> str:
    return hashlib.sha256(("\n".join(sorted(map(str, values))) + "\n").encode()).hexdigest()


def ordered_sha(values) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def save_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Progress:
    def __init__(self, root: Path):
        self.root = root
        self.state = {"phase": "preparing", "work_kind": "preparing", "completed_units": None, "total_units": None}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.beat, daemon=True)

    def emit(self, event: str, **fields):
        with self.lock:
            row = {"schema": "evomind.ariel.replay_progress.v1", "at_utc": datetime.now(timezone.utc).isoformat(),
                   "pid": os.getpid(), "event": event, **self.state, **fields}
            with (self.root / "execution-events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)

    def stage(self, phase: str, kind: str, completed=None, total=None):
        with self.lock:
            self.state = {"phase": phase, "work_kind": kind, "completed_units": completed, "total_units": total}
        self.emit("stage_changed")

    def beat(self):
        while not self.stop.wait(15):
            self.emit("heartbeat")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


def reference_gll(y, mu, sigma):
    """Independent scipy implementation of the public element-wise GLL rule.

    Public reference: ibrahimhabibeg/ariel-2025, revision
    df96b8d6e941cee5c7bd4bc72dd6915c65352526, ariel_pred/metrics.py.
    This is not a hash-bound organizer evaluator or an official final score.
    """
    import numpy as np
    from scipy.stats import norm
    if y.shape != mu.shape or y.shape != sigma.shape or y.shape[1] != 283:
        raise ValueError("scorer_shape_mismatch")
    if not all(np.isfinite(value).all() for value in (y, mu, sigma)) or np.any(mu < 0) or np.any(sigma < 0):
        raise ValueError("scorer_input_invalid")
    mean, std = float(y.mean()), float(y.std())
    if std <= 0:
        raise ValueError("degenerate_reference")
    ideal_sigma = np.r_[1e-6, np.full(282, 1e-5)]
    predicted = norm.logpdf(y, loc=mu, scale=np.maximum(sigma, 1e-15))
    ideal = norm.logpdf(y, loc=y, scale=ideal_sigma)
    reference = norm.logpdf(y, loc=mean, scale=std)
    denominator = ideal - reference
    if not np.isfinite(denominator).all() or np.any(abs(denominator) < 1e-15):
        raise ValueError("invalid_normalizer")
    contributions = np.average((predicted-reference)/denominator, axis=1, weights=np.r_[57.846, np.ones(282)])
    return float(np.clip(contributions.mean(), 0, 1)), contributions


def export_portable(bundle, destination: Path):
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipeline = bundle["pipeline"]
    if type(pipeline) is not Pipeline or list(pipeline.named_steps) != ["imputer", "scaler", "ridge"]:
        raise ValueError("unsupported_pipeline")
    imputer, scaler, ridge = (pipeline.named_steps[name] for name in ("imputer", "scaler", "ridge"))
    if (type(imputer), type(scaler), type(ridge)) != (SimpleImputer, StandardScaler, Ridge):
        raise ValueError("unsupported_estimator")
    if imputer.strategy != "median" or not imputer.add_indicator or not scaler.with_mean or not scaler.with_std:
        raise ValueError("unsupported_preprocessing")
    if getattr(imputer, "keep_empty_features", False):
        raise ValueError("unsupported_empty_feature_policy")
    np.savez_compressed(destination, schema=np.asarray("evomind.ariel.portable_ridge.v1"),
                        imputer_statistics=imputer.statistics_, imputer_keep=~np.isnan(imputer.statistics_),
                        imputer_indicators=imputer.indicator_.features_, scaler_mean=scaler.mean_,
                        scaler_scale=scaler.scale_, ridge_coef=ridge.coef_, ridge_intercept=ridge.intercept_,
                        sigma=np.asarray(bundle["sigma"]), feature_names=np.asarray(bundle["feature_names"], dtype=str),
                        target_columns=np.asarray(bundle["target_columns"], dtype=str))


def child_reload(input_root: Path, output: Path, manifest: Path):
    import joblib
    import numpy as np
    request = json.loads(checked(manifest).read_text(encoding="utf-8"))
    features = checked(output / "development-features.npz")
    if sha(features) != request["features_sha256"]:
        raise ValueError("child_feature_hash_mismatch")
    with np.load(features, allow_pickle=False) as archive:
        x, groups, feature_names = archive["x"], archive["planet_id"], archive["feature_names"]
    receipts = []
    for job in request["models"]:
        path = checked(Path(job["path"]))
        if sha(path) != job["sha256"]:
            raise ValueError("child_model_hash_mismatch")
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            bundle = joblib.load(path)
        if not np.array_equal(np.asarray(bundle["feature_names"], dtype=str), feature_names):
            raise ValueError("child_feature_order_mismatch")
        index = np.asarray(job["indices"], dtype=int)
        mu = np.maximum(bundle["pipeline"].predict(x[index]), 1e-15)
        sigma = np.broadcast_to(bundle["sigma"], mu.shape).copy()
        if mu.shape != (len(index), 283) or not np.isfinite(mu).all() or not np.isfinite(sigma).all():
            raise ValueError("child_prediction_invalid")
        destination = output / (job["name"] + "-predictions.npz")
        np.savez_compressed(destination, mu=mu, sigma=sigma, planet_id=groups[index], indices=index)
        if job["name"] == "final":
            export_portable(bundle, output / "portable-model.npz")
            save_json(output / "frozen-model-identity.json", {
                key: bundle[key] for key in ("development_id_sha256", "parent_holdout_id_sha256", "source_sha256",
                                            "selected_alpha", "selected_sigma_scale", "feature_names", "target_columns")})
        receipts.append({"name": job["name"], "model_sha256": job["sha256"],
                         "prediction_sha256": sha(destination), "rows": len(index),
                         "warnings": sorted({type(notice.message).__name__ for notice in notices}),
                         "training_performed": False, "pid": os.getpid()})
    save_json(output / "independent-reload-receipt.json", {"pid": os.getpid(), "parent_pid": os.getppid(), "models": receipts})


def run_child(command, output: Path, name: str):
    # The managed executor owns the overall 1200-second lifetime and settlement.
    with (output / (name + ".log")).open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(name + "_failed_exit_" + str(result.returncode))


def execute(input_root: Path, output: Path, progress: Progress):
    import joblib
    import numpy as np
    import pandas as pd
    import pyarrow
    import scipy
    import sklearn
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    for name, digest in FROZEN.items():
        if sha(checked(input_root/name)) != digest:
            raise ValueError("frozen_artifact_hash_mismatch_" + name)
    protocol = json.loads((input_root / "replay-protocol.json").read_text(encoding="utf-8"))
    for name, digest in protocol["program_sha256"].items():
        if sha(checked(input_root/name)) != digest:
            raise ValueError("program_hash_mismatch")
    # The managed executor stages this entrypoint as train_gpu.py.
    # Bind its bytes to the frozen logical entrypoint, not the staging name.
    if sha(Path(__file__)) != protocol["program_sha256"]["verify_ariel_independent_replay_v1.py"]:
        raise ValueError("executor_program_hash_mismatch")
    data_manifest = checked(DATA / ".evomind/data-manifest.json")
    if sha(data_manifest) != DATA_SHA:
        raise ValueError("persistent_dataset_manifest_mismatch")
    manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
    inventory = {item["path"].replace("\\", "/"): item for item in manifest["files"]}
    csv_hashes = {}
    for relative in ("data/train.csv", "data/train_star_info.csv"):
        path = checked(DATA / relative)
        digest = sha(path)
        if relative not in inventory or digest != inventory[relative]["sha256"]:
            raise ValueError("csv_manifest_hash_mismatch")
        csv_hashes[relative] = digest
    old_metrics = json.loads((input_root / "metrics.json").read_text())
    old_config = json.loads((input_root / "training-config.json").read_text())
    if len(old_metrics["folds"]) != 5 or old_config["selected_alpha"] != 0.1 or old_config["selected_sigma_scale"] != 1.0:
        raise ValueError("frozen_parameter_contract_changed")
    if any(row["alpha"] != 0.1 or row["sigma_scale"] != 1.0 for row in old_metrics["folds"]):
        raise ValueError("frozen_fold_parameter_contract_changed")

    progress.stage("reconstructing_public_split", "preparing")
    ids_frame = pd.read_csv(DATA/"data/train.csv", usecols=["planet_id"])
    all_ids = ids_frame["planet_id"].astype(str).to_numpy()
    if len(all_ids) != 1100 or len(set(all_ids)) != 1100:
        raise ValueError("planet_id_contract_mismatch")
    development, excluded = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(all_ids, groups=all_ids))
    allowed_rows = set(map(int, development))
    # Excluded rows are never materialized in a label DataFrame.
    labels = pd.read_csv(DATA/"data/train.csv", skiprows=lambda row: row > 0 and row-1 not in allowed_rows)
    target_columns = [name for name in labels if name.startswith("wl_")]
    stars = pd.read_csv(DATA/"data/train_star_info.csv")
    stars = stars.rename(columns={name: "star__"+name for name in stars if name != "planet_id"})
    frame = labels.merge(stars, on="planet_id", validate="one_to_one").sort_values("planet_id").reset_index(drop=True)
    groups = frame["planet_id"].astype(str).to_numpy(dtype=str)
    if len(frame) != 880 or len(excluded) != 220 or len(target_columns) != 283 or set(groups) & set(all_ids[excluded]):
        raise ValueError("excluded_split_contract_failed")
    spec = importlib.util.spec_from_file_location("frozen_ariel_features", input_root/"ariel_2025_nested_gll.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    progress.stage("rebuilding_real_parquet_features", "preparing", 0, 880)
    x, feature_names, feature_audit = legacy.build_features(DATA, frame, output/"feature-events.jsonl")
    y = frame[target_columns].to_numpy(dtype=float)
    if x.shape != (880, 80) or y.shape != (880, 283) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("real_feature_schema_invalid")
    if feature_audit["errors"] != 0 or feature_audit["files"] != 3520:
        raise ValueError("missing_or_invalid_parquet_inputs")
    if feature_names != old_config["feature_names"]:
        raise ValueError("frozen_feature_order_changed")
    np.savez_compressed(output/"development-features.npz", x=x, planet_id=groups,
                        feature_names=np.asarray(feature_names, dtype=str))
    np.savez_compressed(output/"development-labels.npz", y=y, planet_id=groups,
                        target_columns=np.asarray(target_columns, dtype=str))
    split_identity = {"development_id_sha256": ids_sha(groups), "parent_holdout_id_sha256": ids_sha(all_ids[excluded]),
                      "ordered_development_id_sha256": ordered_sha(groups), "development_rows": 880,
                      "excluded_rows": 220, "excluded_labels_materialized": False, "csv_sha256": csv_hashes}
    save_json(output/"split-identity.json", split_identity)
    save_json(output/"environment.json", {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
              "pyarrow": pyarrow.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
              "joblib": joblib.__version__, "fresh_dependency_install": False})

    models = [{"name": "final", "path": str(input_root/"model-bundle.joblib"),
               "sha256": FROZEN["model-bundle.joblib"], "indices": list(range(880))}]
    fold_lineage = []
    fits = 0
    def make_model():
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.1))])
    for fold, (train, valid) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=SEED+1).split(x, y, groups)):
        progress.stage("reconstructing_frozen_fold_"+str(fold), "training", fits, 20)
        inner_mu = np.zeros_like(y[train])
        inner_seen = np.zeros(len(train), dtype=int)
        inner = GroupKFold(n_splits=3, shuffle=True, random_state=SEED+100+fold)
        for inner_train, inner_valid in inner.split(x[train], y[train], groups[train]):
            estimator = make_model()
            estimator.fit(x[train[inner_train]], y[train[inner_train]])
            fits += 1
            inner_mu[inner_valid] = np.maximum(estimator.predict(x[train[inner_valid]]), 1e-15)
            inner_seen[inner_valid] += 1
            progress.emit("fit_completed", completed_units=fits, total_units=20, fold=fold)
        if not np.all(inner_seen == 1):
            raise ValueError("inner_coverage_invalid")
        sigma = np.maximum(np.sqrt(np.mean((y[train]-inner_mu)**2, axis=0)), np.r_[1e-6, np.full(282, 1e-5)])
        estimator = make_model()
        estimator.fit(x[train], y[train])
        fits += 1
        destination = output / ("outer-fold-"+str(fold)+".joblib")
        joblib.dump({"pipeline": estimator, "sigma": sigma, "feature_names": feature_names,
                     "target_columns": target_columns}, destination, compress=3)
        models.append({"name": "fold-"+str(fold), "path": str(destination), "sha256": sha(destination),
                       "indices": valid.tolist()})
        fold_lineage.append({"fold": fold, "train_id_sha256": ids_sha(groups[train]),
                             "valid_id_sha256": ids_sha(groups[valid]), "train_rows": len(train),
                             "valid_rows": len(valid), "alpha": 0.1, "sigma_scale": 1.0,
                             "model_sha256": sha(destination), "model_path": destination.name})
        progress.emit("fit_completed", completed_units=fits, total_units=20, fold=fold)
    save_json(output/"fold-lineage.json", fold_lineage)
    request_path = output/"reload-request.json"
    save_json(request_path, {"models": models, "features_sha256": sha(output/"development-features.npz")})
    progress.stage("independent_checkpoint_reload", "inference", 0, 6)
    run_child([sys.executable, str(Path(__file__)), "--data-dir", str(input_root), "--out-dir", str(output),
               "--reload-manifest", str(request_path)], output, "independent-reload")
    identity = json.loads((output/"frozen-model-identity.json").read_text())
    for key in ("development_id_sha256", "parent_holdout_id_sha256"):
        if identity[key] != split_identity[key]:
            raise ValueError("frozen_model_split_identity_mismatch")
    if identity["source_sha256"] != FROZEN["ariel_2025_nested_gll.py"] or identity["target_columns"] != target_columns:
        raise ValueError("frozen_model_source_or_targets_mismatch")
    shutil.copyfile(input_root/"ariel_portable_predict_v1.py", output/"ariel_portable_predict_v1.py")
    run_child([sys.executable, str(output/"ariel_portable_predict_v1.py"), "--model", str(output/"portable-model.npz"),
               "--features", str(output/"development-features.npz"), "--output", str(output/"portable-predictions.npz")],
              output, "portable-inference")
    with np.load(output/"final-predictions.npz", allow_pickle=False) as reference, np.load(output/"portable-predictions.npz", allow_pickle=False) as portable:
        portable_delta = float(np.max(abs(reference["mu"]-portable["mu"])))
        portable_ok = bool(np.allclose(reference["mu"], portable["mu"], atol=ATOL, rtol=RTOL)
                           and np.array_equal(reference["sigma"], portable["sigma"])
                           and np.array_equal(reference["planet_id"], portable["planet_id"]))
    progress.stage("independent_oof_scoring", "evaluation", 0, 880)
    rebuilt_mu, rebuilt_sigma = np.zeros_like(y), np.zeros_like(y)
    covered = np.zeros(len(y), dtype=int)
    for fold in range(5):
        with np.load(output/("fold-"+str(fold)+"-predictions.npz"), allow_pickle=False) as prediction:
            index = prediction["indices"]
            if not np.array_equal(groups[index], prediction["planet_id"]):
                raise ValueError("independent_fold_id_mismatch")
            rebuilt_mu[index], rebuilt_sigma[index] = prediction["mu"], prediction["sigma"]
            covered[index] += 1
    if not np.all(covered == 1):
        raise ValueError("independent_outer_coverage_invalid")
    with np.load(input_root/"oof-predictions.npz", allow_pickle=False) as old:
        # Never unpickle the legacy object-dtype planet_id array.
        old_mu, old_sigma = old["mu"], old["sigma"]
        if not np.array_equal(old["target_columns"].astype(str), np.asarray(target_columns)):
            raise ValueError("legacy_prediction_target_order_mismatch")
    old_score, contributions = reference_gll(y, old_mu, old_sigma)
    rebuilt_score, _ = reference_gll(y, rebuilt_mu, rebuilt_sigma)
    reference_anchor, _ = reference_gll(y, np.full_like(y, y.mean()), np.full_like(y, y.std()))
    ideal_anchor, _ = reference_gll(y, y, np.broadcast_to(np.r_[1e-6, np.full(282, 1e-5)], y.shape))
    rng = np.random.default_rng(SEED+900)
    bootstrap = [float(np.clip(contributions[rng.integers(0, len(y), len(y))].mean(), 0, 1)) for _ in range(20000)]
    interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    reconstruction_ok = bool(np.allclose(old_mu, rebuilt_mu, atol=ATOL, rtol=RTOL)
                             and np.allclose(old_sigma, rebuilt_sigma, atol=ATOL, rtol=RTOL))
    scorer_ok = abs(old_score-EXPECTED_SCORE) <= 1e-12 and abs(old_score-old_metrics["nested_oof_gll"]) <= 1e-12
    anchors_ok = abs(reference_anchor) <= 1e-12 and abs(ideal_anchor-1) <= 1e-12
    np.savez_compressed(output/"independent-oof-predictions.npz", planet_id=groups, mu=rebuilt_mu,
                        sigma=rebuilt_sigma, target_columns=np.asarray(target_columns, dtype=str))
    shutil.copyfile(input_root/"model-bundle.joblib", output/"frozen-final-model.joblib")
    shutil.copyfile(input_root/"ariel_2025_nested_gll.py", output/"frozen-feature-source.py")
    shutil.copyfile(input_root/"replay-protocol.json", output/"replay-protocol.json")
    result = {"schema": "evomind.ariel.independent_replay.v1", "status": "PASS" if all((portable_ok, reconstruction_ok, scorer_ok, anchors_ok)) else "FAIL",
              "research_verdict": "HOLD", "paper_baseline_comparison": "NOT_COMPARABLE",
              "execution_kind": "frozen_parameter_reconstruction_and_independent_replay", "actual_fit_calls": fits,
              "new_hyperparameter_search": False, "new_independent_holdout": False,
              "original_final_checkpoint_refitted": False, "final_checkpoint_loaded_in_independent_process": True,
              "fold_checkpoints_independently_loaded": 5, "development_rows": 880, "excluded_rows": 220,
              "excluded_labels_materialized": False, "legacy_object_ids_unpickled": False,
              "legacy_row_order_source": "hash_frozen_source_sort_values_planet_id_and_reconstructed_fold_predictions",
              "historical_oof_gll_recomputed": old_score, "reconstructed_oof_gll": rebuilt_score, "descriptive_ci95": interval,
              "reference_zero_anchor": reference_anchor, "ideal_one_anchor": ideal_anchor,
              "gll_recalculation_passed": scorer_ok, "frozen_fold_reconstruction_passed": reconstruction_ok,
              "oof_mu_max_abs_difference": float(np.max(abs(old_mu-rebuilt_mu))),
              "oof_sigma_max_abs_difference": float(np.max(abs(old_sigma-rebuilt_sigma))),
              "portable_prediction_parity_passed": portable_ok, "portable_prediction_max_abs_difference": portable_delta,
              "absolute_tolerance": ATOL, "relative_tolerance": RTOL, "dataset_manifest_sha256": DATA_SHA,
              "full_264gb_rehash_performed": False, "feature_audit": feature_audit, "frozen_artifact_sha256": FROZEN,
              "scorer_provenance": {"kind": "independent_public_formula_implementation", "official_evaluator_bound": False,
                "public_reference": "https://github.com/ibrahimhabibeg/ariel-2025/blob/df96b8d6e941cee5c7bd4bc72dd6915c65352526/ariel_pred/metrics.py"},
              "limitations": ["Retrospective public-development reproduction, not independent research gain.",
                              "2025 R2 paper baseline and organizer scorer identity remain unbound.",
                              "Ariel 2024 paper validation 0.5020 is not a comparable threshold.",
                              "Final-model development predictions are in-sample and not scored.",
                              "Environment receipt describes the existing runtime, not a new preinstall hash lock."],
              "signals_sent": 0, "other_processes_modified": False, "submission_performed": False}
    progress.stage("replay_finished", "evaluation", 880, 880)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reload-manifest", type=Path)
    args = parser.parse_args()
    input_root = checked(args.data_dir)
    output = args.out_dir.resolve()
    output.relative_to(ROOT.resolve(strict=True))
    output.mkdir(parents=True, exist_ok=True)
    if args.reload_manifest:
        child_reload(input_root, output, args.reload_manifest)
        return 0
    # Never overwrite a previously settled execution receipt.
    if (output/"verification.json").exists():
        raise FileExistsError("replay_already_settled")
    started = time.monotonic()
    progress = Progress(output)
    progress.thread.start()
    try:
        progress.emit("started", source_sha256=sha(Path(__file__)))
        result = execute(input_root, output, progress)
    except Exception as error:
        result = {"schema": "evomind.ariel.independent_replay.v1", "status": "FAIL", "research_verdict": "HOLD",
                  "error_class": type(error).__name__, "error": str(error)[:400], "partial_outputs_preserved": True,
                  "paper_baseline_comparison": "NOT_COMPARABLE", "submission_performed": False}
        progress.emit("failed", error_class=type(error).__name__)
    finally:
        progress.close()
    result["elapsed_seconds"] = time.monotonic()-started
    result["verifier_source_sha256"] = sha(Path(__file__))
    save_json(output/"verification.json", result)
    shutil.copyfile(Path(__file__), output/"verify_ariel_independent_replay_v1.py")
    (output/"REUSE.md").write_text(
        "# Ariel frozen-model reproduction\n\n"
        "This package is engineering evidence, not an official competition result or new independent research gain.\n\n"
        "Run inference without pickle:\n\n"
        "```sh\npython ariel_portable_predict_v1.py --model portable-model.npz --features development-features.npz --output fresh-predictions.npz\n```\n\n"
        "Input NPZ keys: x (N x 80), feature_names (exact frozen order), planet_id (Unicode strings). "
        "The frozen-feature-source.py module provides build_features; do not invoke its training main. "
        "development-labels.npz contains only the 880 previously consumed public development rows. "
        "No labels from the 220 excluded rows are included.\n\n"
        "verification.json is a verifier receipt; independent process logs, checkpoints, predictions, hashes, and "
        "recomputed metrics are the evidence. The 2024 paper score 0.5020 must not be compared with this 2025 R2 result.\n",
        encoding="utf-8")
    files = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
             for path in sorted(output.iterdir()) if path.is_file() and path.name != "replay-manifest.json"]
    save_json(output/"replay-manifest.json", {"schema": "evomind.ariel.replay_manifest.v1", "files": files,
              "status": result["status"], "research_verdict": "HOLD"})
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
