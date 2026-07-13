#!/usr/bin/env python3
"""Workstation GPU Batch Trainer v1 — runs tabular Kaggle competitions on GPU.
Target: 75 MLE-Bench competitions, starting with top priority tasks.
All input and output stay under the explicitly configured remote workspace.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Force unbuffered output for real-time log monitoring
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, "reconfigure") else None
# catboost is imported lazily inside train_single so this module stays importable
# (e.g. for unit-testing the pure helpers) on machines without the GPU package.

BASE_DIR: Path | None = None
DATA_DIR: Path | None = None
FALLBACK_DATA_DIR: Path | None = None
RESULTS_DIR: Path | None = None


def configure_runtime_paths(remote_workspace: str = "") -> None:
    """Resolve remote paths at execution time and avoid import-time filesystem writes."""
    value = remote_workspace.strip() or os.environ.get("EVOMIND_HPC_REMOTE_WORKSPACE", "").strip()
    if not value:
        raise RuntimeError("EVOMIND_HPC_REMOTE_WORKSPACE must be configured explicitly")
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise RuntimeError("EVOMIND_HPC_REMOTE_WORKSPACE must be an absolute path")
    data_dir = root / "data"
    fallback_data_dir = root / "mlebench_prepared"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    global BASE_DIR, DATA_DIR, FALLBACK_DATA_DIR, RESULTS_DIR
    BASE_DIR = root
    DATA_DIR = data_dir
    FALLBACK_DATA_DIR = fallback_data_dir
    RESULTS_DIR = results_dir


def _runtime_paths() -> tuple[Path, Path, Path, Path]:
    if any(path is None for path in (BASE_DIR, DATA_DIR, FALLBACK_DATA_DIR, RESULTS_DIR)):
        configure_runtime_paths()
    if BASE_DIR is None or DATA_DIR is None or FALLBACK_DATA_DIR is None or RESULTS_DIR is None:
        raise RuntimeError("HPC runtime paths were not initialized")
    return BASE_DIR, DATA_DIR, FALLBACK_DATA_DIR, RESULTS_DIR

# ── P1: model_selection integration ──
# model_selection.py is pure stdlib. It is uploaded alongside this trainer so the
# recommendation drives every run. If it is somehow absent, we degrade gracefully
# to the proven CatBoost path (never crash a run over the selector).
try:
    from model_selection import (
        DataProfile,
        recommend_model_strategy,
        resolve_training_plan,
    )
    _MODEL_SELECTION_AVAILABLE = True
except Exception as _ms_err:  # pragma: no cover - environment fallback
    print(f"  [model_selection unavailable: {type(_ms_err).__name__}; using CatBoost default]")
    _MODEL_SELECTION_AVAILABLE = False


def _detect_available_models():
    """Return the GBDT families importable in this environment (for plan resolution)."""
    avail = []
    for name in ("catboost", "lightgbm", "xgboost"):
        try:
            __import__(name)
            avail.append(name)
        except Exception:
            pass
    return avail


def _detect_torch():
    """Check if torch is importable (needed for CNN/neural paths)."""
    try:
        import torch
        return torch.cuda.is_available() if torch else False
    except Exception:
        return False


def _neural_available():
    """Check if image_classifier module is importable and torch works."""
    try:
        from image_classifier import ImageClassifier, build_arch_config
        return _detect_torch(), ImageClassifier, build_arch_config
    except Exception:
        return False, None, None


# ── CNN imports (lazy, with fallback) ──
_NEURAL_OK, _ImageClassifier, _build_arch_config = _neural_available()
if not _NEURAL_OK:
    _ImageClassifier = None
    _build_arch_config = None


# ── P2: ensemble engine (multi-seed blending is the small-data gold lever) ──
# ensemble_engine.py is pure numpy+sklearn; uploaded beside this trainer. Falls
# back to a plain equal-weight mean if it is somehow absent (never blocks a run).
try:
    from ensemble_engine import blend as _ensemble_blend
    _ENSEMBLE_AVAILABLE = True
except Exception:  # pragma: no cover - environment fallback
    _ENSEMBLE_AVAILABLE = False

    def _ensemble_blend(model_outputs, weights=None):
        import numpy as _np
        return _np.mean(_np.stack([_np.asarray(a, dtype=float) for a in model_outputs], axis=0), axis=0)

# ── Competition Registry (32 competitions) ──
# type: classification|regression
# metric: accuracy|roc_auc|normalized_gini|rmse|rmsle|mae
# bronze: Kaggle bronze medal threshold (null if unknown)
COMPETITIONS = {
    # ── Existing (11) ──
    "spaceship-titanic": {
        "type": "classification", "metric": "accuracy",
        "target": "Transported", "id_col": "PassengerId",
        "drop_cols": ["PassengerId", "Name"],
        "higher_is_better": True, "bronze": 0.795,
    },
    "titanic": {
        "type": "classification", "metric": "accuracy",
        "target": "Survived", "id_col": "PassengerId",
        "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
        "higher_is_better": True, "bronze": 0.794,
    },
    "house_prices": {
        "type": "regression", "metric": "rmsle",
        "target": "SalePrice", "id_col": "Id",
        "drop_cols": ["Id"],
        "higher_is_better": False, "bronze": 0.140,
        "log_transform_target": True,
    },
    "bike-sharing-demand": {
        "type": "regression", "metric": "rmsle",
        "target": "count", "id_col": "datetime",
        "drop_cols": ["casual", "registered"],
        "higher_is_better": False, "bronze": 0.480,
        "log_transform_target": True,
    },
    "porto-seguro-safe-driver-prediction": {
        "type": "classification", "metric": "normalized_gini",
        "target": "target", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": 0.285,
    },
    "playground-series-s6e6": {
        "type": "classification", "metric": "accuracy",
        "target": "class", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": 0.400,
    },
    "store-sales-time-series-forecasting": {
        "type": "regression", "metric": "rmsle",
        "target": "sales", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": False, "bronze": 0.500,
        "log_transform_target": True,
    },
    "digit-recognizer": {
        "type": "classification", "metric": "accuracy",
        "target": "label", "id_col": None,
        "drop_cols": [],
        "higher_is_better": True, "bronze": 0.986,
        "pixel_data": True,
    },
    "tabular-playground-series-aug-2022": {
        "type": "classification", "metric": "roc_auc",
        "target": "failure", "id_col": "id",
        "drop_cols": ["id", "product_code"],
        "higher_is_better": True, "bronze": None,
    },
    "tabular-playground-series-dec-2021": {
        "type": "classification", "metric": "accuracy",
        "target": "Cover_Type", "id_col": "Id",
        "drop_cols": ["Id"],
        "higher_is_better": True, "bronze": None,
    },
    "tabular-playground-series-may-2022": {
        "type": "classification", "metric": "roc_auc",
        "target": "target", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": None,
    },

    # ── New (19) ──
    "house-prices-advanced-regression-techniques": {
        "type": "regression", "metric": "rmsle",
        "target": "SalePrice", "id_col": "Id",
        "drop_cols": ["Id"],
        "higher_is_better": False, "bronze": 0.140,
        "log_transform_target": True,
    },
    "playground-series-s3e1": {
        "type": "regression", "metric": "rmse",
        "target": "MedHouseVal", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": False, "bronze": None,
    },
    "playground-series-s3e7": {
        "type": "classification", "metric": "accuracy",
        "target": "booking_status", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": None,
    },
    "playground-series-s3e25": {
        "type": "regression", "metric": "rmse",
        "target": "Hardness", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": False, "bronze": None,
    },
    "playground-series-s4e1": {
        "type": "classification", "metric": "roc_auc",
        "target": "Exited", "id_col": "id",
        "drop_cols": ["id", "CustomerId", "Surname"],
        "higher_is_better": True, "bronze": None,
    },
    "playground-series-s4e2": {
        "type": "classification", "metric": "accuracy",
        "target": "NObeyesdad", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": None,
    },
    "playground-series-s4e4": {
        "type": "regression", "metric": "mae",
        "target": "Rings", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": False, "bronze": None,
    },
    "playground-series-s4e6": {
        "type": "classification", "metric": "accuracy",
        "target": "Target", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": None,
    },
    "playground-series-s4e7": {
        "type": "classification", "metric": "roc_auc",
        "target": "Response", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": None,
        "folds": 3,
    },
    "playground-series-s5e1": {
        "type": "regression", "metric": "rmsle",
        "target": "num_sold", "id_col": "id",
        "drop_cols": ["id", "date"],
        "higher_is_better": False, "bronze": None,
        "log_transform_target": True,
    },
    "playground-series-s5e2": {
        "type": "regression", "metric": "rmse",
        "target": "Price", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": False, "bronze": None,
    },
    "playground-series-s5e3": {
        "type": "classification", "metric": "accuracy",
        "target": "rainfall", "id_col": "id",
        "drop_cols": ["id", "day"],
        "higher_is_better": True, "bronze": None,
    },
    "playground-series-s5e4": {
        "type": "regression", "metric": "rmse",
        "target": "Listening_Time_minutes", "id_col": "id",
        "drop_cols": ["id", "Podcast_Name", "Episode_Title"],
        "higher_is_better": False, "bronze": None,
    },
    "playground-series-s5e5": {
        "type": "regression", "metric": "rmse",
        "target": "Calories", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": False, "bronze": None,
    },
    "playground-series-s6e2": {
        "type": "classification", "metric": "roc_auc",
        "target": "Heart Disease", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": None,
    },
    "playground-series-s6e3": {
        "type": "classification", "metric": "roc_auc",
        "target": "Churn", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": None,
    },
    "tabular-playground-series-feb-2022": {
        "type": "classification", "metric": "accuracy",
        "target": "target", "id_col": "row_id",
        "drop_cols": ["row_id"],
        "higher_is_better": True, "bronze": None,
    },
    "tabular-playground-series-jan-2022": {
        "type": "regression", "metric": "rmsle",
        "target": "num_sold", "id_col": "row_id",
        "drop_cols": ["row_id", "date"],
        "higher_is_better": False, "bronze": None,
        "log_transform_target": True,
    },
    "tabular-playground-series-mar-2022": {
        "type": "regression", "metric": "mae",
        "target": "congestion", "id_col": "row_id",
        "drop_cols": ["row_id"],
        "higher_is_better": False, "bronze": None,
    },

    # ── New discovery: playground-series-s4e3 (multi-label classification) ──
    "playground-series-s4e3": {
        "type": "classification", "metric": "roc_auc",
        "target": ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"],
        "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": None,
        "multi_label": True,
    },

    # ── NEW (Run8): home-data-for-ml-course (same dataset as house-prices-advanced-regression-techniques) ──
    "home-data-for-ml-course": {
        "type": "regression", "metric": "rmsle",
        "target": "SalePrice", "id_col": "Id",
        "drop_cols": ["Id"],
        "higher_is_better": False, "bronze": 0.140,
        "log_transform_target": True,
    },

    # ── NEW (Run8): playground-series-s6e7 — Predicting Student Health Risk ──
    # STATUS: NEEDS_MANUAL_ACCEPT — must join at https://www.kaggle.com/competitions/playground-series-s6e7
    # Files: train.csv (62.7MB), test.csv (24.6MB), sample_submission.csv (4.4MB)
    # TODO: After manual join + download, verify target column, metric, and task type.
    # Current config is a PLACEHOLDER based on Playground S6 classification pattern.
    "playground-series-s6e7": {
        "type": "classification", "metric": "roc_auc",
        "target": "target", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": None,
        "_needs_verify": True,  # TODO: verify after manual accept + download
    },

    # ── NEW (Run12): leaf-classification — 99-class leaf species from image features ──
    # Data: mlebench_prepared/leaf-classification/
    # 990 train, 880 test, 192 features (margin1-64, shape1-64, texture1-64) + species target
    "leaf-classification": {
        "type": "classification", "metric": "accuracy",
        "target": "species", "id_col": "id",
        "drop_cols": ["id"],
        "higher_is_better": True, "bronze": None,
        "folds": 5,
        "_note": "99-class multiclass; small data (990 rows) -> multi-seed averaging recommended",
    },

    # ── NEW (Run12): new-york-city-taxi-fare-prediction — regression ──
    # Data: mlebench_prepared/new-york-city-taxi-fare-prediction/
    # 899K train rows, 99K test rows; key is string timestamp ID; fare_amount target
    # pickup_datetime is rich but complex; dropping for now to keep training fast
    "new-york-city-taxi-fare-prediction": {
        "type": "regression", "metric": "rmse",
        "target": "fare_amount", "id_col": "key",
        "drop_cols": ["key", "pickup_datetime"],
        "higher_is_better": False, "bronze": None,
        "folds": 3,  # large dataset (899K rows)
    },

    # NOTE: nomad2018-predict-transparent-conductors SKIPPED — multi-target regression
    # (formation_energy_ev_natom + bandgap_energy_ev) requires architecture changes.
    # NOTE: playground-series-s3e18 SKIPPED — mlebench_prepared directory is empty (no CSV data).

    # ── NEW (Run12): playground-series-s3e26 — Not yet downloaded (placeholder) ──
    # "playground-series-s3e26": {
    #     "type": "classification", "metric": "roc_auc",
    #     "target": "target", "id_col": "id",
    #     "drop_cols": ["id"],
    #     "higher_is_better": True, "bronze": None,
    # },
}


def normalized_gini(y_true, y_pred):
    """2 * roc_auc - 1"""
    return 2 * roc_auc_score(y_true, y_pred) - 1


def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def _align_proba_columns(proba, classes, n_classes):
    """Place a model's predict_proba columns at absolute class ids 0..n_classes-1.

    Targets are LabelEncoded to 0..K-1, so ``classes`` (model.classes_) are direct
    column indices. When a CV fold misses a rare class the matrix is narrower; this
    keeps cross-fold averaging and argmax valid. Pure: takes arrays, returns array.
    """
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    if proba.shape[1] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=float)
    for col, cls in enumerate(classes):
        cls_idx = int(round(float(cls)))
        if 0 <= cls_idx < n_classes:
            full[:, cls_idx] = proba[:, col]
    return full


def _full_width_proba(model, pool, n_classes):
    """Return a (n_samples, n_classes) probability matrix aligned to class ids 0..K-1.

    CatBoost's predict_proba orders columns by ``model.classes_``. Thin wrapper over
    ``_align_proba_columns`` kept for the CatBoost Pool path and existing tests.
    """
    return _align_proba_columns(model.predict_proba(pool), model.classes_, n_classes)


class _GBDTModel:
    """Uniform fit/predict wrapper over catboost | lightgbm | xgboost.

    Keeps the trainer's fold loop family-agnostic. CatBoost uses Pool + cat_features
    on GPU (the proven path). LightGBM runs on CPU (no special GPU build needed);
    XGBoost uses GPU hist. Categoricals are already LabelEncoded by preprocess(),
    so LightGBM/XGBoost consume the numeric frames directly.
    """

    def __init__(self, family, is_clf, n_classes, hyperparams, seed, cat_features):
        self.family = family
        self.is_clf = is_clf
        self.n_classes = int(n_classes) if is_clf else 0
        self.hp = dict(hyperparams or {})
        self.seed = seed
        self.cat_features = cat_features
        self.model = None

    def _build(self):
        hp = self.hp
        if self.family == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor
            common = dict(
                depth=int(hp.get("depth", 6)),
                learning_rate=hp.get("learning_rate", 0.05),
                iterations=int(hp.get("iterations", 2000)),
                early_stopping_rounds=int(hp.get("early_stopping_rounds", 100)),
                task_type="GPU", devices="0", verbose=0, random_seed=self.seed,
            )
            return CatBoostClassifier(**common) if self.is_clf else CatBoostRegressor(**common)
        if self.family == "lightgbm":
            import lightgbm as lgb
            common = dict(
                num_leaves=int(hp.get("num_leaves", 31)),
                learning_rate=hp.get("learning_rate", 0.05),
                n_estimators=int(hp.get("n_estimators", 3000)),
                random_state=self.seed, n_jobs=-1, verbose=-1,
            )
            if self.is_clf:
                objective = "binary" if self.n_classes == 2 else "multiclass"
                return lgb.LGBMClassifier(objective=objective, **common)
            return lgb.LGBMRegressor(**common)
        if self.family == "xgboost":
            import xgboost as xgb
            common = dict(
                max_depth=int(hp.get("max_depth", 6)),
                learning_rate=hp.get("learning_rate", 0.05),
                n_estimators=int(hp.get("n_estimators", 3000)),
                early_stopping_rounds=int(hp.get("early_stopping_rounds", 100)),
                random_state=self.seed, tree_method="hist", device="cuda", verbosity=0,
            )
            return xgb.XGBClassifier(**common) if self.is_clf else xgb.XGBRegressor(**common)
        raise ValueError(f"unknown model family: {self.family}")

    def fit(self, X_tr, y_tr, X_val, y_val):
        self.model = self._build()
        if self.family == "catboost":
            from catboost import Pool
            train_pool = Pool(X_tr, y_tr, cat_features=self.cat_features)
            val_pool = Pool(X_val, y_val, cat_features=self.cat_features)
            self.model.fit(train_pool, eval_set=val_pool)
        elif self.family == "lightgbm":
            import lightgbm as lgb
            rounds = int(self.hp.get("early_stopping_rounds", 100))
            self.model.fit(
                X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(0)],
            )
        else:  # xgboost (early_stopping_rounds set in constructor)
            self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def _input(self, X):
        if self.family == "catboost":
            from catboost import Pool
            return Pool(X, cat_features=self.cat_features)
        return X

    def predict_proba_full(self, X):
        """Full-width (n, n_classes) probabilities aligned to class ids 0..K-1."""
        proba = self.model.predict_proba(self._input(X))
        return _align_proba_columns(proba, self.model.classes_, self.n_classes)

    def predict(self, X):
        return np.asarray(self.model.predict(self._input(X)))


def compute_metric(y_true, y_pred, metric_name, task_type):
    if metric_name == "accuracy":
        if task_type == "classification":
            # y_pred may be probability (binary: values in [0,1]) or class indices (multi-class: values 0,1,2...)
            # Detect: if all values are <= 1.0, treat as probability needing threshold
            if np.max(y_pred) <= 1.0 and np.min(y_pred) >= 0.0:
                y_pred = (y_pred >= 0.5).astype(int)
            return accuracy_score(y_true, y_pred.astype(int))
        else:
            return accuracy_score(y_true, y_pred)
    elif metric_name == "rmsle":
        return rmsle(y_true, y_pred)
    elif metric_name == "roc_auc":
        return roc_auc_score(y_true, y_pred)
    elif metric_name == "normalized_gini":
        return normalized_gini(y_true, y_pred)
    elif metric_name == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric_name == "mae":
        return mean_absolute_error(y_true, y_pred)
    return 0.0


def preprocess(df, cfg, train=True):
    """Clean and encode features."""
    df = df.copy()
    target = cfg["target"]

    # Drop columns
    for col in cfg.get("drop_cols", []):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Separate target
    if train and target in df.columns:
        y = df[target].copy()
        df.drop(columns=[target], inplace=True)
    else:
        y = None

    # Handle pixel data
    if cfg.get("pixel_data") and train:
        pass  # handled separately

    # Encode categoricals
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name == "category":
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
        elif df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Fill NaN — preserve integer dtypes to avoid float contamination
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype in (np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')):
                df[col] = df[col].fillna(int(df[col].median())) if not df[col].isna().all() else df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    return df, y


def _cross_validate(*, family, hyperparams, cv_seed, X_train, y_train, X_test,
                    is_clf, n_classes, metric_name, task_type, cat_features, n_folds):
    """Run one full K-fold CV for a single seed. Returns (oof, test, fold_scores).

    Works in PROBABILITY space for classification so multi-seed averaging is
    mathematically correct for every task:
      * classification: oof=(n, n_classes) full-width OOF proba,
        test=(n_test, n_classes) fold-averaged proba.
      * regression: oof=(n,), test=(n_test,).
    Preserves the dec-2021 fixes (full-width proba alignment via _GBDTModel).
    """
    if is_clf:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
        oof = np.zeros((len(y_train), n_classes))
        test = np.zeros((len(X_test), n_classes))
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
        oof = np.zeros(len(y_train))
        test = np.zeros(len(X_test))
    fold_scores = []

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        try:
            X_tr, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

            # CNN branch (P0) with GBDT fallback
            if family == "cnn" and is_clf and _ImageClassifier is not None and _build_arch_config is not None:
                arch = _build_arch_config(X_train.shape[1], n_classes)
                if hyperparams and hyperparams.get("epochs"):
                    arch.epochs = int(hyperparams["epochs"])
                model = _ImageClassifier(arch)
            elif family == "cnn":
                model = _GBDTModel(family="catboost", is_clf=is_clf, n_classes=n_classes,
                    hyperparams={"depth":6,"lr":0.05,"iters":2000}, seed=cv_seed, cat_features=cat_features)
            else:
                model = _GBDTModel(family=family, is_clf=is_clf, n_classes=n_classes,
                    hyperparams=hyperparams, seed=cv_seed, cat_features=cat_features)
            model.fit(X_tr, y_tr, X_val, y_val)

            if is_clf:
                val_proba = model.predict_proba_full(X_val)   # (n_val, n_classes)
                oof[val_idx] = val_proba
                test += model.predict_proba_full(X_test) / n_folds
                fold_pred = val_proba[:, 1] if n_classes == 2 else np.argmax(val_proba, axis=1)
            else:
                val_pred = model.predict(X_val)
                oof[val_idx] = val_pred
                test += model.predict(X_test) / n_folds
                fold_pred = val_pred

            score = compute_metric(y_val, fold_pred, metric_name, task_type)
            fold_scores.append(score)
            print(f"    fold {fold+1}/{n_folds} [{family} seed={cv_seed}]: {score:.6f}")
            # Free GPU memory between folds (critical for CNN multi-fold)
            if family == "cnn":
                del model
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception as fold_err:
            import traceback
            print(f"    fold {fold+1}/{n_folds} FAILED: {fold_err}")
            traceback.print_exc()
            continue

    return oof, test, fold_scores


def train_single(comp_name, cfg, n_folds=5, seed=42):
    """Train a single competition, return metrics + submission path."""
    # Model libraries are imported lazily inside _GBDTModel (GPU-only deps), so this
    # module stays importable on the control plane for unit tests.
    _, configured_data_dir, fallback_data_dir, results_dir = _runtime_paths()
    data_dir = configured_data_dir / comp_name
    if not data_dir.exists():
        fallback_dir = fallback_data_dir / comp_name
        if fallback_dir.exists():
            data_dir = fallback_dir
            print(f"  [using fallback data path: {data_dir}]")
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    sample_sub_paths = [
        data_dir / "sample_submission.csv",
        data_dir / "sampleSubmission.csv",
        data_dir / "gender_submission.csv",
    ]
    sample_sub_path = None
    for p in sample_sub_paths:
        if p.exists():
            sample_sub_path = p
            break

    if not train_path.exists() or not test_path.exists():
        return {"error": f"Data missing: {comp_name}"}

    print(f"\n{'='*60}")
    print(f"Training: {comp_name}")
    print(f"  Type: {cfg['type']}, Metric: {cfg['metric']}")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        return {"error": f"Failed to load data for {comp_name}: {e}"}
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")

    X_train, y_train = preprocess(train_df, cfg, train=True)
    X_test, _ = preprocess(test_df, cfg, train=False)

    # Align columns
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    task_type = cfg["type"]
    metric_name = cfg["metric"]
    higher_better = cfg.get("higher_is_better", True)
    n_folds = cfg.get("folds", n_folds)  # per-competition override (e.g. 3 folds for large datasets)

    # Validate target
    if y_train is None:
        return {"error": f"Target column '{cfg['target']}' not found in training data"}

    # Encode ALL classification targets to a contiguous 0..K-1 space.
    # CRITICAL FIX (dec-2021 regression): integer labels such as Cover_Type (1-7)
    # were previously left un-encoded because dtype was int64, not object/category.
    # Downstream we use argmax(predict_proba) which is 0-indexed, so raw labels 1-7
    # never matched the 0-6 argmax positions -> a systematic off-by-one that
    # collapsed accuracy to ~1.5%. LabelEncoder is a safe no-op for labels already
    # in 0..K-1 (binary 0/1 and digit-recognizer 0-9 are unchanged).
    # NOTE: requires GPU (Run7) verification; cannot be run on the control plane.
    le_target = None
    if task_type == "classification":
        le_target = LabelEncoder()
        y_train = pd.Series(le_target.fit_transform(y_train), index=y_train.index, name=y_train.name)
        _cls_preview = list(le_target.classes_)[:10]
        print(f"  Encoded target: {len(le_target.classes_)} classes (preview): {_cls_preview}")

    # Log-transform target for RMSLE regression
    log_target = cfg.get("log_transform_target", False)
    if log_target and task_type == "regression":
        y_train = pd.Series(np.log1p(y_train.values), index=y_train.index, name=y_train.name)

    # Cross-validation
    # Determine task shape (folds are built per-seed inside _cross_validate).
    n_classes = 0  # safe default for regression; overwritten for classification
    if task_type == "classification":
        n_classes = len(np.unique(y_train))
        if n_classes > 50:
            n_folds = min(n_folds, 3)
        is_clf = True
    else:
        n_classes = 0
        is_clf = False

    is_multiclass = is_clf and n_classes > 2

    # ── P1: consult model_selection to pick the executable family + hyperparams ──
    # This replaces the fixed-CatBoost default. Neural/CNN recommendations are
    # deferred (P2) with a safe GBDT fallback so a run never becomes inert.
    training_plan = None
    plan_dict = None
    if _MODEL_SELECTION_AVAILABLE:
        try:
            n_high_card = sum(
                1 for c in common_cols
                if X_train[c].dtype == np.dtype('object') or X_train[c].nunique() > 20
            )
            profile = DataProfile(
                task_type=task_type,
                n_rows=len(X_train),
                n_cols=len(common_cols),
                metric=metric_name,
                n_classes=n_classes if is_clf else None,
                categorical_ratio=(n_high_card / max(len(common_cols), 1)),
                is_pixel_like=bool(cfg.get("pixel_data")),
            )
            strategy = recommend_model_strategy(profile)
            training_plan = resolve_training_plan(
                strategy, profile,
                available_models=_detect_available_models(),
                neural_available=_NEURAL_OK,
            )
            plan_dict = training_plan.to_dict()
            print(f"  Plan: {training_plan.executable_model} "
                  f"(deferred={training_plan.deferred_model}, "
                  f"multi_seed={training_plan.multi_seed}, ens={training_plan.ensemble_models})")
        except Exception as plan_err:
            print(f"  [plan resolution failed: {plan_err}; using CatBoost default]")
            training_plan = None

    family = training_plan.executable_model if training_plan else "catboost"
    plan_hp = training_plan.hyperparams if training_plan else {}

    # Only truly categorical columns as cat_features (must work for BOTH train and test)
    # Check train dtype, test dtype, and actual values
    cat_features = []
    for i, c in enumerate(common_cols):
        col_train = X_train[c]
        col_test = X_test[c]
        # If either train or test has float dtype for this column, skip
        if col_train.dtype not in (np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8'), np.dtype('bool'), np.dtype('object')):
            continue
        if col_test.dtype not in (np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8'), np.dtype('bool'), np.dtype('object')):
            continue
        # Object/string columns: always categorical
        if col_train.dtype == np.dtype('object') or col_train.dtype.name == 'category':
            cat_features.append(i)
        # Bool columns: always categorical
        elif col_train.dtype == np.dtype('bool'):
            cat_features.append(i)
        # Integer columns: only if low cardinality in BOTH train and test
        else:
            n_unique_train = col_train.nunique()
            n_unique_test = col_test.nunique()
            if n_unique_train < 50 and n_unique_test < 50:
                cat_features.append(i)
    if not cat_features:
        cat_features = None  # Let CatBoost auto-detect

    # ── P2: multi-seed averaging for small data (variance reduction) ──
    # The plan enables multi_seed for small train sets; otherwise a single seed.
    seeds = (training_plan.seeds if (training_plan and training_plan.multi_seed and training_plan.seeds)
             else [seed])

    oof_list, test_list, all_fold_scores = [], [], []
    for cv_seed in seeds:
        oof_s, test_s, fs = _cross_validate(
            family=family, hyperparams=plan_hp, cv_seed=cv_seed,
            X_train=X_train, y_train=y_train, X_test=X_test,
            is_clf=is_clf, n_classes=n_classes, metric_name=metric_name,
            task_type=task_type, cat_features=cat_features, n_folds=n_folds,
        )
        if fs:
            oof_list.append(oof_s)
            test_list.append(test_s)
            all_fold_scores.extend(fs)
        if len(seeds) > 1:
            print(f"  seed {cv_seed}: {len(fs)} folds, mean={np.mean(fs):.6f}" if fs else f"  seed {cv_seed}: all folds failed")

    if not all_fold_scores:
        return {"error": f"All folds failed for {comp_name}"}

    # Average predictions across seeds (equal weight) via the ensemble engine.
    if len(oof_list) > 1:
        oof_avg = _ensemble_blend(oof_list)
        test_avg = _ensemble_blend(test_list)
        seed_strategy = f"multi_seed_{len(oof_list)}"
    else:
        oof_avg, test_avg = oof_list[0], test_list[0]
        seed_strategy = "single_seed"

    fold_scores = all_fold_scores

    # Derive final oof/test predictions from averaged probabilities/values.
    # (binary -> P(class=1); multi-class -> argmax over averaged proba; regression -> value)
    if is_clf and n_classes == 2:
        oof_preds = np.asarray(oof_avg)[:, 1]
        test_preds = np.asarray(test_avg)[:, 1]
    elif is_clf:
        oof_preds = np.argmax(oof_avg, axis=1).astype(float)
        test_preds = np.argmax(test_avg, axis=1).astype(float)
    else:
        oof_preds = np.asarray(oof_avg, dtype=float)
        test_preds = np.asarray(test_avg, dtype=float)

    # OOF score is computed in the encoded 0..K-1 label space for classification;
    # accuracy/gini/auc are invariant to a bijective relabel, so this is correct.
    oof_score = compute_metric(y_train, oof_preds, metric_name, task_type)
    mean_fold = np.mean(fold_scores)
    std_fold = np.std(fold_scores)

    # Undo log transform
    if log_target and task_type == "regression":
        test_preds = np.expm1(test_preds)
        oof_preds = np.expm1(oof_preds)

    # Decode encoded class indices back to the competition's original labels.
    def _decode(encoded_values):
        if le_target is None:
            return np.asarray(encoded_values)
        idx = np.clip(np.rint(np.asarray(encoded_values)).astype(int), 0, len(le_target.classes_) - 1)
        return le_target.inverse_transform(idx)

    # Generate submission matching sample format
    if sample_sub_path:
        sub_df = pd.read_csv(sample_sub_path)
        target_col = sub_df.columns[1] if len(sub_df.columns) >= 2 else sub_df.columns[0]
        sample_dtype = sub_df[target_col].dtype

        if task_type == "classification" and is_multiclass:
            # test_preds are encoded indices -> decode to original class labels.
            target_values = _decode(test_preds)
            if sample_dtype in (np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')):
                target_values = target_values.astype(int)
        elif task_type == "classification":  # binary
            if sample_dtype == np.dtype("bool"):
                # For bool submissions, output thresholded class indices decoded to bool
                target_values = _decode((test_preds > 0.5).astype(int)).astype(bool)
            elif sample_dtype in (np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')):
                # Output encoded class indices (0/1) directly — matches integer submission format.
                # When target was string-encoded (e.g. "Yes"/"No" → 0/1), _decode would
                # produce strings that can't be cast back to int; the submission already
                # expects 0/1 so we skip the decode.
                target_values = (test_preds > 0.5).astype(int)
            elif le_target is not None and (sample_dtype == np.dtype('object') or sample_dtype.name == 'object'):
                # String-label binary classification: decode 0/1 back to original labels
                target_values = _decode((test_preds > 0.5).astype(int))
            else:
                target_values = test_preds  # probability submission (roc_auc / gini)
        else:  # regression
            if sample_dtype in (np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')):
                target_values = test_preds.astype(int)
            else:
                target_values = test_preds

        sub_df[target_col] = target_values
        sub_path = results_dir / f"submission_{comp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        sub_df.to_csv(sub_path, index=False)
    else:
        sub_path = None

    result = {
        "competition": comp_name,
        "task_type": task_type,
        "metric": metric_name,
        "oof_score": float(oof_score),
        "fold_mean": float(mean_fold),
        "fold_std": float(std_fold),
        "fold_scores": [float(s) for s in fold_scores],
        "n_folds": n_folds,
        "seed": seed,
        "higher_is_better": higher_better,
        "bronze_threshold": cfg.get("bronze"),
        "gate_pass": None,
        "model_family": family,
        "training_plan": plan_dict,
        "seed_strategy": seed_strategy,
        "submission_path": str(sub_path) if sub_path else None,
        "train_shape": list(train_df.shape),
        "timestamp": datetime.now().isoformat(),
    }

    # Gate check (ensure Python bool, not numpy.bool_)
    bronze = cfg.get("bronze")
    if bronze is not None:
        if higher_better:
            result["gate_pass"] = bool(oof_score > bronze)
        else:
            result["gate_pass"] = bool(oof_score < bronze)

    # Save result
    result_path = results_dir / f"gpu_{comp_name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  OOF: {oof_score:.6f} | Fold mean: {mean_fold:.6f} ± {std_fold:.6f}")
    if bronze is not None:
        print(f"  Gate: {'PASS' if result['gate_pass'] else 'FAIL'} (bronze={bronze})")
    if sub_path:
        print(f"  Submission: {sub_path}")

    return result


def run_batch(comp_list=None, n_folds=5, seed=42):
    """Run training for a list of competitions."""
    _, _, _, results_dir = _runtime_paths()
    if comp_list is None:
        comp_list = list(COMPETITIONS.keys())

    results = {}
    for comp_name in comp_list:
        if comp_name not in COMPETITIONS:
            print(f"SKIP {comp_name}: not in registry")
            continue
        cfg = COMPETITIONS[comp_name]
        try:
            r = train_single(comp_name, cfg, n_folds=n_folds, seed=seed)
            results[comp_name] = r
        except Exception as e:
            print(f"FAIL {comp_name}: {e}")
            results[comp_name] = {"error": str(e)}

    # Save summary
    summary_path = results_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:45s} FAIL: {r['error'][:60]}")
        else:
            bronze = r.get("bronze_threshold")
            gate = r.get("gate_pass")
            gate_str = f"gate={'PASS' if gate else 'FAIL'}" if gate is not None else "gate=N/A"
            bronze_str = f"bronze={bronze}" if bronze is not None else ""
            print(f"  {name:45s} oof={r['oof_score']:.6f} {gate_str} {bronze_str}")

    print(f"\nSummary saved to: {summary_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--competitions", nargs="+", default=None, help="Competition names to run")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--priority", action="store_true", help="Run priority tasks only")
    parser.add_argument(
        "--remote-workspace",
        default=os.environ.get("EVOMIND_HPC_REMOTE_WORKSPACE", ""),
        help="dedicated remote workspace (or set EVOMIND_HPC_REMOTE_WORKSPACE)",
    )
    args = parser.parse_args()
    configure_runtime_paths(args.remote_workspace)

    if args.priority:
        priority = [
            "spaceship-titanic", "titanic", "house_prices",
            "bike-sharing-demand", "porto-seguro-safe-driver-prediction",
            "playground-series-s6e6", "store-sales-time-series-forecasting",
            "digit-recognizer", "tabular-playground-series-aug-2022",
            "tabular-playground-series-dec-2021", "tabular-playground-series-may-2022",
            "home-data-for-ml-course",  # NEW Run8: identical to house-prices
            # "playground-series-s6e7",  # NEW: needs manual Kaggle accept first
        ]
        comp_list = [c for c in priority if c in COMPETITIONS]
    else:
        comp_list = args.competitions

    start = time.time()
    results = run_batch(comp_list=comp_list, n_folds=args.folds, seed=args.seed)
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
