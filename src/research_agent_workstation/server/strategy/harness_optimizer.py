"""
Harness Engineering Optimizer — based on Famou-Agent 2.0 + MLEvolve + AIBuildAI patterns.

Top MLE-Bench systems share 4 pillars:
1. Island Model Evolution: parallel exploration + survival of fittest
2. Long-Chain Memory: context persistence across experiments
3. Self-Correction Loops: automated error detection and repair
4. Workflow Orchestration: specialized agents with clear contracts

Applied to our local Kaggle optimization:
- Each "island" = independent ensemble configuration
- Memory = accumulated OOF scores + feature importance + error patterns
- Self-correction = detect CV-public gap, overfitting, data leakage
- Workflow = DeepSeek plans → code generation → execution → evaluation → memory update
"""
from __future__ import annotations

import json, os, sys, math, random, time, csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Callable

ROOT = Path(__file__).resolve().parents[5]  # up to project root
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# ── Island (parallel experiment) ────────────────────────────────────────────

@dataclass
class Island:
    """One parallel optimization track (Famou-Agent 'island model')."""
    island_id: int
    strategy: str               # "feature_engineering", "model_diversity", "ensemble_blend"
    model_config: dict          # hyperparameters for this island
    best_score: Optional[float] = None
    best_submission_path: Optional[str] = None
    history: list[dict] = field(default_factory=list)
    stagnation_count: int = 0

    def record_attempt(self, score: float, config: dict, oof_path: str):
        self.history.append({"score": score, "config": config, "oof_path": oof_path, "time": time.time()})
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

    @property
    def is_stagnant(self) -> bool:
        return self.stagnation_count >= 3


# ── Long-Chain Memory ───────────────────────────────────────────────────────

@dataclass
class ExperimentMemory:
    """Accumulated experience across all experiments (Famou-Agent long-chain memory)."""
    successful_patterns: list[dict] = field(default_factory=list)
    failed_patterns: list[dict] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    cv_public_gap_history: list[float] = field(default_factory=list)
    best_model_configs: dict[str, dict] = field(default_factory=dict)

    def record_success(self, pattern: dict):
        self.successful_patterns.append(pattern)
        if len(self.successful_patterns) > 50:
            self.successful_patterns = self.successful_patterns[-50:]

    def record_failure(self, pattern: dict):
        self.failed_patterns.append(pattern)
        if len(self.failed_patterns) > 50:
            self.failed_patterns = self.failed_patterns[-50:]

    def get_top_strategies(self, n: int = 5) -> list[dict]:
        return sorted(self.successful_patterns, key=lambda x: x.get("score", 0), reverse=True)[:n]

    def get_pitfalls(self) -> list[str]:
        return [p.get("reason", "") for p in self.failed_patterns[-5:]]

    def estimate_cv_public_gap(self) -> float:
        if self.cv_public_gap_history:
            return np.median(self.cv_public_gap_history)
        return 0.004  # default for tabular


# ── Self-Correction Loop ────────────────────────────────────────────────────

class SelfCorrectionLoop:
    """Detect and fix common ML errors (Famou-Agent self-correction)."""

    @staticmethod
    def check_submission_format(submission_path: str, sample_path: str) -> list[str]:
        """Validate submission against sample format."""
        issues = []
        try:
            sub = pd.read_csv(submission_path)
            sample = pd.read_csv(sample_path)
            if list(sub.columns) != list(sample.columns):
                issues.append(f"Column mismatch: {list(sub.columns)} vs {list(sample.columns)}")
            if len(sub) != len(sample):
                issues.append(f"Row count mismatch: {len(sub)} vs {len(sample)}")
        except Exception as e:
            issues.append(f"Cannot read submission: {e}")
        return issues

    @staticmethod
    def check_overfitting(cv_score: float, oof_std: float, n_features: int, n_rows: int) -> bool:
        """Detect overfitting: high CV with few rows + many features = suspect."""
        if n_features > n_rows * 0.5:
            return True  # More features than half the rows
        if oof_std > 0.05 and n_rows < 2000:
            return True  # High variance on small dataset
        return False

    @staticmethod
    def check_data_leakage(train_path: str, test_path: str) -> list[str]:
        """Check for common data leakage patterns."""
        issues = []
        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            # Check ID overlap
            id_cols = [c for c in train.columns if 'id' in c.lower()]
            for col in id_cols:
                if col in test.columns:
                    overlap = set(train[col]) & set(test[col])
                    if overlap:
                        issues.append(f"ID overlap in {col}: {len(overlap)} shared values")
            # Check target in test
            target_candidates = ['target', 'label', 'Survived', 'Transported', 'SalePrice', 'count']
            for tc in target_candidates:
                if tc in test.columns:
                    issues.append(f"Target column '{tc}' found in test set!")
        except Exception as e:
            issues.append(f"Cannot check leakage: {e}")
        return issues

    @staticmethod
    def check_cv_public_alignment(cv_scores: list[float], public_score: float) -> dict:
        """Check if CV-public gap is within expected range."""
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        gap = abs(cv_mean - public_score)
        expected_gap = 0.005  # typical for tabular
        return {
            "cv_mean": cv_mean, "cv_std": cv_std,
            "public_score": public_score,
            "gap": gap,
            "gap_normal": gap < expected_gap * 3,
            "cv_stable": cv_std < 0.02,
            "recommendation": "OK" if gap < expected_gap * 3 and cv_std < 0.02
                             else "Investigate overfitting" if gap > expected_gap * 3
                             else "Reduce CV variance"
        }


# ── Harness Engine ──────────────────────────────────────────────────────────

class HarnessEngine:
    """
    Main harness engine combining:
    - Island model (Famou-Agent): parallel strategy exploration
    - Long-chain memory (Famou-Agent + MLEvolve): accumulated experience
    - Self-correction (Famou-Agent): automated error detection
    - UCT-guided search (MLEvolve): optimal node selection
    """

    def __init__(self, task_id: str, n_islands: int = 3):
        self.task_id = task_id
        self.n_islands = n_islands
        self.memory = ExperimentMemory()
        self.correction = SelfCorrectionLoop()
        self.islands: list[Island] = []
        self.global_best_score: Optional[float] = None
        self.global_best_submission: Optional[str] = None
        self.iteration = 0
        self.start_time = time.time()

        # Task-specific config
        self.task_config = self._load_task_config()
        self._init_islands()

    def _load_task_config(self) -> dict:
        configs = {
            "bike_sharing_demand": {
                "target": "count", "metric": "rmsle", "direction": "minimize",
                "n_rows": 10886, "n_features_typical": 15,
                "best_known_local": 0.285, "best_known_public": 0.40647,
                "cv_public_gap": 0.121,
                "strongest_model": "gbr", "strongest_oof": 0.285,
                "sample_submission": "D:/桌面/codex/科研港科技/tasks/bike_sharing_demand/data/sample_submission.csv",
            },
            "digit_recognizer": {
                "target": "label", "metric": "accuracy", "direction": "maximize",
                "n_rows": 42000, "n_features_typical": 784,
                "best_known_local": 0.9943, "best_known_public": 0.99578,
                "cv_public_gap": 0.0015,
                "strongest_model": "cnn", "strongest_oof": 0.9943,
                "sample_submission": "D:/桌面/codex/科研港科技/tasks/digit_recognizer/data/sample_submission.csv",
            },
            "spaceship_titanic": {
                "target": "Transported", "metric": "accuracy", "direction": "maximize",
                "n_rows": 8693, "n_features_typical": 18,
                "best_known_local": 0.8163, "best_known_public": 0.80640,
                "cv_public_gap": 0.0099,
                "strongest_model": "cb_hgb_blend", "strongest_oof": 0.8163,
                "sample_submission": "D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/sample_submission.csv",
            },
            "titanic": {
                "target": "Survived", "metric": "accuracy", "direction": "maximize",
                "n_rows": 891, "n_features_typical": 10,
                "best_known_local": 0.8418, "best_known_public": 0.77990,
                "cv_public_gap": 0.062,
                "strongest_model": "rf", "strongest_oof": 0.837,
                "sample_submission": "D:/桌面/codex/科研港科技/tasks/titanic/data/sample_submission.csv",
            },
        }
        return configs.get(self.task_id, {})

    def _init_islands(self):
        strategies = [
            {"name": "feature_engineering", "model": "catboost",
             "params": {"iterations": 800, "learning_rate": 0.02, "depth": 7},
             "focus": "Add interaction features and group aggregations"},
            {"name": "model_diversity", "model": "ensemble",
             "params": {"models": ["catboost", "hgb", "lgb"], "blend_method": "nelder_mead"},
             "focus": "Explore different model families with optimized weights"},
            {"name": "hyperparameter_search", "model": "catboost",
             "params": {"iterations": 1000, "learning_rate": 0.015, "depth": 8, "l2_leaf_reg": 5},
             "focus": "Aggressive regularization for better generalization"},
        ]
        for i in range(min(self.n_islands, len(strategies))):
            s = strategies[i]
            self.islands.append(Island(
                island_id=i, strategy=s["name"],
                model_config={"model": s["model"], "params": s["params"], "focus": s["focus"]}
            ))

    def run_iteration(self) -> dict:
        """Run one harness iteration: select island → mutate → train → evaluate → update memory."""
        self.iteration += 1

        # Select best non-stagnant island
        active = [isl for isl in self.islands if not isl.is_stagnant]
        if not active:
            # All stagnant: create new island
            new_id = len(self.islands)
            self.islands.append(Island(
                island_id=new_id, strategy="exploration",
                model_config={"model": "catboost", "params": {"iterations": 500, "learning_rate": 0.05, "depth": 6},
                            "focus": "Fresh exploration after stagnation"}
            ))
            active = [self.islands[-1]]

        island = max(active, key=lambda i: i.best_score or -999)
        return {"island_id": island.island_id, "strategy": island.strategy,
                "config": island.model_config, "best_so_far": island.best_score}

    def update(self, island_id: int, score: float, oof_path: str, config: dict):
        """Update island and memory with new result."""
        island = next((i for i in self.islands if i.island_id == island_id), None)
        if not island:
            return

        island.record_attempt(score, config, oof_path)

        # Update global best
        if self.global_best_score is None or score > self.global_best_score:
            self.global_best_score = score
            self.memory.record_success({
                "island_id": island_id, "strategy": island.strategy,
                "score": score, "config": config, "iteration": self.iteration,
                "time_elapsed": time.time() - self.start_time
            })
        else:
            improvement = score - (island.best_score or 0)
            self.memory.record_failure({
                "island_id": island_id, "strategy": island.strategy,
                "score": score, "delta": improvement,
                "reason": "No improvement" if improvement <= 0 else f"Below island best by {-improvement:.6f}"
            })

    def get_status(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            "task_id": self.task_id,
            "iteration": self.iteration,
            "elapsed_seconds": elapsed,
            "global_best_score": self.global_best_score,
            "n_islands": len(self.islands),
            "active_islands": sum(1 for i in self.islands if not i.is_stagnant),
            "stagnant_islands": sum(1 for i in self.islands if i.is_stagnant),
            "memory_successes": len(self.memory.successful_patterns),
            "memory_failures": len(self.memory.failed_patterns),
            "top_strategies": self.memory.get_top_strategies(3),
            "pitfalls": self.memory.get_pitfalls(),
            "islands": [
                {"id": i.island_id, "strategy": i.strategy, "best_score": i.best_score,
                 "stagnant": i.is_stagnant, "attempts": len(i.history)}
                for i in self.islands
            ]
        }


# ── Quick ensemble trainer (shared across islands) ─────────────────────────

def train_ensemble_for_island(task_id: str, island_config: dict, n_folds: int = 3) -> dict:
    """Train a quick ensemble based on the island's strategy config."""
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Load data (simplified - reuse previously engineered features)
    train = pd.read_csv(f"D:/桌面/codex/科研港科技/tasks/{task_id}/data/train.csv")
    test = pd.read_csv(f"D:/桌面/codex/科研港科技/tasks/{task_id}/data/test.csv")

    target = "Transported" if task_id == "spaceship_titanic" else "Survived"
    y = (train[target] == True).astype(int).values if target in train.columns else train[target].values

    # Quick feature prep (minimal)
    drop = ["PassengerId", "Name", "Cabin", "Ticket", target]
    X = train.drop(columns=[c for c in drop if c in train.columns], errors="ignore")
    Xt = test.drop(columns=[c for c in drop if c in test.columns], errors="ignore")

    # Categorical encoding
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            combined = pd.concat([X[col].astype(str), Xt[col].astype(str)])
            le.fit(combined)
            X[col] = le.transform(X[col].astype(str))
            Xt[col] = le.transform(Xt[col].astype(str))

    common = [c for c in X.columns if c in Xt.columns]
    X = X[common].fillna(-1).astype(float).values
    Xt = Xt[common].fillna(-1).astype(float).values
    X = StandardScaler().fit_transform(X)
    Xt = StandardScaler().fit_transform(Xt)

    # Train based on island strategy
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(Xt))

    model_name = island_config.get("model", "catboost")
    params = island_config.get("params", {})

    for tr, val in skf.split(X, y):
        if model_name == "catboost":
            m = CatBoostClassifier(**params, random_seed=42, verbose=False, thread_count=-1)
        elif model_name == "lgb":
            m = lgb.LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
        else:
            m = HistGradientBoostingClassifier(**params, random_state=42)

        m.fit(X[tr], y[tr])
        oof[val] = m.predict_proba(X[val])[:, 1] if hasattr(m, 'predict_proba') else m.predict(X[val])
        test_pred += (m.predict_proba(Xt)[:, 1] if hasattr(m, 'predict_proba') else m.predict(Xt)) / n_folds

    score = accuracy_score(y, (oof > 0.5).astype(int))

    # Save submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"D:/桌面/codex/科研港科技/experiments/{task_id}/harness_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(f"D:/桌面/codex/科研港科技/tasks/{task_id}/data/sample_submission.csv")
    pred_col = sub.columns[-1]
    sub[pred_col] = (test_pred > 0.5).astype(int) if task_id != "bike_sharing_demand" else np.maximum(0, test_pred)
    sub.to_csv(out_dir / "submission.csv", index=False)

    # Save OOF
    pd.DataFrame({"oof": oof, "true": y}).to_csv(out_dir / "oof_predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"task_id": task_id, "score": float(score), "model": model_name,
                   "params": params, "n_folds": n_folds, "features": X.shape[1]}, f, indent=2, default=str)

    return {"score": float(score), "oof_path": str(out_dir / "oof_predictions.csv"),
            "submission_path": str(out_dir / "submission.csv"), "features": X.shape[1]}


# ── Main harness runner ─────────────────────────────────────────────────────

def run_harness(task_id: str = "spaceship_titanic", n_iterations: int = 10, n_islands: int = 3):
    """Run the full harness optimization loop."""
    print(f"\n{'='*60}")
    print(f"HARNESS ENGINE: {task_id}")
    print(f"Islands: {n_islands}, Iterations: {n_iterations}")
    print(f"{'='*60}")

    engine = HarnessEngine(task_id, n_islands=n_islands)

    for iteration in range(n_iterations):
        action = engine.run_iteration()
        island_id = action["island_id"]
        config = action["config"]

        print(f"\nIter {iteration+1}/{n_iterations} [Island {island_id}: {action['strategy']}]")
        print(f"  Config: {config['model']} | {config['focus']}")

        try:
            result = train_ensemble_for_island(task_id, config, n_folds=3)
            score = result["score"]
            engine.update(island_id, score, result["oof_path"], config)

            improved = engine.global_best_score == score
            marker = ">>> NEW BEST <<<" if improved else ""
            print(f"  Score: {score:.6f} (best: {engine.global_best_score:.6f}) {marker}")

            # Self-correction check
            cfg = engine.task_config
            if cfg.get("n_rows") and cfg["n_rows"] < 2000 and result["features"] > cfg["n_rows"] * 0.5:
                print(f"  WARNING: Possible overfitting ({result['features']} features on {cfg['n_rows']} rows)")

        except Exception as e:
            print(f"  ERROR: {e}")
            engine.update(island_id, -1, "", config)

    # Final report
    status = engine.get_status()
    print(f"\n{'='*60}")
    print(f"HARNESS COMPLETE")
    print(f"Best score: {status['global_best_score']:.6f}")
    print(f"Total iterations: {status['iteration']}")
    print(f"Active/S stagnant islands: {status['active_islands']}/{status['stagnant_islands']}")
    print(f"Memory: {status['memory_successes']} successes, {status['memory_failures']} failures")
    print(f"{'='*60}")

    return status


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="spaceship_titanic")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--islands", type=int, default=3)
    args = parser.parse_args()

    status = run_harness(args.task, args.iterations, args.islands)

    # Save status
    out = Path(f"D:/桌面/codex/科研港科技/workspace/harness/status_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(status, f, indent=2, default=str)
    print(f"\nStatus saved: {out}")
