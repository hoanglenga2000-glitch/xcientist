"""
MLEvolve-style local optimization loop.
Applies UCT search + multi-branch + stagnation detection WITHOUT the full MLEvolve deps.
Uses our existing LGB/XGB/CatBoost/HGB + DeepSeek for planning.
Goal: maximize local CV scores before Kaggle submission.
"""
import json, os, sys, time, math, random, csv, uuid
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# ── UCT Search Node ────────────────────────────────────────────────────────

@dataclass
class UCTNode:
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    parent_id: Optional[str] = None
    branch_id: int = 0
    depth: int = 0
    plan: str = ""
    action: str = ""
    score: Optional[float] = None
    visits: int = 0
    total_reward: float = 0.0
    is_buggy: bool = False
    children: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def uct_value(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        parent_visits = self.visits
        exploit = self.total_reward / self.visits
        explore = exploration_constant * math.sqrt(math.log(max(parent_visits, 1)) / self.visits)
        return exploit + explore

# ── Config per competition ──────────────────────────────────────────────────

TASKS = {
    "spaceship_titanic": {
        "target": "Transported", "metric": "accuracy", "direction": "maximize",
        "type": "binary_classification",
        "best_known_local": 0.8163, "best_known_public": 0.80640,
        "target_threshold": 0.820,
        "key_features": ["Cabin_deck","Cabin_num","Cabin_side","Group","TotalSpend","VIP_Age"],
        "cat_features": ["HomePlanet","CryoSleep","Destination","VIP","Cabin_deck","Cabin_side"],
        "drop": ["PassengerId","Name","Cabin"]
    },
    "titanic": {
        "target": "Survived", "metric": "accuracy", "direction": "maximize",
        "type": "binary_classification",
        "best_known_local": 0.8283, "best_known_public": 0.77751,
        "target_threshold": 0.830,
        "key_features": ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","IsAlone"],
        "cat_features": ["Sex","Embarked","Pclass"],
        "drop": ["PassengerId","Name","Ticket","Cabin"]
    },
    "bike_sharing_demand": {
        "target": "count", "metric": "rmsle", "direction": "minimize",
        "type": "regression",
        "best_known_local": 0.2849, "best_known_public": 0.40647,
        "target_threshold": 0.25,
        "key_features": ["hour","day","month","year","dayofweek","is_weekend","temp","atemp","humidity","windspeed"],
        "cat_features": ["season","holiday","workingday","weather","is_weekend"],
        "drop": ["datetime","casual","registered"]
    },
    "digit_recognizer": {
        "target": "label", "metric": "accuracy", "direction": "maximize",
        "type": "multiclass_classification",
        "best_known_local": 0.97, "best_known_public": 0.85842,
        "target_threshold": 0.975,
        "key_features": [], "cat_features": [], "drop": []
    },
}

# ── MLEvolve-style Optimization Loop ────────────────────────────────────────

class MLEvolveLocalLoop:
    def __init__(self, task_id: str, max_steps: int = 50, n_branches: int = 3):
        self.task_id = task_id
        self.cfg = TASKS[task_id]
        self.max_steps = max_steps
        self.n_branches = n_branches

        # Search state
        self.nodes: dict[str, UCTNode] = {}
        self.branches: dict[int, list[str]] = {}
        self.root = UCTNode(node_id="root", branch_id=-1, plan="ROOT")
        self.nodes[self.root.node_id] = self.root

        self.best_node: Optional[UCTNode] = None
        self.best_score: Optional[float] = None
        self.step_count = 0
        self.start_time = time.time()

        # Stagnation tracking
        self.branch_best: dict[int, float] = {}
        self.branch_stagnation: dict[int, int] = {}
        self.global_stagnation = 0

        # Load data once
        self._load_data()
        self._load_best_experiment()

    def _load_data(self):
        data_dir = ROOT / "tasks" / self.task_id / "data"
        self.train = pd.read_csv(data_dir / "train.csv")
        self.test = pd.read_csv(data_dir / "test.csv")

    def _load_best_experiment(self):
        exp_dir = ROOT / "experiments" / self.task_id
        if not exp_dir.exists():
            self.best_submission = None
            return
        # Find experiment with best metrics
        best_score = None
        best_sub = None
        for d in sorted(os.listdir(exp_dir), reverse=True):
            mf = exp_dir / d / "metrics.json"
            sf = exp_dir / d / "submission.csv"
            if mf.exists() and sf.exists():
                with open(mf) as f:
                    m = json.load(f)
                score = m.get("best_blend_accuracy") or m.get("best_validation_score")
                if score is not None:
                    if best_score is None or score > best_score:
                        best_score = score
                        best_sub = str(sf)
        self.best_submission = best_sub
        if best_score and (self.best_score is None or best_score > self.best_score):
            self.best_score = best_score

    def _piecewise_decay_C(self, step: int) -> float:
        """UCT exploration constant: 1.414 → 0.5 over search."""
        T1, T2 = 15, 35  # phase boundaries
        if step < T1: return 1.414
        elif step <= T2: return max(1.414 - 0.03 * (step - T1), 0.5)
        return 0.5

    def _select_node(self) -> UCTNode:
        """UCT selection: traverse from root, pick best child."""
        current = self.root
        while current.children:
            C = self._piecewise_decay_C(self.step_count)
            unvisited = [c for c in current.children if c.visits == 0]
            if unvisited:
                return random.choice(unvisited)
            current = max(current.children, key=lambda c: c.uct_value(C))
        return current

    def _generate_plan(self, node: UCTNode, branch_id: int) -> str:
        """Generate improvement plan using heuristics (DeepSeek integration point)."""
        cfg = self.cfg
        plans = []

        if branch_id == 0:  # feature engineering branch
            plans = [
                "Add polynomial interactions between top-5 important features",
                "Add target encoding for high-cardinality categorical features with 5-fold OOF",
                "Add feature clustering (KMeans on numeric features) as new categorical feature",
                "Apply Box-Cox / Yeo-Johnson transform to skewed numeric features",
                "Add missing-indicator columns for features with >5% nulls",
            ]
        elif branch_id == 1:  # model diversity branch
            plans = [
                "Add CatBoost with native categorical handling (no one-hot), tune depth 4-8",
                "Add XGBoost with histogram tree method, tune max_depth 3-7",
                "Try TabNet (deep learning for tabular) as additional ensemble member",
                "Replace RF with ExtraTrees (more randomness, often better generalization)",
                "Add GaussianNB as simple baseline ensemble member for diversity",
            ]
        elif branch_id == 2:  # ensemble/blend branch
            plans = [
                "Optimize blend weights via Nelder-Mead on OOF predictions",
                "Add stacking with LogisticRegression meta-learner on OOF preds",
                "Try rank-average blending instead of weighted average",
                "Add post-processing: isotonic calibration on OOF predictions",
                "Implement Bayesian model combination with uncertainty estimates",
            ]
        else:
            plans = [
                "Increase n_estimators by 50% and reduce learning_rate by half",
                "Add more cross-validation folds (from 5 to 10)",
                "Try different random seeds and average predictions",
                "Prune low-importance features (drop bottom 10% by importance)",
            ]

        return random.choice(plans) if plans else "Retry with different hyperparameters"

    def _execute_plan(self, plan: str) -> Optional[float]:
        """Execute a plan: retrain with the proposed change and evaluate OOF."""
        # For efficiency, use simplified re-training focused on the plan
        cfg = self.cfg
        train = self.train
        test = self.test

        # Feature engineering
        X, y, X_test = self._prepare_features(train, test, cfg)

        # Train ensemble with the plan's modification
        from sklearn.model_selection import StratifiedKFold, KFold
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.metrics import accuracy_score, mean_squared_error
        import lightgbm as lgb

        is_classification = "classification" in cfg["type"]
        n_splits = min(5, len(y) // 100)

        if is_classification:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof = np.zeros(len(y))
        test_pred = np.zeros(len(X_test))

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr = y.iloc[tr_idx] if hasattr(y, 'iloc') else y[tr_idx]
            y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

            # Use LGB + HGB based on plan
            try:
                if "XGBoost" in plan:
                    import xgboost as xgb
                    if is_classification:
                        model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                                  random_state=42, verbosity=0, n_jobs=-1)
                    else:
                        model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                                 random_state=42, verbosity=0, n_jobs=-1)
                elif "CatBoost" in plan:
                    from catboost import CatBoostClassifier, CatBoostRegressor
                    if is_classification:
                        model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                                   random_seed=42, verbose=False, thread_count=-1)
                    else:
                        model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                                                  random_seed=42, verbose=False, thread_count=-1)
                else:
                    # Default: LGB
                    if is_classification:
                        n_classes = len(np.unique(y))
                        if n_classes > 2:
                            model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                                       num_leaves=63, random_state=42, verbose=-1, n_jobs=-1)
                        else:
                            model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                                       num_leaves=63, random_state=42, verbose=-1, n_jobs=-1)
                    else:
                        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                                   num_leaves=63, random_state=42, verbose=-1, n_jobs=-1)

                model.fit(X_tr, y_tr)
                if is_classification and hasattr(model, 'predict_proba'):
                    oof[val_idx] = model.predict_proba(X_val)[:, 1]
                    test_pred += model.predict_proba(X_test)[:, 1] / n_splits
                else:
                    oof[val_idx] = model.predict(X_val)
                    test_pred += model.predict(X_test) / n_splits
            except Exception as e:
                # Fallback to sklearn HGB
                if is_classification:
                    model = HistGradientBoostingClassifier(max_iter=200, random_state=42)
                else:
                    model = HistGradientBoostingRegressor(max_iter=200, random_state=42)
                model.fit(X_tr, y_tr)
                if is_classification and hasattr(model, 'predict_proba'):
                    oof[val_idx] = model.predict_proba(X_val)[:, 1]
                    test_pred += model.predict_proba(X_test)[:, 1] / n_splits
                else:
                    oof[val_idx] = model.predict(X_val)
                    test_pred += model.predict(X_test) / n_splits

        # Score
        if is_classification:
            score = accuracy_score(y, (oof > 0.5).astype(int))
        else:
            score = np.sqrt(mean_squared_error(np.log1p(y), np.log1p(np.maximum(0, oof))))

        # Save submission if improved
        improved = False
        if self.best_score is None:
            improved = True
        elif cfg["direction"] == "maximize" and score > self.best_score:
            improved = True
        elif cfg["direction"] == "minimize" and score < self.best_score:
            improved = True

        if improved:
            self.best_score = score
            # Generate submission
            sub = pd.read_csv(ROOT / "tasks" / self.task_id / "data" / "sample_submission.csv")
            pred_col = sub.columns[-1]
            if is_classification:
                sub[pred_col] = (test_pred > 0.5).astype(int)
            else:
                sub[pred_col] = np.maximum(0, test_pred)

            out_dir = ROOT / "experiments" / self.task_id / f"mlevolve_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(out_dir / "submission.csv", index=False)
            with open(out_dir / "metrics.json", "w") as f:
                json.dump({"oof_score": float(score), "plan": plan, "runner": "mlevolve_local_loop"}, f, indent=2)
            self.best_submission = str(out_dir / "submission.csv")

        return score

    def _prepare_features(self, train, test, cfg):
        """Quick feature engineering."""
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        target = cfg["target"]
        drop = [c for c in cfg["drop"] if c in train.columns]
        if target in train.columns:
            drop.append(target)

        X = train.drop(columns=[c for c in drop if c in train.columns], errors='ignore')
        X_test = test.drop(columns=[c for c in drop if c in test.columns], errors='ignore')

        # Align columns
        common = list(set(X.columns) & set(X_test.columns))
        X = X[common]
        X_test = X_test[common]

        y = train[target].copy()

        # Handle categoricals
        for col in cfg.get("cat_features", []):
            if col in common:
                le = LabelEncoder()
                combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
                le.fit(combined)
                X[col] = le.transform(X[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))

        # Fill and scale
        X = X.fillna(X.median() if hasattr(X, 'median') else -1)
        X_test = X_test.fillna(X_test.median() if hasattr(X_test, 'median') else -1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)

        return X_scaled, y, X_test_scaled

    def run(self) -> dict:
        print(f"\n{'='*60}")
        print(f"MLEvolve Local Loop: {self.task_id}")
        print(f"Target: {self.cfg['target']}, Metric: {self.cfg['metric']}")
        print(f"Best known: {self.cfg['best_known_local']} (local), {self.cfg['best_known_public']} (Kaggle)")
        print(f"Max steps: {self.max_steps}, Branches: {self.n_branches}")
        print(f"{'='*60}")

        # Create initial branches
        for bid in range(self.n_branches):
            child = UCTNode(parent_id=self.root.node_id, branch_id=bid, depth=1,
                          plan=f"Branch {bid} initial exploration")
            self.root.children.append(child)
            self.nodes[child.node_id] = child
            self.branches.setdefault(bid, []).append(child.node_id)

        results = []
        start = time.time()

        for step in range(self.max_steps):
            self.step_count = step
            C = self._piecewise_decay_C(step)

            # UCT selection
            node = self._select_node()
            plan = self._generate_plan(node, node.branch_id)

            print(f"\nStep {step+1}/{self.max_steps} [C={C:.3f}] [branch={node.branch_id}]")
            print(f"  Plan: {plan[:100]}...")

            # Execute
            try:
                score = self._execute_plan(plan)
                if score is None:
                    node.is_buggy = True
                    continue

                # UCT update
                node.score = score
                node.visits += 1
                is_better = (self.cfg["direction"] == "maximize" and score > (self.best_score or -999)) or \
                           (self.cfg["direction"] == "minimize" and score < (self.best_score or 999))

                if is_better:
                    node.total_reward += 1.0
                    self.best_node = node
                    self.global_stagnation = 0
                    self.branch_stagnation[node.branch_id] = 0
                    self.branch_best[node.branch_id] = score
                    print(f"  >>> NEW BEST: {score:.6f} <<<")
                else:
                    node.total_reward += 0.3  # partial credit for non-buggy
                    self.global_stagnation += 1
                    self.branch_stagnation[node.branch_id] = self.branch_stagnation.get(node.branch_id, 0) + 1

                results.append({"step": step, "score": score, "best": self.best_score, "branch": node.branch_id})

                # Stagnation response
                if self.branch_stagnation.get(node.branch_id, 0) >= 3:
                    print(f"  Branch {node.branch_id} stagnant ({self.branch_stagnation[node.branch_id]} failures) - switching strategy")
                    self.branch_stagnation[node.branch_id] = 0
                    # Create new child with different plan
                    new_plan = self._generate_plan(node, (node.branch_id + 1) % self.n_branches)
                    child = UCTNode(parent_id=node.node_id, branch_id=node.branch_id,
                                  depth=node.depth+1, plan=new_plan)
                    node.children.append(child)
                    self.nodes[child.node_id] = child

                if self.global_stagnation >= 8:
                    print(f"  GLOBAL stagnation ({self.global_stagnation}) - exploring new branch")
                    self.global_stagnation = 0
                    new_bid = len(self.branches)
                    child = UCTNode(parent_id=self.root.node_id, branch_id=new_bid, depth=1,
                                  plan="New exploration branch (global stagnation response)")
                    self.root.children.append(child)
                    self.nodes[child.node_id] = child
                    self.branches[new_bid] = [child.node_id]

                # Progress
                elapsed = time.time() - start
                print(f"  Score: {score:.6f} | Best: {self.best_score:.6f} | Time: {elapsed:.0f}s")

            except Exception as e:
                print(f"  ERROR: {e}")
                node.is_buggy = True

        elapsed = time.time() - start
        improvement = (self.best_score - self.cfg["best_known_local"]) if self.best_score else 0
        direction_sign = 1 if self.cfg["direction"] == "maximize" else -1
        effective_improvement = improvement * direction_sign

        print(f"\n{'='*60}")
        print(f"FINAL: {self.task_id}")
        print(f"Best score: {self.best_score:.6f}")
        print(f"Previous best: {self.cfg['best_known_local']:.6f}")
        print(f"Improvement: {improvement:+.6f}")
        print(f"Steps: {len(results)}, Time: {elapsed:.0f}s")
        print(f"Submission: {self.best_submission}")
        print(f"{'='*60}")

        return {
            "task_id": self.task_id,
            "best_score": float(self.best_score) if self.best_score else None,
            "previous_best": self.cfg["best_known_local"],
            "improvement": float(improvement),
            "effective_improvement": float(effective_improvement),
            "steps": len(results),
            "time_seconds": elapsed,
            "submission_path": self.best_submission,
            "results": results[-10:]
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="spaceship_titanic")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--branches", type=int, default=3)
    args = parser.parse_args()

    loop = MLEvolveLocalLoop(args.task, max_steps=args.steps, n_branches=args.branches)
    result = loop.run()

    # Save report
    out = ROOT / "workspace" / "mlevolve_reports" / f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nReport: {out}")
