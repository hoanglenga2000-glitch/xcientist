#!/usr/bin/env python3
"""Reformat runner submissions to MLE-Bench expected format and grade them."""
import os, sys, json, pandas as pd, numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GRADE_DIR = SCRIPT_DIR / "mlebench_grading"
os.makedirs(GRADE_DIR, exist_ok=True)

# Competition configs
COMPS = {
    "spaceship-titanic": {
        "id_col": "PassengerId", "target_col": "Transported",
        "is_clf": True, "convert": "bool",  # probabilities → True/False
    },
    "tabular-playground-series-dec-2021": {
        "id_col": "Id", "target_col": "Cover_Type",
        "is_clf": True, "convert": "class",  # probabilities/values → class labels
    },
    "tabular-playground-series-may-2022": {
        "id_col": "id", "target_col": "target",
        "is_clf": True, "convert": "prob",  # keep as probabilities (AUROC)
    },
}

def load_server_predictions(comp_id):
    """Load submissions downloaded from server."""
    comp_dir = GRADE_DIR / comp_id
    test = pd.read_csv(comp_dir / "test.csv")

    submissions = {}
    for seed in [42, 43, 44]:
        sub_path = comp_dir / f"submission_s{seed}.csv"
        if sub_path.exists():
            submissions[seed] = pd.read_csv(sub_path)

    return test, submissions

def reformat(comp_id, test, sub_df, config):
    """Reformat submission to match MLE-Bench expected format."""
    id_col = config["id_col"]
    target_col = config["target_col"]
    conv = config["convert"]

    # Get IDs from test.csv
    test_ids = test[id_col].values

    # The runner's predictions are in order of the test rows
    n_preds = len(sub_df)
    n_test = len(test_ids)

    if n_preds != n_test:
        print(f"  WARNING: {n_preds} predictions vs {n_test} test rows")
        # Use min length
        n = min(n_preds, n_test)
        test_ids = test_ids[:n]
        preds = sub_df["prediction"].values[:n]
    else:
        preds = sub_df["prediction"].values

    if conv == "bool":
        # Convert probabilities to boolean
        preds = (preds > 0.5)
    elif conv == "class":
        # Convert to integer class labels
        preds = np.round(preds).astype(int)
        preds = np.clip(preds, 1, 7)  # Cover_Type is 1-7
    # else "prob": keep as-is

    new_sub = pd.DataFrame({id_col: test_ids, target_col: preds})
    return new_sub

def main():
    results = {}

    for comp_id, config in COMPS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {comp_id}")
        print(f"{'='*60}")

        test, submissions = load_server_predictions(comp_id)

        if not submissions:
            print(f"  No submissions found for {comp_id}")
            continue

        comp_dir = GRADE_DIR / comp_id
        best_seed = None
        best_cv = -1

        for seed in sorted(submissions.keys()):
            sub_df = submissions[seed]
            new_sub = reformat(comp_id, test, sub_df, config)

            # Save reformatted
            out_path = comp_dir / f"mlebench_submission_s{seed}.csv"
            new_sub.to_csv(out_path, index=False)
            print(f"  Reformatted s{seed}: {len(new_sub)} rows → {out_path.name}")

            # Also load CV score from server results
            result_path = comp_dir / f"result_s{seed}.json"
            # (results are on server, skip for now)

        # Use seed 42 as default submission
        default_sub = comp_dir / "mlebench_submission_s42.csv"
        if default_sub.exists():
            print(f"  Default submission: {default_sub}")

            # Try grading with mlebench
            try:
                from mlebench.registry import registry
                from mlebench.grade import grade_csv

                comp = registry.get_competition(comp_id)
                report = grade_csv(default_sub, comp)
                results[comp_id] = {
                    "score": report.score,
                    "any_medal": report.any_medal,
                    "gold": report.gold_medal,
                    "silver": report.silver_medal,
                    "bronze": report.bronze_medal,
                    "above_median": report.above_median,
                    "valid": report.valid_submission,
                }
                print(f"  SCORE: {report.score}")
                print(f"  MEDAL: gold={report.gold_medal} silver={report.silver_medal} bronze={report.bronze_medal}")
                print(f"  ABOVE MEDIAN: {report.above_median}")
            except Exception as e:
                print(f"  GRADE ERROR: {e}")
                results[comp_id] = {"error": str(e)}

    # Save summary
    summary_path = GRADE_DIR / "grading_results.json"
    json.dump(results, open(summary_path, "w"), indent=2, default=str)
    print(f"\nResults saved to {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("GRADING SUMMARY")
    print(f"{'='*60}")
    for comp_id, r in results.items():
        if "error" in r:
            print(f"  {comp_id}: ERROR — {r['error']}")
        else:
            medal = "GOLD" if r["gold"] else ("SILVER" if r["silver"] else ("BRONZE" if r["bronze"] else "NONE"))
            print(f"  {comp_id}: score={r['score']:.4f} medal={medal} above_median={r['above_median']}")

if __name__ == "__main__":
    main()
