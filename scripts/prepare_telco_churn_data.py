from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OFFICIAL_KAGGLE_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    task_dir = Path("tasks/telco_churn")
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / "raw_telco_customer_churn.csv"
    if raw_path.exists() and raw_path.stat().st_size > 0:
        print(f"exists: {raw_path}")
    else:
        print(f"download: {RAW_URL}")
        urlretrieve(RAW_URL, raw_path)
        print(f"written: {raw_path} ({raw_path.stat().st_size} bytes)")

    data = pd.read_csv(raw_path)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    train, test_with_target = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["Churn"],
    )

    train = train.sort_values("customerID").reset_index(drop=True)
    test_with_target = test_with_target.sort_values("customerID").reset_index(drop=True)
    test = test_with_target.drop(columns=["Churn"])
    sample_submission = pd.DataFrame({"customerID": test_with_target["customerID"], "Churn": "No"})

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    holdout_path = data_dir / "holdout_labels.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample_submission.to_csv(sample_path, index=False)
    test_with_target[["customerID", "Churn"]].to_csv(holdout_path, index=False)

    overview_path = task_dir / "overview.txt"
    overview_path.write_text(
        "\n".join(
            [
                "Telco Customer Churn",
                "",
                "Goal: predict whether a telecom customer will churn.",
                "Business workflow: identify high-risk customers so retention teams can prioritize intervention.",
                "Target column: Churn (No/Yes).",
                "Metric: local stratified cross-validation accuracy, with macro F1 recorded for class-balance awareness.",
                "Files: train.csv, test.csv, sample_submission.csv, holdout_labels.csv.",
                "",
                "Local split:",
                f"- random_state: {RANDOM_STATE}",
                f"- test_size: {TEST_SIZE}",
                f"- train rows: {len(train)}",
                f"- test/submission rows: {len(test)}",
                "",
                "Production boundary:",
                "Kaggle API credentials are not configured on this machine, so this round uses a public mirror",
                "to complete the local closed loop. Official Kaggle download/submission remains behind a human gate.",
            ]
        ),
        encoding="utf-8",
    )

    source_note = data_dir / "DATA_SOURCE.md"
    source_note.write_text(
        "\n".join(
            [
                "# Telco Customer Churn Data Source",
                "",
                "Kaggle API credentials and Kaggle CLI are not configured on this local machine yet.",
                "For the real business workflow smoke test, the system uses a public mirror of the Kaggle Telco Customer Churn CSV.",
                "",
                "Official Kaggle dataset:",
                OFFICIAL_KAGGLE_URL,
                "",
                "Public CSV mirror used locally:",
                RAW_URL,
                "",
                "Generated local files:",
                "- train.csv: stratified local training split with Churn.",
                "- test.csv: stratified local test split without Churn, Kaggle-style.",
                "- sample_submission.csv: customerID and Churn columns.",
                "- holdout_labels.csv: retained only for offline business evaluation and audit, not used by the pipeline.",
                "",
                "When Kaggle credentials are added, replace this preparation step with official Kaggle API download and keep the same file names.",
            ]
        ),
        encoding="utf-8",
    )

    for path in [train_path, test_path, sample_path, holdout_path, overview_path, source_note]:
        print(f"written: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
