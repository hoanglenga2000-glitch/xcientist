from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


FILES = {
    "train.csv": "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv",
    "test.csv": "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/test.csv",
}


def main() -> None:
    data_dir = Path("tasks/titanic/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():
        target = data_dir / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"exists: {target}")
            continue
        print(f"download: {url}")
        urlretrieve(url, target)
        print(f"written: {target} ({target.stat().st_size} bytes)")

    sample_target = data_dir / "sample_submission.csv"
    if not sample_target.exists() or sample_target.stat().st_size == 0:
        import pandas as pd

        test = pd.read_csv(data_dir / "test.csv")
        sample = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": 0})
        sample.to_csv(sample_target, index=False)
        print(f"generated: {sample_target} ({sample_target.stat().st_size} bytes)")

    source_note = data_dir / "DATA_SOURCE.md"
    source_note.write_text(
        "\n".join(
            [
                "# Titanic Data Source",
                "",
                "Kaggle API credentials are not configured on this local machine yet.",
                "For local v1 testing, these files use a public mirror of the Kaggle Titanic competition data.",
                "",
                "Official competition:",
                "https://www.kaggle.com/competitions/titanic",
                "",
                "Mirror files:",
                *[f"- {name}: {url}" for name, url in FILES.items()],
                "- sample_submission.csv: generated from test.csv PassengerId with Kaggle-compatible columns.",
                "",
                "When Kaggle credentials are added, replace this step with official Kaggle API download and keep the same file names.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
