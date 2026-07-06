from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "https://raw.githubusercontent.com/zeyongj/House-Prices-Advanced-Regression-Techniques/master"
FILES = {
    "train.csv": f"{BASE_URL}/train.csv",
    "test.csv": f"{BASE_URL}/test.csv",
    "sample_submission.csv": f"{BASE_URL}/sample_submission.csv",
    "data_description.txt": f"{BASE_URL}/data_description.txt",
}


def main() -> None:
    task_dir = Path("tasks/house_prices")
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():
        target = data_dir / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"exists: {target}")
            continue
        print(f"download: {url}")
        urlretrieve(url, target)
        print(f"written: {target} ({target.stat().st_size} bytes)")

    description_path = data_dir / "data_description.txt"
    overview_path = task_dir / "overview.txt"
    overview_lines = [
        "House Prices - Advanced Regression Techniques",
        "",
        "Goal: predict the final SalePrice for each home in the test set.",
        "Metric: Root Mean Squared Logarithmic Error (RMSLE).",
        "Files: train.csv, test.csv, sample_submission.csv.",
        "",
        "Data description excerpt:",
        "",
    ]
    if description_path.exists():
        overview_lines.extend(description_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:120])
    overview_path.write_text("\n".join(overview_lines), encoding="utf-8")
    print(f"written: {overview_path} ({overview_path.stat().st_size} bytes)")

    source_note = data_dir / "DATA_SOURCE.md"
    source_note.write_text(
        "\n".join(
            [
                "# House Prices Data Source",
                "",
                "Kaggle API credentials and Kaggle CLI are not configured on this local machine yet.",
                "For local transferability testing, these files use a public mirror of the Kaggle House Prices competition data.",
                "",
                "Official competition:",
                "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
                "",
                "Mirror files:",
                *[f"- {name}: {url}" for name, url in FILES.items()],
                "",
                "When Kaggle credentials are added, replace this step with official Kaggle API download and keep the same file names.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"written: {source_note} ({source_note.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
