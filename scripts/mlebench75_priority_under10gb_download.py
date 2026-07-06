#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

BASE = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
RAW = BASE / "mlebench_raw_data"
REPORTS = BASE / "reports"
LOGS = BASE / "logs"
MAX_BYTES = 10 * 1024**3
MAX_WORKERS = 2
MAX_LIST_PAGES = 5
PAGE_SIZE = 100

SPLIT75 = [
    line.strip()
    for line in """3d-object-detection-for-autonomous-vehicles
AI4Code
aerial-cactus-identification
alaska2-image-steganalysis
aptos2019-blindness-detection
billion-word-imputation
bms-molecular-translation
cassava-leaf-disease-classification
cdiscount-image-classification-challenge
chaii-hindi-and-tamil-question-answering
champs-scalar-coupling
denoising-dirty-documents
detecting-insults-in-social-commentary
dog-breed-identification
dogs-vs-cats-redux-kernels-edition
facebook-recruiting-iii-keyword-extraction
freesound-audio-tagging-2019
google-quest-challenge
google-research-identify-contrails-reduce-global-warming
h-and-m-personalized-fashion-recommendations
herbarium-2020-fgvc7
herbarium-2021-fgvc8
herbarium-2022-fgvc9
histopathologic-cancer-detection
hms-harmful-brain-activity-classification
hotel-id-2021-fgvc8
hubmap-kidney-segmentation
icecube-neutrinos-in-deep-ice
imet-2020-fgvc7
inaturalist-2019-fgvc6
iwildcam-2019-fgvc6
iwildcam-2020-fgvc7
jigsaw-toxic-comment-classification-challenge
jigsaw-unintended-bias-in-toxicity-classification
kuzushiji-recognition
leaf-classification
learning-agency-lab-automated-essay-scoring-2
lmsys-chatbot-arena
mlsp-2013-birds
multi-modal-gesture-recognition
new-york-city-taxi-fare-prediction
nfl-player-contact-detection
nomad2018-predict-transparent-conductors
osic-pulmonary-fibrosis-progression
petfinder-pawpularity-score
plant-pathology-2020-fgvc7
plant-pathology-2021-fgvc8
predict-volcanic-eruptions-ingv-oe
random-acts-of-pizza
ranzcr-clip-catheter-line-classification
rsna-2022-cervical-spine-fracture-detection
rsna-breast-cancer-detection
rsna-miccai-brain-tumor-radiogenomic-classification
seti-breakthrough-listen
siim-covid19-detection
siim-isic-melanoma-classification
smartphone-decimeter-2022
spooky-author-identification
stanford-covid-vaccine
statoil-iceberg-classifier-challenge
tabular-playground-series-dec-2021
tabular-playground-series-may-2022
tensorflow-speech-recognition-challenge
tensorflow2-question-answering
text-normalization-challenge-english-language
text-normalization-challenge-russian-language
tgs-salt-identification-challenge
the-icml-2013-whale-challenge-right-whale-redux
tweet-sentiment-extraction
us-patent-phrase-to-phrase-matching
uw-madison-gi-tract-image-segmentation
ventilator-pressure-prediction
vesuvius-challenge-ink-detection
vinbigdata-chest-xray-abnormalities-detection
whale-categorization-playground""".splitlines()
    if line.strip()
]


def assert_workspace() -> None:
    if str(BASE) != "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra":
        raise RuntimeError(f"Refusing to run outside Jing workspace: {BASE}")
    for directory in (RAW, REPORTS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    os.chdir(BASE)


def has_useful_files(path: Path) -> bool:
    if not path.exists():
        return False
    files: list[str] = []
    complete_archives = 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            name = candidate.name.lower()
            files.append(name)
            if name.endswith(".zip"):
                marker = candidate.parent / f"{candidate.name}.extracted.ok"
                if marker.exists() or zipfile.is_zipfile(candidate):
                    complete_archives += 1
            elif name.endswith((".7z", ".tar.gz", ".bson")) and candidate.stat().st_size > 0:
                complete_archives += 1
            if len(files) > 5000:
                break
    if not files:
        return False
    train_like = any(name.startswith("train.") or name in {"train.csv", "training.csv"} for name in files)
    test_like = any(name.startswith("test.") or name in {"test.csv", "testing.csv"} for name in files)
    sample_like = any("sample" in name and "submission" in name for name in files)
    media_like = any(
        name.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".wav", ".mp3", ".json", ".parquet", ".npy", ".nii", ".dcm"))
        for name in files
    )
    return (
        (train_like and (test_like or sample_like))
        or (sample_like and media_like)
        or complete_archives > 0
        or (media_like and len(files) > 20)
    )


def extract_zip_archives(path: Path) -> list[str]:
    extracted: list[str] = []
    for archive in path.glob("*.zip"):
        marker = path / f"{archive.name}.extracted.ok"
        if marker.exists():
            continue
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(path)
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")
            extracted.append(archive.name)
        except Exception as exc:
            (path / f"{archive.name}.extract.error.txt").write_text(str(exc), encoding="utf-8")
    return extracted


def list_competition_files(api: KaggleApi, competition_id: str) -> tuple[list[dict], int | None, str]:
    try:
        files: list[dict] = []
        total = 0
        page_token: str | None = None
        seen_tokens: set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            response = api.competition_list_files(competition_id, page_token=page_token, page_size=PAGE_SIZE)
            for item in getattr(response, "files", []) or []:
                name = getattr(item, "name", "") or ""
                size = int(
                    getattr(item, "total_bytes", None)
                    or getattr(item, "totalBytes", None)
                    or getattr(item, "_total_bytes", None)
                    or 0
                )
                files.append({"name": name, "bytes": size})
                total += size
            next_token = (
                getattr(response, "next_page_token", None)
                or getattr(response, "nextPageToken", None)
                or getattr(response, "_next_page_token", None)
                or ""
            )
            if not next_token or next_token in seen_tokens:
                break
            if page_count >= MAX_LIST_PAGES:
                return files, None, f"paginated_unknown_after_{MAX_LIST_PAGES * PAGE_SIZE}_files"
            seen_tokens.add(str(next_token))
            page_token = str(next_token)
        return files, total, "ok"
    except Exception as exc:
        return [], None, f"list_error:{type(exc).__name__}:{exc}"


def classify_download_error(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "403" in text or "forbidden" in text:
        return "needs_api_account_rules_or_permission"
    if "404" in text or "not found" in text:
        return "not_found_or_unavailable"
    if "must accept" in text or ("rules" in text and "accept" in text):
        return "needs_api_account_rules_or_permission"
    return "download_failed"


def run_cli(args: list[str], timeout: int = 4 * 3600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=BASE, text=True, capture_output=True, timeout=timeout)


def download_file_by_file(competition_id: str, files: list[dict], target: Path) -> tuple[str, list[str], str, str]:
    downloaded: list[str] = []
    stdout_tail = ""
    stderr_tail = ""
    for item in files:
        file_name = item["name"]
        if not file_name:
            continue
        local_name = Path(file_name).name
        if (target / local_name).exists() or (target / file_name).exists():
            downloaded.append(file_name)
            continue
        proc = run_cli(["kaggle", "competitions", "download", "-c", competition_id, "-f", file_name, "-p", str(target)])
        stdout_tail = proc.stdout[-800:]
        stderr_tail = proc.stderr[-800:]
        if proc.returncode != 0:
            return classify_download_error(proc.stdout, proc.stderr), downloaded, stdout_tail, stderr_tail
        downloaded.append(file_name)
    return "downloaded_by_file" if downloaded else "downloaded_but_empty_file_list", downloaded, stdout_tail, stderr_tail


def download_one(plan: dict) -> dict:
    competition_id = plan["competition_id"]
    target = RAW / competition_id
    target.mkdir(parents=True, exist_ok=True)
    lock = target / ".priority_under10gb.lock"
    started = time.time()
    if has_useful_files(target):
        return {
            **plan,
            "status": "skipped_existing",
            "elapsed_sec": round(time.time() - started, 2),
            "useful_files_after": True,
            "extracted_archives": extract_zip_archives(target),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    if (target / ".fastlane.lock").exists() or lock.exists():
        return {
            **plan,
            "status": "locked_by_other_downloader",
            "elapsed_sec": round(time.time() - started, 2),
            "useful_files_after": has_useful_files(target),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    lock.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")
    stdout_tail = ""
    stderr_tail = ""
    extracted: list[str] = []
    file_downloads: list[str] = []
    try:
        proc = run_cli(["kaggle", "competitions", "download", "-c", competition_id, "-p", str(target)])
        stdout_tail = proc.stdout[-800:]
        stderr_tail = proc.stderr[-800:]
        if proc.returncode == 0:
            extracted = extract_zip_archives(target)
            status = "downloaded" if has_useful_files(target) else "downloaded_but_incomplete"
        else:
            status = classify_download_error(proc.stdout, proc.stderr)
            if status == "needs_api_account_rules_or_permission" and plan.get("file_count", 0) <= 30:
                fallback_status, file_downloads, stdout_tail, stderr_tail = download_file_by_file(competition_id, plan.get("files", []), target)
                status = fallback_status
                if status in {"downloaded_by_file", "downloaded_but_empty_file_list"}:
                    extracted = extract_zip_archives(target)
                    if not has_useful_files(target):
                        status = "downloaded_but_incomplete"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout_tail = (exc.stdout or "")[-800:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-800:] if isinstance(exc.stderr, str) else ""
    except Exception as exc:
        status = f"exception:{type(exc).__name__}"
        stderr_tail = repr(exc)[-800:]
    finally:
        try:
            lock.unlink()
        except OSError:
            pass
    return {
        **plan,
        "status": status,
        "elapsed_sec": round(time.time() - started, 2),
        "useful_files_after": has_useful_files(target),
        "extracted_archives": extracted,
        "file_downloads": file_downloads[:50],
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main() -> int:
    assert_workspace()
    api = KaggleApi()
    api.authenticate()
    inventory: list[dict] = []
    for index, competition_id in enumerate(SPLIT75, 1):
        files, total_bytes, list_status = list_competition_files(api, competition_id)
        target = RAW / competition_id
        has_data = has_useful_files(target)
        inventory.append(
            {
                "index": index,
                "competition_id": competition_id,
                "estimated_bytes": total_bytes,
                "estimated_gb": round(total_bytes / 1024**3, 3) if total_bytes is not None else None,
                "file_count": len(files),
                "files": files,
                "list_status": list_status,
                "has_data_before": has_data,
                "priority_class": "under_10gb" if total_bytes is not None and total_bytes <= MAX_BYTES else "over_10gb_or_unknown",
            }
        )
    inventory_path = REPORTS / "mlebench75_size_inventory_20260701.json"
    inventory_path.write_text(json.dumps({"schema": "jinghw.mlebench75.size_inventory.v1", "items": inventory}, indent=2), encoding="utf-8")

    candidates = [
        item
        for item in inventory
        if item["priority_class"] == "under_10gb" and not item["has_data_before"] and item["list_status"] == "ok"
    ]
    candidates.sort(key=lambda item: (item["estimated_bytes"] or 0, item["competition_id"]))

    results: list[dict] = []
    progress_path = REPORTS / "mlebench75_under10gb_priority_progress_20260701.json"
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, candidate): candidate["competition_id"] for candidate in candidates}
        for future in cf.as_completed(futures):
            record = future.result()
            results.append(record)
            payload = {
                "schema": "jinghw.mlebench75.under10gb_priority_download.v1",
                "base": str(BASE),
                "max_workers": MAX_WORKERS,
                "total_candidates": len(candidates),
                "completed": len(results),
                "counts": dict(Counter(item["status"] for item in results)),
                "results": results,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "competition_id": record["competition_id"],
                        "status": record["status"],
                        "estimated_gb": record.get("estimated_gb"),
                        "elapsed_sec": record["elapsed_sec"],
                        "useful_files_after": record["useful_files_after"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    final_path = REPORTS / "mlebench75_under10gb_priority_final_20260701.json"
    final_path.write_text(
        json.dumps(
            {
                "schema": "jinghw.mlebench75.under10gb_priority_download.v1",
                "base": str(BASE),
                "max_workers": MAX_WORKERS,
                "total_candidates": len(candidates),
                "completed": len(results),
                "counts": dict(Counter(item["status"] for item in results)),
                "results": results,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
