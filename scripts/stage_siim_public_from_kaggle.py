#!/usr/bin/env python3
"""Resume the SIIM public competition files through the authenticated Kaggle API."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import quote, urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "siim-isic-melanoma-classification"
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / COMPETITION
)
DEFAULT_INVENTORY = DEFAULT_ROOT / "public_staging_inventory.json"
DEFAULT_DESTINATION = DEFAULT_ROOT / "prepared" / "public"
DEFAULT_HEARTBEAT = DEFAULT_ROOT / "public_staging_kaggle_heartbeat.json"
DEFAULT_REPORT = DEFAULT_ROOT / "public_staging_report.json"
class KaggleRateLimitError(RuntimeError):
    def __init__(self, retry_after_seconds: int):
        super().__init__("Kaggle API rate limit")
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class Entry:
    name: str
    size: int


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def safe_target(destination: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe inventory path: {name}")
    allowed_top_level = {"train.csv", "test.csv", "sample_submission.csv"}
    allowed_image_prefixes = {
        ("jpeg", "train"),
        ("jpeg", "test"),
    }
    parts = relative.parts
    allowed = name in allowed_top_level or (
        len(parts) == 3
        and tuple(parts[:2]) in allowed_image_prefixes
        and parts[2].lower().endswith(".jpg")
    )
    if not allowed:
        raise ValueError(f"Non-public SIIM inventory path: {name}")
    # PurePosixPath validation above excludes absolute and parent traversal.
    # Avoid filesystem-backed Path.resolve() for every one of the 33k entries.
    root = destination if destination.is_absolute() else destination.resolve()
    return root.joinpath(*parts)


def load_inventory(path: Path, destination: Path) -> tuple[dict[str, Any], list[Entry]]:
    inventory = read_json(path)
    if inventory.get("schema") != "evomind.siim.public_staging_inventory.v1":
        raise RuntimeError("Unexpected SIIM public inventory schema")
    if inventory.get("competition_id") != COMPETITION:
        raise RuntimeError("Unexpected SIIM competition inventory")
    entries = [
        Entry(str(item["path"]), int(item["size"]))
        for item in inventory.get("entries", [])
    ]
    if len(entries) != int(inventory.get("file_count", -1)):
        raise RuntimeError("SIIM inventory file count mismatch")
    if sum(entry.size for entry in entries) != int(inventory.get("total_bytes", -1)):
        raise RuntimeError("SIIM inventory byte count mismatch")
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise RuntimeError("SIIM inventory contains duplicate paths")
    for entry in entries:
        safe_target(destination, entry.name)
    return inventory, entries


def scan_entries(destination: Path, entries: Sequence[Entry]) -> dict[str, Any]:
    completed_files = 0
    completed_bytes = 0
    missing: list[Entry] = []
    wrong_size: list[dict[str, Any]] = []
    for entry in entries:
        target = safe_target(destination, entry.name)
        if target.is_file():
            actual_size = target.stat().st_size
            if actual_size == entry.size:
                completed_files += 1
                completed_bytes += actual_size
                continue
            wrong_size.append(
                {"path": entry.name, "expected_bytes": entry.size, "actual_bytes": actual_size}
            )
        missing.append(entry)
    return {
        "completed_files": completed_files,
        "completed_bytes": completed_bytes,
        "missing": missing,
        "wrong_size": wrong_size,
    }


def competition_download_file_bounded(
    competition: str,
    file_name: str,
    target_directory: Path,
    *,
    force: bool,
    quiet: bool,
    request_timeout_seconds: int,
    proxy_url: str | None,
) -> Path:
    """Download one Kaggle file with a bounded connect/read timeout.

    The generated Kaggle client follows the redirect inside urllib3 and can
    retain thousands of handles while hiding HTTP 429 responses. This transport
    requests the authenticated API redirect explicitly, handles rate limits,
    then streams the signed storage URL with a real read timeout.
    """
    del quiet  # The resilient transport is deliberately non-verbose.
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    headers = {"User-Agent": "evomind-siim-stager/2.0"}
    auth: tuple[str, str] | None = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif username and key:
        auth = (username, key)
    else:
        raise RuntimeError("Kaggle secure runtime credential is missing")

    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    endpoint = (
        "https://www.kaggle.com/api/v1/competitions/data/download/"
        # Kaggle treats the competition file name as one path segment. Keeping
        # inventory slashes literal yields HTTP 404 for nested JPEG entries;
        # encode them as %2F so the API resolves the exact file key.
        f"{quote(competition, safe='')}/{quote(file_name, safe='')}"
    )
    try:
        with requests.get(
            endpoint,
            headers=headers,
            auth=auth,
            allow_redirects=False,
            stream=True,
            timeout=(30, request_timeout_seconds),
            proxies=proxies,
        ) as api_response:
            if api_response.status_code == 429:
                raw_retry = api_response.headers.get("Retry-After", "")
                retry_after = int(raw_retry) if raw_retry.isdigit() else 300
                raise KaggleRateLimitError(max(30, min(retry_after, 1800)))
            if api_response.status_code not in {301, 302, 303, 307, 308}:
                raise RuntimeError(
                    f"Kaggle download API returned HTTP {api_response.status_code}"
                )
            signed_url = api_response.headers.get("Location", "")
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Kaggle download API transport failed: {type(exc).__name__}"
        ) from None

    parsed_signed_url = urlparse(signed_url)
    if parsed_signed_url.scheme != "https" or not parsed_signed_url.hostname:
        raise RuntimeError(f"Kaggle response has no valid download redirect for {file_name}")

    clean_url = signed_url.split("?", 1)[0]
    output_name = clean_url.rsplit("/", 1)[-1]
    expected_names = {Path(file_name).name, Path(file_name).name + ".zip"}
    if output_name not in expected_names:
        raise RuntimeError(f"Unexpected Kaggle download filename for {file_name}")
    outfile = target_directory / output_name
    temporary = outfile.with_suffix(outfile.suffix + f".{os.getpid()}.download.tmp")
    try:
        # Read the signed storage URL through requests instead of Kaggle's
        # generated urllib3 stream. requests enforces the read timeout for each
        # chunk and honors the explicit loopback proxy.
        with requests.get(
            signed_url,
            stream=True,
            timeout=(30, request_timeout_seconds),
            proxies=proxies,
            headers={"Accept-Encoding": "identity", "User-Agent": "evomind-siim-stager/2.0"},
        ) as download:
            download.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in download.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"Empty Kaggle download for {file_name}")
        if outfile.exists() and not force:
            outfile.unlink()
        os.replace(temporary, outfile)
        return outfile
    except requests.RequestException as exc:
        # Never persist a requests exception string: it can contain the signed
        # storage URL query. The exception class is sufficient for retry logs.
        raise RuntimeError(
            f"Kaggle signed download transport failed: {type(exc).__name__}"
        ) from None
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_download_archive(archive: Path, target: Path, expected_size: int) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"Kaggle download archive is missing: {archive}")
    with zipfile.ZipFile(archive) as handle:
        members = [item for item in handle.infolist() if not item.is_dir()]
        member = next(
            (item for item in members if PurePosixPath(item.filename).name == target.name),
            None,
        )
        if member is None or member.file_size != expected_size:
            raise RuntimeError(
                f"Unexpected Kaggle archive member for {target.name}: "
                f"{[(item.filename, item.file_size) for item in members]}"
            )
        temporary = target.with_suffix(target.suffix + f".{os.getpid()}.kaggle.tmp")
        try:
            with handle.open(member) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            if temporary.stat().st_size != expected_size:
                raise RuntimeError(f"Extracted size mismatch for {target.name}")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()


def download_entry(
    destination: Path,
    competition: str,
    entry: Entry,
    retry_attempts: int,
    request_timeout_seconds: int,
    proxy_url: str | None,
) -> dict[str, Any]:
    target = safe_target(destination, entry.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.with_name(target.name + ".zip")
    before = target.stat().st_size if target.is_file() else 0
    last_error_type: str | None = None
    for attempt in range(1, retry_attempts + 1):
        try:
            force = attempt > 1
            competition_download_file_bounded(
                competition,
                entry.name,
                target.parent,
                force=force,
                quiet=True,
                request_timeout_seconds=request_timeout_seconds,
                proxy_url=proxy_url,
            )
            if not target.is_file() or target.stat().st_size != entry.size:
                extract_download_archive(archive, target, entry.size)
            actual = target.stat().st_size if target.is_file() else -1
            if actual != entry.size:
                raise RuntimeError(
                    f"Downloaded size mismatch for {entry.name}: {actual} != {entry.size}"
                )
            if archive.is_file():
                archive.unlink()
            return {
                "path": entry.name,
                "bytes": actual,
                "downloaded_bytes": max(0, actual - min(before, actual)),
                "attempts": attempt,
            }
        except Exception as exc:  # Transport errors are reduced to safe types.
            last_error_type = type(exc).__name__
            if attempt < retry_attempts and archive.is_file():
                archive.unlink()
            if attempt < retry_attempts:
                if isinstance(exc, KaggleRateLimitError):
                    time.sleep(exc.retry_after_seconds)
                else:
                    time.sleep(min(30, 2**attempt))
    raise RuntimeError(
        f"Failed {entry.name} after {retry_attempts} attempts; "
        f"last_error_type={last_error_type}"
    )


def progress_payload(
    *,
    status: str,
    destination: Path,
    inventory: dict[str, Any],
    completed_files: int,
    completed_bytes: int,
    initial_completed_files: int,
    initial_completed_bytes: int,
    attempted_files: int,
    errors: Sequence[dict[str, Any]],
    workers: int,
    max_in_flight: int,
    request_timeout_seconds: int,
    proxy_url: str | None,
    started_at: float,
) -> dict[str, Any]:
    return {
        "schema": "evomind.siim.public_staging_kaggle_heartbeat.v1",
        "created_at": now_iso(),
        "status": status,
        "pid": os.getpid(),
        "source": "authenticated_kaggle_competition_api",
        "competition_id": COMPETITION,
        "destination": str(destination.resolve()),
        "workers": workers,
        "max_in_flight": max_in_flight,
        "request_timeout_seconds": request_timeout_seconds,
        "network_route": "explicit_loopback_proxy" if proxy_url else "system_default",
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "total_files": int(inventory["file_count"]),
        "total_bytes": int(inventory["total_bytes"]),
        "completed_files": completed_files,
        "completed_bytes": completed_bytes,
        "initial_completed_files": initial_completed_files,
        "initial_completed_bytes": initial_completed_bytes,
        "attempted_files": attempted_files,
        "downloaded_files_this_run": completed_files - initial_completed_files,
        "downloaded_bytes_this_run": completed_bytes - initial_completed_bytes,
        "errors": list(errors),
        "private_paths_requested": False,
        "remote_writes_performed": False,
        "process_signals_sent": 0,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--heartbeat", type=Path, default=DEFAULT_HEARTBEAT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=180,
        help="Bound each Kaggle streaming request; retries resume at file granularity.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Maximum submitted futures; zero uses twice the worker count.",
    )
    parser.add_argument(
        "--proxy-url",
        default=os.environ.get("KAGGLE_PROXY_URL", ""),
        help="Optional explicit HTTP proxy for Kaggle urllib3 traffic.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Bounded smoke limit; zero resumes every missing public file.",
    )
    parser.add_argument("--scan-only", action="store_true")
    return parser


def bounded_future_results(
    executor: ThreadPoolExecutor,
    entries: Sequence[Entry],
    *,
    max_in_flight: int,
    submit: Any,
) -> Any:
    """Yield completed futures while keeping only a small window submitted."""
    iterator = iter(entries)
    pending: dict[Future[Any], Entry] = {}

    def fill_window() -> None:
        while len(pending) < max_in_flight:
            try:
                entry = next(iterator)
            except StopIteration:
                break
            pending[submit(executor, entry)] = entry

    fill_window()
    while pending:
        completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in completed:
            entry = pending.pop(future)
            yield future, entry
        fill_window()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.competition != COMPETITION:
        raise ValueError("Only the frozen SIIM public competition is supported")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if (
        args.retry_attempts < 1
        or args.progress_every < 1
        or args.max_files < 0
        or args.request_timeout_seconds < 30
        or args.max_in_flight < 0
    ):
        raise ValueError("Invalid staging retry/progress limit")
    max_in_flight = args.max_in_flight or args.workers * 2
    if max_in_flight < args.workers or max_in_flight > 64:
        raise ValueError("max_in_flight must be between workers and 64")
    proxy_url = args.proxy_url.strip() or None
    if proxy_url and proxy_url not in {
        "http://127.0.0.1:7892",
        "http://localhost:7892",
    }:
        raise ValueError("proxy_url must be the approved local HTTP proxy")
    destination = args.destination.resolve()
    inventory, entries = load_inventory(args.inventory.resolve(), destination)
    started_at = time.monotonic()
    initial = scan_entries(destination, entries)
    initial_files = int(initial["completed_files"])
    initial_bytes = int(initial["completed_bytes"])
    missing = list(initial["missing"])
    selected = missing[: args.max_files] if args.max_files else missing
    initial_payload = progress_payload(
        status="scan_only" if args.scan_only else "running",
        destination=destination,
        inventory=inventory,
        completed_files=initial_files,
        completed_bytes=initial_bytes,
        initial_completed_files=initial_files,
        initial_completed_bytes=initial_bytes,
        attempted_files=0,
        errors=[],
        workers=args.workers,
        max_in_flight=max_in_flight,
        request_timeout_seconds=args.request_timeout_seconds,
        proxy_url=proxy_url,
        started_at=started_at,
    )
    initial_payload["missing_files"] = len(missing)
    initial_payload["wrong_size_files"] = len(initial["wrong_size"])
    initial_payload["selected_files_this_run"] = len(selected)
    write_json_atomic(args.heartbeat.resolve(), initial_payload)
    if args.scan_only:
        print(json.dumps(initial_payload, ensure_ascii=False, indent=2))
        return 0

    errors: list[dict[str, Any]] = []
    attempted = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        def submit_entry(pool: ThreadPoolExecutor, entry: Entry) -> Future[Any]:
            return pool.submit(
                download_entry,
                destination,
                args.competition,
                entry,
                args.retry_attempts,
                args.request_timeout_seconds,
                proxy_url,
            )

        for future, entry in bounded_future_results(
            executor,
            selected,
            max_in_flight=max_in_flight,
            submit=submit_entry,
        ):
            attempted += 1
            try:
                future.result()
            except Exception as exc:
                errors.append({"path": entry.name, "error": str(exc)})
            if attempted % args.progress_every == 0 or attempted == len(selected):
                snapshot = scan_entries(destination, entries)
                payload = progress_payload(
                    status="running",
                    destination=destination,
                    inventory=inventory,
                    completed_files=int(snapshot["completed_files"]),
                    completed_bytes=int(snapshot["completed_bytes"]),
                    initial_completed_files=initial_files,
                    initial_completed_bytes=initial_bytes,
                    attempted_files=attempted,
                    errors=errors,
                    workers=args.workers,
                    max_in_flight=max_in_flight,
                    request_timeout_seconds=args.request_timeout_seconds,
                    proxy_url=proxy_url,
                    started_at=started_at,
                )
                payload["selected_files_this_run"] = len(selected)
                write_json_atomic(args.heartbeat.resolve(), payload)

    final = scan_entries(destination, entries)
    complete = (
        int(final["completed_files"]) == int(inventory["file_count"])
        and int(final["completed_bytes"]) == int(inventory["total_bytes"])
        and not final["missing"]
        and not errors
    )
    bounded = bool(args.max_files and not complete)
    final_status = (
        "size_verified_complete"
        if complete
        else "bounded_smoke_complete"
        if bounded and not errors
        else "incomplete_with_errors"
    )
    report = {
        "schema": "evomind.siim.public_staging.v1",
        "created_at": now_iso(),
        "status": final_status,
        "source": "authenticated_kaggle_competition_api",
        "competition_id": COMPETITION,
        "destination": str(destination),
        "total_files": int(inventory["file_count"]),
        "total_bytes": int(inventory["total_bytes"]),
        "completed_files": int(final["completed_files"]),
        "completed_bytes": int(final["completed_bytes"]),
        "inventory_manifest_sha256": inventory["manifest_sha256"],
        "initial_completed_files": initial_files,
        "initial_completed_bytes": initial_bytes,
        "attempted_files": attempted,
        "remaining_files": len(final["missing"]),
        "errors": errors,
        "private_paths_requested": False,
        "remote_writes_performed": False,
        "process_signals_sent": 0,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json_atomic(args.report.resolve(), report)
    heartbeat = progress_payload(
        status=final_status,
        destination=destination,
        inventory=inventory,
        completed_files=int(final["completed_files"]),
        completed_bytes=int(final["completed_bytes"]),
        initial_completed_files=initial_files,
        initial_completed_bytes=initial_bytes,
        attempted_files=attempted,
        errors=errors,
        workers=args.workers,
        max_in_flight=max_in_flight,
        request_timeout_seconds=args.request_timeout_seconds,
        proxy_url=proxy_url,
        started_at=started_at,
    )
    heartbeat["remaining_files"] = len(final["missing"])
    write_json_atomic(args.heartbeat.resolve(), heartbeat)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if complete or (bounded and not errors) else 2


if __name__ == "__main__":
    raise SystemExit(main())
