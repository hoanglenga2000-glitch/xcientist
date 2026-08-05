#!/usr/bin/env python3
"""Prepare the MLE-Bench NYC Taxi split without materialising the 55M-row frame.

The pinned upstream implementation reads the complete 5.7 GB ``train.csv``
and then calls ``train_test_split(..., test_size=9914, random_state=0)``.  That
requires far more memory than a normal WSL workstation.  This implementation
preserves the upstream semantics while bounding Python memory:

1. build the exact ``RandomState(0).permutation(n)`` inverse rank in a memmap;
2. parse the source in pandas chunks using MLE-Bench's ``round_trip`` float
   precision and the dtypes inferred by the upstream full-file read;
3. prefix every pandas-serialised row with its split rank;
4. use GNU sort as a disk-backed external sort and stream the sorted rows into
   the four upstream output CSVs;
5. verify all four official MLE-Bench MD5 values before atomically promoting
   the staging directory.

No Kaggle API, grader, campaign, GPU, or submission operation is performed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

COMPETITION_ID = "new-york-city-taxi-fare-prediction"
DEFAULT_TEST_SIZE = 9_914
DEFAULT_RANDOM_STATE = 0
EXPECTED_ZIP_MD5 = "fe01f4ab11ebb0ee2851754516883070"

SOURCE_COLUMNS = (
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
)
PUBLIC_TEST_COLUMNS = tuple(column for column in SOURCE_COLUMNS if column != "fare_amount")
SOURCE_DTYPES: Mapping[str, str] = {
    "key": "object",
    "fare_amount": "float64",
    "pickup_datetime": "object",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}

OFFICIAL_OUTPUT_MD5: Mapping[str, str] = {
    "public/labels.csv": "b1aa60e8d6ca817c18b23bd00e4a2500",
    "public/test.csv": "a7b9d9b7208e9b7cff6891cfd56a0053",
    "public/sample_submission.csv": "10c791174db974943c1c80f38fc86391",
    "private/test.csv": "f51d45e95bcd7565f1175206e604506e",
}

PINNED_RUNTIME: Mapping[str, str] = {
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "scikit-learn": "1.5.1",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_file(path: Path, algorithm: str = "md5", chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceScan:
    rows: int
    sha256: str
    bytes: int
    header: tuple[str, ...]


@dataclass(frozen=True)
class PrepareResult:
    prepared_dir: Path
    rows: int
    test_rows: int
    source_sha256: str
    rank_sha256: str
    output_md5: Mapping[str, str]
    elapsed_seconds: float
    report: Mapping[str, Any]


class JsonlLogger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: object) -> None:
        record = {"at_utc": _utc_now(), "event": event, **fields}
        line = _canonical_json(record)
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(
        {
            "created_at_utc": _utc_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
    )
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        existing = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else "<unreadable>"
        raise RuntimeError(f"another Taxi preparation lock exists at {path}: {existing}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def runtime_versions() -> dict[str, str]:
    import sklearn

    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "python": platform.python_version(),
    }


def validate_pinned_runtime() -> Mapping[str, str]:
    actual = runtime_versions()
    mismatches = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in PINNED_RUNTIME.items()
        if actual.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"pinned preparation runtime mismatch: {_canonical_json(mismatches)}")
    return actual


def scan_source_csv(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> SourceScan:
    """Count data rows and hash the source in one bounded-memory sequential pass."""

    digest = hashlib.sha256()
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header_bytes = handle.readline()
        if not header_bytes:
            raise RuntimeError(f"source CSV is empty: {path}")
        digest.update(header_bytes)
        try:
            header = tuple(header_bytes.rstrip(b"\r\n").decode("utf-8").split(","))
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"source CSV header is not UTF-8: {path}") from exc

        rows = 0
        saw_data = False
        last_byte = b""
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
            rows += chunk.count(b"\n")
            saw_data = True
            last_byte = chunk[-1:]
        if saw_data and last_byte != b"\n":
            rows += 1

    if header != SOURCE_COLUMNS:
        raise RuntimeError(f"unexpected Taxi source columns: expected {SOURCE_COLUMNS}, got {header}")
    if rows <= 0:
        raise RuntimeError(f"source has no data rows: {path}")
    return SourceScan(rows=rows, sha256=digest.hexdigest(), bytes=file_size, header=header)


def build_inverse_rank_memmap(
    rows: int,
    path: Path,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    assignment_chunk_rows: int = 2_000_000,
) -> tuple[np.memmap, str]:
    """Return ``source_row -> train_test_split permutation rank`` as a memmap.

    scikit-learn 1.5.1's ``ShuffleSplit`` uses exactly
    ``RandomState(random_state).permutation(rows)``.  Test rows are the first
    ``test_size`` ranks and train rows are the remainder, in that rank order.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite rank memmap: {path}")
    dtype = np.uint32 if rows <= np.iinfo(np.uint32).max else np.uint64
    rank = np.memmap(path, dtype=dtype, mode="w+", shape=(rows,))
    permutation = np.random.RandomState(random_state).permutation(rows)
    for start in range(0, rows, assignment_chunk_rows):
        stop = min(rows, start + assignment_chunk_rows)
        rank[permutation[start:stop]] = np.arange(start, stop, dtype=dtype)
    rank.flush()
    del permutation
    rank_sha256 = _hash_file(path, "sha256")
    return rank, rank_sha256


def _validate_chunk(chunk: pd.DataFrame, start_row: int) -> None:
    if tuple(chunk.columns) != SOURCE_COLUMNS:
        raise RuntimeError(f"source columns drifted at row {start_row}: {tuple(chunk.columns)}")
    for column, expected_dtype in SOURCE_DTYPES.items():
        if str(chunk[column].dtype) != expected_dtype:
            raise RuntimeError(
                f"source dtype drifted at row {start_row}, column {column}: "
                f"expected {expected_dtype}, got {chunk[column].dtype}"
            )
    # The production GNU/awk splitter deliberately avoids a second CSV parse.
    # These two official object columns contain no CSV metacharacters.  Assert
    # that invariant per chunk rather than silently producing ambiguous rows.
    for column in ("key", "pickup_datetime"):
        values = chunk[column]
        if values.isna().any():
            raise RuntimeError(f"unexpected null in {column} at source row {start_row}")
        if values.str.contains(r'[,"\r\n]', regex=True).any():
            raise RuntimeError(f"CSV metacharacter in {column} at source row {start_row}")


def iter_ranked_chunks(
    source_csv: Path,
    rank: np.memmap | np.ndarray,
    *,
    expected_rows: int,
    chunk_rows: int,
    logger: JsonlLogger | None = None,
) -> Iterator[pd.DataFrame]:
    offset = 0
    reader = pd.read_csv(
        source_csv,
        chunksize=chunk_rows,
        dtype=SOURCE_DTYPES,
        float_precision="round_trip",
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        _validate_chunk(chunk, offset)
        stop = offset + len(chunk)
        if stop > expected_rows:
            raise RuntimeError(f"chunk reader exceeded scanned row count {expected_rows}")
        ranked = chunk.copy(deep=False)
        ranked.insert(0, "__evomind_rank", np.asarray(rank[offset:stop]))
        yield ranked
        offset = stop
        if logger is not None:
            logger.emit(
                "source_chunk_serialized",
                chunk=chunk_number,
                rows_in_chunk=len(chunk),
                rows_complete=offset,
                rows_total=expected_rows,
            )
    if offset != expected_rows:
        raise RuntimeError(f"chunk reader produced {offset} rows; source scan found {expected_rows}")


def _write_headers(public_dir: Path, private_dir: Path) -> Mapping[str, Path]:
    public_dir.mkdir(parents=True, exist_ok=False)
    private_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "labels": public_dir / "labels.csv",
        "public_test": public_dir / "test.csv",
        "sample": public_dir / "sample_submission.csv",
        "private_test": private_dir / "test.csv",
    }
    headers = {
        "labels": SOURCE_COLUMNS,
        "public_test": PUBLIC_TEST_COLUMNS,
        "sample": ("key", "fare_amount"),
        "private_test": SOURCE_COLUMNS,
    }
    for key, path in paths.items():
        path.write_text(",".join(headers[key]) + "\n", encoding="utf-8", newline="\n")
    return paths


def _gnu_sort_command(sort_tmp: Path, sort_memory: str, sort_parallel: int) -> list[str]:
    executable = shutil.which("sort")
    if not executable:
        raise RuntimeError("GNU sort is required for the production backend")
    probe = subprocess.run(
        [executable, "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    if probe.returncode != 0 or "GNU coreutils" not in probe.stdout:
        raise RuntimeError(f"production backend requires GNU sort, got: {probe.stdout.strip()}")
    return [
        executable,
        "--stable",
        "--field-separator=,",
        "--key=1,1n",
        f"--buffer-size={sort_memory}",
        f"--parallel={sort_parallel}",
        f"--temporary-directory={sort_tmp}",
    ]


def _stream_with_gnu_sort(
    chunks: Iterable[pd.DataFrame],
    paths: Mapping[str, Path],
    *,
    expected_rows: int,
    test_size: int,
    work_dir: Path,
    sort_memory: str,
    sort_parallel: int,
    logger: JsonlLogger,
) -> None:
    sort_tmp = work_dir / "gnu-sort-tmp"
    sort_tmp.mkdir(parents=True, exist_ok=False)
    sort_stderr_path = work_dir / "gnu-sort.stderr.log"
    awk_stderr_path = work_dir / "splitter.stderr.log"

    awk_executable = shutil.which("awk")
    if not awk_executable:
        raise RuntimeError("awk is required for the production backend")
    awk_program = r'''
{
    if (($1 + 0) != (NR - 1)) bad_rank = 1
    line = $0
    sub(/^[^,]*,/, "", line)
    if (NR <= test_size) {
        print line >> private_test_path
        printf "%s,%s,%s,%s,%s,%s,%s\n", $2, $4, $5, $6, $7, $8, $9 >> public_test_path
        printf "%s,11.35\n", $2 >> sample_path
    } else {
        print line >> labels_path
    }
}
END {
    if (NR != expected_rows) exit 42
    if (bad_rank) exit 43
}
'''.strip()

    env = os.environ.copy()
    env["LC_ALL"] = "C"
    sort_command = _gnu_sort_command(sort_tmp, sort_memory, sort_parallel)
    awk_command = [
        awk_executable,
        "-F,",
        "-v",
        f"test_size={test_size}",
        "-v",
        f"expected_rows={expected_rows}",
        "-v",
        f"labels_path={paths['labels']}",
        "-v",
        f"public_test_path={paths['public_test']}",
        "-v",
        f"sample_path={paths['sample']}",
        "-v",
        f"private_test_path={paths['private_test']}",
        awk_program,
    ]
    logger.emit(
        "external_sort_started",
        backend="gnu-sort",
        sort_memory=sort_memory,
        sort_parallel=sort_parallel,
        sort_tmp=str(sort_tmp),
    )

    with sort_stderr_path.open("wb") as sort_stderr, awk_stderr_path.open("wb") as awk_stderr:
        awk_process = subprocess.Popen(awk_command, stdin=subprocess.PIPE, stderr=awk_stderr, env=env)
        if awk_process.stdin is None:
            raise RuntimeError("failed to open awk stdin")
        sort_process = subprocess.Popen(
            sort_command,
            stdin=subprocess.PIPE,
            stdout=awk_process.stdin,
            stderr=sort_stderr,
            env=env,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1024 * 1024,
        )
        awk_process.stdin.close()
        if sort_process.stdin is None:
            sort_process.terminate()
            awk_process.terminate()
            raise RuntimeError("failed to open GNU sort stdin")
        try:
            for ranked in chunks:
                ranked.to_csv(sort_process.stdin, index=False, header=False, lineterminator="\n")
            sort_process.stdin.close()
            sort_returncode = sort_process.wait()
            awk_returncode = awk_process.wait()
        except BaseException:
            with contextlib.suppress(Exception):
                sort_process.stdin.close()
            with contextlib.suppress(Exception):
                sort_process.terminate()
            with contextlib.suppress(Exception):
                awk_process.terminate()
            with contextlib.suppress(Exception):
                sort_process.wait(timeout=10)
            with contextlib.suppress(Exception):
                awk_process.wait(timeout=10)
            raise

    if sort_returncode != 0:
        details = sort_stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"GNU sort failed with exit {sort_returncode}: {details}")
    if awk_returncode != 0:
        details = awk_stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"sorted-row splitter failed with exit {awk_returncode}: {details}")
    logger.emit("external_sort_completed", backend="gnu-sort")


def _stream_with_python_sort(
    chunks: Iterable[pd.DataFrame],
    paths: Mapping[str, Path],
    *,
    expected_rows: int,
    test_size: int,
    max_rows: int,
) -> None:
    """In-memory reference backend used only by bounded synthetic tests."""

    if expected_rows > max_rows:
        raise RuntimeError(
            f"python sort backend is limited to {max_rows} rows; use GNU sort for {expected_rows} rows"
        )
    ranked_lines: list[tuple[int, str]] = []
    for ranked in chunks:
        buffer = io.StringIO(newline="")
        ranked.to_csv(buffer, index=False, header=False, lineterminator="\n")
        for line in buffer.getvalue().splitlines(keepends=True):
            rank_text, payload = line.split(",", 1)
            ranked_lines.append((int(rank_text), payload))
    ranked_lines.sort(key=lambda item: item[0])
    if len(ranked_lines) != expected_rows:
        raise RuntimeError(f"python sorter received {len(ranked_lines)} rows, expected {expected_rows}")

    handles: dict[str, IO[str]] = {}
    try:
        for key, path in paths.items():
            handles[key] = path.open("a", encoding="utf-8", newline="")
        for expected_rank, (rank_value, payload) in enumerate(ranked_lines):
            if rank_value != expected_rank:
                raise RuntimeError(f"rank sequence drift: expected {expected_rank}, got {rank_value}")
            if expected_rank < test_size:
                fields = payload.rstrip("\n").split(",")
                if len(fields) != len(SOURCE_COLUMNS):
                    raise RuntimeError(f"unexpected staged field count: {len(fields)}")
                handles["private_test"].write(payload)
                handles["public_test"].write(",".join((fields[0], *fields[2:])) + "\n")
                handles["sample"].write(f"{fields[0]},11.35\n")
            else:
                handles["labels"].write(payload)
    finally:
        for handle in handles.values():
            handle.close()


def _outputs_are_empty(prepared_dir: Path) -> bool:
    return not prepared_dir.exists() or not any(path.is_file() for path in prepared_dir.rglob("*"))


def _verify_output_md5(
    staging_dir: Path,
    expected_md5: Mapping[str, str] | None,
) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative in OFFICIAL_OUTPUT_MD5:
        path = staging_dir / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"prepared output is missing or empty: {path}")
        actual[relative] = _hash_file(path, "md5")
    if expected_md5 is not None:
        mismatches = {
            relative: {"expected": expected_md5[relative], "actual": actual[relative]}
            for relative in expected_md5
            if actual.get(relative) != expected_md5[relative]
        }
        if mismatches:
            raise RuntimeError(f"prepared output MD5 mismatch: {_canonical_json(mismatches)}")
    return actual


def prepare_taxi_streaming(
    *,
    competition_dir: Path,
    prepared_dir: Path,
    work_root: Path,
    test_size: int = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    chunk_rows: int = 250_000,
    backend: str = "gnu-sort",
    sort_memory: str = "2G",
    sort_parallel: int = 4,
    expected_md5: Mapping[str, str] | None = OFFICIAL_OUTPUT_MD5,
    expected_zip_md5: str | None = EXPECTED_ZIP_MD5,
    enforce_pinned_versions: bool = True,
    description_file: Path | None = None,
    log_path: Path | None = None,
    report_path: Path | None = None,
    python_sort_max_rows: int = 1_000_000,
) -> PrepareResult:
    started = time.monotonic()
    competition_dir = competition_dir.expanduser().resolve()
    prepared_dir = prepared_dir.expanduser().resolve()
    work_root = work_root.expanduser().resolve()
    logger = JsonlLogger(log_path.expanduser().resolve() if log_path is not None else None)
    if test_size <= 0 or chunk_rows <= 0:
        raise ValueError("test_size and chunk_rows must be positive")
    official_output_contract = expected_md5 is not None and dict(expected_md5) == OFFICIAL_OUTPUT_MD5
    if random_state != DEFAULT_RANDOM_STATE and official_output_contract:
        raise ValueError("official MD5 verification requires random_state=0")
    if test_size != DEFAULT_TEST_SIZE and official_output_contract:
        raise ValueError("official MD5 verification requires test_size=9914")

    versions = validate_pinned_runtime() if enforce_pinned_versions else runtime_versions()
    raw_dir = competition_dir / "raw"
    source_csv = raw_dir / "train.csv"
    required_raw = (
        source_csv,
        raw_dir / "GCP-Coupons-Instructions.rtf",
    )
    for path in required_raw:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"required raw Taxi file is missing or empty: {path}")

    zip_files = sorted(competition_dir.glob("*.zip"))
    if len(zip_files) != 1:
        raise RuntimeError(f"expected exactly one official ZIP in {competition_dir}; found {len(zip_files)}")
    zip_path = zip_files[0]
    zip_md5 = _hash_file(zip_path, "md5")
    if expected_zip_md5 is not None and zip_md5 != expected_zip_md5:
        raise RuntimeError(f"official ZIP MD5 mismatch: expected {expected_zip_md5}, got {zip_md5}")

    if not _outputs_are_empty(prepared_dir):
        raise RuntimeError(f"refusing to overwrite non-empty prepared directory: {prepared_dir}")
    work_root.mkdir(parents=True, exist_ok=True)
    run_id = f"taxi-streaming-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    run_dir = work_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    staging_dir = prepared_dir.parent / f".{prepared_dir.name}.{run_id}.staging"
    staging_dir.mkdir(parents=False, exist_ok=False)
    public_dir = staging_dir / "public"
    private_dir = staging_dir / "private"

    logger.emit(
        "prepare_started",
        competition=COMPETITION_ID,
        backend=backend,
        competition_dir=str(competition_dir),
        prepared_dir=str(prepared_dir),
        run_dir=str(run_dir),
        staging_dir=str(staging_dir),
        runtime=versions,
        zip_md5=zip_md5,
    )

    report: dict[str, Any] = {
        "schema_version": "evomind.mle.taxi-streaming-report.v1",
        "competition": COMPETITION_ID,
        "status": "running",
        "started_at_utc": _utc_now(),
        "backend": backend,
        "runtime": versions,
        "source": {"zip": str(zip_path), "zip_bytes": zip_path.stat().st_size, "zip_md5": zip_md5},
        "paths": {
            "competition_dir": str(competition_dir),
            "prepared_dir": str(prepared_dir),
            "run_dir": str(run_dir),
            "staging_dir": str(staging_dir),
        },
    }
    internal_report_path = run_dir / "taxi-streaming-report.json"

    def write_report() -> None:
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        internal_report_path.write_text(payload, encoding="utf-8", newline="\n")
        if report_path is not None:
            resolved = report_path.expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(payload, encoding="utf-8", newline="\n")

    try:
        with _exclusive_lock(work_root / "taxi-streaming.lock"):
            scan = scan_source_csv(source_csv)
            if scan.rows <= test_size:
                raise RuntimeError(f"source has {scan.rows} rows; test_size is {test_size}")
            report["source"].update(
                {"train_csv_bytes": scan.bytes, "train_csv_rows": scan.rows, "train_csv_sha256": scan.sha256}
            )
            logger.emit(
                "source_scanned",
                rows=scan.rows,
                bytes=scan.bytes,
                sha256=scan.sha256,
            )

            rank_path = run_dir / "source-to-split-rank.memmap"
            rank, rank_sha256 = build_inverse_rank_memmap(
                scan.rows,
                rank_path,
                random_state=random_state,
            )
            report["split"] = {
                "random_state": random_state,
                "test_size": test_size,
                "train_size": scan.rows - test_size,
                "rank_dtype": str(rank.dtype),
                "rank_sha256": rank_sha256,
            }
            logger.emit(
                "rank_memmap_built",
                rows=scan.rows,
                dtype=str(rank.dtype),
                sha256=rank_sha256,
            )

            paths = _write_headers(public_dir, private_dir)
            chunks = iter_ranked_chunks(
                source_csv,
                rank,
                expected_rows=scan.rows,
                chunk_rows=chunk_rows,
                logger=logger,
            )
            if backend == "gnu-sort":
                _stream_with_gnu_sort(
                    chunks,
                    paths,
                    expected_rows=scan.rows,
                    test_size=test_size,
                    work_dir=run_dir,
                    sort_memory=sort_memory,
                    sort_parallel=sort_parallel,
                    logger=logger,
                )
            elif backend == "python":
                _stream_with_python_sort(
                    chunks,
                    paths,
                    expected_rows=scan.rows,
                    test_size=test_size,
                    max_rows=python_sort_max_rows,
                )
            else:
                raise ValueError(f"unsupported backend: {backend}")
            del rank

            shutil.copy2(raw_dir / "GCP-Coupons-Instructions.rtf", public_dir / "GCP-Coupons-Instructions.rtf")
            if description_file is not None:
                resolved_description = description_file.expanduser().resolve()
                if not resolved_description.is_file():
                    raise RuntimeError(f"description file does not exist: {resolved_description}")
                shutil.copyfile(resolved_description, public_dir / "description.md")

            actual_md5 = _verify_output_md5(staging_dir, expected_md5)
            report["outputs"] = {
                relative: {
                    "md5": digest,
                    "bytes": (staging_dir / relative).stat().st_size,
                }
                for relative, digest in actual_md5.items()
            }
            logger.emit("outputs_verified", md5=actual_md5)

            # Promotion happens only after every checksum passed.  An existing
            # destination is removed only when it contains no files.
            if prepared_dir.exists():
                if not _outputs_are_empty(prepared_dir):
                    raise RuntimeError(f"prepared directory became non-empty during run: {prepared_dir}")
                shutil.rmtree(prepared_dir)
            os.replace(staging_dir, prepared_dir)
            with contextlib.suppress(FileNotFoundError):
                rank_path.unlink()
            with contextlib.suppress(OSError):
                (run_dir / "gnu-sort-tmp").rmdir()

            elapsed = time.monotonic() - started
            report.update(
                {
                    "status": "verified",
                    "completed_at_utc": _utc_now(),
                    "elapsed_seconds": elapsed,
                    "staging_promoted": True,
                }
            )
            write_report()
            logger.emit("prepare_completed", elapsed_seconds=elapsed, prepared_dir=str(prepared_dir))
            return PrepareResult(
                prepared_dir=prepared_dir,
                rows=scan.rows,
                test_rows=test_size,
                source_sha256=scan.sha256,
                rank_sha256=rank_sha256,
                output_md5=actual_md5,
                elapsed_seconds=elapsed,
                report=report,
            )
    except BaseException as exc:
        report.update(
            {
                "status": "failed_closed",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "staging_promoted": False,
            }
        )
        write_report()
        logger.emit("prepare_failed_closed", error_type=type(exc).__name__, error=str(exc), staging_dir=str(staging_dir))
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--sort-memory", default="2G")
    parser.add_argument("--sort-parallel", type=int, default=4)
    parser.add_argument("--backend", choices=("gnu-sort", "python"), default="gnu-sort")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--description-file", type=Path)
    parser.add_argument(
        "--allow-unpinned-runtime",
        action="store_true",
        help="test-only escape hatch; official production preparation must not use it",
    )
    parser.add_argument(
        "--skip-official-md5",
        action="store_true",
        help="test-only escape hatch; official production preparation must not use it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    competition_dir = args.competition_dir.expanduser().resolve()
    prepared_dir = (
        args.prepared_dir.expanduser().resolve()
        if args.prepared_dir is not None
        else competition_dir / "prepared"
    )
    try:
        result = prepare_taxi_streaming(
            competition_dir=competition_dir,
            prepared_dir=prepared_dir,
            work_root=args.work_dir,
            chunk_rows=args.chunk_rows,
            backend=args.backend,
            sort_memory=args.sort_memory,
            sort_parallel=args.sort_parallel,
            expected_md5=None if args.skip_official_md5 else OFFICIAL_OUTPUT_MD5,
            expected_zip_md5=None if args.skip_official_md5 else EXPECTED_ZIP_MD5,
            enforce_pinned_versions=not args.allow_unpinned_runtime,
            description_file=args.description_file,
            log_path=args.log,
            report_path=args.report,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"competition": COMPETITION_ID, "status": "failed_closed", "error": str(exc)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
