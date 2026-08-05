from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts import prepare_mle_taxi_streaming as taxi


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _write_fixture(competition_dir: Path, rows: int = 37) -> pd.DataFrame:
    raw_dir = competition_dir / "raw"
    raw_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "key": [f"2010-01-{(index % 28) + 1:02d} 12:34:56.000000{index}" for index in range(rows)],
            "fare_amount": [float(index) if index % 4 == 0 else index / 7.0 for index in range(rows)],
            "pickup_datetime": [f"2010-01-{(index % 28) + 1:02d} 12:34:56 UTC" for index in range(rows)],
            "pickup_longitude": [-73.9 - index / 10_000 for index in range(rows)],
            "pickup_latitude": [40.7 + index / 20_000 for index in range(rows)],
            "dropoff_longitude": [-73.8 - index / 30_000 for index in range(rows)],
            "dropoff_latitude": [40.8 + index / 40_000 for index in range(rows)],
            "passenger_count": [(index % 6) + 1 for index in range(rows)],
        }
    )
    frame.to_csv(raw_dir / "train.csv", index=False, lineterminator="\n")
    (raw_dir / "GCP-Coupons-Instructions.rtf").write_bytes(b"{\\rtf1 synthetic}\n")
    # The production function requires exactly one non-empty ZIP.  Synthetic
    # tests deliberately disable the official ZIP checksum but retain the
    # source-shape contract.
    (competition_dir / "fixture.zip").write_bytes(b"synthetic official zip fixture")
    return frame


def _write_upstream_reference(source_csv: Path, output_dir: Path, test_size: int) -> None:
    frame = pd.read_csv(source_csv, float_precision="round_trip")
    new_train, new_test = train_test_split(frame, test_size=test_size, random_state=0)
    public = output_dir / "public"
    private = output_dir / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    new_train.to_csv(public / "labels.csv", index=False, lineterminator="\n")
    new_test.drop(columns=["fare_amount"]).to_csv(public / "test.csv", index=False, lineterminator="\n")
    submission = new_test[["key"]].copy()
    submission["fare_amount"] = 11.35
    submission.to_csv(public / "sample_submission.csv", index=False, lineterminator="\n")
    new_test.to_csv(private / "test.csv", index=False, lineterminator="\n")


def test_inverse_rank_memmap_matches_sklearn_shuffle_split(tmp_path: Path) -> None:
    rows = 10_003
    test_size = 17
    rank, _ = taxi.build_inverse_rank_memmap(rows, tmp_path / "rank.memmap")
    frame = pd.DataFrame({"row": np.arange(rows)})
    expected_train, expected_test = train_test_split(frame, test_size=test_size, random_state=0)

    actual_test = np.argsort(np.asarray(rank))[:test_size]
    actual_train = np.argsort(np.asarray(rank))[test_size:]
    np.testing.assert_array_equal(actual_test, expected_test.index.to_numpy())
    np.testing.assert_array_equal(actual_train, expected_train.index.to_numpy())


def test_streaming_outputs_are_byte_identical_to_upstream(tmp_path: Path) -> None:
    competition_dir = tmp_path / "competition"
    _write_fixture(competition_dir)
    reference = tmp_path / "reference"
    _write_upstream_reference(competition_dir / "raw" / "train.csv", reference, test_size=7)
    expected = {
        relative: _md5(reference / relative)
        for relative in taxi.OFFICIAL_OUTPUT_MD5
    }

    result = taxi.prepare_taxi_streaming(
        competition_dir=competition_dir,
        prepared_dir=competition_dir / "prepared",
        work_root=tmp_path / "work",
        test_size=7,
        chunk_rows=5,
        backend="python",
        expected_md5=expected,
        expected_zip_md5=None,
        enforce_pinned_versions=False,
        log_path=tmp_path / "prepare.jsonl",
        report_path=tmp_path / "report.json",
    )

    assert result.rows == 37
    assert result.test_rows == 7
    assert result.report["status"] == "verified"
    for relative in taxi.OFFICIAL_OUTPUT_MD5:
        actual_path = competition_dir / "prepared" / relative
        expected_path = reference / relative
        assert actual_path.read_bytes() == expected_path.read_bytes(), relative


def test_nonempty_destination_fails_closed_without_overwrite(tmp_path: Path) -> None:
    competition_dir = tmp_path / "competition"
    _write_fixture(competition_dir, rows=20)
    prepared = competition_dir / "prepared"
    prepared.mkdir()
    sentinel = prepared / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    try:
        taxi.prepare_taxi_streaming(
            competition_dir=competition_dir,
            prepared_dir=prepared,
            work_root=tmp_path / "work",
            test_size=5,
            backend="python",
            expected_md5=None,
            expected_zip_md5=None,
            enforce_pinned_versions=False,
        )
    except RuntimeError as exc:
        assert "refusing to overwrite non-empty" in str(exc)
    else:
        raise AssertionError("non-empty destination was not rejected")
    assert sentinel.read_text(encoding="utf-8") == "do not replace"
