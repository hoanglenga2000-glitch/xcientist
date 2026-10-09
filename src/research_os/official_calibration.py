"""Local contracts for official-score calibration; no training, network or submission."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_protocol(protocol: dict[str, Any], output: Path) -> str:
    """Exclusive creation: resuming cannot silently replace an experimental protocol."""
    if protocol['competition'] != 'playground-series-s6e9' or protocol['metric'] != 'roc_auc':
        raise ValueError('Unexpected calibration task or metric')
    if protocol['run_seeds'] != [17, 29, 43] or protocol['fold_count'] != 5:
        raise ValueError('Unexpected repetition design')
    if protocol['budget']['total_gpu_hours'] != 24 or protocol['submission']['auto_submit'] is not False:
        raise ValueError('Budget/submission boundary changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode('utf-8')
    with output.open('xb') as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def validate_data_frames(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame,
                         protocol: dict[str, Any]) -> dict[str, Any]:
    id_col, target = protocol['id_column'], protocol['target']
    if train.empty or test.empty:
        raise ValueError('Empty training/test data')
    if any(frame.columns.duplicated().any() for frame in (train, test, sample)):
        raise ValueError('Duplicate column names')
    if target not in train or target in test:
        raise ValueError('Training/test target boundary invalid')
    if sample.columns.tolist() != [id_col, target]:
        raise ValueError('Unexpected submission columns')
    if [col for col in train if col != target] != test.columns.tolist():
        raise ValueError('Train/test feature schema differs')
    for frame in (train, test, sample):
        if id_col not in frame or frame[id_col].isna().any() or not frame[id_col].is_unique:
            raise ValueError('IDs must be unique and nonmissing')
    if set(train[id_col]) & set(test[id_col]):
        raise ValueError('Training/test ID overlap')
    if not np.array_equal(test[id_col].to_numpy(), sample[id_col].to_numpy()):
        raise ValueError('Test/sample ID order differs')
    encoded, encoding = encode_binary_target(train[target])
    if encoded.value_counts().min() < protocol['fold_count']:
        raise ValueError('Too few examples per class for frozen folds')
    return {'train_rows': len(train), 'test_rows': len(test), 'id_unique': True,
            'test_has_no_labels': True, 'schema_and_ids_aligned': True, 'target_encoding': encoding}


def encode_binary_target(target: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Yes is the purchase-positive class; raw source CSVs are never rewritten."""
    if target.isna().any():
        raise ValueError('Missing binary target')
    values = set(target.unique())
    if values == {0, 1}:
        return target.astype(int), {'0': 0, '1': 1}
    if values == {'No', 'Yes'}:
        mapping = {'No': 0, 'Yes': 1}
        return target.map(mapping).astype(int), mapping
    raise ValueError('Expected binary 0/1 or No/Yes target')


def build_frozen_folds(train: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    id_col, target = protocol['id_column'], protocol['target']
    if not train[id_col].is_unique or train[id_col].isna().any():
        raise ValueError('Cannot freeze folds with ambiguous IDs')
    y, _ = encode_binary_target(train[target])
    if y.value_counts().min() < protocol['fold_count']:
        raise ValueError('Cannot create stratified binary folds')
    cv = StratifiedKFold(n_splits=protocol['fold_count'], shuffle=True, random_state=protocol['split_seed'])
    folds = np.full(len(train), -1, dtype=int)
    for fold, (train_indices, valid_indices) in enumerate(cv.split(np.zeros(len(train)), y)):
        if set(train_indices) & set(valid_indices) or (folds[valid_indices] != -1).any():
            raise ValueError('Overlapping validation indices')
        folds[valid_indices] = fold
    if (folds < 0).any():
        raise ValueError('Incomplete fold coverage')
    return pd.DataFrame({id_col: train[id_col].to_numpy(), 'row_index': np.arange(len(train)), 'fold': folds})


def select_frozen_candidate(runs: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [r for r in runs if r.get('arm') == arm]
    if len(selected) != 3 or sorted(r['seed'] for r in selected) != [17, 29, 43]:
        raise ValueError('Three distinct planned seeds are required')
    for key in ['protocol_sha256', 'data_manifest_sha256', 'folds_sha256']:
        values = {r.get(key) for r in selected}
        if len(values) != 1 or not re.fullmatch(r'[0-9a-f]{64}', str(next(iter(values)))):
            raise ValueError('Run data, fold or protocol identity differs or is absent')
    for r in selected:
        score, seconds = r.get('oof_roc_auc'), r.get('runtime_seconds')
        if r.get('status') != 'completed' or r.get('oof_complete') is not True:
            raise ValueError('Incomplete run cannot be selected')
        if r.get('official_score_seen_before_freeze') is not False:
            raise ValueError('Official-score isolation not verified')
        if arm == 'evomind' and r.get('human_modeling_interventions') != 0:
            raise ValueError('Assisted run is not an unassisted candidate')
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid OOF AUC')
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 10800:
            raise ValueError('Run exceeds wall-clock contract')
    return min(selected, key=lambda r: (-r['oof_roc_auc'], r['runtime_seconds'], r['seed']))


def remaining_gpu_hours(entries: list[dict[str, Any]], request_hours: float = 0) -> float:
    """Unsettled reservations remain charged; accounting is not a process watchdog."""
    seen, charged = set(), 0.0
    for row in entries:
        if row['attempt_id'] in seen:
            raise ValueError('Duplicate attempt ID')
        seen.add(row['attempt_id'])
        value = row['actual_gpu_hours'] if row.get('settled') is True else row['reserved_gpu_hours']
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid GPU accounting')
        charged += value
    if not math.isfinite(request_hours) or request_hours < 0 or charged + request_hours > 24:
        raise ValueError('GPU budget exhausted or request invalid')
    return 24 - charged - request_hours


def compare_official_scores(snapshot: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare a complete same-phase team snapshot; never manufacture a candidate rank."""
    identity = ('competition', 'phase', 'metric')
    if tuple(snapshot.get(k) for k in identity) != ('playground-series-s6e9', 'public', 'roc_auc'):
        raise ValueError('Wrong competition, phase or metric')
    if not snapshot.get('captured_at') or not snapshot.get('source_url') or snapshot.get('complete') is not True:
        raise ValueError('Complete sourced leaderboard snapshot required')
    rows = snapshot['rows']
    if not rows or len(rows) != snapshot['team_count']:
        raise ValueError('Leaderboard count mismatch')
    if len({r['team_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate teams')
    ordered = sorted(rows, key=lambda r: r['rank'])
    if [r['rank'] for r in ordered] != list(range(1, len(rows)+1)):
        raise ValueError('Missing or duplicated ranks')
    scores = [r['score'] for r in ordered]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
        raise ValueError('Invalid leaderboard scores')
    if scores != sorted(scores, reverse=True):
        raise ValueError('Scores conflict with official AUC ordering')
    for receipt in (baseline, candidate):
        if any(receipt.get(k) != snapshot[k] for k in identity):
            raise ValueError('Mixed evaluation protocols')
        if receipt.get('status') != 'scored' or not receipt.get('submission_ref') or not receipt.get('source_url'):
            raise ValueError('Official scoring receipt required')
        if not isinstance(receipt.get('score'), (int, float)) or not math.isfinite(receipt['score']) or not 0 <= receipt['score'] <= 1:
            raise ValueError('Invalid official score')
        if receipt.get('prediction_sha256') != receipt.get('frozen_prediction_sha256') or not receipt.get('prediction_sha256'):
            raise ValueError('Submitted predictions do not match frozen candidate')
    if baseline['submission_ref'] == candidate['submission_ref']:
        raise ValueError('Two distinct scoring receipts required')
    boundary_rank = math.ceil(0.25 * len(rows))
    boundary_score = ordered[boundary_rank-1]['score']
    # A team's standing belongs to its best submission, not automatically to this candidate.
    team = next((r for r in ordered if r['team_id'] == candidate.get('team_id')), None)
    linked = team and candidate.get('team_best_submission_ref') == candidate['submission_ref'] and team['score'] == candidate['score']
    return {'competition': snapshot['competition'], 'phase': 'public', 'metric': 'roc_auc',
            'baseline_score': baseline['score'], 'candidate_score': candidate['score'],
            'delta_auc': candidate['score']-baseline['score'], 'top_three': ordered[:3],
            'team_count': len(rows), 'top25_boundary_rank': boundary_rank,
            'top25_boundary_score': boundary_score, 'median_team_score': statistics.median(scores),
            'top25_score_threshold_met': candidate['score'] >= boundary_score,
            'baseline_exceeded': candidate['score'] > baseline['score'],
            'official_candidate_rank': team['rank'] if linked else None,
            'captured_at': snapshot['captured_at'], 'medals_applicable': False}
