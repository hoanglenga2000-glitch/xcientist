"""Verify already downloaded official files and freeze common folds; no fitting/network."""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from research_os.official_calibration import build_frozen_folds, sha256_file, validate_data_frames


def main():
    out=ROOT/'artifacts/ev-public-calibration-20260908'
    data=out/'private-data'
    protocol_path=out/'agent-input/protocol.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if (data/'manifest.json').exists() or (data/'frozen-folds.csv').exists():
        raise ValueError('Frozen data already exists; refusing overwrite')
    train,test,sample=[pd.read_csv(data/f) for f in ['train.csv','test.csv','sample_submission.csv']]
    quality=validate_data_frames(train,test,sample,protocol)
    folds=build_frozen_folds(train,protocol)
    folds.to_csv(data/'frozen-folds.csv',index=False)
    files=['train.csv','test.csv','sample_submission.csv','frozen-folds.csv','playground-series-s6e9.zip']
    manifest={'schema':'evomind.official_calibration.data_manifest.v1','verified_at':datetime.now(timezone.utc).isoformat(),
              'source':'official_kaggle_api_download','competition':protocol['competition'],
              'protocol_sha256':sha256_file(protocol_path),'quality':quality,
              'files':{f:{'bytes':(data/f).stat().st_size,'sha256':sha256_file(data/f)} for f in files},
              'fold_counts':{str(k):int(v) for k,v in folds.fold.value_counts().sort_index().items()},
              'raw_csvs_unchanged':True,'target_encoding_is_in_memory_only':True,
              'raw_rows_sent_to_external_model':False,'training_started':False}
    with (data/'manifest.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2)
    print(json.dumps({'status':'official_data_and_folds_verified','quality':quality,
                      'fold_counts':manifest['fold_counts'],'manifest_sha256':sha256_file(data/'manifest.json'),
                      'training_started':False}))


if __name__=='__main__':main()
