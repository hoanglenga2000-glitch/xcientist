"""Bounded official API file access. No submissions, training or credential logging."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import socket
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from research_os.official_calibration import build_frozen_folds, sha256_file, validate_data_frames


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=['probe','download'],default='probe')
    args=parser.parse_args()
    out=ROOT/'artifacts/ev-public-calibration-20260908'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    receipt={'schema':'evomind.official_calibration.kaggle_access.v1','action':args.action,
             'checked_at':datetime.now(timezone.utc).isoformat(),'competition':'playground-series-s6e9',
             'status':'started','credential_logged':False,'submissions':0,'training_started':False}
    exit_code=0
    try:
        if not os.environ.get('KAGGLE_API_TOKEN'):
            raise ValueError('Managed API credential not supplied')
        logging.disable(logging.CRITICAL)
        socket.setdefaulttimeout(45)
        # Suppress third-party client output, including raw exception payloads/URLs.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
            api=KaggleApi();api.authenticate()
            response=api.competition_list_files(receipt['competition'],page_size=20)
            files=list(response.files)
            names=[f.name for f in files]
            if set(names)!={'train.csv','test.csv','sample_submission.csv'}:
                raise ValueError('Official file set differs from protocol')
            receipt.update(status='file_listing_verified',files=names)
            if args.action=='download':
                data=out/'private-data';data.mkdir(exist_ok=True)
                if (data/'manifest.json').exists():
                    raise ValueError('Frozen data already exists; do not replace')
                api.competition_download_files(receipt['competition'],path=str(data),force=False,quiet=True)
                archive=data/(receipt['competition']+'.zip')
                with zipfile.ZipFile(archive) as z:
                    members=z.infolist()
                    if {m.filename for m in members}!=set(names) or len(members)!=3:
                        raise ValueError('Unexpected archive members')
                    if sum(m.file_size for m in members)>2*1024**3:
                        raise ValueError('Uncompressed archive exceeds staging limit')
                    for m in members:
                        if (m.external_attr >> 16) & 0o170000 == 0o120000:
                            raise ValueError('Archive symlink rejected')
                        with z.open(m) as source, (data/m.filename).open('xb') as target:
                            import shutil
                            shutil.copyfileobj(source,target)
                import pandas as pd
                protocol=json.loads((out/'agent-input/protocol.json').read_text(encoding='utf-8'))
                train,test,sample=[pd.read_csv(data/f) for f in ['train.csv','test.csv','sample_submission.csv']]
                quality=validate_data_frames(train,test,sample,protocol)
                folds=build_frozen_folds(train,protocol)
                folds.to_csv(data/'frozen-folds.csv',index=False)
                manifest={'source':'official_kaggle_api','competition':receipt['competition'],
                          'protocol_sha256':sha256_file(out/'agent-input/protocol.json'),
                          'files':{f:{'bytes':(data/f).stat().st_size,'sha256':sha256_file(data/f)} for f in names+['frozen-folds.csv']},
                          'quality':quality,'raw_data_sent_to_external_llm':False,'training_started':False}
                with (data/'manifest.json').open('x',encoding='utf-8') as stream:
                    json.dump(manifest,stream,indent=2)
                receipt.update(status='official_data_and_folds_verified',quality=quality,
                               manifest_sha256=sha256_file(data/'manifest.json'))
    except BaseException as exc:
        if isinstance(exc,KeyboardInterrupt):raise
        response=getattr(exc,'response',None)
        status=getattr(response,'status_code',None) or getattr(exc,'status',None)
        receipt.update(status='blocked',error_type=type(exc).__name__,http_status=status if isinstance(status,int) else None)
        exit_code=2
    with (out/f'kaggle-{args.action}-{stamp}.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2)
    print(json.dumps(receipt,ensure_ascii=False))
    return exit_code


if __name__=='__main__':raise SystemExit(main())
