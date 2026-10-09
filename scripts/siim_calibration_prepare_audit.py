"""Inspect only the exact preparation output and stopped EV-independent CPU work."""
import json
from pathlib import Path
ROOT=Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/siim_mlebench_calibration_20260908')
result={'root_exists':ROOT.is_dir(),'files':[],'training_started':False,'private_answers_read':False}
if ROOT.is_dir():
    result['files']=[{'name':p.name,'bytes':p.stat().st_size} for p in ROOT.iterdir() if p.is_file()]
    manifest=ROOT/'manifest.json'
    if manifest.is_file():
        data=json.loads(manifest.read_text())
        result['manifest']={k:data[k] for k in data if k!='image_dimensions'}
print(json.dumps(result))
