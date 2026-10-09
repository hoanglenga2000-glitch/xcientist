"""Exact-preparer read-only status; never emits command lines or private data."""
import json,os,time
from pathlib import Path
root=Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/siim_mlebench_calibration_20260908')
result={'root_exists':root.exists(),'workers':[],'progress':None,'manifest_exists':(root/'manifest.json').is_file()}
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or p.name==str(os.getpid()):continue
 try:
  command=(p/'cmdline').read_bytes()
  if b'Freeze JPEG-only SIIM benchmark inputs' not in command:continue
  state=(p/'stat').read_text().split(') ',1)[1].split()[0]
  result['workers'].append({'pid':int(p.name),'state':state})
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
progress=root/'prepare-progress.json'
if progress.is_file():result['progress']=json.loads(progress.read_text())
failures=sorted(root.glob('prepare-failure-*.json')) if root.exists() else []
if failures:result['last_failure']=json.loads(failures[-1].read_text())
print(json.dumps(result))
