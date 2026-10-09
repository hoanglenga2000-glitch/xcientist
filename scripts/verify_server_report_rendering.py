"""Verify isolated server report dependencies and render a synthetic document."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

candidate=Path('C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v2')
deps=Path('C:/EMQA/sys0907-report-deps-v1')
manifest=json.loads((deps/'installed-files.json').read_text(encoding='utf-8'))
for row in manifest['files']:
    path=(deps/'site-packages'/row['name']).resolve()
    path.relative_to((deps/'site-packages').resolve())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise SystemExit('report_dependency_integrity_failed')
sys.path.insert(0,str(deps/'site-packages'));sys.path.insert(0,str(candidate))
sys.argv=[str(candidate.parent/'report_render_probe.py'),'C:/EMQA/sys0907-server-report-v1']
runpy.run_path(sys.argv[0],run_name='__main__')
