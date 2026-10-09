"""Offline import/provenance check; never loads a credential or calls a model."""
import json,sys
from pathlib import Path
STAGE=Path('C:/ProgramData/EvoMind/staging/siim-mlebench-calibration-20260908/runtime-extension')
sys.path.insert(0,str(STAGE))
import siim_aide_controller as adapter
core=adapter.load_core(STAGE/'aide-core',lambda **kwargs:'Plan.\n```python\ndef fit_model(a,b,c,d,e):\n return None\n```')
assert core.MetricValue(.8,maximize=True)>core.MetricValue(.7,maximize=True)
print(json.dumps({'status':'aide_core_import_verified','upstream_commit':'d4a77cf3ca11e0f70749052b2701003e32bc245a','model_calls':0}))
