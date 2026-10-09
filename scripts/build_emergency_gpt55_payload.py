import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
destination=ROOT/'artifacts/emergency-gpt55-20260908'
destination.mkdir(exist_ok=False)
source=ROOT/'artifacts/system-completeness-20260907/gpt55-launcher-candidate-v1/candidate/Start-Node.ps1'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='4af54177257c6cb2336246cfc41863e6413e10e3e1c00afb1a82e63b99c24c6d'
shutil.copyfile(source,destination/'Start-Node.ps1')
shutil.copyfile(ROOT/'scripts/emergency_gpt55_entry.py',destination/'entry.py')
shutil.copyfile(Path('D:/EV12/syscomplete-release-v3/runtime.zip'),destination/'runtime.zip')
assert hashlib.sha256((destination/'runtime.zip').read_bytes()).hexdigest()=='90b7d527de26fb41470c6784d9beee4ce84d1fa03fcf2a804ad064a803938a8d'
shutil.copyfile(ROOT/'scripts/emergency_gpt55_switch.py',destination/'switch.py')
base=Path('D:/EV12/ack07-11727a59/runtime/evomind_runtime')
baseline={p.relative_to(base).as_posix():hashlib.sha256(p.read_text(encoding='utf-8').replace('\r\n','\n').encode()).hexdigest()
    for p in base.rglob('*.py') if '__pycache__' not in p.parts}
manifest={'schema':'evomind.emergency_model_fallback.v1','model':'gpt-5.5','scope':'same_ui_manual_historical_recovery',
    'baseline_launcher_sha256':'0bf9f0a69f2f2f15377cb446a49b12c5811fd9f107633864b9fd6596215a5e22',
    'baseline_runtime':baseline,'payloads':{name:hashlib.sha256((destination/name).read_bytes()).hexdigest()
        for name in ['Start-Node.ps1','entry.py','runtime.zip','switch.py']}}
with (destination/'manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,sort_keys=True,indent=2)
print(json.dumps({'path':str(destination),'manifest_sha256':hashlib.sha256((destination/'manifest.json').read_bytes()).hexdigest()}))
