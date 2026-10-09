"""Copy exact existing distributions into the isolated campaign environment; no downloads."""
import hashlib,importlib.metadata,json,os,shutil,subprocess,sys,time
from pathlib import Path,PurePosixPath
from packaging.requirements import Requirement

ROOT=Path(os.environ['EVOMIND_COMPETITION_DATA_ROOT']);OUT=Path(os.environ['EVOMIND_COMPETITION_RECEIPT'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 began=time.monotonic();r={'schema':'evomind.ev_environment_repair.v1','competition':'playground-series-s6e9','training_started':False,'gpu_training_hours':0,'network_downloads':0}
 try:
  if str(ROOT.resolve())!='/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/ev_public_calibration_20260908':raise ValueError('root_mismatch')
  manifest=ROOT/'.evomind/data-manifest.json';before=sha(manifest);info=json.loads(manifest.read_text())
  target=Path(info['dependency_root']).resolve()
  if target!=ROOT/'.evomind/preinstalled-only':raise ValueError('dependency_root_mismatch')
  env=dict(os.environ);env['PYTHONNOUSERSITE']='1';env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']=str(target)
  probe="import importlib.metadata,json;names="+repr(list(info['dependencies']))+";out={}\nfor n in names:\n try: out[n]=importlib.metadata.version(n)\n except importlib.metadata.PackageNotFoundError: out[n]=None\nprint(json.dumps(out))"
  initial=subprocess.run([sys.executable,'-c',probe],env=env,capture_output=True,text=True,timeout=30)
  if initial.returncode:raise ValueError('isolation_probe_failed')
  r['isolated_before']=json.loads(initial.stdout)
  queue=list(info['dependencies']);distributions={};files={}
  while queue:
   name=queue.pop(0);normalized=name.lower().replace('_','-')
   if normalized in distributions:continue
   distribution=importlib.metadata.distribution(name)
   distributions[normalized]=distribution.version
   for requirement in distribution.requires or []:
    req=Requirement(requirement)
    if req.marker is None or req.marker.evaluate({'extra':''}):queue.append(req.name)
   for item in distribution.files or []:
    rel=PurePosixPath(str(item))
    if rel.is_absolute() or '..' in rel.parts or '__pycache__' in rel.parts:continue
    source=Path(distribution.locate_file(item))
    if not source.is_file():continue
    destination=target/Path(*rel.parts)
    destination.resolve().relative_to(target)
    digest=sha(source)
    if destination.exists():
     if sha(destination)!=digest:raise ValueError('existing_dependency_file_differs')
    else:
     destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    files[rel.as_posix()]={'sha256':digest,'bytes':source.stat().st_size}
  test="import numpy,pandas,sklearn,catboost,scipy,joblib,importlib.metadata,json;print(json.dumps({n:importlib.metadata.version(n) for n in "+repr(list(info['dependencies']))+"}))"
  checked=subprocess.run([sys.executable,'-c',test],env=env,capture_output=True,text=True,timeout=90)
  if checked.returncode:raise ValueError('isolated_import_validation_failed')
  versions=json.loads(checked.stdout)
  if versions!=info['dependencies']:raise ValueError('dependency_versions_differ')
  receipt={'schema':'evomind.ev_isolated_dependencies.v1','packages':distributions,'files':files,'source':'existing_installed_distributions','downloads':0}
  path=ROOT/'.evomind/dependency-files.json'
  with path.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2)
  if sha(manifest)!=before:raise ValueError('data_manifest_changed')
  r.update(status='completed',isolated_after=versions,copied_files=len(files),copied_bytes=sum(x['bytes'] for x in files.values()),
           dependency_manifest_sha256=sha(path),data_manifest_unchanged=True,data_manifest_sha256=before)
 except Exception as e:r.update(status='failed',error_type=type(e).__name__,code=str(e) if isinstance(e,ValueError) else 'dependency_repair_failed')
 r['elapsed_seconds']=time.monotonic()-began;OUT.write_text(json.dumps(r,indent=2))
 return 0 if r['status']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
