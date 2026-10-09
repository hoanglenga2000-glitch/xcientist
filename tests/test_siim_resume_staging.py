import hashlib,importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('siim_runner',Path(__file__).resolve().parents[1]/'src/evomind_runtime/siim_calibration_runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def fixture(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    code=b'pass';(old/'candidate.py').write_bytes(code);(old/'task.json').write_text('{}');(old/'trusted-runner.py').write_text('# pinned')
    identity={'arm':'fixed_baseline','seed':17,'candidate_sha256':hashlib.sha256(code).hexdigest(),'protocol_sha256':'a'*64,'data_manifest_sha256':'b'*64}
    (old/'fold-0.joblib').write_bytes(b'synthetic-model');(old/'fold-0-reference.npy').write_bytes(b'synthetic-reference')
    record={'fold':0}
    for field,name in [('model','fold-0.joblib'),('reference','fold-0-reference.npy')]:
        record[field]={'name':name,'bytes':(old/name).stat().st_size,'sha256':runner.digest(old/name)}
    checkpoint={'source_artifact_root':str(old),'identity':identity,'task_sha256':runner.digest(old/'task.json'),
                'trusted_runner_sha256':runner.digest(old/'trusted-runner.py'),'complete_folds':[record]}
    return old,new,{**identity,'resume_checkpoint':checkpoint}

def test_only_verified_completed_folds_are_staged(tmp_path):
    old,new,task=fixture(tmp_path)
    assert runner.stage_verified_checkpoints(task,new)==[0]
    assert (new/'fold-0.joblib').read_bytes()==(old/'fold-0.joblib').read_bytes()
    assert not (new/'fold-1.joblib').exists()

def test_changed_model_rejected_before_unpickling(tmp_path):
    old,new,task=fixture(tmp_path);(old/'fold-0.joblib').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='resume_file_changed'):runner.stage_verified_checkpoints(task,new)

def test_wrong_candidate_or_seed_not_reused(tmp_path):
    old,new,task=fixture(tmp_path);task['seed']=29
    with pytest.raises(ValueError,match='resume_identity'):runner.stage_verified_checkpoints(task,new)

def test_previous_destination_cannot_be_overwritten(tmp_path):
    old,new,task=fixture(tmp_path);(new/'fold-0.joblib').write_bytes(b'existing')
    with pytest.raises(FileExistsError):runner.stage_verified_checkpoints(task,new)
