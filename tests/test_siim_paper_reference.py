import hashlib,importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('paper_reference',Path(__file__).resolve().parents[1]/'scripts/extract_siim_paper_reference.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def pointer(data):
    return ('version https://git-lfs.github.com/spec/v1\noid sha256:'+hashlib.sha256(data).hexdigest()+'\nsize '+str(len(data))+'\n').encode()

def test_cached_blob_requires_both_hash_and_size():
    data=b'{"source":"synthetic-test"}'
    assert module.verify_cached_lfs(pointer(data),data)==hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError):module.verify_cached_lfs(pointer(data),data+b' ')

def test_conflicted_pointer_not_a_verified_source():
    data=b'{}'
    with pytest.raises(ValueError):module.verify_cached_lfs(b'<<<<<<< HEAD\n'+pointer(data),data)

def test_wrong_bytes_same_size_rejected():
    with pytest.raises(ValueError):module.verify_cached_lfs(pointer(b'abc'),b'cba')
