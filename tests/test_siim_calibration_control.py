from pathlib import Path
import pytest
from evomind_runtime.siim_calibration_control import validate_source

def test_fixed_baseline_source_accepted_without_running_it():
    source=(Path(__file__).resolve().parents[1]/'scripts/siim_fixed_convnext_baseline.py').read_text()
    validate_source(source)

@pytest.mark.parametrize('source',[
    'import socket\ndef fit_model(a,b,c,d,e): pass',
    'def fit_model(a,b,c,d,e): return eval("1")',
    'import torch\ndef fit_model(a,b,c,d,e): return torch.utils.data.DataLoader(a,num_workers=32)',
    'def fit_model(a,b,c,d,e): return a.__class__.__mro__'])
def test_unapproved_interfaces_rejected(source):
    with pytest.raises(ValueError):validate_source(source)
