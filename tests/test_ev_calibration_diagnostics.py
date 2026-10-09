import json
import pytest
from evomind_runtime.ev_calibration_diagnostics import summarize_error


def test_predict_schema_metadata_without_values():
    raw=b'Traceback:\n  File "/private/run.py", line 79, in probabilities\n    p=model.predict_proba(x)\nValueError: columns are missing: {"private_feature"}\n'
    result=summarize_error(raw)
    assert result['phase']=='prediction'
    assert result['diagnostic_codes']==['prediction_feature_schema_mismatch']
    assert 'private' not in json.dumps(result)


def test_exception_messages_and_data_are_not_returned():
    raw=b'ValueError: could not convert string to float: "sensitive-row-value"\n'
    result=summarize_error(raw)
    assert result['exception_type']=='ValueError'
    assert result['diagnostic_codes']==['numeric_conversion_failed']
    assert 'sensitive' not in json.dumps(result)


def test_unknown_exception_name_is_not_leaked():
    result=summarize_error(b'PrivateCredentialNamedException: arbitrary secret\n')
    assert result['exception_type']=='unknown'
    assert 'PrivateCredential' not in json.dumps(result)


def test_library_exception_and_safe_phase():
    result=summarize_error(b'  File "secret", line 1, in predict_proba\n_catboost.CatBoostError: CUDA error\n')
    assert result['exception_type']=='CatBoostError'
    assert result['phase']=='prediction'
    assert result['diagnostic_codes']==['cuda_failure']


def test_oversized_log_rejected():
    with pytest.raises(ValueError):summarize_error(b'x'*65537)
