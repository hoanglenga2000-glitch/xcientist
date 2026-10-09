"""Allowlisted diagnostic metadata; never disclose exception text or raw rows."""
from __future__ import annotations
import hashlib
import re


def summarize_error(raw: bytes) -> dict:
    if len(raw) > 65536:
        raise ValueError('diagnostic_size_exceeded')
    text = raw.decode('utf-8', 'replace')
    lines = text.splitlines()
    last = lines[-1] if lines else ''
    match = re.match(r'(?:[A-Za-z_][A-Za-z0-9_]*\.)*([A-Za-z][A-Za-z0-9_]{0,60}):', last)
    exception = match.group(1) if match else 'unknown'
    allowed_types = {'ValueError','TypeError','KeyError','AttributeError','CatBoostError',
                     'MemoryError','RuntimeError','ImportError','ModuleNotFoundError'}
    if exception not in allowed_types:
        exception = 'unknown'
    # All output strings below are literals, not values recovered from log text.
    rules = {
        'prediction_feature_schema_mismatch': r'columns are missing|feature.*(?:not found|missing|not present)|(?:expected|need).*features?|feature.*model.*(?:pool|dataset)|(?:pool|dataset).*feature.*model',
        'numeric_conversion_failed': r'could not convert.*(?:float|double)|Cannot convert.*(?:float|double)',
        'nonfinite_input': r'infinity|infinite|NaN',
        'categorical_type_invalid': r'Invalid type for cat_feature|cat_features.*must be',
        'cuda_failure': r'CUDA error|cudaError',
        'out_of_memory': r'out of memory|bad allocation',
        'unexpected_argument': r'unexpected keyword argument',
        'shape_mismatch': r'shape mismatch|inconsistent.*samples',
    }
    codes = [code for code, pattern in rules.items() if re.search(pattern, last, re.I)]
    for code in ('invalid_model_classes','invalid_model_probabilities','probability_rows_invalid',
                 'independent_reload_failed','dependency_version_changed','data_file_changed'):
        if last == 'ValueError: '+code:
            codes.append(code)
    phase = 'fit_or_import'
    if re.search(r', in (?:probabilities|predict_proba|_base_predict)\n', text):
        phase = 'prediction'
    return {'exception_type':exception,'phase':phase,'diagnostic_codes':codes,
            'operator_log_sha256':hashlib.sha256(raw).hexdigest(),'raw_text_withheld':True}


def collect_failure(context, task, attempt):
    """Collect a single small private log after a new complete identity check."""
    from .tools import _hpc_verify, _load_bound_hpc_config
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    from xsci.terminal_tools import hpc_identity_evidence_complete
    binding = context.metadata['managed_hpc_identity']
    verified = _hpc_verify({}, context)
    if not verified.ok or not hpc_identity_evidence_complete(
            verified.content, expected_profile=binding['credential_profile'], expected_job_id=binding['job_id']):
        raise ValueError('diagnostic_identity_failed')
    root = task['artifact_root']
    if not re.fullmatch(r'[0-9a-f]{32}', attempt) or not root.endswith('/'+attempt):
        raise ValueError('diagnostic_attempt_mismatch')
    client = connect_ssh(_load_bound_hpc_config(binding['credential_profile'],binding['job_id'],context.session_id),timeout=30)
    try:
        sftp = client.open_sftp()
        try:
            path = root+'/operator-error.txt'
            import stat
            info = sftp.lstat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
                raise ValueError('diagnostic_file_rejected')
            with sftp.open(path,'rb') as stream:
                raw = stream.read(65537)
            return {'schema':'evomind.ev_failure_diagnostic.v1','attempt_id':attempt,
                    **summarize_error(raw),'fresh_identity_samples':5}
        finally:
            sftp.close()
    finally:
        client.close()
