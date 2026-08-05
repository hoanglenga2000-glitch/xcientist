from scripts.evolution_run_cli import _validate_local_gpu_candidate


def test_local_gpu_guard_rejects_leaky_target_encoding_and_missing_review_metrics():
    code = "encoded = train.groupby('city')['is_fraud'].mean()\n"
    violations = _validate_local_gpu_candidate(code)
    assert any("target mean" in item for item in violations)
    assert any("oof_decision_threshold" in item for item in violations)


def test_local_gpu_guard_accepts_chronological_probability_contract():
    code = """
from sklearn.metrics import f1_score, recall_score, precision_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
oof_decision_threshold = 0.37
metrics = {'f1': f1_score, 'recall': recall_score, 'precision': precision_score,
           'brier_score': brier_score_loss, 'oof_decision_threshold': oof_decision_threshold}
"""
    assert _validate_local_gpu_candidate(code) == []
