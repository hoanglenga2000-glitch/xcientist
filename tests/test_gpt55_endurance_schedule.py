import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('gpt55_soak', Path(__file__).parents[1] / 'scripts/run_model_endurance_acceptance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pacing_uses_real_window_without_oversized_blocking_sleep(monkeypatch, tmp_path):
    clock, waits, snapshots = [10.0], [], []
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(module.time, 'sleep', sleep)
    monkeypatch.setattr(module, 'write_json', lambda _path, value: snapshots.append(dict(value)))
    report = {'qualified': False}
    module.wait_for_case_slot(0, 1, 180, 5400, report, tmp_path / 'receipt.json')
    assert clock[0] == 180
    assert max(waits) <= 30
    assert snapshots and all(row['qualified'] is False for row in snapshots)
    assert 5400 // 180 >= 20


def test_slow_case_never_sleeps_past_end_and_continuous_mode_is_unchanged(monkeypatch, tmp_path):
    clock = [5390.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(module, 'write_json', lambda *_: None)
    module.wait_for_case_slot(0, 31, 180, 5400, {}, tmp_path / 'receipt.json')
    assert clock[0] == 5400
    module.wait_for_case_slot(0, 99, 0, 5400, {}, tmp_path / 'receipt.json')
    assert clock[0] == 5400
