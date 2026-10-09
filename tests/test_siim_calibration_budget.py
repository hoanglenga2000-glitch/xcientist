import sqlite3,time
import pytest
from evomind_runtime.siim_calibration_budget import SiimBudget

def ledger(tmp_path,charge=4680.205,status='completed'):
    old=tmp_path/'ev.sqlite3'
    with sqlite3.connect(old) as c:
        c.execute('CREATE TABLE attempts(charged REAL,status TEXT)');c.execute('INSERT INTO attempts VALUES(?,?)',(charge,status))
    return SiimBudget(tmp_path/'siim.sqlite3',old)

def test_previous_ev_usage_counts_and_cannot_reset(tmp_path):
    b=ledger(tmp_path);b.start_case('r')
    with pytest.raises(ValueError):b.start_case('r')
    attempt,seconds=b.reserve('r','a'*64,7200);assert 7100<seconds<=7200
    b.settle(attempt,40,False)
    assert b.summary()['remaining_seconds']==pytest.approx(86400-4680.205-40)

def test_pending_ev_blocks_start(tmp_path):
    b=ledger(tmp_path,status='reserved')
    with pytest.raises(ValueError,match='ev_settlement'):b.start_case('r')

def test_budget_enforces_serial_execution_and_uncertainty(tmp_path):
    b=ledger(tmp_path);b.start_case('r');a,seconds=b.reserve('r','a'*64,500)
    with pytest.raises(ValueError):b.reserve('r','b'*64,500)
    b.settle(a,5,False,uncertain=True)
    assert b.summary()['siim_charged_seconds']==500
    with pytest.raises(ValueError):b.reserve('r','c'*64,500)

def test_total_limit_is_not_restarted_on_competition_switch(tmp_path):
    b=ledger(tmp_path,86300);b.start_case('r')
    with pytest.raises(ValueError,match='combined_gpu_budget'):b.reserve('r','a'*64,7200)

def test_model_stops_before_two_hour_deadline(tmp_path,monkeypatch):
    b=ledger(tmp_path);b.start_case('r');b.require_model_window('r')
    now=time.time();monkeypatch.setattr('evomind_runtime.siim_calibration_budget.time.time',lambda:now+7050)
    with pytest.raises(ValueError):b.require_model_window('r')
