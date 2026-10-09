import json,sqlite3,time
import pytest
from evomind_runtime.siim_calibration_budget import SiimBudget

def fixture(tmp_path):
    old=tmp_path/'ev.sqlite3'
    with sqlite3.connect(old) as c:
        c.execute('CREATE TABLE attempts(charged REAL,status TEXT)');c.execute("INSERT INTO attempts VALUES(4680.205,'completed')")
    limits={'schema':'evomind.siim_source_based_limits.v1','case_ceiling_seconds':86400,'step_ceiling_seconds':32400,
            'global_limit_seconds':86400,'headroom_factor':1.15,'remaining_work_units':{'r0':4,'r1':6,'r2':6}}
    return SiimBudget(tmp_path/'siim.sqlite3',old,limits),limits

def test_official_ceiling_replaces_legacy_two_hours_without_resetting_clock(tmp_path,monkeypatch):
    b,_=fixture(tmp_path);b.start_case('r0')
    with sqlite3.connect(b.path) as c:
        c.execute('UPDATE cases SET started=started-10000');original=c.execute('SELECT started FROM cases').fetchone()[0]
    a,seconds=b.reserve('r0','a'*64,32400)
    assert 7200<seconds<=32400
    with sqlite3.connect(b.path) as c:assert c.execute('SELECT started FROM cases').fetchone()[0]==original
    b.settle(a,50,False)
    with pytest.raises(ValueError):b.start_case('r0')

def test_allocation_is_not_reopened_on_retry(tmp_path):
    b,_=fixture(tmp_path);b.start_case('r0');a,first=b.reserve('r0','a'*64,32400);b.settle(a,500,False)
    a,second=b.reserve('r0','a'*64,32400)
    assert second<=first-499
    b.settle(a,40,True)
    with sqlite3.connect(b.path) as c:assert c.execute('SELECT COUNT(*) FROM adaptive_allocations').fetchone()[0]==1

def test_prior_failed_charges_remain_in_global_total(tmp_path):
    b,_=fixture(tmp_path);b.start_case('r0')
    with sqlite3.connect(b.path) as c:c.execute("INSERT INTO attempts VALUES('old','r0','source',6000,5535.405,'failed')")
    a,seconds=b.reserve('r0','a'*64,32400)
    summary=b.summary()
    assert summary['ev_charged_seconds']==pytest.approx(4680.205)
    assert summary['siim_charged_seconds']==pytest.approx(5535.405)
    assert summary['remaining_seconds']==pytest.approx(86400-4680.205-5535.405-seconds)

def test_unknown_source_limits_cannot_expand_total(tmp_path):
    b,limits=fixture(tmp_path);limits['global_limit_seconds']=129600
    with pytest.raises(ValueError,match='unverified_resource_limits'):SiimBudget(b.path,b.ev_budget_path,limits)

def test_case_completion_releases_only_unused_share(tmp_path):
    b,_=fixture(tmp_path);b.start_case('r0');a,seconds=b.reserve('r0','a'*64,32400)
    with pytest.raises(ValueError):b.mark_complete('r0')
    b.settle(a,100,True);b.mark_complete('r0');b.start_case('r1')
    a,second=b.reserve('r1','b'*64,32400)
    assert second>seconds

def test_model_guard_obeys_allocation_deadline(tmp_path,monkeypatch):
    b,_=fixture(tmp_path);b.start_case('r0');b.require_model_window('r0')
    with sqlite3.connect(b.path) as c:opened,allowance=c.execute('SELECT opened,allowance FROM adaptive_allocations').fetchone()
    monkeypatch.setattr('evomind_runtime.siim_calibration_budget.time.time',lambda:opened+allowance-100)
    with pytest.raises(ValueError,match='model_time_budget'):b.require_model_window('r0')
