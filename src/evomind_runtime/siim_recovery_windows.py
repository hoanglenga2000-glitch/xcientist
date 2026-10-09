"""Append-only continuation windows; existing allocations and charges stay intact."""
import time


def initialize(connection):
    connection.execute('CREATE TABLE IF NOT EXISTS recovery_allocations(amendment_id TEXT,run_id TEXT,opened REAL,allowance REAL,prior_charged REAL,amendment_sha TEXT,PRIMARY KEY(amendment_id,run_id))')
    connection.execute('CREATE TABLE IF NOT EXISTS recovery_outcomes(amendment_id TEXT,run_id TEXT,status TEXT,closed REAL,PRIMARY KEY(amendment_id,run_id))')


def remaining(connection,run_id,ev_charge,amendment):
    if run_id not in amendment['work_units']:
        raise ValueError('recovery_case_not_authorized')
    key=amendment['amendment_id']
    closed={r[0] for r in connection.execute("SELECT run_id FROM recovery_outcomes WHERE amendment_id=? AND status='completed'",(key,))}
    if run_id in closed:
        raise ValueError('recovery_case_already_completed')
    row=connection.execute('SELECT opened,allowance,prior_charged,amendment_sha FROM recovery_allocations WHERE amendment_id=? AND run_id=?',(key,run_id)).fetchone()
    charged=connection.execute('SELECT COALESCE(SUM(charged),0) FROM attempts WHERE run_id=?',(run_id,)).fetchone()[0]
    if row is None:
        total=connection.execute('SELECT COALESCE(SUM(charged),0) FROM attempts').fetchone()[0]
        units={key:value for key,value in amendment['work_units'].items() if key not in closed}
        available=max(0,86400-ev_charge-total-180)
        allowance=min(amendment['step_ceiling_seconds'],available,
                      available*units[run_id]/sum(units.values())*amendment['headroom_factor'])
        row=(time.time(),int(allowance),charged,amendment['amendment_sha256'])
        connection.execute('INSERT INTO recovery_allocations VALUES(?,?,?,?,?,?)',(key,run_id,*row))
    if row[3]!=amendment['amendment_sha256']:
        raise ValueError('recovery_amendment_changed_mid_case')
    return min(row[1]-(time.time()-row[0]),row[1]-(charged-row[2]))


def mark_complete(connection,run_id,amendment):
    connection.execute("INSERT OR IGNORE INTO recovery_outcomes VALUES(?,?,'completed',?)",(amendment['amendment_id'],run_id,time.time()))
