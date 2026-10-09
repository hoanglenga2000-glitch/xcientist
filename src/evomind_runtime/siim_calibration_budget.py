"""Budget for the nine SIIM comparison cases, including previously charged EV work."""
from __future__ import annotations
import math,sqlite3,time,uuid
from pathlib import Path

TOTAL_SECONDS=86400
CASE_SECONDS=7200

class SiimBudget:
    def __init__(self,path,ev_budget_path,limits=None):
        self.path=Path(path);self.ev_budget_path=Path(ev_budget_path)
        self.limits=dict(limits or {})
        from .siim_confirmed_recovery import load_recovery
        self.recovery=load_recovery(self.path)
        if self.recovery and not self.limits:raise ValueError('recovery_requires_source_limits')
        if self.limits:
            if (self.limits.get('schema')!='evomind.siim_source_based_limits.v1'
                    or self.limits.get('case_ceiling_seconds')!=86400
                    or self.limits.get('step_ceiling_seconds')!=32400
                    or self.limits.get('global_limit_seconds')!=TOTAL_SECONDS
                    or self.limits.get('headroom_factor')!=1.15
                    or not isinstance(self.limits.get('remaining_work_units'),dict)
                    or any(type(v) not in {int,float} or not 0<v<=6 for v in self.limits['remaining_work_units'].values())):
                raise ValueError('unverified_resource_limits')
        with sqlite3.connect(self.path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS cases(run_id TEXT PRIMARY KEY,started REAL NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,run_id TEXT,source_sha TEXT,reserved REAL,charged REAL,status TEXT)')
            if self.limits:
                c.execute('CREATE TABLE IF NOT EXISTS adaptive_allocations(run_id TEXT PRIMARY KEY,opened REAL,allowance REAL,prior_charged REAL,limits_sha TEXT)')
                c.execute('CREATE TABLE IF NOT EXISTS case_outcomes(run_id TEXT PRIMARY KEY,status TEXT,closed REAL)')
            if self.recovery:
                from .siim_recovery_windows import initialize
                initialize(c)

    def _adaptive_window(self,c,run_id,ev_charge):
        if self.recovery:
            from .siim_recovery_windows import remaining
            return remaining(c,run_id,ev_charge,self.recovery)
        import hashlib,json
        if run_id not in self.limits['remaining_work_units']:raise ValueError('unregistered_adaptive_case')
        digest=hashlib.sha256(json.dumps(self.limits,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        row=c.execute('SELECT opened,allowance,prior_charged,limits_sha FROM adaptive_allocations WHERE run_id=?',(run_id,)).fetchone()
        charged=c.execute('SELECT COALESCE(SUM(charged),0) FROM attempts WHERE run_id=?',(run_id,)).fetchone()[0]
        if row is None:
            global_used=c.execute('SELECT COALESCE(SUM(charged),0) FROM attempts').fetchone()[0]
            closed={r[0] for r in c.execute('SELECT run_id FROM case_outcomes')}
            if run_id in closed:raise ValueError('adaptive_case_already_closed')
            units={k:v for k,v in self.limits['remaining_work_units'].items() if k not in closed}
            available=max(0,TOTAL_SECONDS-ev_charge-global_used-180)
            allowance=min(self.limits['step_ceiling_seconds'],available,available*units[run_id]/sum(units.values())*self.limits['headroom_factor'])
            row=(time.time(),int(allowance),charged,digest)
            c.execute('INSERT INTO adaptive_allocations VALUES(?,?,?,?,?)',(run_id,*row))
        if row[3]!=digest:raise ValueError('adaptive_limits_changed_mid_case')
        # Neither an old case's start nor its charged attempts is reset. The
        # explicit amendment opens a separately recorded remaining-work window.
        return min(row[1]-(time.time()-row[0]),row[1]-(charged-row[2]))

    def mark_complete(self,run_id):
        if not self.limits:return
        with sqlite3.connect(self.path) as c:
            if c.execute("SELECT 1 FROM attempts WHERE run_id=? AND status IN ('reserved','uncertain','exceeded')",(run_id,)).fetchone():raise ValueError('case_not_settled')
            c.execute("INSERT OR IGNORE INTO case_outcomes VALUES(?,'completed',?)",(run_id,time.time()))
            if self.recovery:
                from .siim_recovery_windows import mark_complete
                mark_complete(c,run_id,self.recovery)

    def _ev_charge(self):
        with sqlite3.connect(self.ev_budget_path.as_uri()+'?mode=ro',uri=True) as c:
            c.execute('PRAGMA query_only=ON');rows=c.execute('SELECT charged,status FROM attempts').fetchall()
        if any(row[1] not in {'completed','failed'} for row in rows):raise ValueError('ev_settlement_incomplete')
        if any(not math.isfinite(row[0]) or row[0]<0 for row in rows):raise ValueError('ev_charge_invalid')
        return sum(row[0] for row in rows)

    def start_case(self,run_id):
        self._ev_charge()
        with sqlite3.connect(self.path) as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute('SELECT 1 FROM cases WHERE run_id=?',(run_id,)).fetchone():raise ValueError('case_already_started')
            c.execute('INSERT INTO cases VALUES(?,?)',(run_id,time.time()))

    def reserve(self,run_id,source_sha,requested):
        step_ceiling=self.limits.get('step_ceiling_seconds',CASE_SECONDS)
        if type(requested) is not int or not 120<=requested<=step_ceiling:raise ValueError('invalid_case_timeout')
        with sqlite3.connect(self.path,timeout=10) as c:
            c.execute('BEGIN IMMEDIATE');ev=self._ev_charge()
            started=c.execute('SELECT started FROM cases WHERE run_id=?',(run_id,)).fetchone()
            if started is None:raise ValueError('case_not_registered')
            elapsed=time.time()-started[0]
            case_ceiling=self.limits.get('case_ceiling_seconds',CASE_SECONDS)
            if not 0<=elapsed<case_ceiling:raise ValueError('case_wall_budget_exhausted')
            rows=c.execute('SELECT charged,status FROM attempts').fetchall()
            if any(r[1] not in {'completed','failed'} for r in rows):raise ValueError('siim_settlement_incomplete')
            seconds=min(requested,int(case_ceiling-elapsed),int(TOTAL_SECONDS-ev-sum(r[0] for r in rows)))
            if self.limits:seconds=min(seconds,int(self._adaptive_window(c,run_id,ev)))
            if seconds<120:raise ValueError('combined_gpu_budget_exhausted')
            attempt=uuid.uuid4().hex
            c.execute("INSERT INTO attempts VALUES(?,?,?,?,0,'reserved')",(attempt,run_id,source_sha,seconds))
        return attempt,seconds

    def settle(self,attempt,seconds,success,uncertain=False):
        if not math.isfinite(seconds) or seconds<0:raise ValueError('invalid_charge')
        with sqlite3.connect(self.path) as c:
            c.execute('BEGIN IMMEDIATE');row=c.execute('SELECT reserved,status FROM attempts WHERE id=?',(attempt,)).fetchone()
            if row is None or row[1]!='reserved':raise ValueError('invalid_settlement')
            status='uncertain' if uncertain else 'exceeded' if seconds>row[0] else 'completed' if success else 'failed'
            c.execute('UPDATE attempts SET charged=?,status=? WHERE id=?',(max(seconds,row[0]) if uncertain else seconds,status,attempt))

    def require_model_window(self,run_id):
        ev=self._ev_charge()
        with sqlite3.connect(self.path) as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT started FROM cases WHERE run_id=?',(run_id,)).fetchone()
            if row is None or not 0<=time.time()-row[0]<self.limits.get('case_ceiling_seconds',CASE_SECONDS)-200:raise ValueError('model_time_budget_exhausted')
            if self.limits and self._adaptive_window(c,run_id,ev)<200:raise ValueError('model_time_budget_exhausted')

    def remaining_case_seconds(self,run_id):
        ev=self._ev_charge()
        with sqlite3.connect(self.path) as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT started FROM cases WHERE run_id=?',(run_id,)).fetchone()
            if row is None:raise ValueError('case_not_registered')
            ceiling=self.limits.get('case_ceiling_seconds',CASE_SECONDS)-(time.time()-row[0])
            return max(0,min(ceiling,self._adaptive_window(c,run_id,ev)))

    def summary(self):
        ev=self._ev_charge()
        with sqlite3.connect(self.path) as c:rows=c.execute('SELECT reserved,charged,status FROM attempts').fetchall()
        charged=sum(r[1] for r in rows);reserved=sum(r[0] for r in rows if r[2]=='reserved')
        result={'total_limit_seconds':TOTAL_SECONDS,'ev_charged_seconds':ev,'siim_charged_seconds':charged,
                'pending_reserved_seconds':reserved,'remaining_seconds':max(0,TOTAL_SECONDS-ev-charged-reserved),
                'settlement_attention':any(r[2] not in {'completed','failed'} for r in rows),'attempt_count':len(rows)}
        if self.limits:
            with sqlite3.connect(self.path) as c:
                allocations=c.execute('SELECT run_id,opened,allowance,prior_charged FROM adaptive_allocations').fetchall()
            result.update(case_ceiling_seconds=86400,step_ceiling_seconds=32400,
                          allocation_rule='remaining_work_share_with_15_percent_headroom',
                          allocations=[dict(zip(['run_id','opened','allowance_seconds','prior_charged_seconds'],row)) for row in allocations])
        if self.recovery:
            with sqlite3.connect(self.path) as c:
                rows=c.execute('SELECT run_id,opened,allowance,prior_charged FROM recovery_allocations WHERE amendment_id=?',(self.recovery['amendment_id'],)).fetchall()
            result.update(recovery_amendment=self.recovery['amendment_id'],
                recovery_windows=[dict(run_id=r[0],opened=r[1],allowance_seconds=r[2],prior_charged_seconds=r[3],remaining_wall_seconds=max(0,r[1]+r[2]-time.time())) for r in rows],
                original_clocks_and_charges_preserved=True)
        return result
