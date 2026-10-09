"""Read-only remaining allocation windows from the application budget ledger."""
import base64
import json
from prepare_siim_web_bridge import connect
from check_siim_training_watch import powershell, PYTHON

code='''import sqlite3,json,time
from pathlib import Path
p=Path('C:/ProgramData/EvoMind/data/workspace/runtime/siim_calibration_budget.sqlite3')
with sqlite3.connect(p.as_uri()+'?mode=ro',uri=True) as c:
 c.execute('PRAGMA query_only=ON')
 rows=c.execute('SELECT run_id,opened,allowance,prior_charged FROM adaptive_allocations').fetchall()
 print(json.dumps([dict(run_id=r[0],opened=r[1],allowance=r[2],remaining_wall_seconds=r[1]+r[2]-time.time()) for r in rows]))
'''
if __name__=='__main__':
    client=connect()
    try:
        encoded=base64.b64encode(code.encode()).decode()
        inline='import base64;exec(base64.b64decode('+repr(encoded)+'))'
        result=powershell(client,"& '"+PYTHON+"' -X utf8 -c '"+inline.replace("'","''")+"'")
        print(json.dumps(result))
    finally:
        client.close()
