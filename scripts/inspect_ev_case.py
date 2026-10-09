"""Read only the requested EV case's tool and model-event metadata."""
import argparse,json,sqlite3,time
from pathlib import Path
BASE=Path('C:/ProgramData/EvoMind')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--compact',action='store_true');args=p.parse_args()
with sqlite3.connect((BASE/'data/workspace/runtime/runtime.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
 c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
 row=c.execute('SELECT id,status,metadata_json FROM sessions WHERE id=?',(args.run,)).fetchone()
 if row is None:raise ValueError('Missing exact case')
 meta=json.loads(row['metadata_json'])
 calls=[];candidates=[]
 for r in c.execute('SELECT id,tool_name,status,result_json FROM tool_calls WHERE session_id=? ORDER BY created_at',(args.run,)):
  result=json.loads(r['result_json'] or '{}')
  calls.append({'id':r['id'],'tool':r['tool_name'],'status':r['status'],'ok':result.get('ok'),'error':result.get('error'),'summary':result.get('summary')})
  content=result.get('content') or {}
  if content.get('metrics'):
   metric=content['metrics'];candidates.append({k:metric.get(k) for k in ['candidate_sha256','oof_roc_auc','runtime_seconds','submission_sha256']})
  if content.get('diagnostic'):calls[-1]['diagnostic']=content['diagnostic']
 attempts=[json.loads(r[0]) for r in c.execute("SELECT payload_json FROM events WHERE session_id=? AND event_type='model.transport_attempt' ORDER BY seq",(args.run,))]
 payload={'run_id':args.run,'status':row['status'],'calibration':meta.get('official_calibration'),'tools':meta.get('run_allowed_tool_names'),'tool_count':len(calls),'tool_calls':calls[-5:] if args.compact else calls,'model_attempts':attempts[-3:] if args.compact else attempts[-5:],'candidates':candidates}
with sqlite3.connect((BASE/'data/workspace/runtime/ev_calibration_budget.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
 c.execute('PRAGMA query_only=ON');clock=c.execute('SELECT started FROM cases WHERE run_id=?',(args.run,)).fetchone()
 rows=c.execute('SELECT reserved,charged,status FROM attempts').fetchall()
 payload['case_elapsed_seconds']=time.time()-clock[0] if clock else None
 payload['budget']={'charged_seconds':sum(r[1] for r in rows),'reserved_seconds':sum(r[0] for r in rows if r[2]=='reserved'),'attempts':len(rows),'statuses':{s:sum(r[2]==s for r in rows) for s in set(r[2] for r in rows)}}
print(json.dumps(payload))
