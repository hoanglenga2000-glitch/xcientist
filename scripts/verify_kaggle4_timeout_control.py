from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
ROOT=Path.cwd()
run_dir=ROOT/'experiments'/'spaceship_titanic'/'wr_2026-06-24T21-07-14.865947_96435552'
out_json=ROOT/'workspace'/'kaggle4_timeout_control_verification_20260624.json'
out_md=ROOT/'reports'/'KAGGLE4_TIMEOUT_CONTROL_VERIFICATION_20260624.md'
required=['launcher_manifest.json','timeout_manifest.json','failure_review.json','task_state_machine.json','gate_engine.json','agent_trace.json','agent_trace.jsonl']
forbidden=['metrics.json','score_promotion_gate.json','submission.csv']
checks=[]
def add(name, passed, detail, evidence):
    checks.append({'name':name,'status':'passed' if passed else 'failed','detail':detail,'evidence':evidence})
add('timeout_run_dir_exists',run_dir.is_dir(),str(run_dir),str(run_dir))
for f in required:
    p=run_dir/f
    add(f'required_{f}',p.exists(),f'exists={p.exists()}',str(p.relative_to(ROOT)) if p.exists() else str(p))
for f in forbidden:
    p=run_dir/f
    add(f'no_score_artifact_{f}',not p.exists(),f'exists={p.exists()}',str(p.relative_to(ROOT)) if p.exists() else str(p))
if (run_dir/'timeout_manifest.json').exists():
    tm=json.loads((run_dir/'timeout_manifest.json').read_text(encoding='utf-8'))
    add('timeout_policy_blocks_promotion','not eligible for promotion' in tm.get('result_policy',''),tm.get('result_policy',''),str((run_dir/'timeout_manifest.json').relative_to(ROOT)))
else:
    tm={}
all_passed=all(c['status']=='passed' for c in checks)
report={'schema':'academic_research_os.timeout_control_verification.v1','created_at':datetime.now().isoformat(timespec='seconds'),'status':'passed' if all_passed else 'failed','run_id':run_dir.name,'task_id':'spaceship_titanic','checks_passed':sum(c['status']=='passed' for c in checks),'checks_total':len(checks),'checks':checks,'claim_boundary':'Timeout run proves controlled failure handling only; it is not score evidence.'}
out_json.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# Kaggle4 Timeout Control Verification','',f"- Status: `{report['status']}`",f"- Checks: `{report['checks_passed']}/{report['checks_total']}`",f"- Run: `{run_dir.relative_to(ROOT).as_posix()}`",'- Claim boundary: timeout control only, not score evidence.','','## Checks','']
for c in checks:
    lines.append(f"- `{c['status']}` {c['name']}: {c['detail']}")
out_md.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'status':report['status'],'checks':f"{report['checks_passed']}/{report['checks_total']}",'json':out_json.relative_to(ROOT).as_posix(),'md':out_md.relative_to(ROOT).as_posix()},ensure_ascii=False,indent=2))
raise SystemExit(0 if all_passed else 1)
