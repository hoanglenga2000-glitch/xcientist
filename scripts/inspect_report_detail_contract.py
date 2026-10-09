"""Read-only comparison of the exact visible Run list and detail contracts."""
import http.client,json,sqlite3
from pathlib import Path
root=Path('C:/ProgramData/EvoMind')
token=(root/'data/workspace/runtime/runtime.token').read_text().strip()
run_id='run_5de9ca8d34b448cfbb6088b50d3c5d7b'
with sqlite3.connect((root/'data/workspace/runtime/principal_access.sqlite3').as_uri()+'?mode=ro',uri=True) as database:
    owner=database.execute("SELECT tenant_id,owner_id FROM resource_owners WHERE kind='session' AND resource_id=?",(run_id,)).fetchone()
assert owner
def get(path):
    c=http.client.HTTPConnection('127.0.0.1',8765,timeout=10)
    try:
        c.request('GET',path,headers={'Authorization':'Bearer '+token,'X-EvoMind-Access-Scope':'user.v1',
            'X-EvoMind-Tenant-Id':owner[0],'X-EvoMind-Principal-Id':owner[1]})
        r=c.getresponse();assert r.status==200;return json.loads(r.read())
    finally:c.close()
listing=get('/v1/runs?limit=100')
row=next(r for r in listing['runs'] if r['id']==run_id)
detail=get('/v1/runs/'+run_id)
print(json.dumps({'run_id':run_id,'list_has_answer':'answer' in row,
    'detail_answer_characters':len(detail.get('answer') or ''),'answer_is_current':detail.get('answer_is_current'),
    'status':detail['status'],'artifact_count':len(detail.get('artifacts') or []),'project_id':detail.get('project_id'),
    'model':detail.get('model')}))
