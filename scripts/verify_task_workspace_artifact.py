"""Verify the browser-observed artifact through the isolated authenticated API.

This is an HTTP download/hash check, not proof of the browser's download folder.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/EV12/task-ui-local-20261008/data/workspace/runtime')
OUTPUT = ROOT / 'artifacts/advanced-tools-migration-20261008'


def verify(artifact_id):
    if not re.fullmatch(r'artifact_[a-f0-9]{32}', artifact_id):
        raise ValueError('invalid_artifact_id')
    with sqlite3.connect((DATA / 'runtime.sqlite3').as_uri() + '?mode=ro', uri=True) as database:
        database.row_factory = sqlite3.Row
        artifact = dict(database.execute('SELECT * FROM deliverables WHERE id=?', (artifact_id,)).fetchone())
    with sqlite3.connect((DATA / 'principal_access.sqlite3').as_uri() + '?mode=ro', uri=True) as database:
        tenant, owner = database.execute("SELECT tenant_id,owner_id FROM resource_owners WHERE kind='session' AND resource_id=?", (artifact['session_id'],)).fetchone()
    with sqlite3.connect((DATA / 'user_tasks.sqlite3').as_uri() + '?mode=ro', uri=True) as database:
        task_id = database.execute('SELECT task_id FROM user_task_runs WHERE run_id=? AND tenant=? AND owner=?', (artifact['run_id'], tenant, owner)).fetchone()[0]
    token = (DATA / 'runtime.token').read_text(encoding='ascii').strip()

    def get(principal):
        connection = http.client.HTTPConnection('127.0.0.1', 8877, timeout=10)
        try:
            connection.request('GET', f'/v1/artifacts/{artifact_id}?download=1', headers={
                'Authorization': 'Bearer ' + token, 'X-EvoMind-Access-Scope': 'user.v1',
                'X-EvoMind-Tenant-Id': tenant, 'X-EvoMind-Principal-Id': principal})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    status, headers, body = get(owner)
    assert status == 200 and len(body) == artifact['bytes']
    digest = hashlib.sha256(body).hexdigest()
    assert digest == artifact['sha256'] == headers['X-Artifact-SHA256']
    assert 'attachment' in headers['Content-Disposition']
    denied, _, _ = get('different-fixture-user')
    assert denied in {403, 404}
    (OUTPUT / 'api-downloaded-artifact.md').write_bytes(body)
    result = {'task_id': task_id, 'run_id': artifact['run_id'], 'artifact_id': artifact_id,
        'bytes': len(body), 'sha256': digest, 'other_user_status': denied,
        'http_download_verified': True, 'browser_saved_download_verified': False,
        'real_model': False, 'gpu': False, 'scientific_claim_verified': False}
    (OUTPUT / 'artifact-download.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('artifact_id')
    verify(parser.parse_args().artifact_id)
