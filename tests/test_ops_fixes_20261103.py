"""2026-11-03 fixes: authenticated model probe, orphan-kill pending queue, invalid-config loop guard."""
import urllib.error

from evomind_runtime import model_probe
from evomind_runtime.gpu_scheduler import GpuScheduler


class _Config:
    def __init__(self, model):
        self.model = model


class _Transport:
    def __init__(self, name, model):
        self.name = name
        self.config = _Config(model)

    def build(self, messages, system, tools, max_tokens, temperature):
        return 'https://example.invalid/v1/chat/completions', {'Authorization': 'Bearer secret-key'}, {'model': self.config.model, 'tools': []}


def test_probe_treats_auth_rejection_as_failure_and_never_records_the_key(tmp_path):
    def post(url, headers, payload, timeout):
        if payload['model'] == 'bad':
            raise urllib.error.HTTPError(url, 401, 'Unauthorized', {}, None)
        assert 'tools' not in payload
        return {'model': payload['model']}

    cache = tmp_path / 'probe.json'
    report = model_probe.probe_providers(cache, force=True, transports=lambda: [_Transport('anthropic', 'claude-opus-5-5'), _Transport('openai', 'bad')], post=post)
    assert report['ok'] is False
    states = {item['model']: item['state'] for item in report['providers']}
    assert states == {'claude-opus-5-5': 'ok', 'bad': 'auth_failed'}
    assert 'secret-key' not in cache.read_text(encoding='utf-8')

    good = model_probe.probe_providers(cache, force=True, transports=lambda: [_Transport('anthropic', 'claude-opus-5-5')], post=post)
    assert good['ok'] is True and good['providers'][0]['identity_ok'] is True
    calls = []
    cached = model_probe.probe_providers(cache, transports=lambda: calls.append(1) or [], post=post)
    assert cached['cached'] is True and calls == []


def test_unconfirmed_kill_is_queued_for_reconcile(tmp_path):
    sched = GpuScheduler(tmp_path / 'gpu.sqlite3')
    try:
        with sched._tx() as db:
            db.execute("INSERT INTO train_jobs(id, tenant_id, owner_id, run_id, vram_mb, max_hours, status, created_at) VALUES('gjob_x','tenant_a','o','run_x',1,1,'cancelled',1)")
        sched.mark_orphan_pending('gjob_x', 'tenant_a')
        row = sched._db().execute("SELECT orphan_kill_at FROM train_jobs WHERE id='gjob_x'").fetchone()
        assert row[0] is not None
        kinds = [r[0] for r in sched._db().execute("SELECT kind FROM gpu_audit WHERE job_id='gjob_x'")]
        assert 'orphan_kill_pending' in kinds
    finally:
        sched.close()


def test_invalid_config_guard_stops_after_three_refusals():
    from types import SimpleNamespace
    from evomind_runtime import tools

    refused = {'tool_name': 'managed_tensor_train', 'status': 'failed', 'result': {'error': 'tenant_training_config_invalid'}}

    class _Store:
        def __init__(self, rows):
            self.rows = rows

        def list_tool_calls(self, session_id, limit=500):
            return self.rows

    context = SimpleNamespace(store=_Store([refused] * 3), session_id='s', approval_verified=True, progress=None)
    result = tools._tenant_tensor_train({}, context)
    assert result.ok is False and result.error == 'tenant_training_config_invalid_repeated'
    assert '停止自动重试' in result.content['user_message']

    context = SimpleNamespace(store=_Store([refused] * 2), session_id='s', approval_verified=False, progress=None)
    assert tools._tenant_tensor_train({}, context).error == 'exact_approval_required'
