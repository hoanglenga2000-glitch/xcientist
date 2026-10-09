from types import SimpleNamespace
from scripts import emergency_gpt55_entry as entry

def test_emergency_entry_never_recovers_historical_work(monkeypatch,tmp_path):
    calls=[]
    runtime=SimpleNamespace(runtime_root=tmp_path,close=lambda:calls.append('close'),
        assistant=SimpleNamespace(recover_incomplete=lambda:(_ for _ in ()).throw(AssertionError('history replay'))))
    monkeypatch.setattr(entry,'AgentRuntime',lambda root:runtime)
    monkeypatch.setattr(entry,'ensure_token',lambda root:'fixture')
    monkeypatch.setattr(entry,'make_handler',lambda *args:None)
    monkeypatch.setattr(entry,'ThreadingHTTPServer',lambda *args:SimpleNamespace(
        serve_forever=lambda **kw:calls.append('serve'),server_close=lambda:calls.append('server_close')))
    entry.serve_manual(tmp_path,54321)
    assert calls==['serve','server_close','close']
