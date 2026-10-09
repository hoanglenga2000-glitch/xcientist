import hashlib
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('soak_harness', Path(__file__).parents[1] / 'scripts/run_model_endurance_acceptance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def records():
    return [{'tool_name': name, 'status': 'completed', 'arguments': {
        'path': 'outputs/summary.json', 'relative_path': 'outputs/summary.json'},
        'result': {'ok': True, 'content': {'receipt': {'ok': True, 'sha256': 'a' * 64}}}}
        for name in ['file_read', 'file_write', 'directory_hash', 'artifact_publish']]


def test_failed_hash_cannot_pass_or_count_as_an_executed_success(tmp_path):
    calls = records()
    calls[2].update(status='failed', result={'ok': False, 'error': 'unknown directory'})
    evidence = module.verify_case_tool_evidence(calls, 'a' * 64, tmp_path)
    assert not evidence['required_tool_steps_passed']
    assert not evidence['hash_check_passed']
    assert evidence['successful_tool_count'] == 3


def test_successful_hash_must_match_the_actual_output_and_file(tmp_path):
    calls = records()
    assert module.verify_case_tool_evidence(calls, 'a' * 64, tmp_path)['required_tool_steps_passed']
    assert not module.verify_case_tool_evidence(calls, 'b' * 64, tmp_path)['required_tool_steps_passed']
    calls[2]['arguments']['relative_path'] = 'inputs/data.json'
    assert not module.verify_case_tool_evidence(calls, 'a' * 64, tmp_path)['required_tool_steps_passed']


def test_hash_recovery_requires_a_later_successful_receipt(tmp_path):
    calls = records()
    failed = {**calls[2], 'status': 'failed', 'result': {'ok': False}}
    calls.insert(2, failed)
    result = module.verify_case_tool_evidence(calls, 'a' * 64, tmp_path)
    assert result['required_tool_steps_passed']
    assert result['successful_tool_count'] == 4


def test_no_readback_or_no_output_never_passes(tmp_path):
    calls = records()
    calls[0]['arguments']['path'] = 'inputs/data.json'
    assert not module.verify_case_tool_evidence(calls, 'a' * 64, tmp_path)['required_tool_steps_passed']
    assert not module.verify_case_tool_evidence(records(), None, tmp_path)['required_tool_steps_passed']


def test_readback_accepts_normalized_absolute_path_only_for_current_run(tmp_path):
    calls = records()
    calls[0]['arguments']['path'] = str(tmp_path / 'outputs' / 'summary.json')
    assert module.verify_case_tool_evidence(calls, 'a' * 64, tmp_path)['readback_passed']
    calls[0]['arguments']['path'] = str(tmp_path.parent / 'other-run' / 'outputs' / 'summary.json')
    assert not module.verify_case_tool_evidence(calls, 'a' * 64, tmp_path)['readback_passed']
    calls[0]['arguments']['path'] = '../other-run/outputs/summary.json'
    assert not module.verify_case_tool_evidence(calls, 'a' * 64, tmp_path)['readback_passed']


def test_real_store_receipts_accept_completed_local_tool_chain(tmp_path):
    from evomind_runtime.runtime import AgentRuntime
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt='Offline synthetic evidence fixture', start=False)
        session = runtime.get_session(run['id'])
        task_root = Path(session['workspace_root'])
        directory_id = session['metadata']['super_agent_directory_ids'][0]
        requests = [
            ('file_write', {'path': 'outputs/summary.json', 'content': '{"fixture":1}'}),
            ('file_read', {'path': 'outputs/summary.json'}),
            ('directory_hash', {'directory_id': directory_id, 'relative_path': 'outputs/summary.json'}),
            ('artifact_publish', {'path': 'outputs/summary.json'}),
        ]
        for name, arguments in requests:
            result = runtime.invoke_tool(run['id'], name, arguments)
            assert result['status'] == 'completed' and result['result']['ok'] is True
        calls = runtime.store.list_tool_calls(run['id'])
        readback = next(call for call in calls if call['tool_name'] == 'file_read')
        assert Path(readback['arguments']['path']).is_absolute()
        digest = hashlib.sha256((task_root / 'outputs/summary.json').read_bytes()).hexdigest()
        evidence = module.verify_case_tool_evidence(calls, digest, task_root)
        assert evidence['successful_tool_count'] == 4
        assert evidence['required_tool_steps_passed']
    finally:
        runtime.close()


def test_bound_directory_is_in_actual_model_prompt_without_rewriting_original(tmp_path):
    from evomind_runtime.runtime import AgentRuntime
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt='Original synthetic fixture objective', start=False)
        session = runtime.get_session(run['id'])
        directory_id = session['metadata']['super_agent_directory_ids'][0]
        scoped = 'Original synthetic fixture objective. Use directory_hash directory_id ' + directory_id + ' and relative_path outputs/summary.json.'
        updated = module.bind_case_instructions(runtime, run, scoped)
        assert updated['prompt'] == 'Original synthetic fixture objective'
        actual = runtime.assistant._execution_prompt(updated, resume=False)
        assert scoped in actual and directory_id in actual
        assert runtime.store.list_tool_calls(run['id']) == []
    finally:
        runtime.close()
