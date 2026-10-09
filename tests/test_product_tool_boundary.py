"""Public runtime/file boundary, using disposable files and no external service."""
import hashlib
from pathlib import Path

from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tenant_access import Principal, current_principal


def test_personal_model_cannot_execute_a_script_that_reads_another_workspace(tmp_path):
    outside = tmp_path / 'other-user.txt'
    outside.write_text('other-user-private-fixture', encoding='utf-8')
    runtime = AgentRuntime(tmp_path)
    token = current_principal.set(Principal('tenant_' + 'a' * 24, 'alice'))
    try:
        run = runtime.assistant.create_run(prompt='isolated boundary', start=False)
    finally:
        current_principal.reset(token)
    try:
        code = f"from pathlib import Path\nprint(Path({str(outside)!r}).read_text())\n"
        write = runtime.invoke_tool(run['id'], 'file_write', {'path': 'work/read_other.py', 'content': code})
        assert write['status'] == 'completed'
        result = runtime.invoke_tool(run['id'], 'shell_exec', {'argv': ['python', 'work/read_other.py'], 'timeout_seconds': 5})
        assert result['status'] == 'failed', result
        assert 'other-user-private-fixture' not in str(result)
        assert result['result']['error'] == 'personal_tool_boundary'
        read = runtime.invoke_tool(run['id'], 'file_read', {'path': str(outside)})
        assert read['status'] == 'failed'
        assert 'other-user-private-fixture' not in str(read)
    finally:
        runtime.close()


def test_upload_library_is_not_rebound_or_hardlinked_to_mutable_run_inputs(tmp_path):
    runtime = AgentRuntime(tmp_path)
    data = b'original library data\n'
    try:
        upload = runtime.assistant.create_upload(name='data.txt', total_bytes=len(data))
        runtime.assistant.put_chunk(upload['id'], 0, data, hashlib.sha256(data).hexdigest())
        source = runtime.assistant.complete_upload(upload['id'])['attachment']
        run = runtime.assistant.create_run(prompt='copy only', attachment_ids=[source['id']], start=False)
        Path(run['task_root'], 'inputs', 'data.txt').write_bytes(b'changed inside one run')
        assert Path(source['path']).read_bytes() == data
        assert runtime.store.get_attachment(source['id'])['path'] == source['path']
        assert run['attachments'][0]['id'] != source['id']
    finally:
        runtime.close()
