"""Isolated provider double only; never ship this file in a production package."""
import json
import time
from task_workspace_provider_fixture import respond as basic_response


def respond(url, headers, payload):
    prompts = '\n'.join(str(message.get('content') or '') for message in payload.get('messages', []) if message.get('role') == 'user')
    if '验收确认' not in prompts:
        return basic_response(url, headers, payload)
    if payload.get('model') != 'isolated-ui-fixture':
        raise ValueError('isolated_model_only')
    steps = [('file_write', {'path': 'work/approval-fixture.txt', 'content': 'disposable local approval fixture'}),
             ('file_delete', {'path': 'work/approval-fixture.txt'})]
    count = sum(message.get('role') == 'tool' for message in payload.get('messages', []))
    if count >= len(steps):
        time.sleep(1)  # Expose the real recovery/SSE transition to browser acceptance.
        message, finish = {'content': '【隔离服务脚本，非真实模型】确认流程已结束；未调用外部服务或 GPU。'}, 'stop'
    else:
        name, arguments = steps[count]
        message, finish = {'content': None, 'tool_calls': [{'id': f'local_confirmation_{count}', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]}, 'tool_calls'
    return {'model': 'isolated-ui-fixture', 'choices': [{'finish_reason': finish, 'message': message}]}
