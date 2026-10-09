"""Pause/approval/resume through owned HTTP APIs; only the provider is replaced."""
import json
import time

from test_user_tasks_http import task_api, no_external_model_credentials


def test_approved_while_paused_resumes_once_without_a_second_approval(tmp_path, monkeypatch):
    from evomind_runtime import personal_model_client

    def provider(_url, _headers, payload):
        count = sum(message.get('role') == 'tool' for message in payload['messages'])
        steps = [('file_write', {'path': 'work/temporary.txt', 'content': 'disposable test input'}),
                 ('file_delete', {'path': 'work/temporary.txt'})]
        if count >= len(steps):
            message, finish = {'content': 'Isolated pause contract completed.'}, 'stop'
        else:
            name, arguments = steps[count]
            message, finish = {'content': None, 'tool_calls': [{'id': f'pause_step_{count}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}, 'tool_calls'
        return {'model': payload['model'], 'choices': [{'finish_reason': finish, 'message': message}]}

    monkeypatch.setattr(personal_model_client, 'post_public_json', provider)
    with task_api(tmp_path) as request:
        profile = request('POST', '/v1/model-profiles', {'name': 'Pause test only', 'provider': 'openai',
            'base_url': 'https://api.example.com/v1', 'model': 'pause-fixture', 'api_key': 'public-fixture-key'})[1]['profile']
        task = request('POST', '/v1/user-tasks', {'title': 'Pause test', 'idempotency_key': 'pause-task'})[1]['task']
        status, run = request('POST', '/v1/runs', {'prompt': 'Exercise the controlled pause sequence.', 'user_task_id': task['id'],
            'model_profile_id': profile['id'], 'model_profile_version': 1, 'idempotency_key': 'pause-run'})
        assert status == 201, run
        path = '/v1/runs/' + run['id']

        def wait_for(states):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = request('GET', path)[1]
                if state['status'] in states:
                    return state
                time.sleep(.03)
            raise AssertionError(f"Run did not reach {states}: {state['status']}")

        pending = wait_for({'waiting_approval'})['approvals'][0]
        assert request('POST', path + '/actions', {'action': 'pause'})[0] == 200
        wait_for({'paused'})
        approval_path = '/v1/approvals/' + pending['id'] + '/decision'
        assert request('POST', approval_path, {'approved': True}, owner='bob')[0] == 404
        assert request('POST', approval_path, {'approved': True})[0] == 200
        assert request('GET', path)[1]['status'] == 'paused'
        assert request('GET', path)[1]['approvals'][0]['status'] == 'approved'
        assert request('POST', path + '/actions', {'action': 'resume', 'instruction': 'Delete a different file instead.'})[0] == 400
        assert request('GET', path)[1]['status'] == 'paused'
        assert request('POST', path + '/actions', {'action': 'resume', 'idempotency_key': 'explicit-resume'})[0] == 200
        state = wait_for({'completed', 'blocked', 'failed'})
        assert state['status'] == 'completed', state['status']
        assert len(state['approvals']) == 1
        assert request('POST', path + '/actions', {'action': 'resume', 'idempotency_key': 'explicit-resume'})[1]['status'] == 'completed'
        assert len(request('GET', path)[1]['approvals']) == 1
