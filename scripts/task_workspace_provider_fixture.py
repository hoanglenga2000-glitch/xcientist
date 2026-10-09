"""Local-only external-provider seam. Never included in the production source package."""
import json

CONTENT = '# 隔离验收文件\n\n这是由隔离服务脚本驱动真实文件工具生成的测试产物。\n不是真实模型分析，不涉及 GPU，不证明科研结论。\n'


def respond(url, headers, payload):
    if payload.get('model') != 'isolated-ui-fixture':
        raise ValueError('model_isolated_network_disabled')
    replies = sum(message.get('role') == 'tool' for message in payload.get('messages', []))
    actions = [
        ('file_write', {'path': 'outputs/isolated-acceptance.md', 'content': CONTENT}),
        ('file_read', {'path': 'outputs/isolated-acceptance.md'}),
        ('artifact_publish', {'path': 'outputs/isolated-acceptance.md', 'name': 'isolated-acceptance.md', 'media_type': 'text/markdown'}),
    ]
    if payload.get('max_tokens') <= 16 or replies >= len(actions):
        message, reason = {'content': '【隔离服务脚本，不是真实模型】测试文件已通过真实文件工具写入、读取并发布。请到“文件与结果”查看与下载。'}, 'stop'
    else:
        name, args = actions[replies]
        if name not in {tool['function']['name'] for tool in payload.get('tools', [])}:
            raise ValueError('model_fixture_tool_not_offered')
        message, reason = {'content': None, 'tool_calls': [{'id': 'fixture_call_' + str(replies), 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls'
    return {'model': 'isolated-ui-fixture', 'choices': [{'finish_reason': reason, 'message': message}]}
