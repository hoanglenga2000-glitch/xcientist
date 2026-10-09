"""Read only allowlisted gateway policy and structural log evidence, no secrets."""
import json
from pathlib import Path
import re

import psutil
import yaml


def main():
    cfg = yaml.safe_load(Path('C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml').read_text(encoding='utf-8-sig'))
    allowed = ['request-retry', 'max-retry-credentials', 'max-retry-interval',
               'transient-error-cooldown-seconds', 'nonstream-keepalive-interval',
               'logging-to-file', 'logs-max-total-size-mb', 'disable-cooling', 'streaming']
    policy = {key: cfg.get(key) for key in allowed}
    providers = [{'name': item.get('name'), 'credential_count': len(item.get('api-key-entries') or []),
                  'models': [{'name': m.get('name'), 'alias': m.get('alias')} for m in item.get('models', [])]}
                 for item in cfg.get('openai-compatibility', [])]
    logs = set()
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'create_time']):
        if 'cliproxy' not in (proc.info['name'] or '').lower():
            continue
        processes.append(proc.info)
        try:
            paths = [x.path for x in proc.open_files() if x.path.endswith('.log')]
            logs.update(paths)
        except psutil.Error:
            pass
    structural = []
    for source in logs:
        path = Path(source)
        # Only parse known access-line fields. Never output raw log lines.
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines()[-250:]:
            match = re.search(r'\[(\d{4}-\d\d-\d\d[^\]]*)\].*?\|\s*(\d{3})\s*\|\s*([^|]+)\|.*?(POST|GET)\s+"?(/v1/(?:responses|chat/completions|models))', line)
            if match:
                structural.append(dict(zip(['timestamp', 'status', 'duration', 'method', 'route'], match.groups())))
    print(json.dumps({'scope': 'gateway_policy_read_only', 'policy': policy, 'providers': providers,
                      'processes': processes, 'log_paths': sorted(logs), 'access_records': structural}, ensure_ascii=True))


if __name__ == '__main__':
    main()
