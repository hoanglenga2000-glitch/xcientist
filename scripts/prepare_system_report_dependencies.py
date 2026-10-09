"""One-shot isolated report dependencies with a durable preparation receipt."""
import json
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT = Path('C:/EMQA/sys0907-report-deps-v1')
CANDIDATE = Path('C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v1')


def save(value):
    target = ROOT / 'preparation-receipt.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')
    os.replace(temporary, target)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    import msvcrt
    import psutil
    with (ROOT / 'preparation.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        status = {'status':'preparing','pid':os.getpid(),'process_started':psutil.Process().create_time(),
                  'candidate':str(CANDIDATE),'production_changed':False,'gpu_actions':0}
        save(status)
        print(json.dumps(status), flush=True)
        started = time.monotonic()
        try:
            sys.path.insert(0, str(CANDIDATE))
            source = CANDIDATE.parent / 'dependency_lock_utf8.py'
            spec = importlib.util.spec_from_file_location('report_dependency_lock', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            status['dependency_source_sha256'] = __import__('hashlib').sha256(source.read_bytes()).hexdigest()
            receipt = module.prepare_target(ROOT, {'matplotlib':'3.10.9','numpy':'2.5.2'})
            status.update(status='ready', dependency_receipt=receipt)
        except Exception as error:
            status.update(status='failed', error_class=type(error).__name__)
        status['elapsed_seconds'] = time.monotonic()-started
        save(status)
        print(json.dumps(status), flush=True)
        return 0 if status['status']=='ready' else 2


if __name__ == '__main__': raise SystemExit(main())
