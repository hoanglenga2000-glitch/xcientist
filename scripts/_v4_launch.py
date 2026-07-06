"""Check V4 status and re-launch if needed."""
import socket, struct, paramiko, os, sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(15)
sock.connect(('127.0.0.1', 7890))
sock.send(b'\x05\x01\x00')
sock.recv(2)
sock.send(b'\x05\x01\x00\x01' + socket.inet_aton('100.85.169.63') + struct.pack('>H', 1235))
sock.recv(10)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname='100.85.169.63', port=1235, username='aimslab-TTA2', password='wM5T1Qfz5l',
            sock=sock, timeout=20, banner_timeout=20, auth_timeout=20,
            allow_agent=False, look_for_keys=False)
print('Connected!')

BASE = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra'

cmds = [
    'which python3 python 2>/dev/null || echo NO_PYTHON3',
    f'ls -la {BASE}/mlebench_catboost_trainer_v4.py 2>/dev/null || echo V4_NOT_FOUND',
    f'ls -la {BASE}/mlebench_trainer_v4_87729.log 2>/dev/null || echo LOG_NOT_FOUND',
    f'tail -10 {BASE}/mlebench_trainer_v4_87729.log 2>/dev/null || echo LOG_EMPTY',
    f'head -5 {BASE}/mlebench_catboost_trainer_v4.py',
]
for cmd in cmds:
    print(f'\n--- {cmd[:80]} ---')
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out: print(out.strip())
    if err: print('STDERR:', err.strip())

# If V4 exists but not running, re-launch
stdin, stdout, stderr = ssh.exec_command(f'ps aux | grep mlebench_catboost_trainer_v4 | grep -v grep')
proc = stdout.read().decode('utf-8', errors='replace').strip()

if not proc:
    print('\n--- V4 not running, launching... ---')
    # Use 'python' (conda env path) instead of 'python3'
    cmd = f'cd {BASE} && nohup python mlebench_catboost_trainer_v4.py > mlebench_trainer_v4_87729.log 2>&1 &'
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out: print(out.strip())
    if err: print('STDERR:', err.strip())
    print(f'Launched: {cmd[:80]}...')
else:
    print(f'\nV4 already running: {proc[:200]}')

ssh.close()
sock.close()
