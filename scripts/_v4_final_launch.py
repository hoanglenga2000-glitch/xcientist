"""Final clean launch — no stdout reading from nohup."""
import socket, struct, paramiko, time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10)
sock.connect(('127.0.0.1', 7890))
sock.send(b'\x05\x01\x00')
sock.recv(2)
sock.send(b'\x05\x01\x00\x01' + socket.inet_aton('100.85.169.63') + struct.pack('>H', 1235))
sock.recv(10)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname='100.85.169.63', port=1235, username='aimslab-TTA2', password='wM5T1Qfz5l',
            sock=sock, timeout=15, banner_timeout=15, auth_timeout=15,
            allow_agent=False, look_for_keys=False)

BASE = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra'
LOG = f'{BASE}/mlebench_trainer_v4_r1.log'

# Check if already running
stdin, stdout, stderr = ssh.exec_command('ps aux | grep "mlebench_catboost_trainer_v4" | grep -v grep', timeout=10)
proc = stdout.read().decode().strip()
print(f'Current processes: {proc if proc else "NONE"}')

# Check GPU
stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits', timeout=10)
print(f'GPU mem used: {stdout.read().decode().strip()} MiB')

# Check log
stdin, stdout, stderr = ssh.exec_command(f'cat {LOG} 2>/dev/null || echo "NO LOG YET"', timeout=10)
log = stdout.read().decode().strip()
print(f'Current log ({len(log)} chars): {log[:500]}')

if 'python' in proc or 'mlebench' in proc:
    print('\nV4 is already running!')
else:
    print('\nLaunching V4...')
    # Use exec_command with a wrapper script approach to avoid stdout issues
    launch_script = f"""#!/bin/bash
cd {BASE}
nohup /opt/miniconda3/bin/python -u mlebench_catboost_trainer_v4.py > {LOG} 2>&1 &
echo "LAUNCHED:$!"
"""
    # Write wrapper script
    sftp = ssh.open_sftp()
    with sftp.file(f'{BASE}/_launch_v4.sh', 'w') as f:
        f.write(launch_script)
    sftp.close()

    # Execute wrapper
    stdin, stdout, stderr = ssh.exec_command(f'bash {BASE}/_launch_v4.sh', timeout=10)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'Launch result: {out}')
    if err: print(f'stderr: {err}')

    time.sleep(3)

    # Verify
    stdin, stdout, stderr = ssh.exec_command('ps aux | grep "mlebench_catboost_trainer_v4" | grep -v grep', timeout=10)
    proc = stdout.read().decode().strip()
    print(f'Process now: {"RUNNING" if "python" in proc else "NOT FOUND"}')

    stdin, stdout, stderr = ssh.exec_command(f'head -20 {LOG}', timeout=10)
    log = stdout.read().decode().strip()
    print(f'Log head: {log[:600]}')

ssh.close()
sock.close()
