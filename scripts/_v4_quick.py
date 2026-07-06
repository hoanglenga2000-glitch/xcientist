"""Minimal: check if V4 is running and launch if not."""
import socket, struct, paramiko, time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10)
try:
    sock.connect(('127.0.0.1', 7890))
    sock.send(b'\x05\x01\x00')
    resp = sock.recv(2)
    print(f'SOCKS5: {resp.hex()}')
    sock.send(b'\x05\x01\x00\x01' + socket.inet_aton('100.85.169.63') + struct.pack('>H', 1235))
    resp = sock.recv(10)
    print(f'CONNECT: {resp.hex()}')
except Exception as e:
    print(f'Bridge error: {e}')
    sock.close()
    exit(1)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(hostname='100.85.169.63', port=1235, username='aimslab-TTA2', password='wM5T1Qfz5l',
                sock=sock, timeout=15, banner_timeout=15, auth_timeout=15,
                allow_agent=False, look_for_keys=False)
    print('SSH connected')
except Exception as e:
    print(f'SSH error: {e}')
    sock.close()
    exit(1)

BASE = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra'

# Quick check
stdin, stdout, stderr = ssh.exec_command(f'tail -5 {BASE}/mlebench_trainer_v4_87729.log', timeout=10)
print('V4 log tail:', stdout.read().decode().strip())

stdin, stdout, stderr = ssh.exec_command('ps aux | grep "catboost_trainer_v4" | grep -v grep', timeout=10)
proc = stdout.read().decode().strip()
if proc:
    print(f'V4 RUNNING: {proc[:200]}')
else:
    print('V4 NOT running — launching...')
    cmd = f'cd {BASE} && rm -f mlebench_trainer_v4_87729.log && nohup /opt/miniconda3/bin/python mlebench_catboost_trainer_v4.py > mlebench_trainer_v4_87729.log 2>&1 & echo PID=$!'
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    print(stdout.read().decode().strip())
    time.sleep(2)
    stdin, stdout, stderr = ssh.exec_command(f'tail -20 {BASE}/mlebench_trainer_v4_87729.log', timeout=10)
    print('New log:', stdout.read().decode().strip())
    stdin, stdout, stderr = ssh.exec_command('ps aux | grep "catboost_trainer_v4" | grep -v grep', timeout=10)
    print('Proc check:', stdout.read().decode().strip())

ssh.close()
sock.close()
