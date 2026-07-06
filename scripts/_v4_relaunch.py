"""Re-launch V4 with correct conda Python."""
import socket, struct, paramiko

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

# Find conda python
stdin, stdout, stderr = ssh.exec_command('which python 2>/dev/null; ls /opt/miniconda3/bin/python 2>/dev/null; which conda 2>/dev/null')
print('Python paths:', stdout.read().decode().strip())

# Check conda python
stdin, stdout, stderr = ssh.exec_command('/opt/miniconda3/bin/python -c "import numpy; import catboost; print(\'OK\')" 2>&1')
print('Conda env check:', stdout.read().decode().strip())

# Also check if there's a conda env with catboost
stdin, stdout, stderr = ssh.exec_command('ls /opt/miniconda3/envs/ 2>/dev/null')
print('Conda envs:', stdout.read().decode().strip())

# Check the Jupyter process python path more carefully
stdin, stdout, stderr = ssh.exec_command('ps aux | grep jupyter | grep -v grep')
jupyter_ps = stdout.read().decode().strip()
print('Jupyter:', jupyter_ps[:200])

# Launch with correct python
cmd = f'cd {BASE} && rm -f mlebench_trainer_v4_87729.log && nohup /opt/miniconda3/bin/python mlebench_catboost_trainer_v4.py > mlebench_trainer_v4_87729.log 2>&1 &'
print(f'\nLaunching: {cmd[:100]}...')
stdin, stdout, stderr = ssh.exec_command(cmd)
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
if out: print('STDOUT:', out.strip())
if err: print('STDERR:', err.strip())

# Verify
import time
time.sleep(3)
stdin, stdout, stderr = ssh.exec_command('ps aux | grep mlebench_catboost_trainer_v4 | grep -v grep')
proc = stdout.read().decode('utf-8', errors='replace').strip()
print(f'\nProcess: {"RUNNING" if "python" in proc else "NOT FOUND"}')
if proc: print(proc)

stdin, stdout, stderr = ssh.exec_command(f'tail -20 {BASE}/mlebench_trainer_v4_87729.log 2>/dev/null')
log = stdout.read().decode('utf-8', errors='replace')
print(f'\nV4 Log ({len(log)} chars):')
print(log)

ssh.close()
sock.close()
