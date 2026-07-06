"""Deploy V4 trainer to 87729 and launch training."""
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

LOCAL = os.path.dirname(os.path.abspath(__file__))
BASE = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra'

# 1. Upload V4 trainer
sftp = ssh.open_sftp()
trainer_local = f'{LOCAL}/mlebench_catboost_trainer_v3.py'
trainer_remote = f'{BASE}/mlebench_catboost_trainer_v4.py'
sftp.put(trainer_local, trainer_remote)
print(f'Uploaded trainer -> {trainer_remote}')

# 2. Upload fixed grading script
grader_local = f'{LOCAL}/_grade_v3.py'
grader_remote = f'{BASE}/_grade_v4.py'
sftp.put(grader_local, grader_remote)
print(f'Uploaded grader -> {grader_remote}')

# 3. Upload deploy script itself for reference
deploy_local = f'{LOCAL}/_deploy_v4.py'
deploy_remote = f'{BASE}/_deploy_v4.py'
sftp.put(deploy_local, deploy_remote)

sftp.close()

# 4. Launch training with nohup
cmd = f'cd {BASE} && nohup python3 mlebench_catboost_trainer_v4.py > mlebench_trainer_v4_87729.log 2>&1 &'
stdin, stdout, stderr = ssh.exec_command(cmd)
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print(f'Launch: {cmd[:80]}...')
if out: print('STDOUT:', out.strip())
if err: print('STDERR:', err.strip())

# 5. Verify process started
stdin, stdout, stderr = ssh.exec_command('ps aux | grep mlebench_catboost_trainer_v4 | grep -v grep')
out = stdout.read().decode('utf-8', errors='replace')
print(f'Process check: {"RUNNING" if "python3" in out else "NOT FOUND"}')
if out: print(out.strip())

ssh.close()
sock.close()
print('\nDone! Training launched.')
