"""Kill all V4 instances and launch a single clean one."""
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

# Kill all existing V4 processes
print('=== Killing all V4 processes ===')
stdin, stdout, stderr = ssh.exec_command(
    'ps aux | grep "mlebench_catboost_trainer_v4" | grep -v grep | awk \'{print $2}\' | xargs -r kill -9 2>/dev/null; echo "Killed"',
    timeout=10
)
print(stdout.read().decode().strip())

time.sleep(2)

# Verify nothing left
stdin, stdout, stderr = ssh.exec_command(
    'ps aux | grep "mlebench_catboost_trainer" | grep -v grep; echo "---"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits',
    timeout=10
)
print('After kill:')
print(stdout.read().decode().strip())

# Wait for GPU memory to clear
time.sleep(3)

# Launch single clean instance
log_name = 'mlebench_trainer_v4_r1.log'
cmd = f'cd {BASE} && nohup /opt/miniconda3/bin/python -u mlebench_catboost_trainer_v4.py > {log_name} 2>&1 & echo PID=$!'
print(f'\n=== Launching: {cmd} ===')
stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
print(stdout.read().decode().strip())

time.sleep(5)

# Verify
print('\n=== Verification ===')
stdin, stdout, stderr = ssh.exec_command(
    f'ps aux | grep "mlebench_catboost_trainer_v4" | grep -v grep; echo "---LOG---"; head -30 {BASE}/{log_name}',
    timeout=10
)
print(stdout.read().decode().strip())

ssh.close()
sock.close()
