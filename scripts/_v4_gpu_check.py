"""Check GPU processes and V4 status."""
import socket, struct, paramiko

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

print('=== nvidia-smi ===')
stdin, stdout, stderr = ssh.exec_command('nvidia-smi', timeout=10)
print(stdout.read().decode().strip())

print('\n=== CatBoost/ML processes ===')
stdin, stdout, stderr = ssh.exec_command('ps aux | grep -E "catboost|mlebench|python.*train" | grep -v grep', timeout=10)
print(stdout.read().decode().strip())

print('\n=== V4 log ===')
stdin, stdout, stderr = ssh.exec_command(f'cat {BASE}/mlebench_trainer_v4_87729.log', timeout=10)
log = stdout.read().decode('utf-8', errors='replace')
print(f'({len(log)} chars):\n{log}')

# Check if the V4 script is alive
print('\n=== V4 script process ===')
stdin, stdout, stderr = ssh.exec_command('ps aux | grep "mlebench_catboost_trainer_v4" | grep -v grep', timeout=10)
out = stdout.read().decode().strip()
print(out if out else 'NOT FOUND')

ssh.close()
sock.close()
