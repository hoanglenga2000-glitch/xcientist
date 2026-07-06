"""Read training log from server."""
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
ssh.connect(hostname='100.85.169.63', port=1235, username='aimslab-TTA1', password='FjODjsq2M2',
            sock=sock, timeout=15, allow_agent=False, look_for_keys=False)

_, stdout, stderr = ssh.exec_command('cat /hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_trainer_87792.log')

print(stdout.read().decode())
err = stderr.read().decode()
if err: print("STDERR:", err[:200])
ssh.close()
