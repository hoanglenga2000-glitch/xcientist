"""Check V3 status via SSHPiper bridge (port 7890)."""
import socket, struct, paramiko, sys

# Connect to local SOCKS5 bridge at 7890
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(15)
sock.connect(('127.0.0.1', 7890))

# SOCKS5 handshake (no-auth)
sock.send(b'\x05\x01\x00')
resp = sock.recv(2)
print(f'SOCKS5 handshake: {resp.hex()}')

# CONNECT to SSHPiper at 100.85.169.63:1235
sock.send(b'\x05\x01\x00\x01' + socket.inet_aton('100.85.169.63') + struct.pack('>H', 1235))
resp = sock.recv(10)
print(f'SOCKS5 connect to SSHPiper: {resp.hex()}')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(
        hostname='100.85.169.63', port=1235,
        username='aimslab-TTA2', password='wM5T1Qfz5l',
        sock=sock, timeout=20,
        banner_timeout=20, auth_timeout=20,
        allow_agent=False, look_for_keys=False
    )
    print('SSH connected!')

    cmds = [
        'ps aux | grep -E "mlebench|catboost|python" | grep -v grep',
        'tail -80 /hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_trainer_v3_87729.log 2>/dev/null || echo LOG_NOT_FOUND',
        'ls -la /hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_proper_results/*/submission_s44.csv 2>/dev/null || echo NO_SUBS',
        'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo NO_GPU',
    ]
    for cmd in cmds:
        print(f'\n--- {cmd[:80]} ---')
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        if out: print(out.strip())
        if err: print('STDERR:', err.strip())

    ssh.close()
except Exception as e:
    print(f'SSH ERROR: {e}')
    import traceback
    traceback.print_exc()
finally:
    sock.close()
