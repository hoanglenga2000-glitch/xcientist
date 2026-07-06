"""Deploy V4 to S7 and launch targeted training."""
import socket, struct, paramiko, time, base64, subprocess, os

# Read V4 script
with open(r'D:\桌面\codex\科研港科技\scripts\gpu_train_v4_targeted.py', 'rb') as f:
    content = f.read()
print("V4 script: %d bytes" % len(content))

# Restart proxy
subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File',
    r'D:\桌面\codex\科研港科技\scripts\manage_hpc_proxy_bridge.ps1', 'stop'],
    capture_output=True, timeout=10)
time.sleep(2)
subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File',
    r'D:\桌面\codex\科研港科技\scripts\manage_hpc_proxy_bridge.ps1', 'start'],
    capture_output=True, timeout=10)
time.sleep(4)

# Connect to S7
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.settimeout(30)
sock.connect(('127.0.0.1', 7890))
sock.sendall(b'\x05\x01\x00'); assert sock.recv(2) == b'\x05\x00'
h = b'100.85.169.63'
sock.sendall(b'\x05\x01\x00\x03' + bytes([len(h)]) + h + struct.pack('!H', 1235))
r = sock.recv(4); assert r[1] == 0
if r[3] == 1: sock.recv(4)
elif r[3] == 3: sock.recv(sock.recv(1)[0]); sock.recv(2)
sock.setblocking(1)

t = paramiko.Transport(sock); t.banner_timeout = 60; t.start_client(30)
t.auth_password('aimslab-zoeXIdNC', 'n6oewebu0p')
print("CONNECTED S7")

# Upload V4 script via base64
b64 = base64.b64encode(content).decode('ascii')
chan = t.open_session(); chan.settimeout(30)
chan.exec_command("echo '%s' | base64 -d > /hpc2hdd/home/aimslab/gpu_train_v4.py && python3 -c 'import ast; ast.parse(open(\"/hpc2hdd/home/aimslab/gpu_train_v4.py\").read()); print(\"SYNTAX OK\")'" % b64)
time.sleep(5)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print("Upload: " + out.decode().strip())
chan.close()

# Launch V4 training — priority targets
PRIORITY = ["titanic", "ps4e1", "ps4e6", "ps3e25", "ps5e1", "tps_dec2021", "tps_jan2022", "tps_mar2022"]

# Create launcher
launcher = """#!/bin/bash
HOME=/hpc2hdd/home/aimslab
for t in %s; do
    echo "=== V4: $t ==="
    python3 $HOME/gpu_train_v4.py "$t" --gpu-device 0 --n-folds 5
    cp /hpc2hdd/home/aimslab/results/v4_result_*.json /hpc2hdd/home/aimslab/results/ 2>/dev/null
    cp /hpc2hdd/home/aimslab/results/v4_submission_*.csv /hpc2hdd/home/aimslab/results/ 2>/dev/null
done
echo "V4 BATCH COMPLETE"
""" % ' '.join(PRIORITY)

b64l = base64.b64encode(launcher.encode()).decode()
chan2 = t.open_session(); chan2.settimeout(15)
chan2.exec_command("echo '%s' | base64 -d > /tmp/v4_launcher.sh && chmod +x /tmp/v4_launcher.sh" % b64l)
time.sleep(2)
chan2.close()

# Launch!
chan3 = t.open_session(); chan3.settimeout(10)
chan3.exec_command("cd /hpc2hdd/home/aimslab && nohup bash /tmp/v4_launcher.sh > /tmp/v4_batch.log 2>&1 & sleep 2 && ps aux|grep gpu_train_v4|grep -v grep|wc -l && nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader")
time.sleep(5)
out3 = b''
while chan3.recv_ready(): out3 += chan3.recv(65536)
print("Launch: " + out3.decode().strip())
chan3.close()

t.close()
print("\nV4 deployed! %d tasks launched on S7" % len(PRIORITY))
