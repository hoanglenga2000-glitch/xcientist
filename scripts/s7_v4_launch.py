"""S7: Upload V4 + Launch training — ONE connection."""
import socket, struct, paramiko, time, base64, os

# Read V4 script
v4_path = r'D:\桌面\codex\科研港科技\scripts\gpu_train_v4_targeted.py'
with open(v4_path, 'rb') as f:
    v4_content = f.read()
v4_b64 = base64.b64encode(v4_content).decode('ascii')
print("V4 script: %d bytes, b64: %d chars" % (len(v4_content), len(v4_b64)))

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

# Upload V4 + launch + verify in ONE command chain
# Use base64 upload (split for large files)
ALL_CMD = """
# Upload V4 script
echo '%s' | base64 -d > /hpc2hdd/home/aimslab/gpu_train_v4.py
python3 -c 'import ast; ast.parse(open("/hpc2hdd/home/aimslab/gpu_train_v4.py").read()); print("V4 SYNTAX OK")'

# Kill old V4 processes
pkill -f gpu_train_v4 2>/dev/null

# Launch V4 for priority targets
cd /hpc2hdd/home/aimslab
echo "=== LAUNCHING V4 ==="
TASKS="titanic ps4e1"
for t in $TASKS; do
    echo "Starting: $t"
    nohup python3 gpu_train_v4.py "$t" --gpu-device 0 --n-folds 5 > /tmp/v4_${t}.log 2>&1 &
    sleep 2
done
sleep 3
echo "=== STATUS ==="
ps aux | grep gpu_train_v4 | grep -v grep | wc -l
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
""" % v4_b64

chan = t.open_session(); chan.settimeout(30)
chan.exec_command(ALL_CMD)
time.sleep(10)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print(out.decode())
chan.close()
t.close()
print("\nDone! V4 training launched on S7")
