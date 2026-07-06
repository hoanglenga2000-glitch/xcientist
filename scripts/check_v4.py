"""Check V4 training status on S7."""
import socket, struct, paramiko, time

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

chan = t.open_session(); chan.settimeout(15)
chan.exec_command("""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
echo "=== V4 PROCS ==="
ps aux | grep gpu_train_v4 | grep -v grep | wc -l
echo "=== V4 SCRIPT ==="
ls -la /hpc2hdd/home/aimslab/gpu_train_v4.py 2>/dev/null || echo NOT_FOUND
echo "=== V4 LOG ==="
tail -10 /tmp/v4_batch.log 2>/dev/null || echo NOT_FOUND
echo "=== V4 RESULTS ==="
ls /hpc2hdd/home/aimslab/results/v4_result_*.json 2>/dev/null | wc -l
for f in /hpc2hdd/home/aimslab/results/v4_result_*.json; do
    python3 -c "import json;d=json.load(open('$f'));print(d['task_id'],round(d['oof_score'],4),'GATE='+str(d.get('gate_passed','?')))" 2>/dev/null
done
""")
time.sleep(6)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print(out.decode())
chan.close(); t.close()
