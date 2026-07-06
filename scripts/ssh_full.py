"""Get full V3 results in one SSH connection."""
import socket, struct, paramiko, time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(30)
sock.connect(('127.0.0.1', 7890))
sock.sendall(b'\x05\x01\x00')
assert sock.recv(2) == b'\x05\x00'

host = b'100.85.169.63'
sock.sendall(b'\x05\x01\x00\x03' + bytes([len(host)]) + host + struct.pack('!H', 1235))
r = sock.recv(4)
assert r[1] == 0
if r[3] == 1: sock.recv(4)
elif r[3] == 3:
    alen = sock.recv(1)[0]
    sock.recv(alen)
sock.recv(2)

sock.setblocking(1)
t = paramiko.Transport(sock)
t.banner_timeout = 60
t.start_client(timeout=30)
t.auth_password('aimslab-fpgTDTSi', 'jXldSnFD6f')
print("CONNECTED\n")

chan = t.open_session()
chan.settimeout(25)
cmd = """echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
echo "=== V3 RESULTS ==="
for f in /hpc2hdd/home/aimslab/results/v3_result_*.json; do
    python3 -c "import json;d=json.load(open('$f'));print(d['task_id'],round(d['oof_score'],4),'GATE='+str(d['gate_passed']),'gap='+str(round(d.get('gate_gap',0),4)))" 2>/dev/null
done
echo "TOTAL: $(ls /hpc2hdd/home/aimslab/results/v3_result_*.json 2>/dev/null | wc -l)"
"""
chan.exec_command(cmd)
time.sleep(10)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print(out.decode())
chan.close()
t.close()
