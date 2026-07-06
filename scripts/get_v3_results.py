"""Get V3 results with gate status."""
import socket, struct, paramiko, time, json, glob, os

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(30)
sock.connect(('127.0.0.1', 7890))
sock.sendall(b'\x05\x01\x00'); sock.recv(2)
h = b'100.85.169.63'
sock.sendall(b'\x05\x01\x00\x03' + bytes([len(h)]) + h + struct.pack('!H', 1235))
r = sock.recv(4)
if r[3] == 1: sock.recv(4)
elif r[3] == 3: sock.recv(sock.recv(1)[0]); sock.recv(2)
sock.setblocking(1)

t = paramiko.Transport(sock)
t.banner_timeout = 60; t.start_client(30)
t.auth_password('aimslab-fpgTDTSi', 'jXldSnFD6f')
print("CONNECTED")

# Save result fetching script to server and run
script = """
import json, glob
fs = sorted(glob.glob("/hpc2hdd/home/aimslab/results/v3_result_*.json"))
passed = 0
for f in fs:
    d = json.load(open(f))
    g = d.get("gate_passed", False)
    if g: passed += 1
    print(f"{d['task_id']:35s} {d.get('metric','?'):10s} {round(d.get('oof_score',0),4):>10.4f}  GATE={'PASS' if g else 'FAIL'}")
print(f"TOTAL: {len(fs)} results | {passed} GATE-PASSED")
"""
import base64
b64 = base64.b64encode(script.encode()).decode()

chan = t.open_session(); chan.settimeout(20)
chan.exec_command(f"echo '{b64}' | base64 -d | python3")
time.sleep(8)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print(out.decode())
chan.close()
t.close()
