"""Get FULL V3 results — one connection, all info."""
import socket, struct, paramiko, time, base64, subprocess, os

# Restart proxy first
subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File',
    r'D:\桌面\codex\科研港科技\scripts\manage_hpc_proxy_bridge.ps1', 'stop'],
    capture_output=True, timeout=15)
time.sleep(2)
subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File',
    r'D:\桌面\codex\科研港科技\scripts\manage_hpc_proxy_bridge.ps1', 'start'],
    capture_output=True, timeout=15)
time.sleep(3)

# Connect
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
print("CONNECTED\n")

# Get detailed results
script = """
import json, glob
fs = sorted(glob.glob("/hpc2hdd/home/aimslab/results/v3_result_*.json"))
passed = 0
print(f"{'Task':35s} {'Metric':10s} {'OOF':>10s}  {'GATE':>6s}  {'Gap':>8s}")
print("-" * 75)
for f in fs:
    d = json.load(open(f))
    g = d.get("gate_passed", False)
    if g: passed += 1
    oof = d.get('oof_score', 0)
    gap = d.get('gate_gap', 0)
    print(f"{d['task_id']:35s} {d.get('metric','?'):10s} {round(oof,4):>10.4f}  {'PASS' if g else 'FAIL':>6s}  {round(gap,4):>+8.4f}")
print(f"\\nTOTAL: {len(fs)} results | {passed} GATE-PASSED | {len(fs)-passed} FAILED")

# Also check S7 and S1 for their results
import os
for node in ['/tmp']:
    extra = glob.glob(f"{node}/v3_batch_*.log")
    if extra:
        print(f"\\nLogs in {node}:")
        for e in extra:
            print(f"  {os.path.basename(e)}")
"""
b64 = base64.b64encode(script.encode()).decode()

chan = t.open_session(); chan.settimeout(20)
chan.exec_command(f"echo '{b64}' | base64 -d | python3")
time.sleep(10)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print(out.decode())
chan.close()

# Also download V3 submissions list
print("\n=== V3 SUBMISSION FILES ===")
chan2 = t.open_session(); chan2.settimeout(15)
chan2.exec_command('ls -la /hpc2hdd/home/aimslab/results/v3_submission_*.csv 2>/dev/null | wc -l && ls /hpc2hdd/home/aimslab/results/v3_submission_*.csv 2>/dev/null | head -30')
time.sleep(5)
out2 = b''
while chan2.recv_ready(): out2 += chan2.recv(65536)
print(out2.decode())
chan2.close()

t.close()
