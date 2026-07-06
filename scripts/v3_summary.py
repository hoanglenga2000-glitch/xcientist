"""Simple V3 summary — one shot."""
import socket, struct, paramiko, time

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

chan = t.open_session(); chan.settimeout(30)
chan.exec_command("python3 -c \"import json,glob;fs=sorted(glob.glob('/hpc2hdd/home/aimslab/results/v3_result_*.json'));p=sum(1 for f in fs if json.load(open(f)).get('gate_passed'));[print(json.load(open(f))['task_id'],round(json.load(open(f))['oof_score'],4),'PASS' if json.load(open(f)).get('gate_passed') else 'FAIL') for f in fs];print('TOTAL:',len(fs),'PASSED:',p)\"")
time.sleep(10)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print(out.decode())
chan.close(); t.close()
