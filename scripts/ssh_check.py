"""Minimal SSH via SOCKS5 proxy."""
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

# Critical: set blocking and pass immediately to paramiko
sock.setblocking(1)
t = paramiko.Transport(sock)
t.banner_timeout = 60
t.start_client(timeout=30)
t.auth_password('aimslab-fpgTDTSi', 'jXldSnFD6f')
print("CONNECTED")

chan = t.open_session()
chan.settimeout(20)
chan.exec_command('hostname && nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader && echo "PROCS:" && ps aux|grep gpu_train_v3|grep -v grep|wc -l && echo "RESULTS:" && ls /hpc2hdd/home/aimslab/results/v3_result_*.json 2>/dev/null|wc -l')
time.sleep(5)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print(out.decode())
chan.close()
t.close()
