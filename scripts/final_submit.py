"""One connection: tar + download V3 submissions, then submit to Kaggle."""
import subprocess, os, time, socket, struct, paramiko, tarfile

# Restart proxy
subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File',
    r'D:\桌面\codex\科研港科技\scripts\manage_hpc_proxy_bridge.ps1', 'stop'],
    capture_output=True, timeout=10)
time.sleep(2)
subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File',
    r'D:\桌面\codex\科研港科技\scripts\manage_hpc_proxy_bridge.ps1', 'start'],
    capture_output=True, timeout=10)
time.sleep(4)

# CONNECT
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.settimeout(60)
sock.connect(('127.0.0.1', 7890))
sock.sendall(b'\x05\x01\x00'); sock.recv(2)
h = b'100.85.169.63'
sock.sendall(b'\x05\x01\x00\x03' + bytes([len(h)]) + h + struct.pack('!H', 1235))
r = sock.recv(4)
if r[3] == 1: sock.recv(4)
elif r[3] == 3: sock.recv(sock.recv(1)[0]); sock.recv(2)
sock.setblocking(1)

t = paramiko.Transport(sock); t.banner_timeout = 60; t.start_client(30)
t.auth_password('aimslab-fpgTDTSi', 'jXldSnFD6f')
print("CONNECTED")

# TAR all V3 submissions
chan = t.open_session(); chan.settimeout(30)
chan.exec_command("cd /hpc2hdd/home/aimslab/results && tar czf /tmp/v3_all.tar.gz v3_submission_*.csv 2>/dev/null; ls -la /tmp/v3_all.tar.gz; echo COUNT:; ls v3_submission_*.csv 2>/dev/null | wc -l")
time.sleep(5)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print(out.decode().strip())
chan.close()

# DOWNLOAD
sftp = paramiko.SFTPClient.from_transport(t)
local_tar = r'D:\桌面\codex\科研港科技\submissions_v3\v3_all.tar.gz'
os.makedirs(os.path.dirname(local_tar), exist_ok=True)
sftp.get('/tmp/v3_all.tar.gz', local_tar)
sftp.close()
t.close()
print("Downloaded: %d bytes" % os.path.getsize(local_tar))

# EXTRACT
os.makedirs(r'D:\桌面\codex\科研港科技\submissions_v3', exist_ok=True)
with tarfile.open(local_tar, 'r:gz') as tar:
    tar.extractall(path=r'D:\桌面\codex\科研港科技\submissions_v3')
import glob
csvs = sorted(glob.glob(r'D:\桌面\codex\科研港科技\submissions_v3\v3_submission_*.csv'))
print("Extracted: %d CSV files" % len(csvs))

# SUBMIT
SLUG_MAP = {
    "titanic": "titanic", "spaceship_titanic": "spaceship-titanic",
    "bike_sharing_demand": "bike-sharing-demand",
    "ps3e1": "playground-series-s3e1", "ps3e7": "playground-series-s3e7",
    "ps4e1": "playground-series-s4e1", "ps4e2": "playground-series-s4e2",
    "ps4e3": "playground-series-s4e3", "ps4e7": "playground-series-s4e7",
    "ps5e3": "playground-series-s5e3", "ps6e2": "playground-series-s6e2",
    "ps6e3": "playground-series-s6e3", "ps6e6": "playground-series-s6e6",
    "porto_seguro": "porto-seguro-safe-driver-prediction",
}
print("\n=== SUBMITTING TO KAGGLE ===")
ok = 0
for tid, slug in SLUG_MAP.items():
    lp = os.path.join(r'D:\桌面\codex\科研港科技\submissions_v3', 'v3_submission_%s.csv' % tid)
    if not os.path.exists(lp):
        print("  MISSING: %s" % slug)
        continue
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", slug, "-f", lp, "-m", "V3-Gate-PASS"],
                      capture_output=True, text=True, timeout=60)
    if "Success" in r.stdout:
        ok += 1
        print("  %s: OK" % slug)
    else:
        print("  %s: %s" % (slug, r.stderr[:60] if r.stderr else r.stdout[:60]))
print("\nSubmitted: %d/%d" % (ok, len(SLUG_MAP)))
