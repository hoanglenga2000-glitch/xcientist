"""V3: proxy restart + tar download + Kaggle submit in one script."""
import subprocess, os, time, socket, struct, paramiko, tarfile

# Step 1: Restart proxy
subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File',
    r'D:\桌面\codex\科研港科技\scripts\manage_hpc_proxy_bridge.ps1', 'stop'],
    capture_output=True, timeout=15)
time.sleep(2)
subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File',
    r'D:\桌面\codex\科研港科技\scripts\manage_hpc_proxy_bridge.ps1', 'start'],
    capture_output=True, timeout=15)
time.sleep(3)

# Step 2: SSH connect
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.settimeout(30)
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

# Step 3: Create tar on server
chan = t.open_session(); chan.settimeout(30)
chan.exec_command("cd /hpc2hdd/home/aimslab/results && tar cf /tmp/v3_subs.tar v3_submission_*.csv 2>/dev/null && ls -la /tmp/v3_subs.tar")
time.sleep(5)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print(out.decode().strip())
chan.close()

# Step 4: SFTP download
sftp = paramiko.SFTPClient.from_transport(t)
local_tar = r'D:\桌面\codex\科研港科技\submissions_v3\v3_subs.tar'
os.makedirs(os.path.dirname(local_tar), exist_ok=True)
sftp.get('/tmp/v3_subs.tar', local_tar)
sftp.close()
t.close()
print("Downloaded: %d bytes" % os.path.getsize(local_tar))

# Step 5: Extract
with tarfile.open(local_tar) as tar:
    tar.extractall(path=r'D:\桌面\codex\科研港科技\submissions_v3')
import glob
csvs = sorted(glob.glob(r'D:\桌面\codex\科研港科技\submissions_v3\v3_submission_*.csv'))
print("Extracted %d CSV files" % len(csvs))

# Step 6: Submit to Kaggle
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

print("\n=== KAGGLE SUBMIT ===")
ok_count = 0
for task_id, slug in SLUG_MAP.items():
    local = os.path.join(r'D:\桌面\codex\科研港科技\submissions_v3', 'v3_submission_%s.csv' % task_id)
    if not os.path.exists(local):
        print("  MISSING: %s" % slug)
        continue
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", slug,
                       "-f", local, "-m", "V3-Gate-PASS"],
                      capture_output=True, text=True, timeout=60)
    ok = "Success" in r.stdout
    if ok: ok_count += 1
    print("  %s: %s" % (slug, "OK" if ok else r.stderr[:80] if r.stderr else r.stdout[:80]))

print("\nSubmitted: %d/%d" % (ok_count, len(SLUG_MAP)))
