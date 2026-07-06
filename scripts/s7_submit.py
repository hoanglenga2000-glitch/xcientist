"""S7: tar + download V3 submissions + Kaggle submit."""
import socket, struct, paramiko, time, os, tarfile, subprocess

# Connect to S7
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
t.auth_password('aimslab-zoeXIdNC', 'n6oewebu0p')
print("CONNECTED S7")

# Create tar.gz
chan = t.open_session(); chan.settimeout(60)
chan.exec_command("cd /hpc2hdd/home/aimslab/results && tar czf /tmp/v3_final.tar.gz v3_submission_*.csv 2>&1 && ls -lh /tmp/v3_final.tar.gz")
time.sleep(10)
out = b''
while chan.recv_ready(): out += chan.recv(65536)
print("Tar: " + out.decode().strip())
chan.close()

# Download via SFTP
sftp = paramiko.SFTPClient.from_transport(t)
local_dir = r'D:\桌面\codex\科研港科技\submissions_v3'
os.makedirs(local_dir, exist_ok=True)
local_tar = os.path.join(local_dir, 'v3_final.tar.gz')
sftp.get('/tmp/v3_final.tar.gz', local_tar)
sz = os.path.getsize(local_tar)
print("Downloaded: %.1f MB" % (sz/1024/1024))
sftp.close()
t.close()

# Extract
with tarfile.open(local_tar, 'r:gz') as tar:
    tar.extractall(path=local_dir)
import glob
csvs = sorted(glob.glob(os.path.join(local_dir, 'v3_submission_*.csv')))
print("Extracted: %d files" % len(csvs))

# Submit gate-passed ones
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
ok = 0
for tid, slug in SLUG_MAP.items():
    lp = os.path.join(local_dir, 'v3_submission_%s.csv' % tid)
    if not os.path.exists(lp):
        print("  MISSING: %s" % slug)
        continue
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", slug, "-f", lp, "-m", "V3-Gate-S7"],
                      capture_output=True, text=True, timeout=60)
    if "Success" in r.stdout:
        ok += 1; print("  OK: %s" % slug)
    else:
        print("  FAIL: %s" % slug)

print("Submitted: %d/%d" % (ok, len(SLUG_MAP)))
