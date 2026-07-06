"""Download V3 submissions and logs from 87729."""
import socket, struct, paramiko, os, sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(15)
sock.connect(('127.0.0.1', 7890))
sock.send(b'\x05\x01\x00')
sock.recv(2)
sock.send(b'\x05\x01\x00\x01' + socket.inet_aton('100.85.169.63') + struct.pack('>H', 1235))
sock.recv(10)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname='100.85.169.63', port=1235, username='aimslab-TTA2', password='wM5T1Qfz5l',
            sock=sock, timeout=20, banner_timeout=20, auth_timeout=20,
            allow_agent=False, look_for_keys=False)
print('Connected!')

BASE = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra'
LOCAL = os.path.dirname(os.path.abspath(__file__))

# 1. Get full training log (last 500 lines)
stdin, stdout, stderr = ssh.exec_command(f'tail -500 {BASE}/mlebench_trainer_v3_87729.log')
log = stdout.read().decode('utf-8', errors='replace')
with open(f'{LOCAL}/mlebench_trainer_v3_87729.log', 'w', encoding='utf-8') as f:
    f.write(log)
print(f'Downloaded log: {len(log)} chars')

# 2. Get training results JSON
stdin, stdout, stderr = ssh.exec_command(f'cat {BASE}/mlebench_proper_results/training_results_v3.json')
results_json = stdout.read().decode('utf-8', errors='replace')
with open(f'{LOCAL}/training_results_v3.json', 'w', encoding='utf-8') as f:
    f.write(results_json)
print(f'Results JSON: {len(results_json)} chars')

# 3. Download all submission CSVs
sftp = ssh.open_sftp()
comps = ['spaceship-titanic', 'tabular-playground-series-dec-2021', 'tabular-playground-series-may-2022',
         'leaf-classification', 'new-york-city-taxi-fare-prediction', 'nomad2018-predict-transparent-conductors',
         'playground-series-s3e18']
os.makedirs(f'{LOCAL}/submissions_v3', exist_ok=True)
for comp in comps:
    remote = f'{BASE}/mlebench_proper_results/{comp}/submission_s44.csv'
    local = f'{LOCAL}/submissions_v3/{comp}_submission.csv'
    try:
        sftp.stat(remote)
        sftp.get(remote, local)
        size = os.path.getsize(local)
        print(f'  Downloaded: {comp} ({size:,} bytes)')
    except:
        print(f'  NOT FOUND: {comp}')

sftp.close()
ssh.close()
sock.close()
print('\nDone!')
