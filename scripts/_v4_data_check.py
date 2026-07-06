"""Check data availability for all 7 competitions."""
import socket, struct, paramiko

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10)
sock.connect(('127.0.0.1', 7890))
sock.send(b'\x05\x01\x00')
sock.recv(2)
sock.send(b'\x05\x01\x00\x01' + socket.inet_aton('100.85.169.63') + struct.pack('>H', 1235))
sock.recv(10)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname='100.85.169.63', port=1235, username='aimslab-TTA2', password='wM5T1Qfz5l',
            sock=sock, timeout=15, banner_timeout=15, auth_timeout=15,
            allow_agent=False, look_for_keys=False)

PREPARED = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_prepared'
comps = [
    'spaceship-titanic', 'tabular-playground-series-dec-2021', 'tabular-playground-series-may-2022',
    'playground-series-s3e18', 'leaf-classification', 'new-york-city-taxi-fare-prediction',
    'nomad2018-predict-transparent-conductors'
]

for comp in comps:
    print(f'\n=== {comp} ===')
    cmds = [
        f'ls {PREPARED}/{comp}/train.csv 2>/dev/null && wc -l {PREPARED}/{comp}/train.csv || echo "NO TRAIN"',
        f'ls {PREPARED}/{comp}/test.csv 2>/dev/null && wc -l {PREPARED}/{comp}/test.csv || echo "NO TEST"',
        f'ls {PREPARED}/{comp}/test_private.csv 2>/dev/null && wc -l {PREPARED}/{comp}/test_private.csv || echo "NO TEST_PRIVATE"',
    ]
    for cmd in cmds:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
        print(f'  {stdout.read().decode().strip()}')

# Also check s3e18 specifically
print('\n=== s3e18 directory listing ===')
stdin, stdout, stderr = ssh.exec_command(f'ls -la {PREPARED}/playground-series-s3e18/ 2>/dev/null || echo "DIR NOT FOUND"', timeout=10)
print(stdout.read().decode().strip())

stdin, stdout, stderr = ssh.exec_command(f'ls -la {PREPARED}/ 2>/dev/null', timeout=10)
print(f'\nAll prepared dirs:\n{stdout.read().decode().strip()}')

ssh.close()
sock.close()
