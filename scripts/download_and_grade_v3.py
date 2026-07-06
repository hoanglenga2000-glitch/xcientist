"""Download V3 submission CSVs from server via proxy, then grade locally."""
import socket
import struct
import paramiko
import os
import sys
import json

SERVER = {
    'host': '10.120.18.240',
    'port': 6988,
    'user': 'aimslab-TTA2',
    'password': 'wM5T1Qfz5l',
}
PROXY = ('127.0.0.1', 7897)
REMOTE_RESULTS = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_proper_results'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

def ssh_connect():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(15)
    sock.connect(PROXY)
    sock.send(b'\x05\x01\x00')
    sock.recv(2)
    target_addr = socket.inet_aton(SERVER['host'])
    sock.send(b'\x05\x01\x00\x01' + target_addr + struct.pack('>H', SERVER['port']))
    sock.recv(10)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=SERVER['host'], port=SERVER['port'],
        username=SERVER['user'], password=SERVER['password'],
        sock=sock, timeout=20, banner_timeout=20, auth_timeout=20
    )
    return ssh, sock

def download_file(ssh, remote_path, local_path):
    sftp = ssh.open_sftp()
    try:
        sftp.stat(remote_path)
        sftp.get(remote_path, local_path)
        size = os.path.getsize(local_path)
        print(f"  Downloaded: {os.path.basename(local_path)} ({size:,} bytes)")
        return True
    except FileNotFoundError:
        print(f"  NOT FOUND: {remote_path}")
        return False
    finally:
        sftp.close()

def main():
    print("Connecting to 87729 via SOCKS5 proxy...")
    ssh, sock = ssh_connect()
    print("Connected!")

    # List all result directories
    stdin, stdout, stderr = ssh.exec_command(f"ls {REMOTE_RESULTS}/")
    comp_dirs = stdout.read().decode().strip().split('\n')
    print(f"Found competitions: {comp_dirs}")

    # Also get training log
    log_remote = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_trainer_v3_87729.log'
    download_file(ssh, log_remote, f'{LOCAL_DIR}/mlebench_trainer_v3_87729.log')

    # Get training results JSON
    results_json = f'{REMOTE_RESULTS}/training_results_v3.json'
    download_file(ssh, results_json, f'{LOCAL_DIR}/training_results_v3.json')

    # Download all submission CSVs
    for comp_dir in comp_dirs:
        comp_dir = comp_dir.strip()
        if not comp_dir:
            continue
        remote_sub = f'{REMOTE_RESULTS}/{comp_dir}/submission_s44.csv'
        local_sub = f'{LOCAL_DIR}/submissions_v3/{comp_dir}_submission.csv'
        os.makedirs(os.path.dirname(local_sub), exist_ok=True)
        download_file(ssh, remote_sub, local_sub)

    # Also download training logs from server
    stdin, stdout, stderr = ssh.exec_command(f"tail -200 {log_remote}")
    log_tail = stdout.read().decode('utf-8', errors='replace')
    print("\n=== LAST 200 LINES OF TRAINING LOG ===")
    print(log_tail[-5000:])  # Last 5000 chars

    ssh.close()
    sock.close()
    print("\nDone!")

if __name__ == '__main__':
    main()
