"""SSH to HPC via SOCKS5 proxy using pexpect."""
import pexpect, sys, os

def ssh_hpc(command, timeout=60):
    """Connect to HPC cluster and run command. Returns (stdout, stderr)."""
    ssh_cmd = (
        'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        '-o ConnectTimeout=15 '
        '-o "ProxyCommand=ncat --proxy-type socks5 --proxy 127.0.0.1:7897 100.85.169.63 1235" '
        'aimslab-deOiwKsB@100.85.169.63 -p 1235 '
        f'"{command}"'
    )

    child = pexpect.popen_spawn.PopenSpawn(ssh_cmd, timeout=timeout, encoding='utf-8')

    # Wait for password prompt or connection
    idx = child.expect(['password:', 'Password:', pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    if idx in [0, 1]:
        child.sendline('31PFmLLb1')

    # Read all output
    try:
        child.expect(pexpect.EOF, timeout=timeout)
    except:
        pass

    output = child.before or ''
    child.close()
    return output

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = ' '.join(sys.argv[1:])
    else:
        cmd = 'hostname && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader'

    result = ssh_hpc(cmd)
    print(result)
