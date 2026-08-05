#!/usr/bin/env python3
"""Check AIMSLAB HPC resources via SSH with auto-password."""
import subprocess, sys, os

# Decrypt password via PowerShell
result = subprocess.run(
    ['powershell.exe', '-NoProfile', '-Command',
     '$s = (Get-Content -LiteralPath "$env:USERPROFILE\\.ssh\\aimslab_vscode_89141_password.dpapi" -Raw).Trim() | ConvertTo-SecureString; '
     '[Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($s))'],
    capture_output=True, text=True, timeout=15
)
PASSWORD = result.stdout.strip()
if not PASSWORD:
    print("FAILED to decrypt password", file=sys.stderr)
    sys.exit(1)

print(f"Password decrypted OK ({len(PASSWORD)} chars)")

# Write the pexpect script directly into WSL /tmp via stdin
wsl_script = f'''import pexpect, sys, os
PASSWORD = {repr(PASSWORD)}
commands = "nvidia-smi && echo === && df -h && echo === && free -h && echo === && lscpu | head -15 && echo === && ls -la /hpc2hdd/home/aimslab/jinghw/ && echo === && ls -la /hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/ && echo ===ALL_DONE==="
child = pexpect.spawn("/usr/bin/ssh", [
    "-tt",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "PubkeyAuthentication=no",
    "-o", "PreferredAuthentications=password",
    "-o", "NumberOfPasswordPrompts=1",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
    "-o", "ProxyCommand=nc -x 127.0.0.1:7890 -X 5 %h %p",
    "-p", "1235",
    "aimslab-wqx-SDBS@100.85.169.63",
    commands
], encoding="utf-8", codec_errors="replace", timeout=90)
child.delaybeforesend = 0.05
idx = child.expect([r"(?i)password:", r"(?i)are you sure", r"(?i)permission denied", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
if idx == 0:
    child.sendline(PASSWORD)
    child.expect(pexpect.EOF, timeout=120)
    sys.stdout.write(child.before)
elif idx == 1:
    child.sendline("yes")
    child.expect(r"(?i)password:", timeout=10)
    child.sendline(PASSWORD)
    child.expect(pexpect.EOF, timeout=120)
    sys.stdout.write(child.before)
elif idx == 2:
    print("PERMISSION DENIED")
    sys.exit(10)
elif idx == 3:
    sys.stdout.write(child.before)
elif idx == 4:
    print("TIMEOUT")
    sys.exit(11)
'''

# Write to WSL /tmp directly
result = subprocess.run(
    ['wsl.exe', 'bash', '-c', 'cat > /tmp/_hpc_check.py'],
    input=wsl_script, text=True, timeout=10
)

# Run the script in WSL
result = subprocess.run(
    ['wsl.exe', 'python3', '/tmp/_hpc_check.py'],
    capture_output=True, text=True, timeout=150
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:1000])
sys.exit(result.returncode)
