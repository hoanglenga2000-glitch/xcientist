from __future__ import annotations

import json

from research_agent_workstation.server.core.gpu_credentials import connect_ssh, load_gpu_ssh_config


def main() -> int:
    config = load_gpu_ssh_config()
    # Current allocations are exposed directly through the local 7890 bridge.
    # A stale historical jump host must never take precedence over that target.
    config.jump_host = None
    client = connect_ssh(config, timeout=20)
    command = (
        "printf 'USER='; whoami; "
        "printf 'HOST='; hostname; "
        "printf 'PWD='; pwd; "
        "printf 'GPU='; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
        "test -d /hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra "
        "&& echo ROOT_OK=1 || echo ROOT_OK=0"
    )
    try:
        _stdin, stdout, stderr = client.exec_command(command, timeout=30)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
    finally:
        client.close()
    print(json.dumps({"exit_code": code, "stdout": out, "stderr": err}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
