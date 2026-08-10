"""SSH via SOCKS5 proxy using paramiko + PySocks. Run commands or upload files."""
import socks
import paramiko
import sys
import time

try:
    from scripts.hpc_connect import secure_ssh_client
except ModuleNotFoundError:  # direct script execution
    from hpc_connect import secure_ssh_client


def ssh_via_socks5(host, port, user, password, command, timeout=60):
    """Execute command on remote host via SOCKS5 proxy."""
    sock = socks.socksocket()
    sock.set_proxy(socks.SOCKS5, "127.0.0.1", 7890)
    sock.settimeout(timeout)
    sock.connect((host, port))

    client = secure_ssh_client()
    try:
        client.connect(
            host, port=port, username=user, password=password, sock=sock,
            allow_agent=False, look_for_keys=False, timeout=timeout,
            banner_timeout=timeout, auth_timeout=timeout,
        )
    except paramiko.AuthenticationException:
        client.close()
        return -1, "", "AUTH FAILED"
    except Exception as e:
        client.close()
        return -2, "", f"CONNECT FAILED: {e}"

    try:
        transport = client.get_transport()
        if transport is None or not transport.is_authenticated():
            return -2, "", "CONNECT FAILED: authenticated transport unavailable"
        session = transport.open_session()
        session.setblocking(True)
        session.exec_command(command)

        stdout = b""
        stderr = b""
        deadline = time.time() + timeout

        while time.time() < deadline:
            if session.recv_ready():
                stdout += session.recv(65536)
            if session.recv_stderr_ready():
                stderr += session.recv_stderr(65536)
            if session.exit_status_ready():
                break
            time.sleep(0.05)

        # Drain remaining
        time.sleep(0.2)
        while session.recv_ready():
            stdout += session.recv(65536)
        while session.recv_stderr_ready():
            stderr += session.recv_stderr(65536)

        exit_code = session.recv_exit_status()
    finally:
        client.close()

    return exit_code, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


def upload_via_socks5(host, port, user, password, local_path, remote_path, timeout=60):
    """Upload a file to remote host via SOCKS5 proxy."""
    sock = socks.socksocket()
    sock.set_proxy(socks.SOCKS5, "127.0.0.1", 7890)
    sock.settimeout(timeout)
    sock.connect((host, port))

    client = secure_ssh_client()
    try:
        client.connect(
            host, port=port, username=user, password=password, sock=sock,
            allow_agent=False, look_for_keys=False, timeout=timeout,
            banner_timeout=timeout, auth_timeout=timeout,
        )
    except paramiko.AuthenticationException:
        client.close()
        return -1, "AUTH FAILED"
    except Exception as e:
        client.close()
        return -2, f"CONNECT FAILED: {e}"

    try:
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        return 0, f"Uploaded {local_path} -> {remote_path}"
    finally:
        client.close()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: secret-on-stdin | python ssh_proxy_helper.py <host> <port> <user> <command>")
        print("       secret-on-stdin | python ssh_proxy_helper.py upload <host> <port> <user> <local> <remote>")
        sys.exit(1)

    secret = sys.stdin.readline().rstrip("\r\n")
    if not secret:
        print("SSH secret must be supplied on stdin", file=sys.stderr)
        sys.exit(2)

    if sys.argv[1] == "upload":
        if len(sys.argv) != 7:
            sys.exit(1)
        _, _, host, port, user, local, remote = sys.argv
        port = int(port)
        code, msg = upload_via_socks5(host, port, user, secret, local, remote)
        print(msg)
        sys.exit(code)
    else:
        host, port, user = sys.argv[1], int(sys.argv[2]), sys.argv[3]
        command = sys.argv[4]
        exit_code, stdout, stderr = ssh_via_socks5(host, port, user, secret, command)
        print(stdout)
        if stderr:
            print(f"STDERR: {stderr}", file=sys.stderr)
        sys.exit(exit_code)
