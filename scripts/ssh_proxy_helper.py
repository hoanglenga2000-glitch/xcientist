"""SSH via SOCKS5 proxy using paramiko + PySocks. Run commands or upload files."""
import socks
import paramiko
import sys
import time


def ssh_via_socks5(host, port, user, password, command, timeout=60):
    """Execute command on remote host via SOCKS5 proxy."""
    sock = socks.socksocket()
    sock.set_proxy(socks.SOCKS5, "127.0.0.1", 7890)
    sock.settimeout(timeout)
    sock.connect((host, port))

    transport = paramiko.Transport(sock)
    try:
        transport.connect(username=user, password=password)
    except paramiko.AuthenticationException:
        transport.close()
        return -1, "", "AUTH FAILED"
    except Exception as e:
        transport.close()
        return -2, "", f"CONNECT FAILED: {e}"

    try:
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
        transport.close()

    return exit_code, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


def upload_via_socks5(host, port, user, password, local_path, remote_path, timeout=60):
    """Upload a file to remote host via SOCKS5 proxy."""
    sock = socks.socksocket()
    sock.set_proxy(socks.SOCKS5, "127.0.0.1", 7890)
    sock.settimeout(timeout)
    sock.connect((host, port))

    transport = paramiko.Transport(sock)
    try:
        transport.connect(username=user, password=password)
    except paramiko.AuthenticationException:
        transport.close()
        return -1, "AUTH FAILED"
    except Exception as e:
        transport.close()
        return -2, f"CONNECT FAILED: {e}"

    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(local_path, remote_path)
        sftp.close()
        return 0, f"Uploaded {local_path} -> {remote_path}"
    finally:
        transport.close()


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python ssh_proxy_helper.py <host> <port> <user> <password> <command>")
        print("       python ssh_proxy_helper.py upload <host> <port> <user> <password> <local> <remote>")
        sys.exit(1)

    if sys.argv[1] == "upload":
        _, host, port, user, password, local, remote = sys.argv
        port = int(port)
        code, msg = upload_via_socks5(host, port, user, password, local, remote)
        print(msg)
        sys.exit(code)
    else:
        host, port, user, password = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
        command = sys.argv[5]
        exit_code, stdout, stderr = ssh_via_socks5(host, port, user, password, command)
        print(stdout)
        if stderr:
            print(f"STDERR: {stderr}", file=sys.stderr)
        sys.exit(exit_code)
