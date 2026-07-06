#!/usr/bin/env python
"""
MLE-Bench Results Grading Script
=================================
Downloads trained model submission CSVs from the HPC training server,
grades them using mlebench.grade.grade_csv(), and reports medal status.

Competitions:
  - spaceship-titanic          (grader: accuracy, higher=better)
  - tabular-playground-series-dec-2021  (grader: multi-class accuracy, higher=better)
  - tabular-playground-series-may-2022  (grader: auc-roc, higher=better)

Usage:
  set GPU_SSH_PASSWORD=31PFmLLb1f
  python grade_all_results.py                     # connect to HPC, download, grade, cleanup
  python grade_all_results.py --keep              # keep downloaded files and reports
  python grade_all_results.py --local <dir>       # grade CSVs already on local disk
  python grade_all_results.py --dry-run           # only list what's on the server

Server connection:
  hpc_connect.py  -- SOCKS5 proxy (127.0.0.1:7897) -> SSHPiper -> HPC cluster (job 87384)
  Falls back to direct paramiko Transport over SOCKS5 if hpc_connect fails.

HPC jobs (from hpc_connect.py JOBS dict):
  87384 = aimslab-TTA-A800-1GPU  (A800x1, primary training server)
"""

import argparse
import json
import os
import shutil
import socket
import struct
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

os.environ.setdefault("GPU_SSH_PASSWORD", "31PFmLLb1f")

HPC_JOB_ID = "87384"
HPC_USER = "aimslab-TTA-A800-1GPU"
SOCKS5_HOST = "100.85.169.63"
SOCKS5_PORT = 1235

SERVER_RESULTS_DIR = "~/jinghw/scripts/gpu_tra/mlebench_proper_results"

# Home directory fallback paths on the server
SERVER_HOME_CANDIDATES = [
    "/root",
    f"/home/{HPC_USER}",
]

COMPETITIONS = [
    "spaceship-titanic",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
]

# Ensure scripts dir in path for hpc_connect import
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# SOCKS5 + SSH connection
# ---------------------------------------------------------------------------

def socks5_connect(proxy_host="127.0.0.1", proxy_port=7897,
                   target_host=SOCKS5_HOST, target_port=SOCKS5_PORT,
                   timeout=30):
    """Create a raw socket tunnelled through the SOCKS5 proxy.
    Returns the connected socket.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((proxy_host, proxy_port))

    # No-auth handshake
    sock.send(b"\x05\x01\x00")
    resp = sock.recv(2)
    if resp != b"\x05\x00":
        raise ConnectionError(f"SOCKS5 handshake rejected: {resp.hex()}")

    # CONNECT to target
    sock.send(b"\x05\x01\x00\x01"
              + socket.inet_aton(target_host)
              + struct.pack(">H", target_port))
    resp = sock.recv(10)
    if resp[1] != 0x00:
        codes = {1: "general failure", 2: "connection not allowed",
                 3: "network unreachable", 4: "host unreachable",
                 5: "connection refused", 6: "TTL expired",
                 7: "command not supported", 8: "address type not supported"}
        raise ConnectionError(f"SOCKS5 connect failed: {codes.get(resp[1], resp[1])}")
    return sock


def sftp_via_hpc_connect():
    """Use the existing hpc_connect.py module (SSHClient + exec_command)."""
    try:
        from hpc_connect import hpc_connect
        ssh = hpc_connect(HPC_JOB_ID)
        sftp = ssh.open_sftp()
        return sftp, ssh
    except ImportError:
        return None, None
    except Exception as e:
        print(f"  [WARN] hpc_connect failed: {e}")
        return None, None


def sftp_via_transport():
    """Use paramiko Transport directly over SOCKS5 (more reliable with SSHPiper)."""
    import paramiko

    try:
        sock = socks5_connect()
    except ConnectionError as e:
        print(f"  [ERROR] SOCKS5 proxy: {e}")
        return None, None

    t = paramiko.Transport(sock)
    try:
        t.connect(username=HPC_USER, password=os.environ["GPU_SSH_PASSWORD"])
    except paramiko.AuthenticationException as e:
        print(f"  [ERROR] SSH auth failed: {e}")
        print(f"  [HINT] Check GPU_SSH_PASSWORD env var (current length={len(os.environ['GPU_SSH_PASSWORD'])}).")
        t.close()
        return None, None
    except EOFError:
        print(f"  [ERROR] SSH session closed during auth (SSHPiper backend may be down).")
        t.close()
        return None, None
    except Exception as e:
        print(f"  [ERROR] Transport connect failed: {type(e).__name__}: {e}")
        t.close()
        return None, None

    if not t.is_authenticated():
        print(f"  [ERROR] Transport not authenticated (password may have rotated).")
        t.close()
        return None, None

    try:
        sftp = paramiko.SFTPClient.from_transport(t)
        return sftp, t
    except Exception as e:
        print(f"  [ERROR] SFTP init failed: {e}")
        t.close()
        return None, None


def connect_and_get_sftp():
    """Try multiple strategies to open an SFTP session.
    Returns (sftp, client_or_transport) or (None, None).
    """
    print("  [1/2] Trying hpc_connect module...")
    sftp, ssh = sftp_via_hpc_connect()
    if sftp is not None:
        print("  [OK]  Connected via hpc_connect.")
        return sftp, ssh

    print("  [2/2] Trying paramiko Transport via SOCKS5...")
    sftp, ssh = sftp_via_transport()
    if sftp is not None:
        print("  [OK]  Connected via Transport.")
        return sftp, ssh

    return None, None


# ---------------------------------------------------------------------------
# Server path resolution
# ---------------------------------------------------------------------------

def resolve_server_dir(sftp):
    """Resolve the server results directory, expanding ~."""
    for home in SERVER_HOME_CANDIDATES + [""]:
        candidate = SERVER_RESULTS_DIR.replace("~", home) if home else SERVER_RESULTS_DIR
        try:
            sftp.listdir(candidate)
            return candidate
        except IOError:
            continue

    # Last resort: try normalizePath
    try:
        return sftp.normalizePath(".") + "/" + SERVER_RESULTS_DIR.replace("~/", "")
    except Exception:
        pass
    return None


def download_csvs(sftp, server_dir, local_dir, dry_run=False):
    """Recursively download CSV files from server_dir to local_dir.
    Returns list of (local_path, competition_dir_name) tuples.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    try:
        entries = sftp.listdir(server_dir)
    except IOError as e:
        print(f"  [WARN] Cannot list '{server_dir}': {e}")
        return downloaded

    for entry in sorted(entries):
        remote_path = f"{server_dir}/{entry}"
        try:
            attr = sftp.lstat(remote_path)
        except IOError:
            continue

        if attr.st_mode & 0o040000:  # directory
            downloaded.extend(download_csvs(sftp, remote_path, local_dir / entry, dry_run))
        elif entry.lower().endswith(".csv"):
            local_path = local_dir / entry
            print(f"  {'[DRY-RUN]' if dry_run else '[DOWNLOAD]'} {remote_path} -> {local_path}")
            if not dry_run:
                sftp.get(remote_path, str(local_path))
            # Determine competition from parent dir name or file name
            comp_from_dir = Path(server_dir).name
            comp_match = None
            for cid in COMPETITIONS:
                if cid == comp_from_dir or cid.replace("-", "") in comp_from_dir.replace("-", ""):
                    comp_match = cid
                    break
            downloaded.append((local_path, comp_match or comp_from_dir))

    return downloaded


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_one(csv_path, competition_id):
    """Grade a single CSV using mlebench. Returns CompetitionReport."""
    from mlebench.grade import grade_csv
    from mlebench.registry import registry

    competition = registry.get_competition(competition_id)
    return grade_csv(Path(csv_path), competition)


def guess_competition(csv_path):
    """Try to determine competition ID from a CSV path."""
    path_str = str(csv_path).lower()
    for cid in COMPETITIONS:
        # e.g. "spaceship-titanic" matches "spaceship_titanic_submission.csv"
        normalized = cid.replace("-", "")
        path_normalized = path_str.replace("-", "").replace("_", "")
        if normalized in path_normalized:
            return cid
    return None


def medal_label(report):
    if report.gold_medal:
        return "GOLD"
    elif report.silver_medal:
        return "SILVER"
    elif report.bronze_medal:
        return "BRONZE"
    return "NONE"


def medal_color(report):
    m = medal_label(report)
    if report.gold_medal:
        return f"\033[1;33m{m:6s}\033[0m"
    elif report.silver_medal:
        return f"\033[1;37m{m:6s}\033[0m"
    elif report.bronze_medal:
        return f"\033[0;33m{m:6s}\033[0m"
    return f"{m:6s}"


def score_fmt(report):
    if report.score is None:
        return "INVALID"
    d = "v" if report.is_lower_better else "^"
    return f"{report.score:.5f} ({d})"


def print_one_report(report, label=""):
    """Pretty-print a grading report for one submission."""
    print()
    print(f"  {'='*64}")
    print(f"  Competition : {report.competition_id}")
    if label:
        print(f"  Submission  : {label}")
    print(f"  Score       : {score_fmt(report)}")
    print(f"  Medal       : {medal_color(report)}")
    print(f"  Above Median: {'Yes' if report.above_median else 'No'}")
    print(f"  Valid       : {'Yes' if report.valid_submission else 'No'}")
    print(f"  {'-'*40}")
    print(f"  Thresholds (higher=better):" if not report.is_lower_better else
          f"  Thresholds (lower=better):")
    print(f"    Gold   : {report.gold_threshold:.5f}")
    print(f"    Silver : {report.silver_threshold:.5f}")
    print(f"    Bronze : {report.bronze_threshold:.5f}")
    print(f"    Median : {report.median_threshold:.5f}")
    print(f"  {'='*64}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MLE-Bench Results Grading Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python grade_all_results.py                    # connect to HPC, download, grade, cleanup
  python grade_all_results.py --keep             # keep downloaded files + JSON report
  python grade_all_results.py --local ./results  # grade CSVs already on disk
  python grade_all_results.py --dry-run          # list server contents without downloading
        """.strip(),
    )
    parser.add_argument("--keep", action="store_true",
                        help="Keep downloaded files and grading report (don't clean up).")
    parser.add_argument("--local", type=str, metavar="DIR",
                        help="Grade CSVs from a local directory instead of downloading.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List server contents without downloading or grading.")
    parser.add_argument("--output", type=str, metavar="FILE",
                        help="Save JSON report to this path (requires --keep).")
    args = parser.parse_args()

    print("=" * 70)
    print("  MLE-Bench Results Grading Script")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Competitions: {', '.join(COMPETITIONS)}")
    print("=" * 70)

    # --local mode: grade files already on disk
    if args.local:
        local_dir = Path(args.local)
        if not local_dir.is_dir():
            print(f"\n[FATAL] Local directory not found: {local_dir}")
            sys.exit(1)

        csv_files = list(local_dir.rglob("*.csv"))
        print(f"\n[INFO] Found {len(csv_files)} CSV file(s) in {local_dir}")

        if not csv_files:
            print("[FATAL] No CSV files to grade.")
            sys.exit(1)

        all_reports = []
        for fp in csv_files:
            comp_id = guess_competition(fp)
            if comp_id is None:
                print(f"  [SKIP]  Cannot determine competition: {fp.name}")
                continue
            print(f"\n[GRADING] {fp.name} as {comp_id} ...")
            try:
                report = grade_one(fp, comp_id)
                print_one_report(report, str(fp))
                all_reports.append(report)
            except ValueError as e:
                print(f"  [ERROR] {e}")
            except Exception as e:
                print(f"  [ERROR] {type(e).__name__}: {e}")

        # Summary
        print_summary(all_reports)
        if args.output:
            save_json(all_reports, args.output)
        elif args.keep:
            save_json(all_reports, local_dir / "grading_summary.json")
        return

    # --dry-run mode: list server without downloading
    if args.dry_run:
        print("\n[DRY-RUN] Connecting to list server contents...")
        sftp, client = connect_and_get_sftp()
        if sftp is None:
            print("\n[FATAL] Cannot connect to server.")
            sys.exit(1)

        server_dir = resolve_server_dir(sftp)
        if server_dir is None:
            print(f"[FATAL] Cannot find results directory on server.")
            client.close()
            sys.exit(1)

        print(f"\n[INFO] Server directory: {server_dir}")
        download_csvs(sftp, server_dir, Path("."), dry_run=True)
        sftp.close()
        client.close()
        return

    # Normal mode: connect, download, grade, clean up
    print("\n[STEP 1] Connecting to HPC cluster...")
    sftp, client = connect_and_get_sftp()
    if sftp is None:
        print("\n" + "=" * 70)
        print("  CONNECTION FAILED")
        print("=" * 70)
        print("  Could not connect to the HPC cluster.")
        print()
        print("  Troubleshooting:")
        print("    1. Verify SOCKS5 proxy is running: netstat -an | findstr 7897")
        print("    2. Check password: echo %GPU_SSH_PASSWORD%")
        print("    3. Try connecting manually: ssh aimslab-TTA-A800-1GPU@100.85.169.63 -p 1235")
        print("       (requires SOCKS5 proxy tool like proxychains or ncat)")
        print("    4. HPC job may have expired; check hpc_connect.py JOBS dict")
        print()
        print("  As a workaround, use --local to grade files already on disk:")
        print("    python grade_all_results.py --local <path_to_csvs>")
        sys.exit(1)

    print("[OK] Connected.")

    # Resolve and scan server directory
    print("\n[STEP 2] Scanning server for submission files...")
    server_dir = resolve_server_dir(sftp)
    if server_dir is None:
        print(f"\n[FATAL] Cannot find results directory on server.")
        print(f"  Tried: {SERVER_RESULTS_DIR}")
        for home in SERVER_HOME_CANDIDATES:
            print(f"         {SERVER_RESULTS_DIR.replace('~', home)}")
        client.close()
        sys.exit(1)

    print(f"  Server directory: {server_dir}")
    try:
        top_entries = sftp.listdir(server_dir)
        print(f"  Top-level entries: {top_entries}")
    except IOError:
        top_entries = []
        print(f"  [WARN] Cannot list top-level directory.")

    # Download
    print(f"\n[STEP 3] Downloading submission CSVs...")
    download_dir = Path(tempfile.mkdtemp(prefix="mlebench_grading_"))
    print(f"  Temp dir: {download_dir}")

    all_downloads = download_csvs(sftp, server_dir, download_dir)
    sftp.close()
    client.close()

    # Group by competition
    submissions = {}  # comp_id -> [Path, ...]
    for local_path, comp_dir in all_downloads:
        comp_id = guess_competition(local_path) or comp_dir
        submissions.setdefault(comp_id, []).append(local_path)

    print(f"\n  Downloaded {len(all_downloads)} file(s) across {len(submissions)} competition(s).")

    if not all_downloads:
        print("\n[FATAL] No CSV files found on server. Nothing to grade.")
        shutil.rmtree(download_dir, ignore_errors=True)
        sys.exit(1)

    # Grade
    print(f"\n[STEP 4] Grading...")
    all_reports = []
    for comp_id in COMPETITIONS:
        if comp_id not in submissions:
            print(f"\n  [SKIP] {comp_id}: no submissions found")
            continue
        for csv_path in submissions[comp_id]:
            print(f"\n  Grading {csv_path.name} as {comp_id}...")
            try:
                report = grade_one(csv_path, comp_id)
                print_one_report(report, csv_path.name)
                all_reports.append(report)
            except ValueError as e:
                print(f"  [ERROR] Dataset not prepared or invalid: {e}")
            except Exception as e:
                print(f"  [ERROR] {type(e).__name__}: {e}")

    # Summary
    print_summary(all_reports)

    # Save report
    if args.output:
        save_json(all_reports, args.output)
        print(f"\n[INFO] JSON report saved to: {args.output}")
    elif args.keep:
        save_json(all_reports, download_dir / "grading_summary.json")
        print(f"\n[INFO] JSON report saved to: {download_dir / 'grading_summary.json'}")
        print(f"[INFO] Downloaded files kept at: {download_dir}")

    # Cleanup
    if not args.keep:
        print(f"\n[STEP 5] Cleaning up {download_dir}...")
        shutil.rmtree(download_dir, ignore_errors=True)
        print("[DONE] Cleanup complete.")
    else:
        print(f"\n[DONE] Files kept at: {download_dir}")

    print()


def print_summary(all_reports):
    """Print a summary table of all grading results."""
    print(f"\n\n{'='*70}")
    print(f"  GRADING SUMMARY")
    print(f"{'='*70}")

    if not all_reports:
        print("  No reports generated.")
        return

    golds   = sum(1 for r in all_reports if r.gold_medal)
    silvers = sum(1 for r in all_reports if r.silver_medal)
    bronzes = sum(1 for r in all_reports if r.bronze_medal)
    nones   = sum(1 for r in all_reports if not r.any_medal)
    valid   = sum(1 for r in all_reports if r.valid_submission)
    total   = len(all_reports)

    print(f"  Total graded : {total}")
    print(f"  Valid        : {valid}")
    print(f"  Gold         : {golds}")
    print(f"  Silver       : {silvers}")
    print(f"  Bronze       : {bronzes}")
    print(f"  No medal     : {nones}")
    if total > 0:
        print(f"  Medal rate   : {(golds + silvers + bronzes) / total * 100:.1f}%")

    # Per-competition table
    print(f"\n  {'Competition':<42s} {'Score':>18s}  {'Medal':6s}  {'>Median':>8s}")
    print(f"  {'-'*42}  {'-'*18}  {'-'*6}  {'-'*8}")
    for r in all_reports:
        s = score_fmt(r) if r.valid_submission else "INVALID"
        m = medal_label(r)
        am = "Yes" if r.above_median else "No"
        print(f"  {r.competition_id:<42s}  {s:>18s}  {medal_color(r)}  {am:>8s}")

    print(f"  {'='*70}\n")


def save_json(all_reports, path):
    """Save reports as JSON."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total": len(all_reports),
        "valid": sum(1 for r in all_reports if r.valid_submission),
        "gold": sum(1 for r in all_reports if r.gold_medal),
        "silver": sum(1 for r in all_reports if r.silver_medal),
        "bronze": sum(1 for r in all_reports if r.bronze_medal),
        "none": sum(1 for r in all_reports if not r.any_medal),
        "reports": [r.to_dict() for r in all_reports],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
