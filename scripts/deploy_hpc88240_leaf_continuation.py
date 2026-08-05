#!/usr/bin/env python3
"""Deploy a fail-closed Jigsaw -> Leaf continuation on AIMSLAB job 88240.

The deployed wrapper is deliberately non-preemptive: it polls the existing
Jigsaw status, waits for three stable idle-GPU observations, and never sends a
signal to any process.  It runs only below the operator's dedicated HPC root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import posixpath
import shlex
import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
gpu_credentials = import_module("research_agent_workstation.server.core.gpu_credentials")
ALLOWED_GPU_REMOTE_ROOT = gpu_credentials.ALLOWED_GPU_REMOTE_ROOT
connect_ssh = gpu_credentials.connect_ssh
BASE_PLAN = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "leaf_multibackbone_frozen_plan.json"
)
LOCAL_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "leaf_multibackbone_hpc88240_frozen_plan_20260727.json"
)
LOCAL_EVIDENCE = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job88240_leaf_continuation_deployment_current.json"
)
REMOTE_BASE = posixpath.join(
    ALLOWED_GPU_REMOTE_ROOT,
    "evomind_mle22",
    "job88240_leaf_s404142_20260727",
)
REMOTE_JIGSAW_BASE = posixpath.join(
    ALLOWED_GPU_REMOTE_ROOT,
    "evomind_mle22",
    "job88240_jigsaw_s41_20260727",
)
REMOTE_JIGSAW_STATUS = posixpath.join(REMOTE_JIGSAW_BASE, "full_seed41_status.json")
REMOTE_SITE_PACKAGES = posixpath.join(REMOTE_JIGSAW_BASE, "env", "site-packages")
REMOTE_DATA_ROOT = posixpath.join(ALLOWED_GPU_REMOTE_ROOT, "mlebench_official_data")
REMOTE_PUBLIC_ROOT = posixpath.join(
    REMOTE_DATA_ROOT,
    "leaf-classification",
    "prepared",
    "public",
)
REMOTE_TORCH_HOME = posixpath.join(ALLOWED_GPU_REMOTE_ROOT, "mlebench_model_cache", "torch")
RUN_ID = "hpc88240_leaf_dualbb_s404142_20260727"
REMOTE_OUTPUT_ROOT = posixpath.join(REMOTE_BASE, "leaf_multibackbone")
REMOTE_RUN_DIR = posixpath.join(REMOTE_OUTPUT_ROOT, "runs", RUN_ID)
REMOTE_PLAN = posixpath.join(
    REMOTE_BASE,
    "plans",
    LOCAL_PLAN.name,
)
REMOTE_WRAPPER = posixpath.join(REMOTE_BASE, "run_leaf_after_jigsaw.sh")
REMOTE_STATUS = posixpath.join(REMOTE_BASE, "full_leaf_status.json")
REMOTE_DEPLOYMENT = posixpath.join(REMOTE_BASE, "deployment_manifest.json")


SOURCE_BINDINGS: tuple[tuple[Path, str], ...] = (
    (
        PROJECT_ROOT / "scripts" / "run_leaf_multibackbone_oof.py",
        "scripts/run_leaf_multibackbone_oof.py",
    ),
    (
        PROJECT_ROOT / "scripts" / "verify_leaf_multibackbone_oof_run.py",
        "scripts/verify_leaf_multibackbone_oof_run.py",
    ),
    (
        PROJECT_ROOT / "scripts" / "mlebench_medal_recovery_adapters.py",
        "scripts/mlebench_medal_recovery_adapters.py",
    ),
    (
        PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py",
        "scripts/mlebench_wave2_adapters.py",
    ),
    (
        PROJECT_ROOT / "scripts" / "run_mlebench_lite_wave0.py",
        "scripts/run_mlebench_lite_wave0.py",
    ),
    (
        PROJECT_ROOT / "scripts" / "russian_transliteration.py",
        "scripts/russian_transliteration.py",
    ),
    (
        PROJECT_ROOT / "src" / "research_os" / "mlebench_phase_a.py",
        "src/research_os/mlebench_phase_a.py",
    ),
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def ensure_remote_path(path: str) -> str:
    root = PurePosixPath(ALLOWED_GPU_REMOTE_ROOT)
    candidate = PurePosixPath(path)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Remote path escaped the dedicated root: {path}") from exc
    return candidate.as_posix()


def remote_source_path(relative: str) -> str:
    return ensure_remote_path(posixpath.join(REMOTE_BASE, relative))


def exec_checked(client: Any, command: str, *, timeout: int = 180) -> str:
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"Remote command failed exit={code}: {error[-600:]}")
    return output


def remote_public_snapshot(client: Any) -> dict[str, Any]:
    command = f"""PUBLIC={shlex.quote(REMOTE_PUBLIC_ROOT)} python3 - <<'PY'
import hashlib,json,os,pathlib
root=pathlib.Path(os.environ['PUBLIC']).resolve()
required=('train.csv','test.csv','sample_submission.csv','description.md')
if not root.is_dir():
    raise SystemExit('Leaf public root is missing')
files=sorted(path for path in root.rglob('*') if path.is_file())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()
core={{name:{{'bytes':(root/name).stat().st_size,'sha256':digest(root/name)}} for name in required}}
payload={{
 'public_root':str(root),
 'file_count':len(files),
 'total_bytes':sum(path.stat().st_size for path in files),
 'image_count':sum(path.parent.name=='images' for path in files),
 'core_files':core,
}}
print(json.dumps(payload,sort_keys=True))
PY"""
    payload = json.loads(exec_checked(client, command, timeout=180))
    if (
        payload.get("public_root") != REMOTE_PUBLIC_ROOT
        or payload.get("file_count") != 994
        or payload.get("image_count") != 990
        or payload.get("total_bytes") != 28_617_101
    ):
        raise RuntimeError("Remote Leaf public inventory differs from the frozen 994-file set")
    return payload


def source_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for local_path, relative in SOURCE_BINDINGS:
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        records.append(
            {
                "local_path": str(local_path.resolve()),
                "remote_path": remote_source_path(relative),
                "bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
            }
        )
    return records


def build_hpc_plan(
    base_plan: Mapping[str, Any],
    public_snapshot: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan = copy.deepcopy(dict(base_plan))
    records = {Path(str(item["local_path"])).name: item for item in sources}
    runner = records["run_leaf_multibackbone_oof.py"]
    verifier = records["verify_leaf_multibackbone_oof_run.py"]
    plan.update(
        {
            "created_at": now_iso(),
            "status": "frozen_waiting_after_siim",
            "execution_target": {
                "cluster": "AI-X86_NVIDIA",
                "job_id": 88240,
                "gpu": "NVIDIA A40",
                "remote_root": ALLOWED_GPU_REMOTE_ROOT,
                "mode": "persistent_serial_after_jigsaw_seed41",
                "site_packages": REMOTE_SITE_PACKAGES,
            },
        }
    )
    plan["training"] = {
        "python": "python3",
        "script": runner["remote_path"],
        "run_id": RUN_ID,
        "data_root": REMOTE_DATA_ROOT,
        "output_root": REMOTE_OUTPUT_ROOT,
        "torch_home": REMOTE_TORCH_HOME,
        "seeds": [40, 41, 42],
        "folds": 5,
        "backbones": ["convnext_small", "efficientnet_v2_s"],
        "image_size": 288,
        "batch_size": 64,
        "workers": 8,
        "manifest_workers": 16,
        "tta": 8,
        "numeric_c": 10.0,
        "image_c": 8.0,
        "multimodal_c": 10.0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    plan["data_contract"] = {
        **dict(public_snapshot),
        "private_paths_requested": False,
        "remote_writes_limited_to_dedicated_root": True,
    }
    plan["implementation"] = {
        "runner": {
            "path": runner["remote_path"],
            "bytes": runner["bytes"],
            "sha256": runner["sha256"],
        },
        "verifier": {
            "path": verifier["remote_path"],
            "bytes": verifier["bytes"],
            "sha256": verifier["sha256"],
        },
        "source_bundle": [
            {
                "path": item["remote_path"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in sources
        ],
    }
    plan["serial_dependency"] = {
        "jigsaw_job_id": 88240,
        "jigsaw_status_path": REMOTE_JIGSAW_STATUS,
        "accepted_verified_terminal_statuses": [
            "verification_passed",
            "verification_complete_gate_failed",
        ],
        "integrity_failure_statuses": [
            "training_failed",
            "verifier_failed",
            "verification_contract_failed",
            "frozen_artifact_drift",
        ],
    }
    plan["launch_contract"] = {
        "gpu_idle_consecutive_checks": 3,
        "gpu_idle_minimum_check_interval_seconds": 10,
        "maximum_idle_memory_mib": 512,
        "maximum_idle_utilization_percent": 10,
        "no_compute_process_required": True,
        "process_signals_allowed": False,
        "automatic_kaggle_submission": False,
        "automatic_official_grader": False,
    }
    plan["claim_boundary"] = (
        "HPC public-OOF continuation plan; CV and OOF are not official medals. "
        "Human Gate remains mandatory."
    )
    plan["operational_revision"] = {
        "created_at": now_iso(),
        "reason": "Move every post-seed40 GPU training stage from the local RTX 4060 to job 88240 A40.",
        "model_data_folds_seeds_backbones_and_promotion_thresholds_changed": False,
        "throughput_only_changes": {
            "batch_size": 64,
            "workers": 8,
            "manifest_workers": 16,
        },
        "process_signals_sent": 0,
    }
    return plan


def render_wrapper(
    *, plan_sha256: str, sources: Sequence[Mapping[str, Any]], public_snapshot: Mapping[str, Any]
) -> str:
    source_by_name = {Path(str(item["remote_path"])).name: item for item in sources}
    runner = source_by_name["run_leaf_multibackbone_oof.py"]["remote_path"]
    verifier = source_by_name["verify_leaf_multibackbone_oof_run.py"]["remote_path"]
    frozen: list[tuple[str, str]] = [(plan_sha256, REMOTE_PLAN)]
    frozen.extend((str(item["sha256"]), str(item["remote_path"])) for item in sources)
    for name, record in dict(public_snapshot["core_files"]).items():
        frozen.append((str(record["sha256"]), posixpath.join(REMOTE_PUBLIC_ROOT, name)))
    frozen.extend(
        (
            (
                "0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a",
                posixpath.join(
                    REMOTE_TORCH_HOME,
                    "hub/checkpoints/convnext_small-0c510722.pth",
                ),
            ),
            (
                "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74",
                posixpath.join(
                    REMOTE_TORCH_HOME,
                    "hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth",
                ),
            ),
        )
    )
    frozen_lines = "\n".join(f"{digest}|{path}" for digest, path in frozen)
    return f"""#!/usr/bin/env bash
set -uo pipefail
BASE={shlex.quote(REMOTE_BASE)}
JIGSAW_STATUS={shlex.quote(REMOTE_JIGSAW_STATUS)}
PLAN={shlex.quote(REMOTE_PLAN)}
RUN_ID={shlex.quote(RUN_ID)}
RUN_DIR={shlex.quote(REMOTE_RUN_DIR)}
STATUS={shlex.quote(REMOTE_STATUS)}
DATA_ROOT={shlex.quote(REMOTE_DATA_ROOT)}
PUBLIC_ROOT={shlex.quote(REMOTE_PUBLIC_ROOT)}
OUTPUT_ROOT={shlex.quote(REMOTE_OUTPUT_ROOT)}
TORCH_HOME_DIR={shlex.quote(REMOTE_TORCH_HOME)}
RUNNER={shlex.quote(str(runner))}
VERIFIER={shlex.quote(str(verifier))}
SITE={shlex.quote(REMOTE_SITE_PACKAGES)}
PYTHONPATH_VALUE="$SITE:$BASE/src:$BASE"

write_status() {{
  state="$1"
  code="${{2:-}}"
  stable="${{3:-0}}"
  EVOMIND_STATE="$state" EVOMIND_EXIT_CODE="$code" EVOMIND_STABLE="$stable" \
  EVOMIND_WRAPPER_PID="$$" EVOMIND_STATUS_PATH="$STATUS" EVOMIND_RUN_ID="$RUN_ID" \
  EVOMIND_RUN_DIR="$RUN_DIR" EVOMIND_PLAN="$PLAN" EVOMIND_JIGSAW_STATUS="$JIGSAW_STATUS" \
  python3 - <<'PY'
import datetime,json,os,pathlib
path=pathlib.Path(os.environ['EVOMIND_STATUS_PATH'])
tmp=path.with_suffix(path.suffix+'.tmp')
raw=os.environ.get('EVOMIND_EXIT_CODE','')
payload={{
 'schema':'evomind.hpc_leaf_persistent_run.v1',
 'created_at':datetime.datetime.now().astimezone().isoformat(),
 'status':os.environ['EVOMIND_STATE'],
 'wrapper_pid':int(os.environ['EVOMIND_WRAPPER_PID']),
 'run_id':os.environ['EVOMIND_RUN_ID'],
 'run_dir':os.environ['EVOMIND_RUN_DIR'],
 'plan_path':os.environ['EVOMIND_PLAN'],
 'jigsaw_status_path':os.environ['EVOMIND_JIGSAW_STATUS'],
 'stable_idle_observations':int(os.environ.get('EVOMIND_STABLE','0')),
 'exit_code':int(raw) if raw else None,
 'private_labels_used':False,
 'official_grader_executed':False,
 'kaggle_submission_executed':False,
 'process_signals_sent':0,
 'other_processes_modified':False,
}}
tmp.write_text(json.dumps(payload,indent=2)+'\\n')
tmp.replace(path)
PY
}}

verify_frozen() {{
  while IFS='|' read -r expected path; do
    [ -n "$expected" ] || continue
    if [ ! -f "$path" ]; then
      write_status frozen_artifact_missing 22
      return 22
    fi
    actual=$(sha256sum -- "$path" | awk '{{print $1}}')
    if [ "$actual" != "$expected" ]; then
      write_status frozen_artifact_drift 23
      return 23
    fi
  done <<'EOF'
{frozen_lines}
EOF
}}

verify_frozen || exit $?
if [ -e "$RUN_DIR" ]; then
  write_status target_run_exists 21
  exit 21
fi

write_status waiting_for_jigsaw_seed41
while true; do
  if [ ! -f "$JIGSAW_STATUS" ]; then
    sleep 30
    continue
  fi
  jigsaw_state=$(JIGSAW_STATUS="$JIGSAW_STATUS" python3 - <<'PY'
import json,os
print(json.load(open(os.environ['JIGSAW_STATUS'],encoding='utf-8')).get('status',''))
PY
)
  case "$jigsaw_state" in
    verification_passed|verification_complete_gate_failed) break ;;
    training_failed|verifier_failed|verification_contract_failed|frozen_artifact_drift)
      write_status blocked_by_jigsaw_integrity_failure 31
      exit 31
      ;;
    *) write_status waiting_for_jigsaw_seed41; sleep 30 ;;
  esac
done

verify_frozen || exit $?
stable=0
while [ "$stable" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,pstate --format=csv,noheader,nounits)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' | wc -l)
  used=$(echo "$line" | awk -F',' '{{gsub(/ /,"",$4);print $4}}')
  util=$(echo "$line" | awk -F',' '{{gsub(/ /,"",$6);print $6}}')
  if [ "$apps" -eq 0 ] && [ "$used" -le 512 ] && [ "$util" -le 10 ]; then
    stable=$((stable+1))
  else
    stable=0
  fi
  write_status waiting_for_gpu_idle "" "$stable"
  if [ "$stable" -lt 3 ]; then sleep 10; fi
done

verify_frozen || exit $?
if [ -e "$RUN_DIR" ]; then
  write_status target_run_exists 21
  exit 21
fi
mkdir -p -- "$OUTPUT_ROOT" "$BASE/logs"
write_status training
PYTHONPATH="$PYTHONPATH_VALUE" TORCH_HOME="$TORCH_HOME_DIR" CUDA_VISIBLE_DEVICES=0 \
python3 "$RUNNER" \
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --run-id "$RUN_ID" \
  --seeds 40,41,42 --folds 5 --backbones convnext_small,efficientnet_v2_s \
  --image-size 288 --batch-size 64 --workers 8 --manifest-workers 16 --tta 8 \
  --numeric-c 10.0 --image-c 8.0 --multimodal-c 10.0 \
  --promotion-mean-log-loss 0.0135 --promotion-max-seed-log-loss 0.0135 \
  --torch-home "$TORCH_HOME_DIR" \
  >"$BASE/logs/leaf_runner.stdout.log" 2>"$BASE/logs/leaf_runner.stderr.log"
runner_rc=$?
if [ "$runner_rc" -ne 0 ] && [ "$runner_rc" -ne 3 ]; then
  write_status training_failed "$runner_rc"
  exit "$runner_rc"
fi

write_status verifying "$runner_rc"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PYTHONPATH_VALUE" python3 "$VERIFIER" \
  --run-dir "$RUN_DIR" --plan "$PLAN" --output "$RUN_DIR/independent_verification.json" \
  >"$BASE/logs/leaf_verifier.stdout.log" 2>"$BASE/logs/leaf_verifier.stderr.log"
verify_rc=$?
if [ "$verify_rc" -ne 0 ]; then
  write_status verifier_failed "$verify_rc"
  exit "$verify_rc"
fi

final_state=$(REPORT="$RUN_DIR/independent_verification.json" python3 - <<'PY'
import json,os
r=json.load(open(os.environ['REPORT'],encoding='utf-8'))
if r.get('ok') is not True:
    print('verification_contract_failed')
elif r.get('candidate_ready') is True:
    print('verification_passed')
else:
    print('verification_complete_gate_failed')
PY
)
write_status "$final_state" 0
exit 0
"""


def upload_file(client: Any, local_path: Path, remote_path: str) -> None:
    remote_path = ensure_remote_path(remote_path)
    temporary = f"{remote_path}.{os.getpid()}.tmp"
    with client.open_sftp() as sftp:
        sftp.put(str(local_path), temporary)
    exec_checked(
        client,
        f"mkdir -p -- {shlex.quote(posixpath.dirname(remote_path))}; "
        f"mv -f -- {shlex.quote(temporary)} {shlex.quote(remote_path)}",
    )


def upload_bytes(client: Any, value: bytes, remote_path: str) -> None:
    remote_path = ensure_remote_path(remote_path)
    temporary = f"{remote_path}.{os.getpid()}.tmp"
    exec_checked(client, f"mkdir -p -- {shlex.quote(posixpath.dirname(remote_path))}")
    with client.open_sftp() as sftp:
        with sftp.open(temporary, "wb") as handle:
            handle.write(value)
            handle.flush()
    exec_checked(client, f"mv -f -- {shlex.quote(temporary)} {shlex.quote(remote_path)}")


def remote_json(client: Any, path: str) -> dict[str, Any] | None:
    command = (
        f"if [ -f {shlex.quote(path)} ]; then cat -- {shlex.quote(path)}; "
        "else printf 'null'; fi"
    )
    payload = json.loads(exec_checked(client, command, timeout=60))
    return payload if isinstance(payload, dict) else None


def deploy(*, start: bool = True) -> dict[str, Any]:
    base = json.loads(BASE_PLAN.read_text(encoding="utf-8-sig"))
    sources = source_records()
    client = connect_ssh(timeout=30)
    try:
        public = remote_public_snapshot(client)
        plan = build_hpc_plan(base, public, sources)
        write_json_atomic(LOCAL_PLAN, plan)
        plan_sha = sha256_file(LOCAL_PLAN)
        wrapper = render_wrapper(
            plan_sha256=plan_sha,
            sources=sources,
            public_snapshot=public,
        )
        wrapper_bytes = wrapper.encode("utf-8")

        exec_checked(
            client,
            "mkdir -p -- "
            + " ".join(
                shlex.quote(value)
                for value in (
                    REMOTE_BASE,
                    posixpath.join(REMOTE_BASE, "scripts"),
                    posixpath.join(REMOTE_BASE, "src", "research_os"),
                    posixpath.join(REMOTE_BASE, "plans"),
                    posixpath.join(REMOTE_BASE, "logs"),
                )
            ),
        )
        for record in sources:
            upload_file(client, Path(str(record["local_path"])), str(record["remote_path"]))
        upload_bytes(
            client,
            b'"""Minimal package marker for the frozen Leaf HPC bundle."""\n',
            remote_source_path("src/research_os/__init__.py"),
        )
        upload_file(client, LOCAL_PLAN, REMOTE_PLAN)
        upload_bytes(client, wrapper_bytes, REMOTE_WRAPPER)
        exec_checked(client, f"chmod 700 -- {shlex.quote(REMOTE_WRAPPER)}")

        smoke = exec_checked(
            client,
            f"PYTHONPATH={shlex.quote(REMOTE_SITE_PACKAGES + ':' + REMOTE_BASE + '/src:' + REMOTE_BASE)} "
            "CUDA_VISIBLE_DEVICES='' python3 - <<'PY'\n"
            "import torch,torchvision,numpy,pandas,sklearn,PIL,scipy\n"
            "import scripts.run_leaf_multibackbone_oof as runner\n"
            "import scripts.verify_leaf_multibackbone_oof_run as verifier\n"
            "print('import_smoke=passed')\n"
            "print('torch='+torch.__version__)\n"
            "print('torchvision='+torchvision.__version__)\n"
            "print('runner='+runner.sha256_file(runner.Path(runner.__file__).resolve()))\n"
            "print('verifier='+verifier.sha256_file(verifier.Path(verifier.__file__).resolve()))\n"
            "PY",
            timeout=180,
        )
        remote_hashes = json.loads(
            exec_checked(
                client,
                "PLAN="
                + shlex.quote(REMOTE_PLAN)
                + " WRAPPER="
                + shlex.quote(REMOTE_WRAPPER)
                + " python3 - <<'PY'\n"
                "import hashlib,json,os,pathlib\n"
                "def d(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()\n"
                "print(json.dumps({'plan':d(os.environ['PLAN']),'wrapper':d(os.environ['WRAPPER'])}))\n"
                "PY",
            )
        )
        if remote_hashes["plan"] != plan_sha or remote_hashes["wrapper"] != sha256_bytes(
            wrapper_bytes
        ):
            raise RuntimeError("Remote Leaf plan or wrapper hash differs after upload")

        prior_status = remote_json(client, REMOTE_STATUS)
        launch_pid: int | None = None
        launch_action = "deployed_without_start"
        if start:
            prior_pid = int((prior_status or {}).get("wrapper_pid") or 0)
            prior_active = False
            if prior_pid > 0:
                output = exec_checked(
                    client,
                    f"if ps -p {prior_pid} -o args= | grep -F -- {shlex.quote(REMOTE_WRAPPER)} >/dev/null; "
                    "then echo yes; else echo no; fi",
                ).strip()
                prior_active = output == "yes"
            if prior_active:
                launch_pid = prior_pid
                launch_action = "existing_wrapper_reused"
            elif prior_status and prior_status.get("status") in {
                "training",
                "verifying",
                "verification_passed",
                "verification_complete_gate_failed",
            }:
                launch_action = "terminal_or_active_status_preserved"
            else:
                output = exec_checked(
                    client,
                    f"nohup bash {shlex.quote(REMOTE_WRAPPER)} "
                    f">{shlex.quote(REMOTE_BASE + '/logs/leaf_wrapper.stdout.log')} "
                    f"2>{shlex.quote(REMOTE_BASE + '/logs/leaf_wrapper.stderr.log')} "
                    f"</dev/null & echo $!",
                ).strip()
                launch_pid = int(output.splitlines()[-1])
                launch_action = "new_waiting_wrapper_started"
                time.sleep(3)

        status = remote_json(client, REMOTE_STATUS)
        gpu_csv = exec_checked(
            client,
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,"
            "utilization.gpu,temperature.gpu,pstate --format=csv,noheader,nounits",
            timeout=60,
        ).strip()
        deployment = {
            "schema": "evomind.hpc88240_leaf_continuation_deployment.v1",
            "created_at": now_iso(),
            "status": "deployed_and_waiting" if start else "deployed_not_started",
            "cluster": "AI-X86_NVIDIA",
            "job_id": 88240,
            "remote_base": REMOTE_BASE,
            "remote_plan": {
                "path": REMOTE_PLAN,
                "bytes": LOCAL_PLAN.stat().st_size,
                "sha256": plan_sha,
            },
            "remote_wrapper": {
                "path": REMOTE_WRAPPER,
                "bytes": len(wrapper_bytes),
                "sha256": sha256_bytes(wrapper_bytes),
            },
            "source_bundle": [
                {
                    "path": item["remote_path"],
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                }
                for item in sources
            ],
            "public_data": public,
            "import_smoke": smoke.strip().splitlines(),
            "gpu_snapshot_csv": gpu_csv,
            "launch_action": launch_action,
            "wrapper_pid": launch_pid or (status or {}).get("wrapper_pid"),
            "wrapper_status": status,
            "jigsaw_dependency": remote_json(client, REMOTE_JIGSAW_STATUS),
            "remote_workspace_confined": True,
            "local_gpu_new_training_allowed": False,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "human_gate_preserved": True,
        }
        upload_bytes(
            client,
            (json.dumps(deployment, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            REMOTE_DEPLOYMENT,
        )
        write_json_atomic(LOCAL_EVIDENCE, deployment)
        return deployment
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-start", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = deploy(start=not args.no_start)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
