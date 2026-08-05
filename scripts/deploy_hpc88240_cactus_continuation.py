#!/usr/bin/env python3
"""Deploy the RANZCR -> multi-seed high-resolution Cactus continuation on job 88240.

The deployment is deliberately candidate-only.  It stages a frozen public-data
plan and a non-preemptive wrapper, then (optionally) starts that wrapper.  The
wrapper waits for the RANZCR terminal verifier and three observed idle GPU
checks before it uses the GPU.  It never invokes an official grader or a
Kaggle submission command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (PROJECT_ROOT, SRC_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts import deploy_hpc88240_leaf_continuation as common  # noqa: E402
from scripts.deploy_hpc88240_siim_continuation import connect_with_retry  # noqa: E402

ALLOWED_ROOT = common.ALLOWED_GPU_REMOTE_ROOT
REMOTE_BASE = posixpath.join(
    ALLOWED_ROOT,
    "evomind_mle22",
    "job88240_cactus_after_ranzcr_s404142_20260727",
)
REMOTE_RANZCR_STATUS = posixpath.join(
    ALLOWED_ROOT,
    "evomind_mle22",
    "job88240_ranzcr_highres_s42_20260727",
    "full_ranzcr_status.json",
)
REMOTE_SITE_PACKAGES = posixpath.join(
    ALLOWED_ROOT,
    "evomind_mle22",
    "job88240_jigsaw_s41_20260727",
    "env",
    "site-packages",
)
REMOTE_DATA_ROOT = posixpath.join(ALLOWED_ROOT, "mlebench_official_data")
REMOTE_PUBLIC_ROOT = posixpath.join(
    REMOTE_DATA_ROOT,
    "aerial-cactus-identification",
    "prepared",
    "public",
)
REMOTE_TORCH_HOME = posixpath.join(ALLOWED_ROOT, "mlebench_model_cache", "torch")
REMOTE_OFFICIAL_SOURCE = posixpath.join(ALLOWED_ROOT, "mle-bench")
RUN_PREFIX = "hpc88240_cactus_convnext384"
SEEDS = (40, 41, 42)
REMOTE_OUTPUT_ROOT = posixpath.join(REMOTE_BASE, "mlebench_lite_runs")
REMOTE_CANDIDATE = posixpath.join(REMOTE_BASE, "multiseed_candidate")
REMOTE_V1_STATUS = posixpath.join(REMOTE_BASE, "full_cactus_status.json")
REMOTE_STATUS = posixpath.join(REMOTE_BASE, "full_cactus_status_v2.json")
REMOTE_WRAPPER = posixpath.join(REMOTE_BASE, "run_cactus_after_ranzcr_v2.sh")
REMOTE_V1_WRAPPER = posixpath.join(REMOTE_BASE, "run_cactus_after_ranzcr.sh")
REMOTE_PLAN = posixpath.join(
    REMOTE_BASE, "plans", "cactus_multiseed_highres_hpc88240_frozen_plan_v2_20260727.json"
)
REMOTE_OPTIMIZATION_PLAN = posixpath.join(
    REMOTE_BASE, "plans", "medal_recovery_gpt56_current.json"
)
REMOTE_DEPLOYMENT = posixpath.join(REMOTE_BASE, "deployment_manifest_v2.json")
LOCAL_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "cactus_multiseed_highres_hpc88240_frozen_plan_v2_20260727.json"
)
LOCAL_OPTIMIZATION_PLAN = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_recovery_gpt56_current.json"
)
LOCAL_EVIDENCE = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job88240_cactus_continuation_deployment_current.json"
)

# ConvNeXt-Small ImageNet-1K weight as used by the current vision adapter.
CONVNEXT_SMALL_WEIGHT = posixpath.join(
    REMOTE_TORCH_HOME, "hub", "checkpoints", "convnext_small-0c510722.pth"
)
CONVNEXT_SMALL_WEIGHT_SHA256 = (
    "0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a"
)

RANZCR_READY_TERMINAL_STATUSES = frozenset(
    {"verification_passed", "verification_complete_gate_failed"}
)
RANZCR_FAILURE_TERMINAL_STATUSES = frozenset(
    {
        "frozen_artifact_missing",
        "frozen_artifact_drift",
        "target_run_exists",
        "blocked_by_siim_integrity_failure",
        "highres_training_failed",
        "highres_bundle_missing",
        "aggregation_failed",
        "verifier_failed",
        "verification_contract_failed",
    }
)
SEED_CANDIDATE_STATUSES = frozenset(
    {
        "promotion_gate_passed_confirmation_pending",
        "candidate_ready_confirmation_pending",
    }
)
PRESERVE_EXISTING_V2_STATUSES = frozenset(
    {
        "waiting_for_v1_retirement",
        "waiting_for_ranzcr",
        "waiting_for_gpu_idle",
        "training_seed_40",
        "training_seed_41",
        "training_seed_42",
        "aggregating",
        "verifying",
        "verification_passed",
        "verification_complete_gate_failed",
        "v1_candidate_preserved",
        "target_candidate_exists",
    }
)

SOURCE_BINDINGS: tuple[tuple[Path, str], ...] = (
    (PROJECT_ROOT / "scripts" / "mlebench_compute_policy.py", "scripts/mlebench_compute_policy.py"),
    (PROJECT_ROOT / "scripts" / "run_mlebench_lite_full.py", "scripts/run_mlebench_lite_full.py"),
    (PROJECT_ROOT / "scripts" / "run_mlebench_lite_wave0.py", "scripts/run_mlebench_lite_wave0.py"),
    (PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py", "scripts/mlebench_wave2_adapters.py"),
    (PROJECT_ROOT / "scripts" / "mlebench_medal_recovery_adapters.py", "scripts/mlebench_medal_recovery_adapters.py"),
    (PROJECT_ROOT / "scripts" / "russian_transliteration.py", "scripts/russian_transliteration.py"),
    (PROJECT_ROOT / "src" / "research_os" / "mlebench_phase_a.py", "src/research_os/mlebench_phase_a.py"),
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def remote_path(relative: str) -> str:
    return common.ensure_remote_path(posixpath.join(REMOTE_BASE, relative))


def source_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for local, relative in SOURCE_BINDINGS:
        resolved = local.resolve()
        records.append(
            {
                "local_path": str(resolved),
                "remote_path": remote_path(relative),
                "bytes": resolved.stat().st_size,
                "sha256": common.sha256_file(resolved),
            }
        )
    return records


def artifact_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("remote_path", "bytes", "sha256")}


def source_map(sources: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {Path(str(record["remote_path"])).name: record for record in sources}


def remote_file_record(path: Path, remote_path_value: str) -> dict[str, Any]:
    """Build an immutable upload record for a plan/configuration artifact."""

    resolved = path.resolve()
    return {
        "local_path": str(resolved),
        "remote_path": common.ensure_remote_path(remote_path_value),
        "bytes": resolved.stat().st_size,
        "sha256": common.sha256_file(resolved),
    }


def deployment_evidence_status(
    *, start: bool, action: str, wrapper_status: Mapping[str, Any] | None
) -> str:
    """Derive evidence state from observed launch action and remote state."""

    if not start:
        return "deployed_not_started"
    observed = str((wrapper_status or {}).get("status") or "").strip()
    if action == "new_waiting_wrapper_started":
        return f"started_{observed}" if observed else "started_status_pending"
    if action == "existing_wrapper_reused":
        return f"reused_{observed}" if observed else "reused_status_pending"
    if action == "terminal_or_active_status_preserved":
        return f"preserved_{observed}" if observed else "preserved_status_pending"
    return action


def remote_data_snapshot(client: Any) -> dict[str, Any]:
    """Freeze the public Cactus inputs without touching extracted/private data."""

    command = f"""PUBLIC={shlex.quote(REMOTE_PUBLIC_ROOT)} python3 - <<'PY'
import hashlib,json,os,pathlib
p=pathlib.Path(os.environ['PUBLIC'])
required=('train.csv','sample_submission.csv','train.zip','test.zip')
missing=[name for name in required if not (p/name).is_file()]
if missing:
 raise SystemExit('missing public Cactus inputs: '+','.join(missing))
files=sorted(x for x in p.rglob('*') if x.is_file())
def sha(x):
 h=hashlib.sha256()
 with x.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
core={{name:{{'bytes':(p/name).stat().st_size,'sha256':sha(p/name)}} for name in required}}
print(json.dumps({{'file_count':len(files),'total_bytes':sum(x.stat().st_size for x in files),'core':core}}))
PY"""
    snapshot = json.loads(common.exec_checked(client, command, timeout=180))
    if snapshot["file_count"] < 4 or snapshot["total_bytes"] <= 0:
        raise RuntimeError("Remote Cactus public dataset inventory is empty")
    return snapshot


def build_plan(
    sources: Sequence[Mapping[str, Any]], data: Mapping[str, Any]
) -> dict[str, Any]:
    records = source_map(sources)
    optimization_plan = remote_file_record(
        LOCAL_OPTIMIZATION_PLAN, REMOTE_OPTIMIZATION_PLAN
    )
    return {
        "schema": "evomind.cactus.multiseed_highres_frozen_plan.v1",
        "created_at": now_iso(),
        "status": "frozen_waiting_for_ranzcr",
        "competition_id": "aerial-cactus-identification",
        "objective": {
            "metric": "roc_auc",
            "direction": "maximize",
            "historical_official_score": 0.99201,
            "bronze_threshold": 1.0,
            "single_seed_stop_loss_auc": 0.9995,
            "multiseed_promotion_auc": 0.9997,
            "official_score_claimed": False,
        },
        "planner": {
            "requested_model": "gpt-5.6-sol",
            "served_model": "gpt-5.6-sol",
            "execution_plan": artifact_binding(optimization_plan),
        },
        "execution_target": {
            "cluster": "AI-X86_NVIDIA",
            "job_id": 88240,
            "gpu": "NVIDIA A40",
            "remote_root": ALLOWED_ROOT,
            "process_signals_allowed": False,
        },
        "serial_dependency": {
            "status_path": REMOTE_RANZCR_STATUS,
            "accepted_verified_terminal_statuses": [
                *sorted(RANZCR_READY_TERMINAL_STATUSES),
            ],
            "blocking_terminal_statuses": sorted(RANZCR_FAILURE_TERMINAL_STATUSES),
        },
        "training": {
            "run_prefix": RUN_PREFIX,
            "seeds": list(SEEDS),
            "python": "python3",
            "script": records["run_mlebench_lite_full.py"]["remote_path"],
            "data_root": REMOTE_DATA_ROOT,
            "output_root": REMOTE_OUTPUT_ROOT,
            "allowed_root": ALLOWED_ROOT,
            "official_source_root": REMOTE_OFFICIAL_SOURCE,
            "epochs": 12,
            "folds": 5,
            "backbone": "convnext_small",
            "image_size": 384,
            "batch_size": 48,
            "workers": 24,
            "tta_flips": True,
            "candidate_only": True,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
        "stopping_policy": {
            "kind": "serial_multiseed_with_public_oof_stop_loss",
            "per_seed_minimum_auc": 0.9995,
            "all_seeds_required_before_aggregate": True,
            "abort_on_missing_bundle": True,
            "abort_on_non_candidate_result": True,
        },
        "aggregation": {
            "method": "equal_weight_probability_mean_across_seeds",
            "candidate_dir": REMOTE_CANDIDATE,
            "promotion_auc": 0.9997,
            "private_labels_used": False,
        },
        "data_contract": {
            **dict(data),
            "public_root": REMOTE_PUBLIC_ROOT,
            "visibility_mode": "PUBLIC_ONLY",
            "public_only_environment": "EVOMIND_MLEBENCH_PUBLIC_ONLY=1",
            "private_paths_requested": False,
        },
        "implementation": {
            name: artifact_binding(record) for name, record in records.items()
        },
        "launch_contract": {
            "gpu_idle_consecutive_checks": 3,
            "minimum_check_interval_seconds": 10,
            "process_signals_allowed": False,
            "automatic_official_grader": False,
            "automatic_kaggle_submission": False,
            "human_confirmation_required": True,
            "v1_supersession": {
                "mode": "observe_only_natural_retirement",
                "v1_status_path": REMOTE_V1_STATUS,
                "v1_wrapper_path": REMOTE_V1_WRAPPER,
                "process_signals_allowed": False,
            },
        },
        "claim_boundary": "Frozen public-OOF Cactus candidate; not an official medal.",
    }


def render_wrapper(
    plan_sha256: str,
    sources: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any],
) -> str:
    records = source_map(sources)
    optimization_plan = remote_file_record(
        LOCAL_OPTIMIZATION_PLAN, REMOTE_OPTIMIZATION_PLAN
    )
    frozen = [
        (plan_sha256, REMOTE_PLAN),
        (CONVNEXT_SMALL_WEIGHT_SHA256, CONVNEXT_SMALL_WEIGHT),
        (str(optimization_plan["sha256"]), str(optimization_plan["remote_path"])),
    ]
    frozen.extend((str(record["sha256"]), str(record["remote_path"])) for record in sources)
    for name, record in data["core"].items():
        frozen.append((str(record["sha256"]), posixpath.join(REMOTE_PUBLIC_ROOT, name)))
    frozen_lines = "\n".join(f"{digest}|{path}" for digest, path in frozen)
    runner = records["run_mlebench_lite_full.py"]["remote_path"]
    seed_lines = " ".join(str(seed) for seed in SEEDS)
    ranzcr_ready_cases = "|".join(sorted(RANZCR_READY_TERMINAL_STATUSES))
    ranzcr_failure_cases = "|".join(sorted(RANZCR_FAILURE_TERMINAL_STATUSES))
    candidate_statuses_literal = repr(sorted(SEED_CANDIDATE_STATUSES))
    return f'''#!/usr/bin/env bash
set -uo pipefail
BASE={shlex.quote(REMOTE_BASE)}
RANZCR_STATUS={shlex.quote(REMOTE_RANZCR_STATUS)}
V1_STATUS={shlex.quote(REMOTE_V1_STATUS)}
V1_WRAPPER={shlex.quote(REMOTE_V1_WRAPPER)}
STATUS={shlex.quote(REMOTE_STATUS)}
PLAN={shlex.quote(REMOTE_PLAN)}
OPTIMIZATION_PLAN={shlex.quote(REMOTE_OPTIMIZATION_PLAN)}
CANDIDATE={shlex.quote(REMOTE_CANDIDATE)}
OUTPUT={shlex.quote(REMOTE_OUTPUT_ROOT)}
PUBLIC={shlex.quote(REMOTE_PUBLIC_ROOT)}
DATA_ROOT={shlex.quote(REMOTE_DATA_ROOT)}
TORCH_HOME_DIR={shlex.quote(REMOTE_TORCH_HOME)}
SITE={shlex.quote(REMOTE_SITE_PACKAGES)}
RUNNER={shlex.quote(str(runner))}
PYTHONPATH_VALUE="$SITE:$BASE/src:$BASE"
export EVOMIND_MLEBENCH_PUBLIC_ONLY=1

write_status() {{
 state="$1"; code="${{2:-}}"; stable="${{3:-0}}"; completed="${{4:-0}}"
 EVOMIND_STATE="$state" EVOMIND_EXIT_CODE="$code" EVOMIND_STABLE="$stable" EVOMIND_COMPLETED="$completed" \\
 EVOMIND_PID="$$" EVOMIND_STATUS="$STATUS" EVOMIND_CANDIDATE="$CANDIDATE" \\
 python3 - <<'PY'
import datetime,json,os,pathlib
p=pathlib.Path(os.environ['EVOMIND_STATUS']); t=p.with_suffix(p.suffix+'.tmp'); raw=os.environ.get('EVOMIND_EXIT_CODE','')
d={{'schema':'evomind.hpc_cactus_persistent_run.v2','created_at':datetime.datetime.now().astimezone().isoformat(),'status':os.environ['EVOMIND_STATE'],'wrapper_pid':int(os.environ['EVOMIND_PID']),'candidate_dir':os.environ['EVOMIND_CANDIDATE'],'stable_idle_observations':int(os.environ.get('EVOMIND_STABLE','0')),'completed_seeds':int(os.environ.get('EVOMIND_COMPLETED','0')),'exit_code':int(raw) if raw else None,'visibility_mode':'PUBLIC_ONLY','private_labels_used':False,'official_grader_executed':False,'kaggle_submission_executed':False,'process_signals_sent':0,'other_processes_modified':False,'v1_supersession':'observe_only_natural_retirement'}}
t.write_text(json.dumps(d,indent=2)+'\\n'); t.replace(p)
PY
}}

verify_frozen() {{
 while IFS='|' read -r expected path; do
  [ -n "$expected" ] || continue
  [ -f "$path" ] || {{ write_status frozen_artifact_missing 22; return 22; }}
  actual=$(sha256sum -- "$path" | awk '{{print $1}}')
  [ "$actual" = "$expected" ] || {{ write_status frozen_artifact_drift 23; return 23; }}
 done <<'EOF'
{frozen_lines}
EOF
}}

read_dependency() {{
 RANZCR_STATUS="$RANZCR_STATUS" python3 - <<'PY'
import json,os
print(json.load(open(os.environ['RANZCR_STATUS'],encoding='utf-8')).get('status',''))
PY
}}

v1_wrapper_is_active() {{
 [ -f "$V1_STATUS" ] || return 1
 pid=$(V1_STATUS="$V1_STATUS" python3 - <<'PY'
import json,os
try:
 print(int(json.load(open(os.environ['V1_STATUS'],encoding='utf-8')).get('wrapper_pid') or 0))
except Exception:
 print(0)
PY
)
 [ "$pid" -gt 0 ] || return 1
 ps -p "$pid" -o args= 2>/dev/null | grep -F -- "$V1_WRAPPER" >/dev/null
}}

verify_frozen || exit $?
[ ! -e "$CANDIDATE" ] || {{ write_status target_candidate_exists 21; exit 21; }}
while v1_wrapper_is_active; do
 write_status waiting_for_v1_retirement
 sleep 30
done
[ ! -e "$CANDIDATE" ] || {{ write_status v1_candidate_preserved 25; exit 25; }}
write_status waiting_for_ranzcr
while true; do
 [ -f "$RANZCR_STATUS" ] || {{ sleep 30; continue; }}
 state=$(read_dependency)
 case "$state" in
  {ranzcr_ready_cases}) break ;;
  {ranzcr_failure_cases})
   write_status blocked_by_ranzcr_integrity_failure 31; exit 31 ;;
  *) write_status waiting_for_ranzcr; sleep 30 ;;
 esac
done
verify_frozen || exit $?
wait_for_gpu_idle() {{
 completed_seed_count="$1"; stable=0
 while [ "$stable" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' | wc -l)
  used=$(echo "$line" | awk -F',' '{{gsub(/ /,"",$1);print $1}}'); util=$(echo "$line" | awk -F',' '{{gsub(/ /,"",$2);print $2}}')
  if [ "$apps" -eq 0 ] && [ "$used" -le 512 ] && [ "$util" -le 10 ]; then stable=$((stable+1)); else stable=0; fi
  write_status waiting_for_gpu_idle "" "$stable" "$completed_seed_count"; [ "$stable" -ge 3 ] || sleep 10
 done
}}
mkdir -p -- "$OUTPUT" "$BASE/logs"
completed=0
for seed in {seed_lines}; do
 wait_for_gpu_idle "$completed"
 run_id={shlex.quote(RUN_PREFIX)}"_s${{seed}}_20260727"
 run_dir="$OUTPUT/$run_id"
 task="$run_dir/aerial-cactus-identification/attempts/attempt_001"
 [ ! -e "$run_dir" ] || {{ write_status target_run_exists 24 3 "$completed"; exit 24; }}
 write_status "training_seed_${{seed}}" "" 3 "$completed"
 PYTHONPATH="$PYTHONPATH_VALUE" TORCH_HOME="$TORCH_HOME_DIR" CUDA_VISIBLE_DEVICES=0 EVOMIND_MLEBENCH_PUBLIC_ONLY=1 \\
 python3 "$RUNNER" --data-root "$DATA_ROOT" --output-root "$OUTPUT" \\
  --allowed-root {shlex.quote(ALLOWED_ROOT)} --official-source-root {shlex.quote(REMOTE_OFFICIAL_SOURCE)} \\
  --waves Wave0 --competitions aerial-cactus-identification --run-id "$run_id" \\
  --seed "$seed" --phase-a-scope requested --optimization-plan "$OPTIMIZATION_PLAN" --candidate-only --hold-cuda-lease --wave2-fast-kernels \\
  --wave2-workers 24 --aerial-epochs 12 --aerial-backbone convnext_small --aerial-image-size 384 \\
  --aerial-batch-size 48 --aerial-folds 5 \\
  >"$BASE/logs/seed_${{seed}}.stdout.log" 2>"$BASE/logs/seed_${{seed}}.stderr.log"
 rc=$?
 if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then write_status "seed_${{seed}}_training_failed" "$rc" 3 "$completed"; exit "$rc"; fi
 [ -f "$task/vision_oof_and_test.npz" ] && [ -f "$run_dir/summary.json" ] && [ -f "$task/result.json" ] || {{ write_status "seed_${{seed}}_bundle_missing" 32 3 "$completed"; exit 32; }}
 SEED_RESULT="$task/result.json" SEED_SUMMARY="$run_dir/summary.json" SEED_BUNDLE="$task/vision_oof_and_test.npz" python3 - <<'PY'
import json,os
import numpy as np
from sklearn.metrics import roc_auc_score
result=json.load(open(os.environ['SEED_RESULT'],encoding='utf-8'))
summary=json.load(open(os.environ['SEED_SUMMARY'],encoding='utf-8'))
allowed_statuses=set({candidate_statuses_literal})
if result.get('status') not in allowed_statuses:
 raise SystemExit('candidate result status contract failed: '+str(result.get('status')))
if result.get('candidate_only') is not True:
 raise SystemExit('candidate_only was not preserved')
if result.get('valid_submission') is not True:
 raise SystemExit('candidate submission validation failed')
if result.get('official_grader_executed') is not False:
 raise SystemExit('official grader execution detected')
if result.get('official_grader_withheld') is not True:
 raise SystemExit('official grader withholding contract failed')
if result.get('official_grader_confirmation_pending') is not True:
 raise SystemExit('human confirmation contract failed')
if summary.get('status') != 'candidate_complete' or summary.get('competition_count') != 1:
 raise SystemExit('candidate summary terminal contract failed')
if summary.get('candidate_confirmation_pending') != 1 or summary.get('failed') != 0:
 raise SystemExit('candidate summary completion counts failed')
if summary.get('official_private_grader_rate') != 0.0:
 raise SystemExit('candidate summary reports official grading')
with np.load(os.environ['SEED_BUNDLE'],allow_pickle=False) as z:
 required={{'truth','selected_oof_probability','selected_test_probability','fold'}}
 if not required <= set(z.files): raise SystemExit('candidate bundle missing required arrays')
 truth=np.asarray(z['truth']).reshape(-1); prob=np.asarray(z['selected_oof_probability'],dtype=np.float64).reshape(-1)
 if truth.shape != prob.shape or not np.isfinite(prob).all(): raise SystemExit('invalid OOF arrays')
 auc=float(roc_auc_score(truth,prob))
 if auc < 0.9995: raise SystemExit(f'single-seed public OOF stop loss: {{auc:.8f}} < 0.99950000')
PY
 rc=$?
 if [ "$rc" -ne 0 ]; then write_status "seed_${{seed}}_stop_loss" "$rc" 3 "$completed"; exit "$rc"; fi
 completed=$((completed+1))
done
write_status aggregating "" 3 "$completed"
CANDIDATE="$CANDIDATE" OUTPUT="$OUTPUT" python3 - <<'PY'
import datetime,hashlib,json,os,pathlib
import numpy as np
from sklearn.metrics import roc_auc_score
candidate=pathlib.Path(os.environ['CANDIDATE']); output=pathlib.Path(os.environ['OUTPUT'])
candidate.mkdir(parents=True,exist_ok=False)
seeds=(40,41,42); bundles=[]
for seed in seeds:
 path=output/f'{RUN_PREFIX}_s{{seed}}_20260727'/'aerial-cactus-identification'/'attempts'/'attempt_001'/'vision_oof_and_test.npz'
 bundles.append(path)
arrays=[]
for path in bundles:
 with np.load(path,allow_pickle=False) as z:
  arrays.append({{'truth':np.asarray(z['truth']).reshape(-1),'selected_oof_probability':np.asarray(z['selected_oof_probability'],dtype=np.float64).reshape(-1),'selected_test_probability':np.asarray(z['selected_test_probability'],dtype=np.float64).reshape(-1),'fold':np.asarray(z['fold']).reshape(-1)}})
truth=np.asarray(arrays[0]['truth']).reshape(-1)
if any(not np.array_equal(truth,np.asarray(a['truth']).reshape(-1)) for a in arrays[1:]): raise RuntimeError('seed truth ordering mismatch')
if any(len(a['selected_oof_probability'])!=len(truth) or len(a['selected_test_probability'])==0 for a in arrays): raise RuntimeError('seed probability cardinality mismatch')
oof=np.mean(np.stack([a['selected_oof_probability'] for a in arrays],axis=0),axis=0,dtype=np.float64)
test=np.mean(np.stack([a['selected_test_probability'] for a in arrays],axis=0),axis=0,dtype=np.float64)
if not np.isfinite(oof).all() or not np.isfinite(test).all() or np.any((oof<0.0)|(oof>1.0)) or np.any((test<0.0)|(test>1.0)): raise RuntimeError('invalid multiseed probability')
auc=float(roc_auc_score(truth,oof))
bundle=candidate/'cactus_multiseed_oof_and_test.npz'
np.savez_compressed(bundle,truth=truth,selected_oof_probability=oof,selected_test_probability=test,seeds=np.asarray(seeds,dtype=np.int64))
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
report={{'schema':'evomind.cactus.multiseed_candidate.v1','created_at':datetime.datetime.now().astimezone().isoformat(),'seeds':list(seeds),'public_oof_auc':auc,'promotion_auc':0.9997,'candidate_ready':bool(auc>=0.9997),'candidate_only':True,'valid_submission':True,'official_grader_executed':False,'official_grader_withheld':True,'official_grader_confirmation_pending':True,'private_labels_used':False,'kaggle_submission_executed':False,'probability_dtype':'float64','prediction_bundle':{{'path':str(bundle),'bytes':bundle.stat().st_size,'sha256':digest}},'source_bundles':[str(x) for x in bundles]}}
(candidate/'candidate_report.json').write_text(json.dumps(report,indent=2)+'\\n',encoding='utf-8')
PY
rc=$?
if [ "$rc" -ne 0 ]; then write_status aggregation_failed "$rc" 3 "$completed"; exit "$rc"; fi
write_status verifying "" 3 "$completed"
CANDIDATE="$CANDIDATE" python3 - <<'PY'
import hashlib,json,os,pathlib
import numpy as np
from sklearn.metrics import roc_auc_score
candidate=pathlib.Path(os.environ['CANDIDATE']); report=json.loads((candidate/'candidate_report.json').read_text(encoding='utf-8')); bundle=pathlib.Path(report['prediction_bundle']['path'])
if hashlib.sha256(bundle.read_bytes()).hexdigest()!=report['prediction_bundle']['sha256']: raise SystemExit('candidate bundle hash drift')
with np.load(bundle,allow_pickle=False) as z:
 required={{'truth','selected_oof_probability','selected_test_probability','seeds'}}
 if not required <= set(z.files): raise SystemExit('candidate bundle incomplete')
 truth=np.asarray(z['truth']).reshape(-1); oof=np.asarray(z['selected_oof_probability'],dtype=np.float64).reshape(-1); test=np.asarray(z['selected_test_probability'],dtype=np.float64).reshape(-1)
 if z['selected_oof_probability'].dtype != np.float64 or z['selected_test_probability'].dtype != np.float64: raise SystemExit('candidate probabilities are not float64')
 if len(truth)!=len(oof) or len(test)==0 or not np.isfinite(oof).all() or not np.isfinite(test).all() or np.any((oof<0.0)|(oof>1.0)) or np.any((test<0.0)|(test>1.0)): raise SystemExit('candidate bundle invalid')
 auc=float(roc_auc_score(truth,oof))
 if abs(auc-float(report['public_oof_auc']))>1e-12: raise SystemExit('candidate report AUC mismatch')
if report.get('candidate_only') is not True or report.get('valid_submission') is not True or report.get('official_grader_executed') is not False or report.get('official_grader_withheld') is not True or report.get('official_grader_confirmation_pending') is not True: raise SystemExit('candidate report contract invalid')
verification={{'schema':'evomind.cactus.multiseed_candidate_verification.v1','passed':True,'candidate_ready':bool(report['candidate_ready']),'public_oof_auc':auc,'prediction_bundle_sha256':report['prediction_bundle']['sha256'],'probability_dtype':'float64','candidate_only':True,'valid_submission':True,'official_grader_executed':False,'official_grader_withheld':True,'official_grader_confirmation_pending':True,'private_labels_used':False,'kaggle_submission_executed':False}}
(candidate/'independent_verification.json').write_text(json.dumps(verification,indent=2)+'\\n',encoding='utf-8')
PY
rc=$?
if [ "$rc" -ne 0 ]; then write_status verifier_failed "$rc" 3 "$completed"; exit "$rc"; fi
final=$(REPORT="$CANDIDATE/independent_verification.json" python3 - <<'PY'
import json,os
r=json.load(open(os.environ['REPORT'],encoding='utf-8'))
print('verification_passed' if r.get('candidate_ready') is True else 'verification_complete_gate_failed')
PY
)
write_status "$final" 0 3 "$completed"
exit 0
'''


def remote_sha256(client: Any, path: str) -> str | None:
    output = common.exec_checked(
        client,
        f"if [ -f {shlex.quote(path)} ]; then sha256sum -- {shlex.quote(path)} | awk '{{print $1}}'; fi",
    ).strip()
    return output or None


def v2_wrapper_is_active(client: Any, status: Mapping[str, Any] | None) -> bool:
    """Check the existing v2 wrapper without mutating it."""

    pid = int((status or {}).get("wrapper_pid") or 0)
    if not pid:
        return False
    return (
        common.exec_checked(
            client,
            f"if ps -p {pid} -o args= | grep -F -- {shlex.quote(REMOTE_WRAPPER)} >/dev/null; then echo yes; else echo no; fi",
        ).strip()
        == "yes"
    )


def preserve_existing_v2_action(
    status: Mapping[str, Any] | None, *, active: bool
) -> str | None:
    """Return the no-upload action when an existing v2 run must be left untouched."""

    observed = str((status or {}).get("status") or "").strip()
    if active:
        return "existing_wrapper_reused"
    if observed in PRESERVE_EXISTING_V2_STATUSES:
        return "terminal_or_active_status_preserved"
    return None


def existing_v2_evidence(
    client: Any,
    *,
    action: str,
    status: Mapping[str, Any] | None,
    start: bool,
) -> dict[str, Any]:
    """Return remote deployment evidence without uploading new frozen artifacts."""

    deployed = common.remote_json(client, REMOTE_DEPLOYMENT)
    if deployed:
        return deployed
    return {
        "schema": "evomind.hpc88240_cactus_continuation_deployment.v2",
        "created_at": now_iso(),
        "status": deployment_evidence_status(
            start=start, action=action, wrapper_status=status
        ),
        "cluster": "AI-X86_NVIDIA",
        "job_id": 88240,
        "gpu": "NVIDIA A40",
        "remote_base": REMOTE_BASE,
        "launch_action": action,
        "wrapper_pid": (status or {}).get("wrapper_pid"),
        "wrapper_status": status,
        "remote_uploads_skipped": True,
        "idempotence_reason": "existing_v2_wrapper_or_terminal_status_preserved",
        "process_signals_sent": 0,
        "other_processes_modified": False,
    }


def upload_file_if_changed(client: Any, local_path: Path, remote_path_value: str) -> str:
    """Upload a file only after a remote content-hash preflight."""

    local = local_path.resolve()
    expected = common.sha256_file(local)
    if remote_sha256(client, remote_path_value) == expected:
        return "reused"
    common.upload_file(client, local, remote_path_value)
    if remote_sha256(client, remote_path_value) != expected:
        raise RuntimeError(f"Remote artifact hash differs after upload: {remote_path_value}")
    return "uploaded"


def upload_bytes_if_changed(client: Any, payload: bytes, remote_path_value: str) -> str:
    """Upload generated content only if its remote digest is not already exact."""

    expected = hashlib.sha256(payload).hexdigest()
    if remote_sha256(client, remote_path_value) == expected:
        return "reused"
    common.upload_bytes(client, payload, remote_path_value)
    if remote_sha256(client, remote_path_value) != expected:
        raise RuntimeError(f"Remote generated artifact hash differs: {remote_path_value}")
    return "uploaded"


def deploy(*, start: bool = True) -> dict[str, Any]:
    """Stage the immutable deployment, starting only when explicitly requested."""

    sources = source_records()
    client = connect_with_retry()
    try:
        prior = common.remote_json(client, REMOTE_STATUS)
        preserve_action = preserve_existing_v2_action(
            prior, active=v2_wrapper_is_active(client, prior)
        )
        if preserve_action:
            evidence = existing_v2_evidence(
                client, action=preserve_action, status=prior, start=start
            )
            common.write_json_atomic(LOCAL_EVIDENCE, evidence)
            return evidence

        data = remote_data_snapshot(client)
        plan = build_plan(sources, data)
        common.write_json_atomic(LOCAL_PLAN, plan)
        plan_sha = common.sha256_file(LOCAL_PLAN)
        wrapper = render_wrapper(plan_sha, sources, data)
        wrapper_bytes = wrapper.encode("utf-8")
        common.exec_checked(
            client,
            "mkdir -p -- "
            + " ".join(
                shlex.quote(path)
                for path in (
                    REMOTE_BASE,
                    posixpath.join(REMOTE_BASE, "scripts"),
                    posixpath.join(REMOTE_BASE, "src", "research_os"),
                    posixpath.join(REMOTE_BASE, "plans"),
                    posixpath.join(REMOTE_BASE, "logs"),
                )
            ),
        )
        uploads: dict[str, str] = {}
        for record in sources:
            remote = str(record["remote_path"])
            uploads[remote] = upload_file_if_changed(
                client, Path(str(record["local_path"])), remote
            )
        package_marker = (
            client,
            b'"""Minimal package marker for the frozen Cactus HPC bundle."""\n',
            remote_path("src/research_os/__init__.py"),
        )
        uploads[package_marker[2]] = upload_bytes_if_changed(*package_marker)
        uploads[REMOTE_OPTIMIZATION_PLAN] = upload_file_if_changed(
            client, LOCAL_OPTIMIZATION_PLAN, REMOTE_OPTIMIZATION_PLAN
        )
        uploads[REMOTE_PLAN] = upload_file_if_changed(client, LOCAL_PLAN, REMOTE_PLAN)
        uploads[REMOTE_WRAPPER] = upload_bytes_if_changed(client, wrapper_bytes, REMOTE_WRAPPER)
        common.exec_checked(client, f"chmod 700 -- {shlex.quote(REMOTE_WRAPPER)}")
        smoke = common.exec_checked(
            client,
            f"PYTHONPATH={shlex.quote(REMOTE_SITE_PACKAGES + ':' + REMOTE_BASE + '/src:' + REMOTE_BASE)} "
            "CUDA_VISIBLE_DEVICES='' python3 - <<'PY'\n"
            "import hashlib\n"
            "from pathlib import Path\n"
            "import scripts.run_mlebench_lite_full as full\n"
            "import scripts.run_mlebench_lite_wave0 as wave0\n"
            "import scripts.mlebench_wave2_adapters as vision\n"
            "print('import_smoke=passed')\n"
            "print('full='+hashlib.sha256(Path(full.__file__).read_bytes()).hexdigest())\n"
            "print('wave0='+hashlib.sha256(Path(wave0.__file__).read_bytes()).hexdigest())\n"
            "print('vision='+hashlib.sha256(Path(vision.__file__).read_bytes()).hexdigest())\n"
            "PY",
            timeout=180,
        )
        action = "deployed_without_start"
        pid: int | None = None
        if start:
            prior_pid = int((prior or {}).get("wrapper_pid") or 0)
            active = False
            if prior_pid:
                active = common.exec_checked(
                    client,
                    f"if ps -p {prior_pid} -o args= | grep -F -- {shlex.quote(REMOTE_WRAPPER)} >/dev/null; then echo yes; else echo no; fi",
                ).strip() == "yes"
            if active:
                action = "existing_wrapper_reused"
                pid = prior_pid
            elif prior and prior.get("status") in {
                "training_seed_40",
                "training_seed_41",
                "training_seed_42",
                "aggregating",
                "verifying",
                "verification_passed",
                "verification_complete_gate_failed",
            }:
                action = "terminal_or_active_status_preserved"
            else:
                output = common.exec_checked(
                    client,
                    f"nohup bash {shlex.quote(REMOTE_WRAPPER)} >{shlex.quote(REMOTE_BASE + '/logs/wrapper_v2.stdout.log')} 2>{shlex.quote(REMOTE_BASE + '/logs/wrapper_v2.stderr.log')} </dev/null & echo $!",
                ).strip()
                pid = int(output.splitlines()[-1])
                action = "new_waiting_wrapper_started"
                time.sleep(3)
        status = common.remote_json(client, REMOTE_STATUS)
        v1_status = common.remote_json(client, REMOTE_V1_STATUS)
        evidence = {
            "schema": "evomind.hpc88240_cactus_continuation_deployment.v2",
            "created_at": now_iso(),
            "status": deployment_evidence_status(
                start=start, action=action, wrapper_status=status
            ),
            "cluster": "AI-X86_NVIDIA",
            "job_id": 88240,
            "gpu": "NVIDIA A40",
            "remote_base": REMOTE_BASE,
            "data_contract": data,
            "plan": {"path": REMOTE_PLAN, "bytes": LOCAL_PLAN.stat().st_size, "sha256": plan_sha},
            "wrapper": {"path": REMOTE_WRAPPER, "bytes": len(wrapper_bytes), "sha256": hashlib.sha256(wrapper_bytes).hexdigest()},
            "source_uploads": uploads,
            "import_smoke": smoke.strip().splitlines(),
            "launch_action": action,
            "wrapper_pid": pid or (status or {}).get("wrapper_pid"),
            "wrapper_status": status,
            "v1_supersession": {
                "mode": "observe_only_natural_retirement",
                "v1_status_path": REMOTE_V1_STATUS,
                "v1_wrapper_path": REMOTE_V1_WRAPPER,
                "v1_status": v1_status,
                "process_signals_sent": 0,
            },
            "ranzcr_dependency": common.remote_json(client, REMOTE_RANZCR_STATUS),
            "visibility_mode": "PUBLIC_ONLY",
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "human_gate_preserved": True,
        }
        evidence_bytes = (json.dumps(evidence, ensure_ascii=False, indent=2) + "\n").encode()
        upload_bytes_if_changed(client, evidence_bytes, REMOTE_DEPLOYMENT)
        common.write_json_atomic(LOCAL_EVIDENCE, evidence)
        return evidence
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
