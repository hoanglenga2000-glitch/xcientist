#!/usr/bin/env python3
"""Deploy the Leaf -> SIIM public-data candidate chain on job 88240 A40."""

from __future__ import annotations

import argparse
import copy
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
from scripts import stage_siim_public_from_hpc as siim_stage  # noqa: E402

ALLOWED_ROOT = common.ALLOWED_GPU_REMOTE_ROOT
REMOTE_BASE = posixpath.join(
    ALLOWED_ROOT,
    "evomind_mle22",
    "job88240_siim_s42_20260727",
)
REMOTE_LEAF_BASE = posixpath.join(
    ALLOWED_ROOT,
    "evomind_mle22",
    "job88240_leaf_s404142_20260727",
)
REMOTE_LEAF_STATUS = posixpath.join(REMOTE_LEAF_BASE, "full_leaf_status.json")
REMOTE_JIGSAW_BASE = posixpath.join(
    ALLOWED_ROOT,
    "evomind_mle22",
    "job88240_jigsaw_s41_20260727",
)
REMOTE_SITE_PACKAGES = posixpath.join(REMOTE_JIGSAW_BASE, "env", "site-packages")
REMOTE_DATA_ROOT = posixpath.join(ALLOWED_ROOT, "mlebench_official_data")
REMOTE_PUBLIC_ROOT = posixpath.join(
    REMOTE_DATA_ROOT,
    "siim-isic-melanoma-classification",
    "prepared",
    "public",
)
REMOTE_TORCH_HOME = posixpath.join(ALLOWED_ROOT, "mlebench_model_cache", "torch")
REMOTE_OFFICIAL_SOURCE = posixpath.join(ALLOWED_ROOT, "mle-bench")
ABLATION_RUN_ID = "hpc88240_siim_preproc5_s42_20260727"
FINAL_RUN_ID = "hpc88240_siim_final_s42_20260727"
REMOTE_ABLATION_OUTPUT = posixpath.join(REMOTE_BASE, "siim_preprocessing_ablation")
REMOTE_ABLATION_RUN = posixpath.join(
    REMOTE_ABLATION_OUTPUT,
    "runs",
    ABLATION_RUN_ID,
)
REMOTE_ABLATION_REPORT = posixpath.join(
    REMOTE_ABLATION_RUN,
    "siim_preprocessing_ablation.json",
)
REMOTE_FINAL_OUTPUT = posixpath.join(REMOTE_BASE, "mlebench_lite_runs")
REMOTE_FINAL_RUN = posixpath.join(REMOTE_FINAL_OUTPUT, FINAL_RUN_ID)
REMOTE_STATUS = posixpath.join(REMOTE_BASE, "full_siim_status.json")
REMOTE_WRAPPER = posixpath.join(REMOTE_BASE, "run_siim_after_leaf.sh")
REMOTE_DEPLOYMENT = posixpath.join(REMOTE_BASE, "deployment_manifest.json")
REMOTE_INVENTORY = posixpath.join(REMOTE_BASE, "data_contract", "public_staging_inventory.json")
REMOTE_STAGING_REPORT = posixpath.join(REMOTE_BASE, "data_contract", "public_staging_report.json")
REMOTE_OPTIMIZATION_PLAN = posixpath.join(
    REMOTE_BASE,
    "plans",
    "medal_recovery_gpt56_current.json",
)

LOCAL_ABLATION_BASE = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "siim_preprocessing_ablation_frozen_plan.json"
)
LOCAL_FINAL_BASE = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "siim_final_candidate_frozen_plan.json"
)
LOCAL_ABLATION_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "siim_preprocessing_ablation_hpc88240_frozen_plan_20260727.json"
)
LOCAL_FINAL_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "siim_final_candidate_hpc88240_frozen_plan_20260727.json"
)
LOCAL_OPTIMIZATION_PLAN = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_recovery_gpt56_current.json"
)
LOCAL_INVENTORY = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / "siim-isic-melanoma-classification"
    / "public_staging_inventory.json"
)
LOCAL_COMPACT_INVENTORY = (
    PROJECT_ROOT / "workspace" / "hpc" / "job88240_siim_public_inventory_compact.json"
)
LOCAL_EVIDENCE = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job88240_siim_continuation_deployment_current.json"
)
REMOTE_ABLATION_PLAN = posixpath.join(REMOTE_BASE, "plans", LOCAL_ABLATION_PLAN.name)
REMOTE_FINAL_PLAN = posixpath.join(REMOTE_BASE, "plans", LOCAL_FINAL_PLAN.name)


SOURCE_BINDINGS: tuple[tuple[Path, str], ...] = (
    (PROJECT_ROOT / "scripts" / "run_siim_preprocessing_ablation.py", "scripts/run_siim_preprocessing_ablation.py"),
    (PROJECT_ROOT / "scripts" / "run_mlebench_lite_full.py", "scripts/run_mlebench_lite_full.py"),
    (PROJECT_ROOT / "scripts" / "run_mlebench_lite_wave0.py", "scripts/run_mlebench_lite_wave0.py"),
    (PROJECT_ROOT / "scripts" / "mlebench_medal_recovery_adapters.py", "scripts/mlebench_medal_recovery_adapters.py"),
    (PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py", "scripts/mlebench_wave2_adapters.py"),
    (PROJECT_ROOT / "scripts" / "russian_transliteration.py", "scripts/russian_transliteration.py"),
    (PROJECT_ROOT / "scripts" / "probe_siim_candidate_batch.py", "scripts/probe_siim_candidate_batch.py"),
    (PROJECT_ROOT / "scripts" / "verify_siim_final_candidate.py", "scripts/verify_siim_final_candidate.py"),
    (PROJECT_ROOT / "scripts" / "verify_siim_final_candidate_hpc.py", "scripts/verify_siim_final_candidate_hpc.py"),
    (PROJECT_ROOT / "scripts" / "queue_siim_final_candidate.py", "scripts/queue_siim_final_candidate.py"),
    (PROJECT_ROOT / "scripts" / "local_rtx4060_idle_gate.py", "scripts/local_rtx4060_idle_gate.py"),
    (PROJECT_ROOT / "src" / "research_os" / "mlebench_phase_a.py", "src/research_os/mlebench_phase_a.py"),
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def remote_path(relative: str) -> str:
    return common.ensure_remote_path(posixpath.join(REMOTE_BASE, relative))


def file_record(path: Path, remote: str) -> dict[str, Any]:
    path = path.resolve()
    return {
        "local_path": str(path),
        "remote_path": common.ensure_remote_path(remote),
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def source_records() -> list[dict[str, Any]]:
    return [file_record(local, remote_path(relative)) for local, relative in SOURCE_BINDINGS]


def source_map(sources: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {Path(str(item["remote_path"])).name: item for item in sources}


def artifact_binding(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": item["remote_path"],
        "bytes": item["bytes"],
        "sha256": item["sha256"],
    }


def read_public_inventory() -> dict[str, Any]:
    entries = siim_stage.read_remote_manifest()
    inventory = siim_stage.build_inventory(entries)
    if (
        inventory["file_count"] != 33_129
        or inventory["total_bytes"] != 25_765_345_055
        or inventory["train_jpeg_count"] != 28_984
        or inventory["test_jpeg_count"] != 4_142
        or inventory["manifest_sha256"]
        != "8fef392368fcc433f4532129312f3b2ef7118d76cad04dc70f694246c34eff13"
    ):
        raise RuntimeError("Remote SIIM public inventory differs from the frozen dataset")
    return inventory


def compact_inventory_payload(inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "evomind.siim.remote_public_inventory_compact.v1",
        "created_at": "2026-07-27T00:00:00+08:00",
        "competition_id": "siim-isic-melanoma-classification",
        "remote_public_root": REMOTE_PUBLIC_ROOT,
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "train_jpeg_count": inventory["train_jpeg_count"],
        "test_jpeg_count": inventory["test_jpeg_count"],
        "manifest_sha256": inventory["manifest_sha256"],
        "source_full_inventory": {
            "path": str(LOCAL_INVENTORY.resolve()),
            "bytes": LOCAL_INVENTORY.stat().st_size,
            "sha256": common.sha256_file(LOCAL_INVENTORY),
        },
        "private_paths_requested": False,
        "remote_writes_performed": False,
    }


def compact_inventory_bytes(inventory: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(compact_inventory_payload(inventory), ensure_ascii=False, indent=2)
        + "\n"
    ).encode("utf-8")


def compact_inventory_record(inventory: Mapping[str, Any]) -> dict[str, Any]:
    value = compact_inventory_bytes(inventory)
    return {
        "path": REMOTE_INVENTORY,
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def build_ablation_plan(
    base: Mapping[str, Any],
    inventory: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan = copy.deepcopy(dict(base))
    records = source_map(sources)
    plan["created_at"] = now_iso()
    plan["status"] = "frozen_waiting_prerequisites"
    plan["execution_target"] = {
        "cluster": "AI-X86_NVIDIA",
        "job_id": 88240,
        "gpu": "NVIDIA A40",
        "site_packages": REMOTE_SITE_PACKAGES,
        "remote_root": ALLOWED_ROOT,
        "process_signals_allowed": False,
    }
    plan["training"] = {
        "python": "python3",
        "script": records["run_siim_preprocessing_ablation.py"]["remote_path"],
        "run_id": ABLATION_RUN_ID,
        "data_root": REMOTE_DATA_ROOT,
        "output_root": REMOTE_ABLATION_OUTPUT,
        "torch_home": REMOTE_TORCH_HOME,
        "seed": 42,
        "evaluation_seeds": [40, 41, 42],
        "folds": 3,
        "profiles": list(base["training"]["profiles"]),
        "image_size": 224,
        "batch_size": 128,
        "workers": 16,
        "manifest_workers": 32,
        "full_backbone": "convnext_small",
        "lesion_backbone": "efficientnet_v2_s",
        "linear_alpha": 0.0001,
        "linear_max_iter": 2000,
        "minimum_mean_gain": 0.0005,
        "maximum_worst_fold_regression": 0.002,
        "maximum_seed_mean_regression": 0.001,
        "minimum_seed_pass_fraction": 2.0 / 3.0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    plan["data_contract"] = {
        "inventory": compact_inventory_record(inventory),
        "manifest_sha256": inventory["manifest_sha256"],
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "train_jpeg_count": inventory["train_jpeg_count"],
        "test_jpeg_count": inventory["test_jpeg_count"],
        "remote_public_root": REMOTE_PUBLIC_ROOT,
        "private_paths_requested": False,
    }
    plan["implementation"] = {
        "runner": artifact_binding(records["run_siim_preprocessing_ablation.py"]),
        "adapter": artifact_binding(records["mlebench_medal_recovery_adapters.py"]),
        "wave2": artifact_binding(records["mlebench_wave2_adapters.py"]),
    }
    plan["serial_dependency"] = {
        "leaf_status_path": REMOTE_LEAF_STATUS,
        "accepted_verified_terminal_statuses": [
            "verification_passed",
            "verification_complete_gate_failed",
        ],
    }
    plan["launch_contract"] = {
        "gpu_idle_consecutive_checks": 3,
        "minimum_check_interval_seconds": 10,
        "no_compute_process_required": True,
        "process_signals_allowed": False,
        "automatic_official_grader": False,
        "automatic_kaggle_submission": False,
    }
    plan["operational_revision"] = {
        "created_at": now_iso(),
        "reason": "Use complete remote SIIM public data directly and move the ablation from RTX 4060 to A40.",
        "model_data_folds_seeds_profiles_and_thresholds_changed": False,
        "throughput_only_changes": {"batch_size": 128, "workers": 16, "manifest_workers": 32},
        "process_signals_sent": 0,
    }
    return plan


def build_final_plan(
    base: Mapping[str, Any],
    inventory: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    ablation_plan_sha256: str,
) -> dict[str, Any]:
    plan = copy.deepcopy(dict(base))
    records = source_map(sources)
    optimization = file_record(LOCAL_OPTIMIZATION_PLAN, REMOTE_OPTIMIZATION_PLAN)
    plan["created_at"] = now_iso()
    plan["status"] = "frozen_waiting_prerequisites"
    plan["execution_target"] = {
        "cluster": "AI-X86_NVIDIA",
        "job_id": 88240,
        "gpu": "NVIDIA A40",
        "remote_root": ALLOWED_ROOT,
        "site_packages": REMOTE_SITE_PACKAGES,
        "process_signals_allowed": False,
    }
    plan["planner"] = {
        "provider": "openai_compatible_local_gateway",
        "requested_model": "gpt-5.6-sol",
        "served_model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "evidence": {"execution_plan": artifact_binding(optimization)},
    }
    plan["serial_dependency"] = {
        "plan": {
            "path": REMOTE_ABLATION_PLAN,
            "bytes": LOCAL_ABLATION_PLAN.stat().st_size,
            "sha256": ablation_plan_sha256,
        },
        "report": REMOTE_ABLATION_REPORT,
        "run_id": ABLATION_RUN_ID,
        "evaluation_seeds": [40, 41, 42],
        "folds": 3,
        "profiles": list(base["serial_dependency"]["profiles"]),
        "selected_profile_source": "verified_ablation_report_only",
    }
    plan["resource"] = {
        "expected_gpu_name": "NVIDIA A40",
        "observed_memory_total_mib": 46068,
        "single_gpu_serial": True,
        "no_process_preemption": True,
        "no_process_signals": True,
    }
    plan["training"] = {
        "python": "python3",
        "script": records["run_mlebench_lite_full.py"]["remote_path"],
        "data_root": REMOTE_DATA_ROOT,
        "output_root": REMOTE_FINAL_OUTPUT,
        "allowed_root": ALLOWED_ROOT,
        "official_source_root": REMOTE_OFFICIAL_SOURCE,
        "run_id": FINAL_RUN_ID,
        "seed": 42,
        "epochs": 8,
        "outer_folds": 5,
        "inner_folds": 3,
        "image_size": 384,
        "batch_probe_candidates": [64, 32, 16],
        "probe_metadata_width": 16,
        "probe_output_root": posixpath.join(REMOTE_BASE, "batch_probes"),
        "workers": 16,
        "backbone": "convnext_small",
        "secondary_backbone": "efficientnet_v2_s",
        "learning_rate": 0.0003,
        "metadata_iterations": 700,
        "catboost_task_type": "GPU",
        "torch_home": REMOTE_TORCH_HOME,
        "candidate_only": True,
        "hold_cuda_lease": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    plan["data_contract"] = {
        "inventory": compact_inventory_record(inventory),
        "staging_report": REMOTE_STAGING_REPORT,
        "public_dir": REMOTE_PUBLIC_ROOT,
        "sample_submission": posixpath.join(REMOTE_PUBLIC_ROOT, "sample_submission.csv"),
        "manifest_sha256": inventory["manifest_sha256"],
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "train_rows": inventory["train_jpeg_count"],
        "test_rows": inventory["test_jpeg_count"],
        "train_jpeg_count": inventory["train_jpeg_count"],
        "test_jpeg_count": inventory["test_jpeg_count"],
        "private_paths_requested": False,
    }
    plan["implementation"] = {
        "full_runner": artifact_binding(records["run_mlebench_lite_full.py"]),
        "wave0": artifact_binding(records["run_mlebench_lite_wave0.py"]),
        "adapter": artifact_binding(records["mlebench_medal_recovery_adapters.py"]),
        "wave2": artifact_binding(records["mlebench_wave2_adapters.py"]),
        "ablation_runner": artifact_binding(records["run_siim_preprocessing_ablation.py"]),
        "batch_probe": artifact_binding(records["probe_siim_candidate_batch.py"]),
        "verifier": artifact_binding(records["verify_siim_final_candidate_hpc.py"]),
    }
    plan["launch_contract"] = {
        "queue_status": REMOTE_STATUS,
        "gpu_idle_consecutive_checks": 3,
        "minimum_check_interval_seconds": 10,
        "batch_probe_required": True,
        "batch_selection_rule": "largest_passing_frozen_candidate",
        "process_signals_allowed": False,
        "automatic_official_grader": False,
        "automatic_kaggle_submission": False,
        "human_confirmation_required_for_official_grader": True,
    }
    plan["verification_contract"]["script"] = records[
        "verify_siim_final_candidate_hpc.py"
    ]["remote_path"]
    plan["operational_revision"] = {
        "created_at": now_iso(),
        "reason": "Run the frozen SIIM nested candidate on A40 after the verified Leaf terminal stage.",
        "model_data_folds_seed_backbones_epochs_and_thresholds_changed": False,
        "throughput_only_changes": {"batch_probe_candidates": [64, 32, 16], "workers": 16},
        "process_signals_sent": 0,
    }
    return plan


def render_wrapper(
    *,
    ablation_plan_sha256: str,
    final_plan_sha256: str,
    sources: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> str:
    records = source_map(sources)
    optimization_sha = common.sha256_file(LOCAL_OPTIMIZATION_PLAN)
    inventory_sha = compact_inventory_record(inventory)["sha256"]
    frozen: list[tuple[str, str]] = [
        (ablation_plan_sha256, REMOTE_ABLATION_PLAN),
        (final_plan_sha256, REMOTE_FINAL_PLAN),
        (optimization_sha, REMOTE_OPTIMIZATION_PLAN),
        (inventory_sha, REMOTE_INVENTORY),
    ]
    frozen.extend((str(item["sha256"]), str(item["remote_path"])) for item in sources)
    frozen.extend(
        (
            ("d1827411b90c36fc9d68994360b3aab4a5c123e348e872511f338bef4f3f18ac", posixpath.join(REMOTE_PUBLIC_ROOT, "train.csv")),
            ("4050281af8c6ed04d64b8ac24d1b151b64dbb1d4a4160ee72b0900012e5282ea", posixpath.join(REMOTE_PUBLIC_ROOT, "test.csv")),
            ("0045a0af17b1b68c2f600c3e18dba0b5f5410fd47f47476cee1b6784897f2da9", posixpath.join(REMOTE_PUBLIC_ROOT, "sample_submission.csv")),
            ("0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a", posixpath.join(REMOTE_TORCH_HOME, "hub/checkpoints/convnext_small-0c510722.pth")),
            ("dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74", posixpath.join(REMOTE_TORCH_HOME, "hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth")),
        )
    )
    frozen_lines = "\n".join(f"{digest}|{path}" for digest, path in frozen)
    runner = records["run_mlebench_lite_full.py"]["remote_path"]
    ablation = records["run_siim_preprocessing_ablation.py"]["remote_path"]
    probe = records["probe_siim_candidate_batch.py"]["remote_path"]
    verifier = records["verify_siim_final_candidate_hpc.py"]["remote_path"]
    adapter_sha = records["mlebench_medal_recovery_adapters.py"]["sha256"]
    return f"""#!/usr/bin/env bash
set -uo pipefail
BASE={shlex.quote(REMOTE_BASE)}
LEAF_STATUS={shlex.quote(REMOTE_LEAF_STATUS)}
STATUS={shlex.quote(REMOTE_STATUS)}
ABLATION_PLAN={shlex.quote(REMOTE_ABLATION_PLAN)}
FINAL_PLAN={shlex.quote(REMOTE_FINAL_PLAN)}
ABLATION_REPORT={shlex.quote(REMOTE_ABLATION_REPORT)}
ABLATION_RUN={shlex.quote(REMOTE_ABLATION_RUN)}
FINAL_RUN={shlex.quote(REMOTE_FINAL_RUN)}
DATA_ROOT={shlex.quote(REMOTE_DATA_ROOT)}
PUBLIC_ROOT={shlex.quote(REMOTE_PUBLIC_ROOT)}
TORCH_HOME_DIR={shlex.quote(REMOTE_TORCH_HOME)}
OFFICIAL_SOURCE={shlex.quote(REMOTE_OFFICIAL_SOURCE)}
SITE={shlex.quote(REMOTE_SITE_PACKAGES)}
PYTHONPATH_VALUE="$SITE:$BASE/src:$BASE"
RUNNER={shlex.quote(str(runner))}
ABLATION={shlex.quote(str(ablation))}
PROBE={shlex.quote(str(probe))}
VERIFIER={shlex.quote(str(verifier))}
OPTIMIZATION={shlex.quote(REMOTE_OPTIMIZATION_PLAN)}
SELECTED_PROFILE=""
SELECTED_BATCH=""

write_status() {{
  state="$1"; code="${{2:-}}"; stable="${{3:-0}}"
  EVOMIND_STATE="$state" EVOMIND_EXIT_CODE="$code" EVOMIND_STABLE="$stable" \
  EVOMIND_PROFILE="$SELECTED_PROFILE" EVOMIND_BATCH="$SELECTED_BATCH" \
  EVOMIND_WRAPPER_PID="$$" EVOMIND_STATUS_PATH="$STATUS" EVOMIND_FINAL_RUN="$FINAL_RUN" \
  EVOMIND_ABLATION_RUN="$ABLATION_RUN" EVOMIND_FINAL_PLAN="$FINAL_PLAN" \
  python3 - <<'PY'
import datetime,json,os,pathlib
path=pathlib.Path(os.environ['EVOMIND_STATUS_PATH']); tmp=path.with_suffix(path.suffix+'.tmp')
raw=os.environ.get('EVOMIND_EXIT_CODE',''); batch=os.environ.get('EVOMIND_BATCH','')
payload={{
 'schema':'evomind.hpc_siim_persistent_run.v1',
 'created_at':datetime.datetime.now().astimezone().isoformat(),
 'status':os.environ['EVOMIND_STATE'],
 'wrapper_pid':int(os.environ['EVOMIND_WRAPPER_PID']),
 'ablation_run_dir':os.environ['EVOMIND_ABLATION_RUN'],
 'final_run_dir':os.environ['EVOMIND_FINAL_RUN'],
 'plan_path':os.environ['EVOMIND_FINAL_PLAN'],
 'selected_profile':os.environ.get('EVOMIND_PROFILE') or None,
 'selected_batch_size':int(batch) if batch else None,
 'stable_idle_observations':int(os.environ.get('EVOMIND_STABLE','0')),
 'exit_code':int(raw) if raw else None,
 'private_labels_used':False,
 'official_grader_executed':False,
 'kaggle_submission_executed':False,
 'process_signals_sent':0,
 'other_processes_modified':False,
}}
tmp.write_text(json.dumps(payload,indent=2)+'\\n'); tmp.replace(path)
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

verify_public_inventory() {{
  PUBLIC_ROOT="$PUBLIC_ROOT" python3 - <<'PY'
import os,pathlib,sys
p=pathlib.Path(os.environ['PUBLIC_ROOT'])
files=[p/n for n in ('train.csv','test.csv','sample_submission.csv')]
files+=list((p/'jpeg/train').glob('*.jpg'))+list((p/'jpeg/test').glob('*.jpg'))
ok=len(files)=={int(inventory['file_count'])} and sum(x.stat().st_size for x in files)=={int(inventory['total_bytes'])}
sys.exit(0 if ok else 41)
PY
}}

verify_frozen || exit $?
verify_public_inventory || {{ write_status public_inventory_drift 41; exit 41; }}
if [ -e "$ABLATION_RUN" ] || [ -e "$FINAL_RUN" ]; then write_status target_run_exists 21; exit 21; fi

write_status waiting_for_leaf
while true; do
  [ -f "$LEAF_STATUS" ] || {{ sleep 30; continue; }}
  leaf_state=$(LEAF_STATUS="$LEAF_STATUS" python3 - <<'PY'
import json,os
print(json.load(open(os.environ['LEAF_STATUS'],encoding='utf-8')).get('status',''))
PY
)
  case "$leaf_state" in
    verification_passed|verification_complete_gate_failed) break ;;
    training_failed|verifier_failed|verification_contract_failed|frozen_artifact_drift|blocked_by_jigsaw_integrity_failure)
      write_status blocked_by_leaf_integrity_failure 31; exit 31 ;;
    *) write_status waiting_for_leaf; sleep 30 ;;
  esac
done

verify_frozen || exit $?
verify_public_inventory || {{ write_status public_inventory_drift 41; exit 41; }}
stable=0
while [ "$stable" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' | wc -l)
  used=$(echo "$line" | awk -F',' '{{gsub(/ /,"",$1);print $1}}')
  util=$(echo "$line" | awk -F',' '{{gsub(/ /,"",$2);print $2}}')
  if [ "$apps" -eq 0 ] && [ "$used" -le 512 ] && [ "$util" -le 10 ]; then stable=$((stable+1)); else stable=0; fi
  write_status waiting_for_gpu_idle "" "$stable"
  [ "$stable" -ge 3 ] || sleep 10
done

mkdir -p -- {shlex.quote(REMOTE_ABLATION_OUTPUT)} {shlex.quote(REMOTE_FINAL_OUTPUT)} "$BASE/logs" "$BASE/batch_probes"
write_status ablation_training
PYTHONPATH="$PYTHONPATH_VALUE" TORCH_HOME="$TORCH_HOME_DIR" CUDA_VISIBLE_DEVICES=0 \
python3 "$ABLATION" --data-root "$DATA_ROOT" --output-root {shlex.quote(REMOTE_ABLATION_OUTPUT)} \
  --run-id {shlex.quote(ABLATION_RUN_ID)} --seed 42 --evaluation-seeds 40,41,42 --folds 3 \
  --image-size 224 --batch-size 128 --workers 16 --manifest-workers 32 \
  --full-backbone convnext_small --lesion-backbone efficientnet_v2_s \
  --linear-alpha 0.0001 --linear-max-iter 2000 --minimum-mean-gain 0.0005 \
  --maximum-worst-fold-regression 0.002 --maximum-seed-mean-regression 0.001 \
  --minimum-seed-pass-fraction 0.6666666666666666 --torch-home "$TORCH_HOME_DIR" \
  >"$BASE/logs/siim_ablation.stdout.log" 2>"$BASE/logs/siim_ablation.stderr.log"
ablation_rc=$?
if [ "$ablation_rc" -ne 0 ]; then write_status ablation_failed "$ablation_rc"; exit "$ablation_rc"; fi
SELECTED_PROFILE=$(REPORT="$ABLATION_REPORT" python3 - <<'PY'
import json,os,sys
r=json.load(open(os.environ['REPORT'],encoding='utf-8'))
if r.get('schema')!='evomind.siim.preprocessing_ablation.v1' or r.get('passed') is not True:
    sys.exit(3)
print(r['selected_profile'])
PY
)
if [ "$?" -ne 0 ] || [ -z "$SELECTED_PROFILE" ]; then write_status ablation_gate_failed 32; exit 32; fi

write_status batch_probing
for batch in 64 32 16; do
  report="$BASE/batch_probes/batch_${{batch}}.json"
  PYTHONPATH="$PYTHONPATH_VALUE" TORCH_HOME="$TORCH_HOME_DIR" CUDA_VISIBLE_DEVICES=0 \
  python3 "$PROBE" --report "$report" --batch-size "$batch" --image-size 384 \
    --metadata-width 16 --full-backbone convnext_small --lesion-backbone efficientnet_v2_s \
    --torch-home "$TORCH_HOME_DIR" --expected-gpu-name 'NVIDIA A40' \
    >"$BASE/logs/probe_${{batch}}.stdout.log" 2>"$BASE/logs/probe_${{batch}}.stderr.log"
  rc=$?
  if [ "$rc" -eq 0 ] && REPORT="$report" python3 - <<'PY'
import json,os,sys
r=json.load(open(os.environ['REPORT'],encoding='utf-8'))
sys.exit(0 if r.get('status')=='passed' and r.get('adapter_source_sha256')=='{adapter_sha}' else 1)
PY
  then SELECTED_BATCH="$batch"; break; fi
done
if [ -z "$SELECTED_BATCH" ]; then write_status batch_probe_failed 33; exit 33; fi

write_status final_training
PYTHONPATH="$PYTHONPATH_VALUE" TORCH_HOME="$TORCH_HOME_DIR" CUDA_VISIBLE_DEVICES=0 \
python3 "$RUNNER" --data-root "$DATA_ROOT" --output-root {shlex.quote(REMOTE_FINAL_OUTPUT)} \
  --allowed-root {shlex.quote(ALLOWED_ROOT)} --official-source-root "$OFFICIAL_SOURCE" \
  --waves Wave0 --competitions siim-isic-melanoma-classification --run-id {shlex.quote(FINAL_RUN_ID)} \
  --seed 42 --phase-a-scope requested --optimization-plan "$OPTIMIZATION" --candidate-only \
  --hold-cuda-lease --siim-preprocessing-profile "$SELECTED_PROFILE" \
  --siim-preprocessing-ablation-report "$ABLATION_REPORT" --siim-backbone convnext_small \
  --siim-secondary-backbone efficientnet_v2_s --siim-epochs 8 --siim-folds 5 \
  --siim-inner-folds 3 --siim-batch-size "$SELECTED_BATCH" --siim-image-size 384 \
  --siim-workers 16 --siim-learning-rate 0.0003 --siim-metadata-iterations 700 \
  --siim-catboost-task-type GPU --wave2-fast-kernels \
  >"$BASE/logs/siim_final.stdout.log" 2>"$BASE/logs/siim_final.stderr.log"
final_rc=$?
if [ "$final_rc" -ne 0 ]; then write_status final_training_failed "$final_rc"; exit "$final_rc"; fi

write_status verifying
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PYTHONPATH_VALUE" python3 "$VERIFIER" \
  --plan "$FINAL_PLAN" --run-dir "$FINAL_RUN" --queue-status "$STATUS" \
  --output "$FINAL_RUN/independent_verification.json" \
  >"$BASE/logs/siim_verifier.stdout.log" 2>"$BASE/logs/siim_verifier.stderr.log"
verify_rc=$?
if [ "$verify_rc" -ne 0 ] && [ "$verify_rc" -ne 3 ]; then write_status verifier_failed "$verify_rc"; exit "$verify_rc"; fi
final_state=$(REPORT="$FINAL_RUN/independent_verification.json" python3 - <<'PY'
import json,os
r=json.load(open(os.environ['REPORT'],encoding='utf-8'))
if r.get('passed') is not True: print('verification_contract_failed')
elif r.get('candidate_ready') is True: print('verification_passed')
else: print('verification_complete_gate_failed')
PY
)
write_status "$final_state" "$verify_rc"
exit 0
"""


def build_staging_report(inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "evomind.siim.public_staging.v1",
        "created_at": now_iso(),
        "competition_id": "siim-isic-melanoma-classification",
        "status": "size_verified_complete",
        "completed_files": inventory["file_count"],
        "total_files": inventory["file_count"],
        "completed_bytes": inventory["total_bytes"],
        "total_bytes": inventory["total_bytes"],
        "inventory_manifest_sha256": inventory["manifest_sha256"],
        "errors": [],
        "private_paths_requested": False,
        "remote_writes_performed": False,
        "process_signals_sent": 0,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def remote_sha256(client: Any, path: str) -> str | None:
    output = common.exec_checked(
        client,
        f"if [ -f {shlex.quote(path)} ]; then sha256sum -- {shlex.quote(path)} | awk '{{print $1}}'; fi",
    ).strip()
    return output or None


def upload_source_if_needed(client: Any, record: Mapping[str, Any]) -> str:
    remote = str(record["remote_path"])
    if remote_sha256(client, remote) == record["sha256"]:
        return "reused"
    common.upload_file(client, Path(str(record["local_path"])), remote)
    if remote_sha256(client, remote) != record["sha256"]:
        raise RuntimeError(f"Remote SIIM source hash differs after upload: {remote}")
    return "uploaded"


def connect_with_retry(attempts: int = 4) -> Any:
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            return common.connect_ssh(timeout=30)
        except Exception as exc:  # pragma: no cover - exercised only on flaky SSH links
            last_error = exc
            if index + 1 < attempts:
                time.sleep(5 * (index + 1))
    assert last_error is not None
    raise last_error


def deploy(*, start: bool = True) -> dict[str, Any]:
    inventory = read_public_inventory()
    compact_bytes = compact_inventory_bytes(inventory)
    LOCAL_COMPACT_INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_COMPACT_INVENTORY.write_bytes(compact_bytes)
    sources = source_records()
    ablation_base = json.loads(LOCAL_ABLATION_BASE.read_text(encoding="utf-8-sig"))
    final_base = json.loads(LOCAL_FINAL_BASE.read_text(encoding="utf-8-sig"))
    ablation_plan = build_ablation_plan(ablation_base, inventory, sources)
    common.write_json_atomic(LOCAL_ABLATION_PLAN, ablation_plan)
    ablation_sha = common.sha256_file(LOCAL_ABLATION_PLAN)
    final_plan = build_final_plan(
        final_base,
        inventory,
        sources,
        ablation_plan_sha256=ablation_sha,
    )
    common.write_json_atomic(LOCAL_FINAL_PLAN, final_plan)
    final_sha = common.sha256_file(LOCAL_FINAL_PLAN)
    wrapper = render_wrapper(
        ablation_plan_sha256=ablation_sha,
        final_plan_sha256=final_sha,
        sources=sources,
        inventory=inventory,
    )
    wrapper_bytes = wrapper.encode("utf-8")

    client = connect_with_retry()
    try:
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
                    posixpath.join(REMOTE_BASE, "data_contract"),
                )
            ),
        )
        source_uploads = {
            str(record["remote_path"]): upload_source_if_needed(client, record)
            for record in sources
        }
        common.upload_bytes(
            client,
            b'"""Minimal package marker for the frozen SIIM HPC bundle."""\n',
            remote_path("src/research_os/__init__.py"),
        )
        common.upload_bytes(client, compact_bytes, REMOTE_INVENTORY)
        staging = build_staging_report(inventory)
        common.upload_bytes(
            client,
            (json.dumps(staging, ensure_ascii=False, indent=2) + "\n").encode(),
            REMOTE_STAGING_REPORT,
        )
        common.upload_file(client, LOCAL_OPTIMIZATION_PLAN, REMOTE_OPTIMIZATION_PLAN)
        common.upload_file(client, LOCAL_ABLATION_PLAN, REMOTE_ABLATION_PLAN)
        common.upload_file(client, LOCAL_FINAL_PLAN, REMOTE_FINAL_PLAN)
        common.upload_bytes(client, wrapper_bytes, REMOTE_WRAPPER)
        common.exec_checked(client, f"chmod 700 -- {shlex.quote(REMOTE_WRAPPER)}")

        smoke = common.exec_checked(
            client,
            f"PYTHONPATH={shlex.quote(REMOTE_SITE_PACKAGES + ':' + REMOTE_BASE + '/src:' + REMOTE_BASE)} "
            "CUDA_VISIBLE_DEVICES='' python3 - <<'PY'\n"
            "import torch,torchvision,numpy,pandas,sklearn,PIL,scipy,catboost\n"
            "import scripts.run_siim_preprocessing_ablation as a\n"
            "import scripts.run_mlebench_lite_full as f\n"
            "import scripts.verify_siim_final_candidate_hpc as v\n"
            "print('import_smoke=passed')\n"
            "print('torch='+torch.__version__)\n"
            "print('torchvision='+torchvision.__version__)\n"
            "print('ablation='+a.sha256_file(a.Path(a.__file__).resolve()))\n"
            "print('full='+f.sha256_file(f.Path(f.__file__).resolve()))\n"
            "print('verifier='+v.sha256_file(v.Path(v.__file__).resolve()))\n"
            "PY",
            timeout=180,
        )
        prior = common.remote_json(client, REMOTE_STATUS)
        action = "deployed_without_start"
        pid: int | None = None
        if start:
            prior_pid = int((prior or {}).get("wrapper_pid") or 0)
            active = False
            if prior_pid:
                active = (
                    common.exec_checked(
                        client,
                        f"if ps -p {prior_pid} -o args= | grep -F -- {shlex.quote(REMOTE_WRAPPER)} >/dev/null; then echo yes; else echo no; fi",
                    ).strip()
                    == "yes"
                )
            if active:
                pid = prior_pid
                action = "existing_wrapper_reused"
            elif prior and prior.get("status") in {
                "ablation_training",
                "batch_probing",
                "final_training",
                "verifying",
                "verification_passed",
                "verification_complete_gate_failed",
            }:
                action = "terminal_or_active_status_preserved"
            else:
                output = common.exec_checked(
                    client,
                    f"nohup bash {shlex.quote(REMOTE_WRAPPER)} "
                    f">{shlex.quote(REMOTE_BASE + '/logs/wrapper.stdout.log')} "
                    f"2>{shlex.quote(REMOTE_BASE + '/logs/wrapper.stderr.log')} </dev/null & echo $!",
                ).strip()
                pid = int(output.splitlines()[-1])
                action = "new_waiting_wrapper_started"
                time.sleep(3)
        status = common.remote_json(client, REMOTE_STATUS)
        gpu = common.exec_checked(
            client,
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,pstate --format=csv,noheader,nounits",
        ).strip()
        evidence = {
            "schema": "evomind.hpc88240_siim_continuation_deployment.v1",
            "created_at": now_iso(),
            "status": "deployed_and_waiting" if start else "deployed_not_started",
            "cluster": "AI-X86_NVIDIA",
            "job_id": 88240,
            "gpu": "NVIDIA A40",
            "remote_base": REMOTE_BASE,
            "public_inventory": {key: inventory[key] for key in ("file_count", "total_bytes", "train_jpeg_count", "test_jpeg_count", "manifest_sha256")},
            "ablation_plan": {"path": REMOTE_ABLATION_PLAN, "bytes": LOCAL_ABLATION_PLAN.stat().st_size, "sha256": ablation_sha},
            "final_plan": {"path": REMOTE_FINAL_PLAN, "bytes": LOCAL_FINAL_PLAN.stat().st_size, "sha256": final_sha},
            "wrapper": {"path": REMOTE_WRAPPER, "bytes": len(wrapper_bytes), "sha256": hashlib.sha256(wrapper_bytes).hexdigest()},
            "source_bundle": [artifact_binding(item) for item in sources],
            "source_uploads": source_uploads,
            "import_smoke": smoke.strip().splitlines(),
            "launch_action": action,
            "wrapper_pid": pid or (status or {}).get("wrapper_pid"),
            "wrapper_status": status,
            "leaf_dependency": common.remote_json(client, REMOTE_LEAF_STATUS),
            "gpu_snapshot_csv": gpu,
            "remote_workspace_confined": True,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "human_gate_preserved": True,
        }
        common.upload_bytes(
            client,
            (json.dumps(evidence, ensure_ascii=False, indent=2) + "\n").encode(),
            REMOTE_DEPLOYMENT,
        )
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
