#!/usr/bin/env python3
"""Deploy the SIIM -> high-resolution RANZCR continuation on job 88240."""

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
    "job88240_ranzcr_highres_s42_20260727",
)
REMOTE_SIIM_STATUS = posixpath.join(
    ALLOWED_ROOT,
    "evomind_mle22",
    "job88240_siim_s42_20260727",
    "full_siim_status.json",
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
    "ranzcr-clip-catheter-line-classification",
    "prepared",
    "public",
)
REMOTE_TORCH_HOME = posixpath.join(ALLOWED_ROOT, "mlebench_model_cache", "torch")
REMOTE_OFFICIAL_SOURCE = posixpath.join(ALLOWED_ROOT, "mle-bench")
BASELINE_BUNDLE = posixpath.join(
    ALLOWED_ROOT,
    "mlebench_lite_runs",
    "a800_ranzcr_recovery_s42_20260726_145840",
    "ranzcr-clip-catheter-line-classification",
    "attempts",
    "attempt_001",
    "vision_oof_and_test.npz",
)
BASELINE_SHA256 = "0c1660497fd7a24744f63ee0de3abb2d68d045d3cbf5e982df9db1bfaf80dc21"
RUN_ID = "hpc88240_ranzcr_convnext768_s42_20260727"
REMOTE_OUTPUT_ROOT = posixpath.join(REMOTE_BASE, "mlebench_lite_runs")
REMOTE_RUN = posixpath.join(REMOTE_OUTPUT_ROOT, RUN_ID)
REMOTE_TASK = posixpath.join(
    REMOTE_RUN,
    "ranzcr-clip-catheter-line-classification",
    "attempts",
    "attempt_001",
)
REMOTE_HIGHRES_BUNDLE = posixpath.join(REMOTE_TASK, "vision_oof_and_test.npz")
REMOTE_CANDIDATE = posixpath.join(REMOTE_BASE, "crossrun_candidate")
REMOTE_STATUS = posixpath.join(REMOTE_BASE, "full_ranzcr_status.json")
REMOTE_WRAPPER = posixpath.join(REMOTE_BASE, "run_ranzcr_after_siim.sh")
REMOTE_PLAN = posixpath.join(REMOTE_BASE, "plans", "ranzcr_highres_crossrun_hpc88240_frozen_plan_20260727.json")
REMOTE_OPTIMIZATION_PLAN = posixpath.join(REMOTE_BASE, "plans", "medal_recovery_gpt56_current.json")
REMOTE_DEPLOYMENT = posixpath.join(REMOTE_BASE, "deployment_manifest.json")
LOCAL_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "ranzcr_highres_crossrun_hpc88240_frozen_plan_20260727.json"
)
LOCAL_OPTIMIZATION_PLAN = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_recovery_gpt56_current.json"
)
LOCAL_EVIDENCE = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job88240_ranzcr_continuation_deployment_current.json"
)


SOURCE_BINDINGS: tuple[tuple[Path, str], ...] = (
    (PROJECT_ROOT / "scripts" / "run_mlebench_lite_full.py", "scripts/run_mlebench_lite_full.py"),
    (PROJECT_ROOT / "scripts" / "run_mlebench_lite_wave0.py", "scripts/run_mlebench_lite_wave0.py"),
    (PROJECT_ROOT / "scripts" / "mlebench_medal_recovery_adapters.py", "scripts/mlebench_medal_recovery_adapters.py"),
    (PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py", "scripts/mlebench_wave2_adapters.py"),
    (PROJECT_ROOT / "scripts" / "russian_transliteration.py", "scripts/russian_transliteration.py"),
    (PROJECT_ROOT / "scripts" / "aggregate_ranzcr_crossrun_candidate.py", "scripts/aggregate_ranzcr_crossrun_candidate.py"),
    (PROJECT_ROOT / "scripts" / "verify_ranzcr_crossrun_candidate.py", "scripts/verify_ranzcr_crossrun_candidate.py"),
    (PROJECT_ROOT / "src" / "research_os" / "mlebench_phase_a.py", "src/research_os/mlebench_phase_a.py"),
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def remote_path(relative: str) -> str:
    return common.ensure_remote_path(posixpath.join(REMOTE_BASE, relative))


def source_records() -> list[dict[str, Any]]:
    records = []
    for local, relative in SOURCE_BINDINGS:
        local = local.resolve()
        records.append(
            {
                "local_path": str(local),
                "remote_path": remote_path(relative),
                "bytes": local.stat().st_size,
                "sha256": common.sha256_file(local),
            }
        )
    return records


def artifact_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("remote_path", "bytes", "sha256")}


def source_map(sources: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {Path(str(record["remote_path"])).name: record for record in sources}


def remote_data_snapshot(client: Any) -> dict[str, Any]:
    command = f"""PUBLIC={shlex.quote(REMOTE_PUBLIC_ROOT)} python3 - <<'PY'
import hashlib,json,os,pathlib
p=pathlib.Path(os.environ['PUBLIC'])
root_files=[x for x in p.iterdir() if x.is_file()]
images=list((p/'train').glob('*.jpg'))+list((p/'test').glob('*.jpg'))
def sha(x):
 h=hashlib.sha256()
 with x.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
print(json.dumps({{
 'file_count':len(root_files)+len(images),
 'total_bytes':sum(x.stat().st_size for x in root_files+images),
 'train_count':len(list((p/'train').glob('*.jpg'))),
 'test_count':len(list((p/'test').glob('*.jpg'))),
 'core':{{x.name:{{'bytes':x.stat().st_size,'sha256':sha(x)}} for x in root_files}},
}}))
PY"""
    snapshot = json.loads(common.exec_checked(client, command, timeout=180))
    if (
        snapshot["file_count"] != 30_087
        or snapshot["total_bytes"] != 6_919_980_199
        or snapshot["train_count"] != 27_074
        or snapshot["test_count"] != 3_009
    ):
        raise RuntimeError("Remote RANZCR public dataset differs from the frozen inventory")
    return snapshot


def build_plan(
    sources: Sequence[Mapping[str, Any]], data: Mapping[str, Any]
) -> dict[str, Any]:
    records = source_map(sources)
    return {
        "schema": "evomind.ranzcr.highres_crossrun_frozen_plan.v1",
        "created_at": now_iso(),
        "status": "frozen_waiting_for_siim",
        "competition_id": "ranzcr-clip-catheter-line-classification",
        "objective": {
            "metric": "mean_columnwise_roc_auc",
            "direction": "maximize",
            "historical_official_score": 0.95889,
            "bronze_threshold": 0.9709,
            "public_oof_promotion_auc": 0.9725,
            "official_score_claimed": False,
        },
        "planner": {
            "requested_model": "gpt-5.6-sol",
            "served_model": "gpt-5.6-sol",
            "execution_plan": {
                "path": REMOTE_OPTIMIZATION_PLAN,
                "bytes": LOCAL_OPTIMIZATION_PLAN.stat().st_size,
                "sha256": common.sha256_file(LOCAL_OPTIMIZATION_PLAN),
            },
        },
        "execution_target": {
            "cluster": "AI-X86_NVIDIA",
            "job_id": 88240,
            "gpu": "NVIDIA A40",
            "remote_root": ALLOWED_ROOT,
            "process_signals_allowed": False,
        },
        "serial_dependency": {
            "status_path": REMOTE_SIIM_STATUS,
            "accepted_verified_terminal_statuses": [
                "verification_passed",
                "verification_complete_gate_failed",
            ],
        },
        "baseline": {
            "model_family": "efficientnet_v2_s_512_grouped_5fold",
            "prediction_bundle": {
                "path": BASELINE_BUNDLE,
                "sha256": BASELINE_SHA256,
                "bytes": 2_599_523,
            },
            "public_oof_auc": 0.9388353612205999,
            "official_score": 0.95889,
        },
        "training": {
            "run_id": RUN_ID,
            "python": "python3",
            "script": records["run_mlebench_lite_full.py"]["remote_path"],
            "data_root": REMOTE_DATA_ROOT,
            "output_root": REMOTE_OUTPUT_ROOT,
            "allowed_root": ALLOWED_ROOT,
            "official_source_root": REMOTE_OFFICIAL_SOURCE,
            "seed": 42,
            "epochs": 12,
            "folds": 5,
            "backbone": "convnext_small",
            "image_size": 768,
            "batch_size": 8,
            "workers": 32,
            "learning_rate": 0.0001,
            "candidate_only": True,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
        "ensemble": {
            "method": "per_label_fold_crossfit_probability_blend",
            "grid_step": 0.025,
            "output_dir": REMOTE_CANDIDATE,
            "aggregator": artifact_binding(records["aggregate_ranzcr_crossrun_candidate.py"]),
            "verifier": artifact_binding(records["verify_ranzcr_crossrun_candidate.py"]),
        },
        "data_contract": {
            **dict(data),
            "public_root": REMOTE_PUBLIC_ROOT,
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
        },
        "claim_boundary": "Frozen public-OOF recovery plan; not an official medal.",
    }


def render_wrapper(
    plan_sha256: str,
    sources: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any],
) -> str:
    records = source_map(sources)
    frozen = [(plan_sha256, REMOTE_PLAN), (BASELINE_SHA256, BASELINE_BUNDLE)]
    frozen.extend((str(r["sha256"]), str(r["remote_path"])) for r in sources)
    for name, record in data["core"].items():
        frozen.append((record["sha256"], posixpath.join(REMOTE_PUBLIC_ROOT, name)))
    frozen.append(
        (
            "0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a",
            posixpath.join(REMOTE_TORCH_HOME, "hub/checkpoints/convnext_small-0c510722.pth"),
        )
    )
    frozen_lines = "\n".join(f"{digest}|{path}" for digest, path in frozen)
    runner = records["run_mlebench_lite_full.py"]["remote_path"]
    aggregator = records["aggregate_ranzcr_crossrun_candidate.py"]["remote_path"]
    verifier = records["verify_ranzcr_crossrun_candidate.py"]["remote_path"]
    return f"""#!/usr/bin/env bash
set -uo pipefail
BASE={shlex.quote(REMOTE_BASE)}
SIIM_STATUS={shlex.quote(REMOTE_SIIM_STATUS)}
STATUS={shlex.quote(REMOTE_STATUS)}
PLAN={shlex.quote(REMOTE_PLAN)}
RUN={shlex.quote(REMOTE_RUN)}
TASK={shlex.quote(REMOTE_TASK)}
CANDIDATE={shlex.quote(REMOTE_CANDIDATE)}
PUBLIC={shlex.quote(REMOTE_PUBLIC_ROOT)}
DATA_ROOT={shlex.quote(REMOTE_DATA_ROOT)}
TORCH_HOME_DIR={shlex.quote(REMOTE_TORCH_HOME)}
SITE={shlex.quote(REMOTE_SITE_PACKAGES)}
PYTHONPATH_VALUE="$SITE:$BASE/src:$BASE"
RUNNER={shlex.quote(str(runner))}
AGGREGATOR={shlex.quote(str(aggregator))}
VERIFIER={shlex.quote(str(verifier))}
BASELINE={shlex.quote(BASELINE_BUNDLE)}
HIGHRES={shlex.quote(REMOTE_HIGHRES_BUNDLE)}

write_status() {{
  state="$1"; code="${{2:-}}"; stable="${{3:-0}}"
  EVOMIND_STATE="$state" EVOMIND_EXIT_CODE="$code" EVOMIND_STABLE="$stable" \
  EVOMIND_PID="$$" EVOMIND_STATUS="$STATUS" EVOMIND_RUN="$RUN" EVOMIND_CANDIDATE="$CANDIDATE" \
  python3 - <<'PY'
import datetime,json,os,pathlib
p=pathlib.Path(os.environ['EVOMIND_STATUS']); t=p.with_suffix(p.suffix+'.tmp'); raw=os.environ.get('EVOMIND_EXIT_CODE','')
d={{'schema':'evomind.hpc_ranzcr_persistent_run.v1','created_at':datetime.datetime.now().astimezone().isoformat(),'status':os.environ['EVOMIND_STATE'],'wrapper_pid':int(os.environ['EVOMIND_PID']),'run_dir':os.environ['EVOMIND_RUN'],'candidate_dir':os.environ['EVOMIND_CANDIDATE'],'stable_idle_observations':int(os.environ.get('EVOMIND_STABLE','0')),'exit_code':int(raw) if raw else None,'private_labels_used':False,'official_grader_executed':False,'kaggle_submission_executed':False,'process_signals_sent':0,'other_processes_modified':False}}
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

verify_frozen || exit $?
[ ! -e "$RUN" ] && [ ! -e "$CANDIDATE" ] || {{ write_status target_run_exists 21; exit 21; }}
write_status waiting_for_siim
while true; do
 [ -f "$SIIM_STATUS" ] || {{ sleep 30; continue; }}
 state=$(SIIM_STATUS="$SIIM_STATUS" python3 - <<'PY'
import json,os
print(json.load(open(os.environ['SIIM_STATUS'],encoding='utf-8')).get('status',''))
PY
)
 case "$state" in
  verification_passed|verification_complete_gate_failed) break ;;
  ablation_failed|ablation_gate_failed|batch_probe_failed|final_training_failed|verifier_failed|verification_contract_failed|frozen_artifact_drift)
   write_status blocked_by_siim_integrity_failure 31; exit 31 ;;
  *) write_status waiting_for_siim; sleep 30 ;;
 esac
done
verify_frozen || exit $?
stable=0
while [ "$stable" -lt 3 ]; do
 line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
 apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' | wc -l)
 used=$(echo "$line" | awk -F',' '{{gsub(/ /,"",$1);print $1}}'); util=$(echo "$line" | awk -F',' '{{gsub(/ /,"",$2);print $2}}')
 if [ "$apps" -eq 0 ] && [ "$used" -le 512 ] && [ "$util" -le 10 ]; then stable=$((stable+1)); else stable=0; fi
 write_status waiting_for_gpu_idle "" "$stable"; [ "$stable" -ge 3 ] || sleep 10
done
mkdir -p -- {shlex.quote(REMOTE_OUTPUT_ROOT)} "$BASE/logs"
write_status highres_training
PYTHONPATH="$PYTHONPATH_VALUE" TORCH_HOME="$TORCH_HOME_DIR" CUDA_VISIBLE_DEVICES=0 \
python3 "$RUNNER" --data-root "$DATA_ROOT" --output-root {shlex.quote(REMOTE_OUTPUT_ROOT)} \
 --allowed-root {shlex.quote(ALLOWED_ROOT)} --official-source-root {shlex.quote(REMOTE_OFFICIAL_SOURCE)} \
 --waves Wave2 --competitions ranzcr-clip-catheter-line-classification --run-id {shlex.quote(RUN_ID)} \
 --seed 42 --phase-a-scope requested --candidate-only --hold-cuda-lease --wave2-fast-kernels \
 --wave2-workers 32 --wave2-ranzcr-epochs 12 --wave2-ranzcr-backbone convnext_small \
 --wave2-ranzcr-image-size 768 --wave2-ranzcr-batch-size 8 --wave2-ranzcr-folds 5 \
 --wave2-ranzcr-learning-rate 0.0001 \
 >"$BASE/logs/highres.stdout.log" 2>"$BASE/logs/highres.stderr.log"
rc=$?
if [ "$rc" -ne 0 ]; then write_status highres_training_failed "$rc"; exit "$rc"; fi
[ -f "$HIGHRES" ] || {{ write_status highres_bundle_missing 32; exit 32; }}
write_status aggregating
PYTHONPATH="$PYTHONPATH_VALUE" CUDA_VISIBLE_DEVICES="" python3 "$AGGREGATOR" \
 --baseline "$BASELINE" --highres "$HIGHRES" --public-dir "$PUBLIC" \
 --output-dir "$CANDIDATE" --grid-step 0.025 --promotion-auc 0.9725 \
 >"$BASE/logs/aggregate.stdout.log" 2>"$BASE/logs/aggregate.stderr.log"
rc=$?
if [ "$rc" -ne 0 ]; then write_status aggregation_failed "$rc"; exit "$rc"; fi
write_status verifying
PYTHONPATH="$PYTHONPATH_VALUE" CUDA_VISIBLE_DEVICES="" python3 "$VERIFIER" \
 --candidate-dir "$CANDIDATE" --public-dir "$PUBLIC" \
 --output "$CANDIDATE/independent_verification.json" \
 >"$BASE/logs/verifier.stdout.log" 2>"$BASE/logs/verifier.stderr.log"
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then write_status verifier_failed "$rc"; exit "$rc"; fi
final=$(REPORT="$CANDIDATE/independent_verification.json" python3 - <<'PY'
import json,os
r=json.load(open(os.environ['REPORT'],encoding='utf-8'))
if r.get('passed') is not True: print('verification_contract_failed')
elif r.get('candidate_ready') is True: print('verification_passed')
else: print('verification_complete_gate_failed')
PY
)
write_status "$final" "$rc"
exit 0
"""


def remote_sha256(client: Any, path: str) -> str | None:
    output = common.exec_checked(
        client,
        f"if [ -f {shlex.quote(path)} ]; then sha256sum -- {shlex.quote(path)} | awk '{{print $1}}'; fi",
    ).strip()
    return output or None


def deploy(*, start: bool = True) -> dict[str, Any]:
    sources = source_records()
    client = connect_with_retry()
    try:
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
        uploads = {}
        for record in sources:
            remote = str(record["remote_path"])
            if remote_sha256(client, remote) == record["sha256"]:
                uploads[remote] = "reused"
            else:
                common.upload_file(client, Path(str(record["local_path"])), remote)
                if remote_sha256(client, remote) != record["sha256"]:
                    raise RuntimeError(f"Remote RANZCR source hash differs: {remote}")
                uploads[remote] = "uploaded"
        common.upload_bytes(
            client,
            b'"""Minimal package marker for the frozen RANZCR HPC bundle."""\n',
            remote_path("src/research_os/__init__.py"),
        )
        common.upload_file(client, LOCAL_OPTIMIZATION_PLAN, REMOTE_OPTIMIZATION_PLAN)
        common.upload_file(client, LOCAL_PLAN, REMOTE_PLAN)
        common.upload_bytes(client, wrapper_bytes, REMOTE_WRAPPER)
        common.exec_checked(client, f"chmod 700 -- {shlex.quote(REMOTE_WRAPPER)}")
        smoke = common.exec_checked(
            client,
            f"PYTHONPATH={shlex.quote(REMOTE_SITE_PACKAGES + ':' + REMOTE_BASE + '/src:' + REMOTE_BASE)} "
            "CUDA_VISIBLE_DEVICES='' python3 - <<'PY'\n"
            "import scripts.run_mlebench_lite_full as f\n"
            "import scripts.aggregate_ranzcr_crossrun_candidate as a\n"
            "import scripts.verify_ranzcr_crossrun_candidate as v\n"
            "print('import_smoke=passed')\n"
            "print('full='+f.sha256_file(f.Path(f.__file__).resolve()))\n"
            "print('aggregate='+a.sha256_file(a.Path(a.__file__).resolve()))\n"
            "print('verify='+v.aggregate.sha256_file(v.Path(v.__file__).resolve()))\n"
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
                active = common.exec_checked(
                    client,
                    f"if ps -p {prior_pid} -o args= | grep -F -- {shlex.quote(REMOTE_WRAPPER)} >/dev/null; then echo yes; else echo no; fi",
                ).strip() == "yes"
            if active:
                action = "existing_wrapper_reused"
                pid = prior_pid
            elif prior and prior.get("status") in {
                "highres_training",
                "aggregating",
                "verifying",
                "verification_passed",
                "verification_complete_gate_failed",
            }:
                action = "terminal_or_active_status_preserved"
            else:
                output = common.exec_checked(
                    client,
                    f"nohup bash {shlex.quote(REMOTE_WRAPPER)} >{shlex.quote(REMOTE_BASE + '/logs/wrapper.stdout.log')} 2>{shlex.quote(REMOTE_BASE + '/logs/wrapper.stderr.log')} </dev/null & echo $!",
                ).strip()
                pid = int(output.splitlines()[-1])
                action = "new_waiting_wrapper_started"
                time.sleep(3)
        status = common.remote_json(client, REMOTE_STATUS)
        evidence = {
            "schema": "evomind.hpc88240_ranzcr_continuation_deployment.v1",
            "created_at": now_iso(),
            "status": "deployed_and_waiting" if start else "deployed_not_started",
            "cluster": "AI-X86_NVIDIA",
            "job_id": 88240,
            "gpu": "NVIDIA A40",
            "remote_base": REMOTE_BASE,
            "data_contract": data,
            "plan": {"path": REMOTE_PLAN, "bytes": LOCAL_PLAN.stat().st_size, "sha256": plan_sha},
            "wrapper": {"path": REMOTE_WRAPPER, "bytes": len(wrapper_bytes), "sha256": hashlib.sha256(wrapper_bytes).hexdigest()},
            "baseline": {"path": BASELINE_BUNDLE, "sha256": BASELINE_SHA256},
            "source_uploads": uploads,
            "import_smoke": smoke.strip().splitlines(),
            "launch_action": action,
            "wrapper_pid": pid or (status or {}).get("wrapper_pid"),
            "wrapper_status": status,
            "siim_dependency": common.remote_json(client, REMOTE_SIIM_STATUS),
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
