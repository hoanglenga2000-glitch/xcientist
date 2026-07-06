#!/usr/bin/env python3
"""
GPU BATCH TRAINING ORCHESTRATOR V3
Distributes all Kaggle competitions across all GPU instances in parallel.
Runs as supervisor — deploys, monitors, collects, and reports.

Architecture:
  Local (supervisor) → SOCKS5 → 9 GPU instances → each runs gpu_train_v3.py
  Training is fully on GPU cluster. Local is zero-compute, supervision only.
"""
import sys, os, json, time, argparse, threading, queue, traceback
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from hpc_connect_v2 import INSTANCES, connect, exec_cmd, find_home, deploy_script, close_all

# ── Competition Priority & Status ──────────────────────────────────────
# Priority: P0=close to bronze, P1=needs work, P2=unknown/exploratory

BRONZE_CONFIRMED = {
    "digit_recognizer", "spaceship_titanic", "house_prices", "bike_sharing_demand",
    "ps3e1", "ps3e7", "ps4e2", "ps4e3", "ps4e4", "ps4e6", "ps4e7",
    "ps5e3", "ps6e2", "ps6e3", "ps6e6",
    "tps_feb2022", "tps_may2022", "ps5e1", "ps5e5",
}

TRAINING_QUEUE = [
    # (task_id, priority, notes, extra_args)
    ("titanic",               "P0", "差0.005到铜牌", "--depth 4 --l2 8 --lr 0.015 --iter 2000 --n-folds 10"),
    ("ps4e1",                 "P0", "差0.004到铜牌", "--depth 4 --l2 10 --lr 0.015 --iter 2000 --n-folds 7"),
    ("porto_seguro",          "P1", "gini=0.03→0.285", "--depth 6 --l2 3 --lr 0.03 --iter 2000 --n-folds 5"),
    ("store_sales",           "P1", "RMSLE=2.12→0.50", "--depth 6 --l2 3 --lr 0.03 --iter 1000 --n-folds 3"),
    ("ps5e2",                 "P1", "回归赛", "--depth 5 --l2 5 --lr 0.02 --iter 1000"),
    ("ps5e4",                 "P1", "回归赛", "--depth 5 --l2 5 --lr 0.02 --iter 1000"),
    ("ps3e25",                "P2", "多分类", "--depth 5 --l2 5 --lr 0.02 --iter 1000"),
    ("tps_aug2022",           "P2", "roc_auc", "--depth 5 --l2 8 --lr 0.015 --iter 2000"),
    ("tps_dec2021",           "P2", "多分类", "--depth 5 --l2 5 --lr 0.02 --iter 1000"),
    ("tps_jan2022",           "P2", "回归赛", "--depth 5 --l2 5 --lr 0.02 --iter 1000"),
    ("tps_mar2022",           "P2", "多分类", "--depth 5 --l2 5 --lr 0.02 --iter 1000"),
    # Re-try with optimized params for close-but-not-confirmed
    ("titanic",               "P0", "V2: stacking ensemble attempt", "--depth 3 --l2 15 --lr 0.01 --iter 3000 --n-folds 10"),
]

# ── State tracking ─────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(ROOT, 'workspace', 'batch_v3_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

class BatchOrchestrator:
    def __init__(self, job_ids=None, dry_run=False):
        self.job_ids = job_ids or sorted(INSTANCES.keys())
        self.dry_run = dry_run
        self.results: dict = {}        # task_id → result dict
        self.queue = list(TRAINING_QUEUE)
        self.completed = []
        self.failed = []
        self.bronze_new = []
        self.batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lock = threading.Lock()
        self.start_time = None

        # Filter out already-bronze if not explicitly re-training
        self.queue = [(t, p, n, a) for t, p, n, a in self.queue
                      if t not in BRONZE_CONFIRMED or "V2" in n or "V3" in n]

    def check_instance(self, job_id):
        """Verify instance is reachable and GPU available."""
        try:
            out, err, code = exec_cmd(job_id, 'nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1', timeout=15)
            if code == 0 and out.strip():
                return {"ok": True, "gpu": out.strip(), "home": find_home(job_id)}
        except Exception as e:
            pass
        return {"ok": False, "error": "unreachable or no GPU"}

    def deploy_to_instance(self, job_id):
        """Deploy gpu_train_v3.py to the instance."""
        local = os.path.join(SCRIPT_DIR, 'gpu_train_v3.py')
        return deploy_script(job_id, local, 'gpu_train_v3.py')

    def run_training(self, job_id, task_id, priority, notes, extra_args, gpu_device=0):
        """Run a single training task on a specific GPU instance."""
        try:
            # Build command
            cmd = f"cd $(dirname $(which python3 2>/dev/null || echo /usr/bin/python3)) 2>/dev/null; cd ~; python3 gpu_train_v3.py {task_id} --gpu-device {gpu_device} {extra_args}"

            print(f"  [{job_id}][{task_id}] Starting training ({priority})...")
            t0 = time.time()

            out, err, code = exec_cmd(job_id, cmd, timeout=3600)

            elapsed = time.time() - t0

            # Parse result JSON from output
            result = {"task_id": task_id, "job_id": job_id, "priority": priority,
                      "elapsed": round(elapsed, 1), "exit_code": code}

            # Extract JSON from output
            json_start = out.find('{"task_id"')
            if json_start >= 0:
                json_end = out.find('\n', json_start)
                if json_end < 0: json_end = len(out)
                try:
                    parsed = json.loads(out[json_start:json_end])
                    result.update(parsed)
                except: pass

            # Look for SUMMARY line
            for line in out.split('\n'):
                if 'SUMMARY:' in line:
                    result['summary'] = line.strip()
                    print(f"  [{job_id}][{task_id}] {line.strip()} [{elapsed:.0f}s]")
                    break

            if err:
                result['stderr_tail'] = err[-500:]

            return result

        except Exception as e:
            return {"task_id": task_id, "job_id": job_id, "priority": priority,
                    "status": "failed", "error": str(e)[:500]}

    def run_batch(self):
        """Main batch execution: deploy, then run all tasks in parallel."""
        self.start_time = time.time()

        # Phase 1: Health check all instances
        print("=" * 70)
        print(f"BATCH {self.batch_id} — GPU Cluster Health Check")
        print("=" * 70)

        online = {}
        for jid in self.job_ids:
            status = self.check_instance(jid)
            online[jid] = status
            icon = "✓" if status["ok"] else "✗"
            gpu = status.get("gpu", status.get("error", "N/A"))[:70]
            print(f"  [{jid}] {icon} {INSTANCES[jid]['gpu']} | {gpu}")

        online_jobs = [jid for jid, s in online.items() if s["ok"]]
        print(f"\n  Online: {len(online_jobs)}/{len(self.job_ids)} instances")

        if not online_jobs:
            print("FATAL: No GPU instances available. Check proxy bridge.")
            return {"status": "no_gpu_available"}

        if self.dry_run:
            return {"status": "dry_run", "online": len(online_jobs), "queue": len(self.queue)}

        # Phase 2: Deploy training script to all online instances
        print(f"\n{'='*70}")
        print(f"PHASE 2: Deploy gpu_train_v3.py to {len(online_jobs)} instances")
        print(f"{'='*70}")

        for jid in online_jobs:
            try:
                remote = self.deploy_to_instance(jid)
                print(f"  [{jid}] Deployed → {remote}")
            except Exception as e:
                print(f"  [{jid}] DEPLOY FAILED: {e}")
                online_jobs.remove(jid)

        if not online_jobs:
            return {"status": "deploy_failed"}

        # Phase 3: Distribute and run training in parallel
        print(f"\n{'='*70}")
        print(f"PHASE 3: Parallel Training — {len(self.queue)} tasks on {len(online_jobs)} GPUs")
        print(f"{'='*70}")

        # Round-robin distribution
        task_assignments = []
        for i, (task_id, priority, notes, extra_args) in enumerate(self.queue):
            jid = online_jobs[i % len(online_jobs)]
            # Alternate GPU device if multi-GPU instance
            gpu_dev = 0
            if INSTANCES[jid]['gpu'].startswith('2×'):
                gpu_dev = i % 2  # alternate between 0 and 1 for dual-GPU
            task_assignments.append((jid, task_id, priority, notes, extra_args, gpu_dev))

        for jid, tid, pri, notes, args, gpu in task_assignments:
            print(f"  [{jid}:GPU{gpu}] ← {tid} ({pri}) {notes}")

        # Execute in parallel (max concurrent = number of online instances × GPUs)
        max_workers = sum(2 if INSTANCES[j]['gpu'].startswith('2×') else 1 for j in online_jobs)
        max_workers = min(max_workers, len(task_assignments))
        print(f"\n  Running with {max_workers} parallel workers...")

        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for jid, tid, pri, notes, args, gpu in task_assignments:
                future = executor.submit(self.run_training, jid, tid, pri, notes, args, gpu)
                futures[future] = (jid, tid)

            for future in as_completed(futures):
                jid, tid = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)

                    with self.lock:
                        if result.get('gate_passed'):
                            self.bronze_new.append(result)
                            print(f"\n  ★ BRONZE GATE PASS: {tid} score={result.get('oof_score')} [GPU: {jid}]")
                        elif result.get('status') == 'completed':
                            self.completed.append(result)
                        else:
                            self.failed.append(result)
                        self.results[tid] = result

                except Exception as e:
                    self.failed.append({"task_id": tid, "job_id": jid, "error": str(e)})
                    print(f"  ✗ [{jid}][{tid}] Worker crashed: {e}")

        # Phase 4: Summary
        elapsed_total = time.time() - self.start_time
        print(f"\n{'='*70}")
        print(f"BATCH COMPLETE — {elapsed_total/60:.1f} min")
        print(f"{'='*70}")
        print(f"  Total tasks: {len(all_results)}")
        print(f"  Gate PASS: {len(self.bronze_new)}")
        print(f"  Completed (no pass): {len(self.completed)}")
        print(f"  Failed: {len(self.failed)}")

        if self.bronze_new:
            print(f"\n  ★ NEW BRONZE CANDIDATES:")
            for r in self.bronze_new:
                print(f"    {r['task_id']}: {r.get('metric','?')}={r.get('oof_score','?')} "
                      f"bronze={r.get('bronze_threshold','?')} gap={r.get('gate_gap','?')}")

        # Save manifest
        manifest = {
            "batch_id": self.batch_id,
            "started": datetime.fromtimestamp(self.start_time).isoformat(),
            "elapsed_minutes": round(elapsed_total / 60, 1),
            "online_instances": online_jobs,
            "total_tasks": len(all_results),
            "gate_pass_count": len(self.bronze_new),
            "completed_count": len(self.completed),
            "failed_count": len(self.failed),
            "bronze_confirmed_before": sorted(BRONZE_CONFIRMED),
            "bronze_new": [r.get('task_id') for r in self.bronze_new],
            "results": all_results,
        }
        manifest_path = os.path.join(RESULTS_DIR, f"batch_{self.batch_id}.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  Manifest: {manifest_path}")

        # Cleanup connections
        close_all()

        return manifest


def main():
    parser = argparse.ArgumentParser(description="GPU Batch Training Orchestrator V3")
    parser.add_argument('--jobs', nargs='+', default=None, help='Specific job IDs to use')
    parser.add_argument('--dry-run', action='store_true', help='Check health only, no training')
    parser.add_argument('--task', help='Run single task only')
    parser.add_argument('--skip-bronze', action='store_true', default=True,
                        help='Skip already bronze competitions (default: True)')
    args = parser.parse_args()

    orch = BatchOrchestrator(job_ids=args.jobs, dry_run=args.dry_run)

    if args.task:
        # Single task mode
        orch.queue = [(args.task, "manual", "manual run",
                       "--depth 5 --l2 5 --lr 0.02 --iter 1000")]
        if orch.queue[0][0] in BRONZE_CONFIRMED and args.skip_bronze:
            print(f"SKIP: {args.task} already bronze. Use --skip-bronze=0 to force.")
            return

    if args.skip_bronze:
        orch.queue = [(t, p, n, a) for t, p, n, a in orch.queue
                      if t not in BRONZE_CONFIRMED]

    if not orch.queue:
        print("No tasks to run. All competitions already bronze or queue empty.")
        return

    result = orch.run_batch()
    if result.get('status') in ('no_gpu_available', 'deploy_failed'):
        sys.exit(1)


if __name__ == '__main__':
    main()
