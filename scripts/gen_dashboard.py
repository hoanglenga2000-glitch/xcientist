import json, os, datetime

base = "D:/桌面/codex/科研港科技"

with open(f"{base}/workspace/kaggle_experiment_inventory_20260624.json") as f:
    inv = json.load(f)
with open(f"{base}/workspace/top30_next_evolution_orders_20260625.json") as f:
    top30 = json.load(f)
with open(f"{base}/workspace/mlebench_style_current_leaderboard_20260625.json") as f:
    lb = json.load(f)
with open(f"{base}/workspace/mlevolve_alignment_matrix_20260625.json") as f:
    align = json.load(f)

now = datetime.datetime.now().isoformat()

dash = {
    "schema": "academic_research_os.teacher_medal_rate_dashboard.v1",
    "generated_at": now,
    "generated_by": "Claude Code via AI Research Workstation @ http://127.0.0.1:8088",
    "headline": {
        "total_tasks_with_experiments": inv["task_count_with_experiments"],
        "total_runs_observed": inv["total_runs_observed"],
        "total_scored_runs": inv["total_scored_runs"],
        "total_promoted_runs": inv["total_promoted_runs"],
        "total_held_runs": inv["total_held_runs"],
        "total_timeout_or_failed": inv["total_timeout_or_failed_runs"],
        "kaggle10_runnable": f"{inv['kaggle10_runnable_count']}/{inv['kaggle10_task_count']}",
        "kaggle10_completion": inv["kaggle10_completion_status"],
        "official_score_known_count": inv["official_score_known_count"],
        "official_rank_known_count": inv["official_rank_known_count"],
        "official_top30_count": inv["official_top30_count"],
        "official_top30_rate": inv["official_top30_rate"],
        "current_top30_rate_pct": f"{inv['official_top30_rate'] * 100:.1f}%",
        "teacher_medal_target": "50%+ top30 rate across all Kaggle10 tasks"
    },
    "leaderboard": [],
    "top30_evolution_orders": [],
    "governance": {},
    "resource_status": {},
    "priority_actions": []
}

for r in lb.get("leaderboard_rows", []):
    dash["leaderboard"].append({
        "task_id": r["task_id"],
        "score": r.get("score"),
        "metric": r.get("metric"),
        "official_rank": r.get("official_rank"),
        "leaderboard_team_count": r.get("leaderboard_team_count"),
        "rank_percentile": r.get("rank_percentile"),
        "top30": r.get("top30"),
        "top30_gap": r.get("top30_gap"),
        "medal_zone": r.get("medal_zone"),
        "status": r.get("status")
    })

for o in top30.get("orders", []):
    dash["top30_evolution_orders"].append({
        "task_id": o["task_id"],
        "priority": o["priority"],
        "status": o.get("current_official_status"),
        "official_rank": o.get("official_rank"),
        "rank_percentile": o.get("rank_percentile"),
        "top30_gap": o.get("top30_gap"),
        "current_best_score": o.get("current_best_score"),
        "metric": o.get("metric"),
        "submit_budget": o.get("official_submit_budget"),
        "branches": [{
            "branch_id": b["branch_id"],
            "branch_type": b["branch_type"],
            "hypothesis": b["hypothesis"][:120],
            "expected_delta": b["expected_delta"]
        } for b in o.get("selected_branches", [])]
    })

dash["governance"] = {
    "artifact_coverage": inv.get("governance_artifact_coverage", {}),
    "kaggle4_verification": inv.get("kaggle4_verification"),
    "timeout_control": inv.get("timeout_control_verification"),
    "claim_boundary": inv.get("claim_boundary"),
    "alignment_source": align.get("source_repo"),
    "alignment_version": align.get("schema")
}

dash["resource_status"] = {
    "gpu": {
        "type": "NVIDIA A800-SXM4-80GB",
        "count": 1,
        "ssh_gateway": "passed",
        "torch_installed": False,
        "blocker": "torch not installed on remote; no loaded SSH credential for automated jobs",
        "current_gate_ready": False
    },
    "kaggle": {
        "configured": True,
        "credential_type": "DPAPI access_token",
        "python_version": "2.2.1",
        "submission_gate": "human_approval_required"
    },
    "deepseek": {
        "configured": False,
        "model": "deepseek-v4-flash",
        "blocker": "DEEPSEEK_API_KEY not set"
    },
    "code_agent": {
        "configured": False,
        "blocker": "Neither ANTHROPIC_API_KEY nor DEEPSEEK_API_KEY set"
    }
}

dash["priority_actions"] = [
    {
        "rank": 1, "task": "spaceship_titanic", "priority": "P0",
        "action": "Run 3-branch optimization to push from 36.8% to <=30% percentile",
        "current_score": 0.80674, "target": "rank <=625 (top 30%)",
        "blocker": "GPU torch missing; needs local CPU ensemble or GPU torch fix first"
    },
    {
        "rank": 2, "task": "global", "priority": "P0",
        "action": "Install PyTorch on remote A800 GPU via SSH gateway",
        "blocker": "All GPU training blocked; No module named torch"
    },
    {
        "rank": 3, "task": "global", "priority": "P1",
        "action": "Run Kaggle DPAPI smoke test to verify official download/submission",
        "command": "powershell scripts/manage_kaggle_secret.ps1 smoke -AllowRealExternal"
    },
    {
        "rank": 4, "task": "global", "priority": "P1",
        "action": "Configure DEEPSEEK_API_KEY for research LLM and strategy generation"
    },
    {
        "rank": 5, "task": "global", "priority": "P2",
        "action": "Run full statistics refresh via workstation-summary API"
    }
]

json_path = f"{base}/workspace/teacher_medal_rate_dashboard.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(dash, f, indent=2, ensure_ascii=False, default=str)
print(f"JSON: {json_path} ({os.path.getsize(json_path)} bytes)")

# Generate Markdown
md = []
md.append("# TEACHER MEDAL RATE DASHBOARD")
md.append("")
md.append(f"> Generated: {now}")
md.append(f"> Source: AI Research Workstation @ http://127.0.0.1:8088")
md.append(f"> Schema: academic_research_os.teacher_medal_rate_dashboard.v1")
md.append("")
md.append("---")
md.append("")
md.append("## Headline Metrics")
md.append("")
md.append("| Metric | Value |")
md.append("|--------|-------|")
md.append(f"| Total Tasks (with experiments) | {inv['task_count_with_experiments']} |")
md.append(f"| Total Runs Observed | {inv['total_runs_observed']} |")
md.append(f"| Scored Runs | {inv['total_scored_runs']} |")
md.append(f"| Promoted Runs | {inv['total_promoted_runs']} |")
md.append(f"| Held (low-score) Runs | {inv['total_held_runs']} |")
md.append(f"| Timeout/Failed | {inv['total_timeout_or_failed_runs']} |")
md.append(f"| Kaggle10 Runnable | {inv['kaggle10_runnable_count']}/{inv['kaggle10_task_count']} ({inv['kaggle10_completion_status']}) |")
md.append(f"| Official Top30 Count | {inv['official_top30_count']} |")
md.append(f"| **Official Top30 Rate** | **{inv['official_top30_rate']*100:.1f}%** |")
md.append(f"| Scores Known | {inv['official_score_known_count']} |")
md.append(f"| Ranks Known | {inv['official_rank_known_count']} |")
md.append("")
md.append("---")
md.append("")
md.append("## Leaderboard (MLEBench Style)")
md.append("")
md.append("| Task | Score | Rank | Percentile | Top30 | Gap | Status |")
md.append("|------|-------|------|------------|-------|-----|--------|")
for r in lb.get("leaderboard_rows", []):
    task = r["task_id"].replace("_", " ")[:30]
    score = str(r.get("score", "?"))[:10]
    rank_str = f"{r.get('official_rank','?')}/{r.get('leaderboard_team_count','?')}" if r.get('official_rank') else "?"
    pct = f"{r.get('rank_percentile',0)*100:.1f}%" if r.get('rank_percentile') else "?"
    top30_s = "YES" if r.get('top30') else ("NO" if r.get('top30') is False else "?")
    tgap = r.get('top30_gap')
    gap = f"{tgap*100:.1f}pp" if tgap is not None else "?"
    status_s = str(r.get("status", "?"))[:15]
    md.append(f"| {task} | {score} | {rank_str} | {pct} | {top30_s} | {gap} | {status_s} |")
md.append("")

md.append("---")
md.append("")
md.append("## Top30 Evolution Orders (Priority Queue)")
md.append("")
for o in top30.get("orders", []):
    md.append(f"### {o['task_id']} [{o['priority']}]")
    md.append("")
    md.append(f"- **Status**: {o.get('current_official_status', '?')}")
    rp = o.get('rank_percentile')
    rp_str = f"{rp*100:.1f}%" if rp is not None else "?"
    tg = o.get('top30_gap')
    tg_str = f"{tg*100:.1f}pp" if tg is not None else "?"
    md.append(f"- **Rank**: {o.get('official_rank', '?')} / percentile={rp_str}")
    md.append(f"- **Top30 Gap**: {tg_str}")
    md.append(f"- **Current Best Score**: {o.get('current_best_score', '?')}")
    md.append(f"- **Submit Budget**: {o.get('official_submit_budget', '?')}")
    md.append("")
    md.append("| Branch | Type | Hypothesis | Expected Delta |")
    md.append("|--------|------|------------|----------------|")
    for b in o.get("selected_branches", []):
        h = b['hypothesis'][:80] + "..." if len(b['hypothesis']) > 80 else b['hypothesis']
        md.append(f"| {b['branch_id'][:40]} | {b['branch_type']} | {h} | {b['expected_delta']} |")
    md.append("")

md.append("---")
md.append("")
md.append("## Resource Status")
md.append("")
md.append("| Resource | Status | Detail |")
md.append("|----------|--------|--------|")
md.append("| GPU (A800) | BLOCKED | SSH passed but torch not installed; no loaded credential for auto jobs |")
md.append("| Kaggle DPAPI | READY | Token configured, Python 2.2.1, Human Gate for submission |")
md.append("| DeepSeek LLM | NOT CONFIGURED | DEEPSEEK_API_KEY not set |")
md.append("| Code Agent | NOT CONFIGURED | No ANTHROPIC_API_KEY or DEEPSEEK_API_KEY |")
md.append("")

md.append("---")
md.append("")
md.append("## Governance")
md.append("")
k4 = inv.get('kaggle4_verification', {})
tc = inv.get('timeout_control_verification', {})
md.append(f"- **Kaggle4 Verification**: {k4.get('status', '?')} ({k4.get('checks', '?')})")
md.append(f"- **Timeout Control**: {tc.get('status', '?')} ({tc.get('checks', '?')})")
md.append(f"- **Claim Boundary**: {inv.get('claim_boundary', '')[:300]}")
md.append("")
md.append("### Artifact Coverage")
for k, v in inv.get("governance_artifact_coverage", {}).items():
    md.append(f"- **{k}**: {v}")
md.append("")

md.append("---")
md.append("")
md.append("## Priority Actions (Next Steps)")
md.append("")
md.append("| # | Priority | Task | Action | Blocker |")
md.append("|---|----------|------|--------|---------|")
for a in dash["priority_actions"]:
    b = a.get("blocker", "")[:60] if a.get("blocker") else ""
    act = a["action"][:100]
    md.append(f"| {a['rank']} | {a['priority']} | {a['task']} | {act} | {b} |")
md.append("")

md.append("---")
md.append("")
md.append("## Evidence Integrity")
md.append("")
md.append("- All scores are backed by Kaggle response artifacts or local CV/OOF evidence with explicit claim binding.")
md.append("- No rank, top30, or medal claim is made without corresponding Kaggle leaderboard artifacts.")
md.append("- GPU/SSH connections are verified via nvidia-smi smoke tests before accepting training jobs.")
md.append("- Kaggle submissions require: submission_audit + validation_contract + claim_audit + human gate approval.")
md.append("")
md.append(f"*Dashboard auto-generated by AI Research Workstation. JSON artifact: workspace/teacher_medal_rate_dashboard.json*")

md_path = f"{base}/TEACHER_MEDAL_RATE_DASHBOARD.md"
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print(f"MD: {md_path} ({os.path.getsize(md_path)} bytes)")
print("Done!")
