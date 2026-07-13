from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures" / "three_layer_evidence_20260623"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_all(fig: plt.Figure, name: str) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    svg = FIG_DIR / f"{name}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png.relative_to(ROOT)), "svg": str(svg.relative_to(ROOT))}


def figure_architecture() -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_axis_off()
    layers = [
        ("Layer 3: XCIENTIST-style Research Harness", "Hypothesis -> Validation Contract -> Risk Check -> Claim Audit\nPrevents overclaim and binds every conclusion to artifacts", "#FEF3C7", "#B45309"),
        ("Layer 2: MLEvolve-style Search Controller", "Search Graph -> Branch Selection -> Retrospective Memory -> Best-so-far Gate\nTurns baseline runs into self-evolving optimization loops", "#DBEAFE", "#1D4ED8"),
        ("Layer 1: Multi-Agent Research OS", "Task -> Agent Workflow -> Code/Training -> Metrics/OOF -> Submission Gate -> Report\nMakes every step executable, reproducible and auditable", "#D1FAE5", "#047857"),
    ]
    y_positions = [0.72, 0.42, 0.12]
    for (title, body, fill, edge), y in zip(layers, y_positions):
        rect = plt.Rectangle((0.06, y), 0.88, 0.20, transform=ax.transAxes, facecolor=fill, edgecolor=edge, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.09, y + 0.135, title, transform=ax.transAxes, fontsize=16, fontweight="bold", color=edge)
        ax.text(0.09, y + 0.055, body, transform=ax.transAxes, fontsize=11.5, color="#0F172A", linespacing=1.45)
    for y1, y2 in [(0.70, 0.62), (0.40, 0.32)]:
        ax.annotate("", xy=(0.50, y2), xytext=(0.50, y1), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#334155", lw=2))
    ax.text(0.5, 0.965, "Self-Evolving and Auditable MLE Research OS", transform=ax.transAxes, ha="center", fontsize=20, fontweight="bold", color="#0F172A")
    ax.text(0.5, 0.015, "Current evidence: local proxy validation; no official Kaggle/GPU/MLE-Bench medal claim.", transform=ax.transAxes, ha="center", fontsize=10, color="#64748B")
    return save_all(fig, "figure_1_three_layer_architecture")


def normalized(row: dict[str, Any], value: float) -> float:
    base = float(row["round1_baseline"])
    if row["direction"] == "minimize":
        return (base - value) / base * 100.0
    return (value - base) / base * 100.0


def trajectory_values(row: dict[str, Any]) -> tuple[list[str], list[float]]:
    if "round4_score" in row:
        return ["Round1", "Round2", "Round3", "Round4", "Final"], [
            row["round1_baseline"],
            row["round2_best_so_far"],
            row["round3_best_so_far"],
            row["round4_score"],
            row["final_best_so_far"],
        ]
    return ["Round1", "Round2", "Round3", "Final"], [
        row["round1_baseline"],
        row["round2_best_so_far"],
        row["round3_score"],
        row["final_best_so_far"],
    ]


def figure_trajectory(bundle: dict[str, Any]) -> dict[str, str]:
    rows = bundle.get("active_trajectory") or bundle.get("round4_trajectory") or bundle["trajectory"]
    latest_round = bundle.get("latest_round", "round3")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.30, 1]})
    for row in rows:
        rounds, raw = trajectory_values(row)
        axes[0].plot(rounds, raw, marker="o", linewidth=2.4, label=f"{row['task_id']} ({row['metric']})")
        if "round4_decision" in row and "preserve" in row["round4_decision"]:
            axes[0].scatter(["Round4"], [row["round4_score"]], s=90, marker="x", color="#EF4444", zorder=5)
    axes[0].set_title(f"Raw local proxy score trajectory through {latest_round}", fontweight="bold")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(fontsize=8, loc="best")
    axes[0].set_xlabel("Evolution round")
    axes[0].set_ylabel("Task metric")

    task_labels = [row["task_id"] for row in rows]
    x = np.arange(len(rows))
    improvements = [normalized(row, row["final_best_so_far"]) for row in rows]
    bars = axes[1].bar(x, improvements, color=["#10B981" if v >= 0 else "#EF4444" for v in improvements])
    axes[1].axhline(0, color="#334155", linewidth=1)
    axes[1].set_xticks(x, task_labels, rotation=15, ha="right")
    axes[1].set_title("Final best-so-far gain vs Round1 baseline (%)", fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.22)
    for bar, value in zip(bars, improvements):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    promoted = bundle.get("headline_results", {}).get("round4_promoted")
    preserved = bundle.get("headline_results", {}).get("round4_preserved_parent")
    subtitle = f"Round4 promoted={promoted}, preserved_parent={preserved}" if promoted is not None else "best-so-far never regresses"
    fig.suptitle(f"Round1 -> {latest_round}: best-so-far never regresses ({subtitle})", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_all(fig, "figure_2_best_so_far_trajectory")


def figure_claim_boundary(bundle: dict[str, Any]) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_axis_off()
    allowed = bundle["claim_boundary"]["allowed"]
    not_allowed = bundle["claim_boundary"]["not_allowed"]
    ax.text(0.5, 0.96, "Claim Boundary and XCIENTIST-style Audit", ha="center", transform=ax.transAxes, fontsize=18, fontweight="bold")
    left = plt.Rectangle((0.06, 0.12), 0.41, 0.72, transform=ax.transAxes, facecolor="#ECFDF5", edgecolor="#059669", linewidth=2)
    right = plt.Rectangle((0.53, 0.12), 0.41, 0.72, transform=ax.transAxes, facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=2)
    ax.add_patch(left)
    ax.add_patch(right)
    ax.text(0.09, 0.78, "Allowed claims", transform=ax.transAxes, fontsize=15, fontweight="bold", color="#047857")
    ax.text(0.56, 0.78, "Blocked overclaims", transform=ax.transAxes, fontsize=15, fontweight="bold", color="#B91C1C")
    allowed_short = [
        "Three-layer Research OS shown on 3 local tasks.",
        "Best-so-far is protected across rounds.",
        "Round4 uses memory-guided branch search.",
        "Contracts and claim audits exist for all branches.",
    ]
    blocked_short = [
        "No official Kaggle leaderboard improvement claim.",
        "No GPU/HPC execution claim for these rounds.",
        "No MLE-Bench medal or MLEvolve parity claim.",
    ]
    for i, item in enumerate(allowed_short):
        ax.text(0.09, 0.70 - i * 0.12, f"[OK] {item}", transform=ax.transAxes, fontsize=10.5, color="#064E3B", wrap=True)
    for i, item in enumerate(blocked_short):
        ax.text(0.56, 0.70 - i * 0.14, f"[BLOCKED] {item}", transform=ax.transAxes, fontsize=10.5, color="#7F1D1D", wrap=True)
    ax.text(0.5, 0.045, "Audit rule: unsupported or proxy-only evidence is reported as boundary-limited, not as official benchmark success.", transform=ax.transAxes, ha="center", fontsize=10, color="#475569")
    return save_all(fig, "figure_3_claim_boundary_audit")


def main() -> None:
    bundle = read_json(ROOT / "workspace" / "paper_evidence_bundle_20260623.json")
    figures = [
        {
            "figure_id": "figure_1",
            "title": "Three-layer architecture",
            "caption": "The proposed AI Research Workstation integrates orchestration/execution, self-evolving search, and research-harness audit into a single evidence-first system.",
            "paths": figure_architecture(),
        },
        {
            "figure_id": "figure_2",
            "title": "Best-so-far trajectory",
            "caption": "Across three local proxy tabular tasks, Round2 and Round3 branches preserve or improve best-so-far scores rather than overwriting the best candidate with weaker runs.",
            "paths": figure_trajectory(bundle),
        },
        {
            "figure_id": "figure_3",
            "title": "Claim boundary audit",
            "caption": "XCIENTIST-style claim audit separates supported local-proxy claims from blocked overclaims such as official leaderboard, GPU/HPC, or MLE-Bench medal assertions.",
            "paths": figure_claim_boundary(bundle),
        },
    ]
    manifest = {
        "schema": "academic_research_os.paper_figure_manifest.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_bundle": "workspace/paper_evidence_bundle_20260623.json",
        "figures": figures,
    }
    manifest_path = FIG_DIR / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), "figures": figures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
