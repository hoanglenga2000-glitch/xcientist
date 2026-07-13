"use client";

import { useEffect, useState } from "react";
import { BookOpen, CheckCircle2, FileText, GitBranch, RefreshCcw, ShieldCheck, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge, type StatusTone } from "@/components/ui/status-badge";
import * as api from "@/lib/api/client";
import type { PaperEvidenceBundle as PaperEvidenceBundleType } from "@/lib/api/types";

function score(value: number, digits = 6) {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function tone(decision: string): StatusTone {
  if (/promote/i.test(decision)) return "green";
  if (/preserve/i.test(decision)) return "amber";
  if (/reject|fail/i.test(decision)) return "red";
  return "slate";
}

function candidateScore(row: { round4_score?: number; round3_score?: number }) {
  return typeof row.round4_score === "number" ? row.round4_score : row.round3_score;
}

function decisionLabel(row: { round4_decision?: string; round3_decision?: string }) {
  return row.round4_decision ?? row.round3_decision ?? "pending";
}

export function PaperEvidenceBundleCard({
  onGenerated
}: {
  onGenerated?: (payload: unknown) => void;
}) {
  const [bundle, setBundle] = useState<PaperEvidenceBundleType | null>(null);
  const [bundlePath, setBundlePath] = useState<string>("workspace/paper_evidence_bundle_20260623.json");
  const [message, setMessage] = useState("Loading paper evidence bundle...");
  const [busy, setBusy] = useState(false);

  async function load() {
    try {
      const payload = await api.getPaperEvidenceBundle();
      setBundle(payload.bundle ?? null);
      setBundlePath(payload.bundle_path ?? bundlePath);
      setMessage(payload.bundle ? "Evidence bundle loaded from artifact." : "Evidence bundle is not generated yet.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Failed to load evidence bundle.");
    }
  }

  async function generate() {
    setBusy(true);
    try {
      const payload = await api.generatePaperEvidenceBundle();
      onGenerated?.(payload);
      await load();
      setMessage(`Generated: ${payload.paper_report ?? payload.bundle ?? "paper evidence bundle"}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Failed to generate evidence bundle.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const headline = bundle?.headline_results;
  const trajectory = bundle?.active_trajectory ?? bundle?.round4_trajectory ?? bundle?.trajectory ?? [];
  const latestRound = bundle?.latest_round ?? (bundle?.round4_trajectory?.length ? "round4" : "round3");

  return (
    <Card className="border-blue-100">
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-blue-700" />
            Paper Evidence Bundle
          </CardTitle>
          <CardDescription>
            Three-layer architecture evidence for the thesis: Research OS + MLEvolve-style search + XCIENTIST-style audit.
          </CardDescription>
        </div>
        <StatusBadge tone={headline?.best_so_far_never_regressed ? "green" : "amber"}>
          {headline?.best_so_far_never_regressed ? "best protected" : "pending"}
        </StatusBadge>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <EvidenceMetric icon={GitBranch} label="Tasks" value={headline?.tasks ?? "n/a"} tone="blue" />
          <EvidenceMetric icon={CheckCircle2} label="Round2 promoted" value={headline?.round2_promoted ?? "n/a"} tone="green" />
          <EvidenceMetric icon={CheckCircle2} label="Round3 promoted" value={headline?.round3_promoted ?? "n/a"} tone="green" />
          <EvidenceMetric icon={TrendingUp} label="Round4 promoted" value={headline?.round4_promoted ?? "n/a"} tone="green" />
        </div>

        <div className="overflow-hidden rounded-md border border-slate-200">
          <div className="grid grid-cols-[1.1fr_.9fr_.9fr_.9fr_.9fr_1.2fr] bg-slate-50 px-3 py-2 text-[11px] font-bold uppercase text-slate-500">
            <span>Task</span>
            <span>Round1</span>
            <span>Round2</span>
            <span>{latestRound === "round4" ? "Round4" : "Round3"}</span>
            <span>Final</span>
            <span>Decision</span>
          </div>
          {trajectory.length ? trajectory.map((row) => (
            <div key={row.task_id} className="grid grid-cols-[1.1fr_.9fr_.9fr_.9fr_.9fr_1.2fr] items-center border-t border-slate-100 px-3 py-2 text-xs">
              <span className="font-bold text-slate-950">{row.task_id}</span>
              <span>{score(row.round1_baseline)}</span>
              <span>{score(row.round2_best_so_far)}</span>
              <span>{score(candidateScore(row) ?? Number.NaN)}</span>
              <span className="font-bold">{score(row.final_best_so_far)}</span>
              <StatusBadge tone={tone(decisionLabel(row))}>{decisionLabel(row)}</StatusBadge>
            </div>
          )) : (
            <div className="border-t border-slate-100 px-3 py-6 text-sm text-slate-500">No trajectory loaded.</div>
          )}
        </div>

        <SteadyImprovementPanel bundle={bundle} />

        <div className="grid gap-3 lg:grid-cols-3">
          <LayerCard title="Layer 1" subtitle="Multi-Agent Research OS" body="Tasks execute through artifacts, agent traces, reports, gates and submissions." />
          <LayerCard title="Layer 2" subtitle="MLEvolve-style Search" body="Round4 consumes retrospective memory and promotes only strict improvements; weaker branches preserve parent best." />
          <LayerCard title="Layer 3" subtitle="XCIENTIST-style Harness" body="Validation contracts and claim audits limit what the report can claim." />
        </div>

        <FigureManifestPanel bundle={bundle} />

        <div className="rounded-md border border-amber-100 bg-amber-50 p-3 text-xs leading-5 text-amber-900">
          <div className="font-bold">Claim boundary</div>
          <div className="mt-1">
            Local proxy evidence only. No official Kaggle leaderboard score, GPU/HPC success claim, MLE-Bench medal rate, or MLEvolve parity claim is made here.
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={generate} disabled={busy} data-testid="generate-paper-evidence-bundle">
            <RefreshCcw className="h-4 w-4" />
            {busy ? "Generating..." : "Generate Bundle"}
          </Button>
          <span className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-600">
            <FileText className="mr-1 inline h-3.5 w-3.5" />
            {bundlePath}
          </span>
          <span className="text-xs text-slate-500">{message}</span>
        </div>
      </CardContent>
    </Card>
  );
}

function EvidenceMetric({
  icon: Icon,
  label,
  value,
  tone
}: {
  icon: typeof CheckCircle2;
  label: string;
  value: unknown;
  tone: StatusTone;
}) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3">
      <div className="flex items-center justify-between">
        <Icon className="h-4 w-4 text-slate-500" />
        <StatusBadge tone={tone}>{tone}</StatusBadge>
      </div>
      <div className="mt-3 text-[11px] font-bold uppercase text-slate-500">{label}</div>
      <div className="mt-1 text-xl font-bold text-slate-950">{String(value)}</div>
    </div>
  );
}

function LayerCard({ title, subtitle, body }: { title: string; subtitle: string; body: string }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4">
      <div className="text-xs font-bold uppercase text-blue-700">{title}</div>
      <div className="mt-1 text-sm font-bold text-slate-950">{subtitle}</div>
      <p className="mt-2 text-xs leading-5 text-slate-600">{body}</p>
    </div>
  );
}

function SteadyImprovementPanel({ bundle }: { bundle: PaperEvidenceBundleType | null }) {
  const protocol = bundle?.steady_improvement_protocol;
  const certificate = protocol?.monotonicity_certificate;
  const branches = bundle?.round4_search_plan?.branches ?? [];
  return (
    <div className="rounded-md border border-emerald-100 bg-emerald-50/60 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-bold text-emerald-950">
            <TrendingUp className="h-4 w-4" />
            Steady Improvement Protocol
          </div>
          <p className="mt-1 max-w-4xl text-xs leading-5 text-emerald-900">
            {protocol?.paper_core_claim ?? "Best-so-far protection and failure-to-memory conversion are loaded after generating the protocol."}
          </p>
        </div>
        <StatusBadge tone={certificate?.all_tasks_best_so_far_never_regressed ? "green" : "amber"}>
          {certificate?.all_tasks_best_so_far_never_regressed ? "monotonic" : "pending"}
        </StatusBadge>
      </div>
      <div className="mt-3 grid gap-3 md:grid-cols-4">
        <EvidenceMetric icon={ShieldCheck} label="No regression" value={String(certificate?.all_tasks_best_so_far_never_regressed ?? "n/a")} tone="green" />
        <EvidenceMetric icon={CheckCircle2} label="Improved tasks" value={certificate?.tasks_final_improved_over_round1 ?? "n/a"} tone="green" />
        <EvidenceMetric icon={GitBranch} label="Round4 branches" value={branches.length || "n/a"} tone="blue" />
        <EvidenceMetric icon={FileText} label="Next plan" value={protocol?.next_round_plan ?? "pending"} tone="amber" />
      </div>
      <VerificationGatePanel bundle={bundle} />
      {branches.length ? (
        <div className="mt-3 grid gap-2 lg:grid-cols-3">
          {branches.map((branch) => (
            <div key={branch.task_id} className="rounded-md border border-emerald-100 bg-white p-3 text-xs">
              <div className="flex items-center justify-between gap-2">
                <div className="font-bold text-slate-950">{branch.task_id}</div>
                <StatusBadge tone="blue">{branch.code_generation_mode ?? "mode"}</StatusBadge>
              </div>
              <div className="mt-2 text-[11px] font-bold uppercase text-slate-500">{branch.search_stage ?? "search"}</div>
              <div className="mt-1 font-semibold text-slate-800">{branch.branch_type ?? "branch"}</div>
              <p className="mt-2 leading-5 text-slate-600">{branch.hypothesis ?? "Hypothesis pending."}</p>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function FigureManifestPanel({ bundle }: { bundle: PaperEvidenceBundleType | null }) {
  const figures = bundle?.figure_manifest_payload?.figures ?? [];
  if (!figures.length) return null;
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div>
          <div className="text-sm font-bold text-slate-950">Paper Figures</div>
          <div className="mt-1 text-xs text-slate-500">Figure manifest is attached to the same evidence bundle.</div>
        </div>
        <StatusBadge tone="green">figures ready</StatusBadge>
      </div>
      <div className="grid gap-3 lg:grid-cols-3">
        {figures.map((figure) => (
          <div key={figure.figure_id} className="rounded-md border border-slate-200 bg-slate-50 p-3 text-xs">
            <div className="font-bold text-slate-950">{figure.title}</div>
            <p className="mt-2 min-h-14 leading-5 text-slate-600">{figure.caption}</p>
            <div className="mt-2 truncate rounded border border-slate-200 bg-white px-2 py-1 font-mono text-[11px] text-slate-500">
              {figure.paths?.png ?? figure.paths?.svg ?? "figure path pending"}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function VerificationGatePanel({ bundle }: { bundle: PaperEvidenceBundleType | null }) {
  const verification = bundle?.steady_improvement_verification;
  const checks = verification?.checks ?? [];
  if (!verification) return null;
  const passed = verification.status === "passed";
  return (
    <div className="mt-3 rounded-md border border-blue-100 bg-white p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="text-xs font-bold uppercase text-blue-700">Machine Verification Gate</div>
          <p className="mt-1 text-xs leading-5 text-slate-600">{verification.claim_verified ?? "Verification artifact loaded."}</p>
        </div>
        <StatusBadge tone={passed ? "green" : "red"}>{verification.status ?? "pending"}</StatusBadge>
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-3">
        {checks.slice(0, 9).map((check) => (
          <div key={check.id} className="rounded border border-slate-200 bg-slate-50 px-2 py-2 text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-bold text-slate-900">{check.id}</span>
              <StatusBadge tone={check.passed ? "green" : "red"}>{check.passed ? "pass" : "fail"}</StatusBadge>
            </div>
            <p className="mt-1 leading-4 text-slate-500">{check.description}</p>
          </div>
        ))}
      </div>
      <div className="mt-2 truncate rounded border border-slate-200 bg-slate-50 px-2 py-1 font-mono text-[11px] text-slate-500">
        {verification.report_path ?? "workspace/three_layer_steady_improvement_verification_20260623.json"}
      </div>
    </div>
  );
}
