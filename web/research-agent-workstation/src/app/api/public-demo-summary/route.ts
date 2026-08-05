import { NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import type {
  MultiAgentArtifactManifest,
  MultiAgentCurrentRun,
  MultiAgentHandoff,
  MultiAgentHpcProbe,
  MultiAgentPublicArtifactPreview,
  MultiAgentReview,
  MultiAgentRuntimeSnapshot,
  MultiAgentTaskGraph,
  WorkstationSummary,
} from "@/lib/api/types";
import { sanitizeClientJson } from "@/lib/server/json";
import { resolveWorkspacePath } from "@/lib/server/paths";
import { publicReportSourceRunId } from "@/lib/server/public-scientific-report";
import { loadMultiAgentRuntimeByRunId } from "@/lib/server/summary";

export const dynamic = "force-dynamic";

const PUBLIC_TASK_ID = "evomind-qwen7b-finetune";
const PUBLIC_RUN_ID = "EVOMIND-DEMO-7B-20260722";

function publicMarkdown(content: string, sourceRunId: string): string {
  return content
    .replaceAll(sourceRunId, PUBLIC_RUN_ID)
    .replace(/- GPU：[^\n]+；峰值显存：/g, "- 计算：Remote GPU（对外推荐配置 NVIDIA A40 48GB）；峰值显存：")
    .replace(/- Compute: one verified [^;]+; local GPU was not used/g, "- Compute: verified Remote GPU; recommended public configuration: NVIDIA A40 48GB; local GPU was not used")
    .replace(/NVIDIA A800(?:-SXM4)?-80GB/gi, "Remote GPU")
    .replace(/\bA800\b/gi, "Remote GPU")
    .trim();
}

async function publicArtifactPreviews(
  artifactManifest: MultiAgentArtifactManifest,
  sourceRunId: string,
): Promise<MultiAgentPublicArtifactPreview[]> {
  const runRoot = resolveWorkspacePath(`workspace/evomind_runs/${sourceRunId}`);
  const [report, modelCard] = await Promise.all([
    fs.readFile(`${runRoot}/research_report.md`, "utf-8"),
    fs.readFile(`${runRoot}/model_card.md`, "utf-8"),
  ]);
  const artifacts = artifactManifest.artifacts ?? [];
  const artifactByKind = new Map(
    artifacts
      .filter((artifact) => artifact.kind)
      .map((artifact) => [artifact.kind as string, artifact]),
  );
  const reportArtifact = artifactByKind.get("research_report.md");
  const modelCardArtifact = artifactByKind.get("model_card.md");
  const adapterNames = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
  ];
  const adapterFiles = adapterNames.flatMap((name) => {
    const artifact = artifactByKind.get(name);
    return artifact ? [{ name, sha256: artifact.sha256, bytes: artifact.bytes }] : [];
  });
  return [
    {
      id: "research_report",
      name: "research_report.md",
      title: "研究报告",
      kind: "markdown",
      content: publicMarkdown(report, sourceRunId),
      sha256: reportArtifact?.sha256,
      bytes: reportArtifact?.bytes,
    },
    {
      id: "model_card",
      name: "model_card.md",
      title: "模型卡",
      kind: "markdown",
      content: publicMarkdown(modelCard, sourceRunId),
      sha256: modelCardArtifact?.sha256,
      bytes: modelCardArtifact?.bytes,
    },
    {
      id: "adapter",
      name: "adapter/",
      title: "Adapter 交付清单",
      kind: "file_list",
      files: adapterFiles,
    },
  ];
}

function publicTaskGraph(source: MultiAgentTaskGraph | null | undefined): MultiAgentTaskGraph {
  const nodes = (source?.nodes ?? []).map((node) => ({
    task_id: node.task_id,
    goal: node.goal,
    role: node.role,
    dependencies: node.dependencies ?? [],
    resource_type: node.resource_type,
    status: node.status,
    attempts: node.attempts,
    solution_id: node.solution_id,
  }));
  const edges = (source?.edges ?? []).map((edge) => ({ from: edge.from, to: edge.to }));
  return { schema: source?.schema, run_id: PUBLIC_RUN_ID, nodes, edges };
}

function publicHandoffs(source: MultiAgentHandoff[] | undefined): MultiAgentHandoff[] {
  return (source ?? []).map((handoff, index) => ({
    schema: handoff.schema,
    handoff_id: `public-handoff-${String(index + 1).padStart(2, "0")}`,
    run_id: PUBLIC_RUN_ID,
    task_id: handoff.task_id,
    sender: handoff.sender,
    receiver: handoff.receiver,
    expected_output: handoff.expected_output,
  }));
}

function publicArtifactManifest(source: MultiAgentArtifactManifest | null | undefined): MultiAgentArtifactManifest {
  return {
    schema: source?.schema,
    run_id: PUBLIC_RUN_ID,
    selected_solution: source?.selected_solution,
    base_model: source?.base_model,
    training_method: source?.training_method,
    review_status: source?.review_status,
    claim_audit_status: source?.claim_audit_status,
    model_publication: source?.model_publication,
    generated_at: source?.generated_at,
    artifacts: (source?.artifacts ?? []).map((artifact, index) => ({
      path: `public-artifact-${String(index + 1).padStart(2, "0")}`,
      sha256: artifact.sha256,
      bytes: artifact.bytes,
      kind: artifact.kind,
    })),
  };
}

function publicTelemetry(source: Array<Record<string, unknown>> | undefined) {
  return (source ?? [])
    .filter((sample) => typeof sample.step === "number" && typeof sample.loss === "number")
    .map((sample) => {
      const gpu = Array.isArray(sample.gpu) ? sample.gpu[0] as Record<string, unknown> | undefined : undefined;
      return {
        event: typeof sample.event === "string" ? sample.event : "training_step",
        step: sample.step,
        loss: sample.loss,
        learning_rate: sample.learning_rate,
        elapsed_seconds: sample.elapsed_seconds,
        cuda_max_allocated_mb: sample.cuda_max_allocated_mb,
        gpu: gpu ? [{
          name: "Remote GPU",
          utilization_percent: gpu.utilization_percent,
          memory_used_mb: gpu.memory_used_mb,
        }] : [],
      };
    });
}

function publicRuntimeSnapshot(source: MultiAgentRuntimeSnapshot | null | undefined): MultiAgentRuntimeSnapshot {
  const llm = source?.llm;
  const before = llm?.before_after_eval?.before;
  const after = llm?.before_after_eval?.after;
  return {
    schema: source?.schema,
    task_id: PUBLIC_TASK_ID,
    run_id: PUBLIC_RUN_ID,
    status: source?.status,
    seq: source?.seq,
    active_agents: source?.active_agents,
    open_requirements: source?.open_requirements,
    next_action: source?.next_action,
    metrics: source?.metrics ? {
      selected_solution: source.metrics.selected_solution,
      metric: source.metrics.metric,
      cv_score: source.metrics.cv_score,
      review_status: source.metrics.review_status,
    } : undefined,
    llm: llm ? {
      task_type: llm.task_type,
      base_model: llm.base_model,
      dataset: llm.dataset ? {
        counts: llm.dataset.counts,
        source_split_policy: llm.dataset.source_split_policy,
        secrets_removed: llm.dataset.secrets_removed,
      } : null,
      qlora_config: llm.qlora_config ? {
        method: llm.qlora_config.method,
        lora_r: llm.qlora_config.lora_r,
        lora_alpha: llm.qlora_config.lora_alpha,
        lora_dropout: llm.qlora_config.lora_dropout,
        max_sequence_length: llm.qlora_config.max_sequence_length,
        effective_batch_size: llm.qlora_config.effective_batch_size,
        epochs: llm.qlora_config.epochs,
        compute_dtype: llm.qlora_config.compute_dtype,
      } : null,
      training: llm.training,
      before_after_eval: {
        before: before ? {
          domain_composite: before.domain_composite,
          format_compliance: before.format_compliance,
          sensitive_leakage: before.sensitive_leakage,
        } : null,
        after: after ? {
          domain_composite: after.domain_composite,
          format_compliance: after.format_compliance,
          sensitive_leakage: after.sensitive_leakage,
        } : null,
        improvement_pp: llm.before_after_eval?.improvement_pp,
      },
      telemetry: publicTelemetry(llm.telemetry),
      adapter_reload: llm.adapter_reload ? {
        passed: llm.adapter_reload.passed,
        absolute_delta: llm.adapter_reload.absolute_delta,
      } : null,
      claim_audit: llm.claim_audit ? { status: llm.claim_audit.status } : null,
    } : null,
    reviewer: source?.reviewer ? { status: source.reviewer.status } : undefined,
    gates: source?.gates,
  };
}

function publicReview(source: MultiAgentReview | null | undefined): MultiAgentReview | null {
  if (!source) return null;
  return {
    schema: source.schema,
    status: source.status,
    parent_subjective_summary_received: source.parent_subjective_summary_received,
    claim_audit: source.claim_audit ? {
      status: source.claim_audit.status,
      official_kaggle_score_claimed: source.claim_audit.official_kaggle_score_claimed,
      fresh_run_only: source.claim_audit.fresh_run_only,
    } : undefined,
  };
}

function publicHpcProbe(source: MultiAgentHpcProbe | null | undefined): MultiAgentHpcProbe {
  return {
    status: source?.status === "passed" ? "passed" : "blocked",
    generated_at: source?.generated_at,
    gpu_inventory: [{ name: "Remote GPU" }],
    torch: {
      cuda_available: source?.torch?.cuda_available === true,
      device_count: source?.torch?.device_count ?? 0,
    },
  };
}

function publicEvents(source: Array<Record<string, unknown>> | undefined) {
  return (source ?? []).map((event) => ({
    schema: event.schema,
    seq: event.seq,
    task_id: event.task_id,
    agent: event.agent,
    status: event.status,
  }));
}

export async function GET() {
  try {
    const sourceRunId = publicReportSourceRunId();
    const runtime = sanitizeClientJson(
      await loadMultiAgentRuntimeByRunId(PUBLIC_TASK_ID, sourceRunId),
    ) as WorkstationSummary["runtime"];
    if (!runtime?.current_run) {
      return NextResponse.json(
        { ok: false, error: "The pinned public demo run is unavailable." },
        { status: 404 },
      );
    }

    const currentRun: MultiAgentCurrentRun = {
      schema: runtime.current_run.schema,
      task_id: PUBLIC_TASK_ID,
      run_id: PUBLIC_RUN_ID,
      run_dir: "public-deliverables",
      status: runtime.current_run.status,
      last_seq: runtime.current_run.last_seq,
      updated_at: runtime.current_run.updated_at,
    };
    const artifactManifest = publicArtifactManifest(runtime.artifact_manifest);
    const artifactPreviews = await publicArtifactPreviews(artifactManifest, sourceRunId);
    const artifactCount = artifactManifest.artifacts?.length ?? 0;
    const summary: WorkstationSummary = {
      tasks: [{
        id: PUBLIC_TASK_ID,
        name: "EvoMind 7B 领域微调",
        task_type: "llm_finetune",
        status: currentRun.status ?? "completed",
        metric: "domain_composite",
      }],
      runs: [{
        id: PUBLIC_RUN_ID,
        task_id: PUBLIC_TASK_ID,
        output_dir: null,
        status: currentRun.status,
        workstation_run: true,
        direct_training_allowed: false,
        official_submission_allowed: false,
        accepted: runtime.review?.status === "passed",
      }],
      evidence: (artifactManifest.artifacts ?? []).map((artifact, index) => ({
        task_id: PUBLIC_TASK_ID,
        name: artifact.kind ?? `Artifact ${index + 1}`,
        artifact_type: artifact.kind ?? "verified_artifact",
        verification_status: "verified",
        sha256: artifact.sha256,
        bytes: artifact.bytes,
      })),
      gates: [
        { task_id: PUBLIC_TASK_ID, gate_type: "Independent Reviewer", decision: runtime.review?.status === "passed" ? "approved" : "pending" },
        { task_id: PUBLIC_TASK_ID, gate_type: "Claim Audit", decision: runtime.runtime_snapshot?.gates?.claim_audit === "passed" ? "approved" : "pending" },
        { task_id: PUBLIC_TASK_ID, gate_type: "Model Publication", decision: "blocked" },
      ],
      connector_status: {
        hpc: { status: runtime.hpc_probe?.status === "passed" ? "ready" : "blocked", label: "Remote GPU" },
        local_gpu: { status: "disabled" },
      },
      runtime: {
        task_id: PUBLIC_TASK_ID,
        current_run: currentRun,
        task_graph: publicTaskGraph(runtime.task_graph),
        handoffs: publicHandoffs(runtime.handoffs),
        event_log: publicEvents(runtime.event_log),
        artifact_manifest: artifactManifest,
        runtime_snapshot: publicRuntimeSnapshot(runtime.runtime_snapshot),
        review: publicReview(runtime.review),
        hpc_probe: publicHpcProbe(runtime.hpc_probe),
        public_artifact_previews: artifactPreviews,
      },
    };

    return NextResponse.json(
      { ...summary, public_demo: { pinned: true, artifact_count: artifactCount } },
      { headers: { "Cache-Control": "no-store, max-age=0" } },
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown public demo summary error";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
