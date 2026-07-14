import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const require = createRequire(import.meta.url);
const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const fixtureRoot = path.join(webRoot, ".hpc-gate-adversarial-fixture");

function sha256(text) {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

function normalizedRelative(value) {
  return String(value).replaceAll("\\", "/").replace(/^\/+/, "");
}

function loadTypeScriptModule(relativePath, mocks) {
  const filename = path.join(webRoot, relativePath);
  const source = fs.readFileSync(filename, "utf8");
  const compiled = ts.transpileModule(source, {
    fileName: filename,
    compilerOptions: {
      esModuleInterop: true,
      module: ts.ModuleKind.CommonJS,
      moduleResolution: ts.ModuleResolutionKind.NodeJs,
      target: ts.ScriptTarget.ES2022
    },
    reportDiagnostics: true
  });
  const errors = (compiled.diagnostics ?? []).filter(
    (diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error
  );
  assert.equal(errors.length, 0, `TypeScript transpilation failed for ${relativePath}`);

  const module = { exports: {} };
  const localRequire = (specifier) => {
    if (Object.hasOwn(mocks, specifier)) return mocks[specifier];
    if (specifier.startsWith("node:")) return require(specifier);
    throw new Error(`Unexpected dependency ${specifier} while loading ${relativePath}`);
  };
  const wrapper = new vm.Script(
    `(function (require, module, exports, __filename, __dirname) { ${compiled.outputText}\n})`,
    { filename }
  ).runInThisContext();
  wrapper(localRequire, module, module.exports, filename, path.dirname(filename));
  return module.exports;
}

function makeHarness() {
  const files = new Map();
  const actions = [];
  const resolveWorkspacePath = (relativePath) => {
    const resolved = path.resolve(fixtureRoot, normalizedRelative(relativePath));
    const boundary = path.relative(fixtureRoot, resolved);
    if (boundary.startsWith("..") || path.isAbsolute(boundary)) {
      throw new Error("fixture path escaped root");
    }
    return resolved;
  };
  const toRelativePath = (absolutePath) => {
    const relative = path.relative(fixtureRoot, absolutePath);
    if (relative.startsWith("..") || path.isAbsolute(relative)) return null;
    return normalizedRelative(relative);
  };
  const readStableRegularTextFile = async (absolutePath, options) => {
    const allowedBoundary = path.relative(options.allowedRoot, absolutePath);
    if (allowedBoundary.startsWith("..") || path.isAbsolute(allowedBoundary)) {
      throw new Error("artifact escaped allowed root");
    }
    const relative = toRelativePath(absolutePath);
    if (!relative || !files.has(relative)) throw new Error("artifact missing");
    const text = files.get(relative);
    if (Buffer.byteLength(text, "utf8") > options.maxBytes) throw new Error("artifact too large");
    return { text, sha256: sha256(text) };
  };
  const prisma = {
    actionLog: {
      findMany: async ({ where, take }) => actions
        .filter((record) => record.taskId === where.taskId && record.action === where.action)
        .sort((left, right) => (
          right.createdAt.getTime() - left.createdAt.getTime()
          || right.id.localeCompare(left.id)
        ))
        .slice(0, take),
      findUnique: async ({ where }) => actions.find((record) => record.id === where.id) ?? null
    }
  };
  const decodeJson = (value) => {
    if (typeof value !== "string") return null;
    try {
      return JSON.parse(value);
    } catch {
      return null;
    }
  };
  const codeQuality = loadTypeScriptModule("src/lib/server/code-quality-gate.ts", {
    "@/lib/db": { prisma },
    "@/lib/server/json": { decodeJson },
    "@/lib/server/paths": { resolveWorkspacePath, toRelativePath },
    "@/lib/server/stable-file": { readStableRegularTextFile }
  });
  const hpc = loadTypeScriptModule("src/lib/server/hpc-execution-gate.ts", {
    "@/lib/server/code-quality-gate": codeQuality,
    "@/lib/server/json": { decodeJson },
    "@/lib/server/paths": { resolveWorkspacePath },
    "@/lib/server/stable-file": { readStableRegularTextFile }
  });

  function addReview({ id, taskId = "task-alpha", createdAt, passed = true, suffix = id }) {
    if (!passed) {
      actions.push({
        id,
        taskId,
        action: "review_agent_patch",
        createdAt,
        metadataJson: JSON.stringify({ overall_status: "failed" })
      });
      return null;
    }
    const prefix = `workspace/tasks/${taskId}/code/patches`;
    const patchPath = `${prefix}/candidate_${suffix}.diff`;
    const patchText = `diff --git a/src/value.py b/src/value.py\n+value = ${JSON.stringify(suffix)}\n`;
    const patchSha256 = sha256(patchText);
    const qualityArtifactPath = `${prefix}/code_quality_check_${suffix}.json`;
    const qualityPayload = {
      task_id: taskId,
      overall_status: "passed",
      original_data_check: "passed",
      command_risk_check: "passed",
      credential_check: "passed",
      human_gate_required: true,
      patch_path: patchPath,
      patch_sha256: patchSha256,
      affected_files: ["src/value.py"],
      patch_python_syntax_check: { status: "passed" }
    };
    const qualityText = `${JSON.stringify(qualityPayload)}\n`;
    const qualityArtifactSha256 = sha256(qualityText);
    files.set(patchPath, patchText);
    files.set(qualityArtifactPath, qualityText);
    actions.push({
      id,
      taskId,
      action: "review_agent_patch",
      createdAt,
      metadataJson: JSON.stringify({
        overall_status: "passed",
        quality_artifact: qualityArtifactPath,
        quality_artifact_sha256: qualityArtifactSha256,
        patch_sha256: patchSha256
      })
    });
    return {
      actionId: id,
      qualityArtifactPath,
      qualityArtifactSha256,
      patchPath,
      patchSha256
    };
  }

  function makePendingGate(dependency, { taskId = "task-alpha", runId = "run-1", template = "template-a" } = {}) {
    const manifestPath = `workspace/workstation_runs/${taskId}/${runId}/hpc_execution_gate_manifest.json`;
    const manifest = {
      schema: "academic_research_os.hpc_execution_gate.v1",
      task_id: taskId,
      workstation_run_id: runId,
      requested_template: template,
      status: "pending_approval",
      resource_policy: "whitelist_templates_only",
      remote_training_allowed: false,
      approval_required_before: "POST /api/gpu/jobs",
      code_agent_dependency: {
        status: "code_quality_passed",
        quality_gate_path: dependency.qualityArtifactPath,
        quality_gate_sha256: dependency.qualityArtifactSha256,
        quality_gate_action_id: dependency.actionId,
        patch_path: dependency.patchPath,
        patch_sha256: dependency.patchSha256,
        required_before_training: true
      }
    };
    const manifestText = `${JSON.stringify(manifest)}\n`;
    const manifestSha256 = sha256(manifestText);
    files.set(manifestPath, manifestText);
    return {
      id: `${runId}_hpc_execution_approval`,
      taskId,
      runId,
      gateType: "hpc_execution_approval",
      decision: "pending",
      reviewer: null,
      decidedAt: null,
      evidenceJson: JSON.stringify({
        manifest_path: manifestPath,
        manifest_sha256: manifestSha256,
        requested_template: template,
        code_agent_dependency: {
          quality_gate_path: dependency.qualityArtifactPath,
          quality_gate_sha256: dependency.qualityArtifactSha256,
          quality_gate_action_id: dependency.actionId,
          patch_path: dependency.patchPath,
          patch_sha256: dependency.patchSha256
        }
      })
    };
  }

  return { actions, addReview, codeQuality, files, hpc, makePendingGate };
}

let scenarios = 0;

{
  const harness = makeHarness();
  const dependency = harness.addReview({ id: "review-a", createdAt: new Date("2026-07-14T00:00:00.001Z") });
  assert.equal((await harness.codeQuality.latestPassedCodeQualityGate("task-alpha")).actionId, "review-a");
  assert.equal((await harness.codeQuality.passedCodeQualityGateByBinding("task-alpha", dependency)).actionId, "review-a");
  assert.equal(await harness.codeQuality.passedCodeQualityGateByBinding("task-beta", dependency), null);
  scenarios += 2;

  const qualityText = harness.files.get(dependency.qualityArtifactPath);
  harness.files.set(dependency.qualityArtifactPath, "tampered quality artifact\n");
  assert.equal(await harness.codeQuality.passedCodeQualityGateByBinding("task-alpha", dependency), null);
  harness.files.set(dependency.qualityArtifactPath, qualityText);
  scenarios += 1;

  harness.files.set(dependency.patchPath, "tampered patch\n");
  assert.equal(await harness.codeQuality.passedCodeQualityGateByBinding("task-alpha", dependency), null);
  scenarios += 1;
}

{
  const harness = makeHarness();
  harness.addReview({ id: "review-old", createdAt: new Date("2026-07-14T00:00:00.001Z") });
  harness.addReview({ id: "review-new-failed", createdAt: new Date("2026-07-14T00:00:00.002Z"), passed: false });
  assert.equal(await harness.codeQuality.latestPassedCodeQualityGate("task-alpha"), null);
  scenarios += 1;
}

{
  const harness = makeHarness();
  const timestamp = new Date("2026-07-14T00:00:00.001Z");
  harness.addReview({ id: "review-a", createdAt: timestamp });
  harness.addReview({ id: "review-b", createdAt: timestamp });
  assert.equal(await harness.codeQuality.latestPassedCodeQualityGate("task-alpha"), null);
  scenarios += 1;
}

{
  const harness = makeHarness();
  const dependency = harness.addReview({ id: "review-a", createdAt: new Date("2026-07-14T00:00:00.001Z") });
  const gate = harness.makePendingGate(dependency);
  const pending = await harness.hpc.validateHpcExecutionGate(gate, {
    taskId: "task-alpha",
    runId: "run-1",
    template: "template-a",
    requireApproved: false
  });
  assert.equal(pending.ok, true, JSON.stringify(pending.reasons));
  assert.deepEqual(
    (await harness.hpc.validateHpcExecutionGate(null, {
      taskId: "task-alpha",
      runId: "run-1",
      template: "template-a"
    })).reasons,
    ["hpc_gate_missing"]
  );
  const wrongTemplate = await harness.hpc.validateHpcExecutionGate(gate, {
    taskId: "task-alpha",
    runId: "run-1",
    template: "template-b",
    requireApproved: false
  });
  assert.equal(wrongTemplate.ok, false);
  assert.equal(wrongTemplate.reasons.includes("hpc_gate_template_mismatch"), true);
  scenarios += 3;

  const decidedAt = new Date("2026-07-14T00:01:00.000Z");
  const approvedEvidence = harness.hpc.buildHpcApprovalEvidence(gate, {
    reviewer: "Research Admin",
    reason: "Approved after bound evidence review.",
    artifactPath: "workspace/gates/approval.json",
    decidedAt
  });
  const approvedGate = {
    ...gate,
    decision: "approved",
    reviewer: "Research Admin",
    decidedAt,
    evidenceJson: JSON.stringify(approvedEvidence)
  };
  const approved = await harness.hpc.validateHpcExecutionGate(approvedGate, {
    taskId: "task-alpha",
    runId: "run-1",
    template: "template-a"
  });
  assert.equal(approved.ok, true, JSON.stringify(approved.reasons));
  scenarios += 1;

  const overwritten = {
    ...approvedGate,
    evidenceJson: JSON.stringify({
      ...approvedEvidence,
      approval: { ...approvedEvidence.approval, reviewer: "Different Reviewer" }
    })
  };
  const overwrittenResult = await harness.hpc.validateHpcExecutionGate(overwritten, {
    taskId: "task-alpha",
    runId: "run-1",
    template: "template-a"
  });
  assert.equal(overwrittenResult.ok, false);
  assert.equal(overwrittenResult.reasons.includes("hpc_gate_approval_binding_mismatch"), true);
  assert.throws(
    () => harness.hpc.buildHpcApprovalEvidence({ ...gate, decision: "rejected" }, {
      reviewer: "Research Admin",
      reason: "Invalid transition.",
      decidedAt
    }),
    /hpc_gate_not_pending/
  );
  scenarios += 2;
}

{
  const harness = makeHarness();
  const dependency = harness.addReview({ id: "review-a", createdAt: new Date("2026-07-14T00:00:00.001Z") });
  const gate = harness.makePendingGate(dependency);
  const evidence = JSON.parse(gate.evidenceJson);
  harness.files.set(evidence.manifest_path, "tampered manifest\n");
  const result = await harness.hpc.validateHpcExecutionGate(gate, {
    taskId: "task-alpha",
    runId: "run-1",
    template: "template-a",
    requireApproved: false
  });
  assert.equal(result.ok, false);
  assert.equal(result.reasons.includes("hpc_gate_manifest_hash_mismatch"), true);
  scenarios += 1;
}

{
  const harness = makeHarness();
  const dependency = harness.addReview({ id: "review-a", createdAt: new Date("2026-07-14T00:00:00.001Z") });
  const gate = harness.makePendingGate(dependency);
  const evidence = JSON.parse(gate.evidenceJson);
  const forgedManifest = JSON.parse(harness.files.get(evidence.manifest_path));
  forgedManifest.schema = "forged.schema.v9";
  forgedManifest.status = "approved_without_gate";
  forgedManifest.resource_policy = "arbitrary_commands";
  forgedManifest.remote_training_allowed = true;
  const forgedText = `${JSON.stringify(forgedManifest)}\n`;
  harness.files.set(evidence.manifest_path, forgedText);
  gate.evidenceJson = JSON.stringify({ ...evidence, manifest_sha256: sha256(forgedText) });
  const result = await harness.hpc.validateHpcExecutionGate(gate, {
    taskId: "task-alpha",
    runId: "run-1",
    template: "template-a",
    requireApproved: false
  });
  assert.equal(result.ok, false);
  for (const reason of [
    "hpc_gate_manifest_schema_mismatch",
    "hpc_gate_manifest_status_mismatch",
    "hpc_gate_manifest_resource_policy_mismatch",
    "hpc_gate_manifest_remote_training_policy_mismatch"
  ]) {
    assert.equal(result.reasons.includes(reason), true, `${reason} was not enforced`);
  }
  scenarios += 1;
}

{
  const harness = makeHarness();
  const dependency = harness.addReview({ id: "review-old", createdAt: new Date("2026-07-14T00:00:00.001Z") });
  const gate = harness.makePendingGate(dependency);
  harness.addReview({ id: "review-new", createdAt: new Date("2026-07-14T00:00:00.002Z") });
  const result = await harness.hpc.validateHpcExecutionGate(gate, {
    taskId: "task-alpha",
    runId: "run-1",
    template: "template-a",
    requireApproved: false
  });
  assert.equal(result.ok, false);
  assert.equal(result.reasons.includes("hpc_gate_code_dependency_stale"), true);
  scenarios += 1;
}

{
  const harness = makeHarness();
  const dependency = harness.addReview({ id: "review-old", createdAt: new Date("2026-07-14T00:00:00.001Z") });
  const gate = harness.makePendingGate(dependency);
  harness.addReview({ id: "review-new-failed", createdAt: new Date("2026-07-14T00:00:00.002Z"), passed: false });
  const result = await harness.hpc.validateHpcExecutionGate(gate, {
    taskId: "task-alpha",
    runId: "run-1",
    template: "template-a",
    requireApproved: false
  });
  assert.equal(result.ok, false);
  assert.equal(result.reasons.includes("hpc_gate_code_dependency_stale"), true);
  scenarios += 1;
}

console.log(`hpc gate adversarial checks passed (${scenarios} scenarios)`);
