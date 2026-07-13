import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { createClaudeSession, probeDeepSeekCodeCache, sessionToDraftPayload } from "@/lib/server/claude-agent-sessions";
import { logAction } from "@/lib/server/actions";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { latestExperimentPath, normalizeTaskId, readJsonFile, resolveWorkspacePath, toRelativePath } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

function safeAgent(value: unknown) {
  const agent = String(value ?? "claude_code").toLowerCase();
  return ["claude_code", "deepseek_code_agent", "codex", "local_template"].includes(agent) ? agent : "local_template";
}

function metricLines(metrics: Record<string, unknown> | null | undefined) {
  const entries = Object.entries(metrics ?? {});
  if (!entries.length) return ["# Latest metrics are not available yet."];
  return entries.map(([key, value]) => `# ${key}: ${typeof value === "number" ? value.toFixed(6) : String(value)}`);
}

function buildDraftCode(taskId: string, sourceAgent: string, metrics: Record<string, unknown> | null | undefined) {
  const isRegression = taskId.includes("house");
  const metricName = isRegression ? "RMSLE" : "accuracy";
  return [
    `"""Agent draft generated for ${taskId}.`,
    "",
    "This file is intentionally written as a reviewable draft. It is not applied",
    "to the production pipeline until the Code Quality Gate and Manual Gate pass.",
    `Source agent bridge: ${sourceAgent}`,
    `Primary metric: ${metricName}`,
    `Generated at: ${new Date().toISOString()}`,
    `"""`,
    "",
    "from __future__ import annotations",
    "",
    "from dataclasses import dataclass",
    "from pathlib import Path",
    "from typing import Any",
    "",
    "import numpy as np",
    "import pandas as pd",
    "from sklearn.compose import ColumnTransformer",
    "from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier",
    "from sklearn.impute import SimpleImputer",
    "from sklearn.metrics import accuracy_score, mean_squared_log_error",
    "from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split",
    "from sklearn.pipeline import Pipeline",
    "from sklearn.preprocessing import OneHotEncoder",
    "",
    "",
    "@dataclass(frozen=True)",
    "class AgentRunConfig:",
    "    task_id: str",
    "    target: str",
    "    random_state: int = 42",
    "    n_splits: int = 5",
    "",
    "",
    "def build_preprocessor(frame: pd.DataFrame, target: str) -> ColumnTransformer:",
    "    features = frame.drop(columns=[target], errors='ignore')",
    "    numeric_features = features.select_dtypes(include=['number']).columns.tolist()",
    "    categorical_features = [column for column in features.columns if column not in numeric_features]",
    "    return ColumnTransformer(",
    "        transformers=[",
    "            ('num', SimpleImputer(strategy='median'), numeric_features),",
    "            ('cat', Pipeline([",
    "                ('imputer', SimpleImputer(strategy='most_frequent')),",
    "                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),",
    "            ]), categorical_features),",
    "        ],",
    "        remainder='drop',",
    "        verbose_feature_names_out=False,",
    "    )",
    "",
    "",
    "def build_model(config: AgentRunConfig) -> Pipeline:",
    "    if config.task_id == 'house_prices':",
    "        estimator: Any = HistGradientBoostingRegressor(",
    "            learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.08, random_state=config.random_state",
    "        )",
    "    else:",
    "        estimator = RandomForestClassifier(n_estimators=260, max_depth=7, random_state=config.random_state)",
    "    return Pipeline([('preprocess', 'passthrough'), ('model', estimator)])",
    "",
    "",
    "def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:",
    "    clipped = np.maximum(y_pred, 1.0)",
    "    return float(np.sqrt(mean_squared_log_error(y_true, clipped)))",
    "",
    "",
    "def run_agent_draft(train_csv: str | Path, config: AgentRunConfig) -> dict[str, float]:",
    "    frame = pd.read_csv(train_csv)",
    "    y = frame[config.target]",
    "    x = frame.drop(columns=[config.target])",
    "    preprocessor = build_preprocessor(frame, config.target)",
    "    model = build_model(config)",
    "    model.steps[0] = ('preprocess', preprocessor)",
    "    if config.task_id == 'house_prices':",
    "        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=config.random_state)",
    "        model.fit(x_train, y_train)",
    "        prediction = model.predict(x_valid)",
    "        return {'holdout_rmsle': rmsle(y_valid.to_numpy(), prediction)}",
    "    splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)",
    "    scores = cross_val_score(model, x, y, cv=splitter, scoring='accuracy')",
    "    return {'cv_accuracy_mean': float(scores.mean())}",
    "",
    "",
    "if __name__ == '__main__':",
    `    cfg = AgentRunConfig(task_id='${taskId}', target='${isRegression ? "SalePrice" : "Survived"}')`,
    "    print(cfg)",
    "",
    ...metricLines(metrics)
  ].join("\n");
}

function buildPatch(taskId: string, sourceAgent: string, draftPath: string) {
  return [
    "diff --git a/workspace_agent_manifest.md b/workspace_agent_manifest.md",
    "--- a/workspace_agent_manifest.md",
    "+++ b/workspace_agent_manifest.md",
    "@@",
    `+Code agent draft generated for ${taskId}.`,
    `+Source agent: ${sourceAgent}.`,
    `+Draft file: ${draftPath}.`,
    "+Next required action: review the draft, import a concrete diff, then run the local experiment."
  ].join("\n");
}

export async function POST(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  await ensureWorkstationSeeded();
  const { taskId: rawTaskId } = await params;
  const taskId = normalizeTaskId(rawTaskId);
  const body = await request.json().catch(() => ({}));
  const sourceAgent = safeAgent(body.source_agent);
  if (body.cache_probe === true) {
    const probe = await probeDeepSeekCodeCache({
      taskId,
      prompt: typeof body.prompt === "string" ? body.prompt : "Generate a gated research pipeline improvement patch from current task evidence.",
      model: typeof body.model === "string" ? body.model : undefined
    });
    return NextResponse.json(probe);
  }
  if (sourceAgent === "claude_code" || sourceAgent === "deepseek_code_agent") {
    const record = await createClaudeSession({
      taskId,
      prompt: typeof body.prompt === "string" ? body.prompt : "Generate a gated research pipeline improvement patch from current task evidence.",
      model: typeof body.model === "string" ? body.model : undefined,
      maxTurns: Number.isFinite(Number(body.max_turns)) ? Number(body.max_turns) : undefined,
      timeoutSeconds: Number.isFinite(Number(body.timeout_seconds)) ? Number(body.timeout_seconds) : undefined,
      cacheOnly: body.cache_only === true,
      provider: sourceAgent === "deepseek_code_agent" ? "deepseek_code_agent" : undefined
    });
    return NextResponse.json(sessionToDraftPayload(record));
  }

  const latest = await latestExperimentPath(taskId);
  const validation = latest ? await readJsonFile(resolveWorkspacePath(path.join(latest, "validation_gate.json"))) as Record<string, unknown> | null : null;
  const experiment = latest ? await readJsonFile(resolveWorkspacePath(path.join(latest, "experiment_log.json"))) as Record<string, unknown> | null : null;
  const metrics = (validation?.metrics as Record<string, unknown> | undefined) ?? (experiment?.best_metrics as Record<string, unknown> | undefined) ?? null;

  const codeDir = resolveWorkspacePath(path.join("workspace", "tasks", taskId, "code", "current_code"));
  const patchDir = resolveWorkspacePath(path.join("workspace", "tasks", taskId, "code", "patches"));
  await fs.mkdir(codeDir, { recursive: true });
  await fs.mkdir(patchDir, { recursive: true });

  const draftPath = path.join(codeDir, "agent_draft.py");
  const patchPath = path.join(patchDir, "agent_patch.diff");
  const manifestPath = path.join(codeDir, "agent_draft_manifest.json");
  const generatedCode = buildDraftCode(taskId, sourceAgent, metrics);
  const patchDiff = buildPatch(taskId, sourceAgent, toRelativePath(draftPath) ?? "agent_draft.py");

  await fs.writeFile(draftPath, generatedCode, "utf-8");
  await fs.writeFile(patchPath, patchDiff, "utf-8");
  await fs.writeFile(
    manifestPath,
    JSON.stringify(
      {
        task_id: taskId,
        source_agent: sourceAgent,
        cli_status: "bridge_mode",
        latest_experiment: latest,
        draft_path: toRelativePath(draftPath),
        patch_path: toRelativePath(patchPath),
        generated_at: new Date().toISOString(),
        safety_note: "Host Claude/Codex CLI execution is not invoked from the container. Use export/import gate for production safety."
      },
      null,
      2
    ),
    "utf-8"
  );

  await logAction({
    action: "generate_code_agent_draft",
    taskId,
    message: `${sourceAgent.replaceAll("_", " ")} draft generated for IDE review.`,
    artifactPath: toRelativePath(draftPath),
    metadata: {
      source_agent: sourceAgent,
      latest_experiment: latest,
      patch_path: toRelativePath(patchPath),
      manifest_path: toRelativePath(manifestPath),
      cli_status: "bridge_mode"
    }
  });

  return NextResponse.json({
    ok: true,
    task_id: taskId,
    source_agent: sourceAgent,
    draft_path: toRelativePath(draftPath),
    patch_path: toRelativePath(patchPath),
    manifest_path: toRelativePath(manifestPath),
    generated_code: generatedCode,
    patch_diff: patchDiff,
    cli_status: "bridge_mode"
  });
}
