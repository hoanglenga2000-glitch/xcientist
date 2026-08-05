import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";
import ts from "typescript";

type GateDecisionView = {
  canonical: "approved" | "rejected" | "pending" | "unknown";
  raw: string;
  display: string;
  tone: string;
  isPending: boolean;
};

type GatesScreenModule = {
  normalizeGateDecision: (value: unknown) => GateDecisionView;
  gateBelongsToSelectedTask: (gate: unknown, selectedTask: string) => boolean;
  GatesScreen: (props: Record<string, unknown>) => unknown;
};

type RenderedElement = {
  type?: unknown;
  props?: Record<string, unknown>;
  key?: unknown;
};

function loadGatesScreenModule(): GatesScreenModule {
  const filename = fileURLToPath(new URL("./GatesScreen.tsx", import.meta.url));
  const source = readFileSync(filename, "utf8");
  const output = ts.transpileModule(source, {
    fileName: filename,
    compilerOptions: {
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;

  const jsx = (type: unknown, props: Record<string, unknown> | null, key?: unknown): RenderedElement => ({
    type,
    props: props ?? {},
    key,
  });
  const lucideIcons = new Proxy<Record<string, unknown>>({}, {
    get: (_target, property) => String(property),
  });
  const stubs: Record<string, unknown> = {
    "react/jsx-runtime": { jsx, jsxs: jsx, Fragment: "Fragment" },
    "@/lib/utils": { cn: (...values: unknown[]) => values.filter(Boolean).join(" ") },
    "../primitives/Layout": { PageHeader: "PageHeader", Panel: "Panel", MetricTile: "MetricTile" },
    "../primitives/StatusBadge": { StatusBadgeV2: "StatusBadgeV2", StatusDot: "StatusDot" },
    "../localization": {
      t: (locale: string, english: string, chinese: string) => locale === "zh-CN" ? chinese : english,
    },
  };
  const requireStub = (specifier: string): unknown => {
    if (specifier === "lucide-react") return lucideIcons;
    if (Object.prototype.hasOwnProperty.call(stubs, specifier)) return stubs[specifier];
    throw new Error(`Unexpected GatesScreen import: ${specifier}`);
  };
  const moduleRecord: { exports: Record<string, unknown> } = { exports: {} };
  runInNewContext(output, {
    exports: moduleRecord.exports,
    module: moduleRecord,
    require: requireStub,
  });
  return moduleRecord.exports as GatesScreenModule;
}

function collectElements(node: unknown, type: string, result: RenderedElement[] = []): RenderedElement[] {
  if (Array.isArray(node)) {
    for (const child of node) collectElements(child, type, result);
    return result;
  }
  if (!node || typeof node !== "object") return result;
  const element = node as RenderedElement;
  if (element.type === type) result.push(element);
  collectElements(element.props?.children, type, result);
  return result;
}

function collectText(node: unknown, result: string[] = []): string[] {
  if (typeof node === "string" || typeof node === "number") {
    result.push(String(node));
    return result;
  }
  if (Array.isArray(node)) {
    for (const child of node) collectText(child, result);
    return result;
  }
  if (node && typeof node === "object") collectText((node as RenderedElement).props?.children, result);
  return result;
}

const gatesScreen = loadGatesScreenModule();

test("GatesScreen normalizes canonical and legacy gate decisions", () => {
  assert.deepEqual(
    ["approved", "rejected", "pending"].map((decision) => gatesScreen.normalizeGateDecision(decision).canonical),
    ["approved", "rejected", "pending"],
  );

  const promote = gatesScreen.normalizeGateDecision("PROMOTE");
  assert.equal(promote.canonical, "approved");
  assert.equal(promote.display, "approved / promote");
  assert.equal(promote.tone, "verified");

  const hold = gatesScreen.normalizeGateDecision(" hold ");
  assert.equal(hold.canonical, "rejected");
  assert.equal(hold.display, "rejected / hold");
  assert.equal(hold.tone, "failed");
  assert.equal(hold.isPending, false);

  const unknown = gatesScreen.normalizeGateDecision("reviewing");
  assert.equal(unknown.canonical, "unknown");
  assert.equal(unknown.display, "reviewing");
});

test("GatesScreen task predicate fails closed for unscoped and other-task gates", () => {
  assert.equal(gatesScreen.gateBelongsToSelectedTask({ task_id: "task-a" }, "task-a"), true);
  assert.equal(gatesScreen.gateBelongsToSelectedTask({ taskId: "task-a" }, "task-a"), true);
  assert.equal(gatesScreen.gateBelongsToSelectedTask({ task_id: "task-b" }, "task-a"), false);
  assert.equal(gatesScreen.gateBelongsToSelectedTask({ decision: "approved" }, "task-a"), false);
  assert.equal(gatesScreen.gateBelongsToSelectedTask({ task_id: "task-a" }, ""), false);
});

test("GatesScreen renders KPIs and rows from the selected task only", () => {
  const noOp = () => undefined;
  const tree = gatesScreen.GatesScreen({
    selectedTask: "task-a",
    setSelectedTask: noOp,
    selectedStage: "",
    setSelectedStage: noOp,
    selectedExperiment: "",
    setSelectedExperiment: noOp,
    gateStatus: "Pending",
    setGateStatus: noOp,
    patchApplied: false,
    setPatchApplied: noOp,
    reportSubmitted: false,
    setReportSubmitted: noOp,
    locale: "en-US",
    summary: {
      gates: [
        { id: "a-approved", task_id: "task-a", gate_type: "Canonical approved", decision: "approved" },
        { id: "a-promote", task_id: "task-a", gate_type: "Legacy promote", decision: "promote" },
        { id: "a-hold", task_id: "task-a", gate_type: "Legacy hold", decision: "hold" },
        { id: "a-pending", task_id: "task-a", gate_type: "Canonical pending", decision: "pending" },
        { id: "b-approved", task_id: "task-b", gate_type: "Other task gate", decision: "approved" },
      ],
    },
  });

  const metrics = Object.fromEntries(
    collectElements(tree, "MetricTile").map((element) => [
      String(element.props?.label),
      element.props?.value,
    ]),
  );
  assert.equal(metrics["Total Gates"], 4);
  assert.equal(metrics.Pending, 1);
  assert.equal(metrics.Approved, 2);

  const text = collectText(tree).join(" ");
  assert.match(text, /approved \/ promote/);
  assert.match(text, /rejected \/ hold/);
  assert.doesNotMatch(text, /Other task gate/);
});
