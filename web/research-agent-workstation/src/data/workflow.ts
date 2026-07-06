import type { Edge, Node } from "@xyflow/react";

export type WorkflowNodeData = {
  label: string;
  subtitle: string;
  status: "passed" | "active" | "pending" | "disabled";
  kind: "input" | "agent" | "tool" | "integrity" | "output";
};

export const nodeLibrary = {
  "Research Inputs": ["Research Question", "Dataset", "Paper / Literature", "Evaluation Metric"],
  "Agent Nodes": [
    "Literature Agent",
    "Data Agent",
    "Planner Agent",
    "Developer Agent",
    "Trainer Agent",
    "Reviewer Agent",
    "Writer Agent"
  ],
  "Tool Nodes": ["Python Runner", "CPU Sandbox", "Kaggle API", "Figure Generator", "Report Generator"],
  "Integrity Nodes": ["Leakage Check", "Submission Check", "Human Gate", "Evidence Archive"]
};

export const workflowNodes: Node<WorkflowNodeData>[] = [
  { id: "question", position: { x: 80, y: 30 }, data: { label: "Research Question", subtitle: "research_question.md", status: "passed", kind: "input" } },
  { id: "dataset", position: { x: 360, y: 30 }, data: { label: "Dataset", subtitle: "data_profile.md", status: "passed", kind: "input" } },
  { id: "literature", position: { x: 220, y: 145 }, data: { label: "Literature Agent", subtitle: "lit_review.md", status: "passed", kind: "agent" } },
  { id: "data", position: { x: 220, y: 255 }, data: { label: "Data Agent", subtitle: "data_profile.md", status: "passed", kind: "agent" } },
  { id: "planner", position: { x: 220, y: 365 }, data: { label: "Planner Agent", subtitle: "plan.md", status: "passed", kind: "agent" } },
  { id: "developer", position: { x: 220, y: 475 }, data: { label: "Developer Agent", subtitle: "code/", status: "passed", kind: "agent" } },
  { id: "python", position: { x: 220, y: 585 }, data: { label: "Python Runner", subtitle: "run_logs/", status: "passed", kind: "tool" } },
  { id: "trainer", position: { x: 220, y: 695 }, data: { label: "Trainer Agent", subtitle: "models/ metrics.json", status: "passed", kind: "agent" } },
  { id: "reviewer", position: { x: 220, y: 805 }, data: { label: "Reviewer Agent", subtitle: "review_report.md", status: "active", kind: "agent" } },
  { id: "leakage", position: { x: 520, y: 805 }, data: { label: "Leakage Check", subtitle: "leak_check.md", status: "passed", kind: "integrity" } },
  { id: "human", position: { x: 780, y: 805 }, data: { label: "Human Gate", subtitle: "approval.json", status: "pending", kind: "integrity" } },
  { id: "archive", position: { x: 160, y: 940 }, data: { label: "Evidence Archive", subtitle: "evidence_index.json", status: "passed", kind: "integrity" } },
  { id: "writer", position: { x: 480, y: 940 }, data: { label: "Writer Agent", subtitle: "report_draft.md", status: "passed", kind: "agent" } },
  { id: "output", position: { x: 780, y: 940 }, data: { label: "Report Output", subtitle: "report.pdf", status: "passed", kind: "output" } }
];

export const workflowEdges: Edge[] = [
  { id: "q-lit", source: "question", target: "literature" },
  { id: "d-lit", source: "dataset", target: "literature" },
  { id: "lit-data", source: "literature", target: "data" },
  { id: "data-plan", source: "data", target: "planner" },
  { id: "plan-dev", source: "planner", target: "developer" },
  { id: "dev-python", source: "developer", target: "python" },
  { id: "python-trainer", source: "python", target: "trainer" },
  { id: "trainer-reviewer", source: "trainer", target: "reviewer" },
  { id: "reviewer-leak", source: "reviewer", target: "leakage" },
  { id: "leak-human", source: "leakage", target: "human" },
  { id: "reviewer-archive", source: "reviewer", target: "archive" },
  { id: "archive-writer", source: "archive", target: "writer" },
  { id: "writer-output", source: "writer", target: "output" }
].map((edge) => ({
  ...edge,
  type: "smoothstep",
  animated: edge.target === "human",
  style: { stroke: "#94A3B8", strokeWidth: 1.5 }
}));
