export const agents = [
  {
    name: "Orchestrator Agent",
    role: "Planner",
    status: "Idle",
    currentTask: "Waiting for validation decision",
    tokenBudget: "14.2k / 32k",
    toolBudget: "3 / 8",
    latency: "820 ms",
    lastAction: "Opened human gate"
  },
  {
    name: "Analyst Agent",
    role: "Evidence",
    status: "Waiting",
    currentTask: "Review residual analysis",
    tokenBudget: "18.6k / 32k",
    toolBudget: "5 / 8",
    latency: "1.1 s",
    lastAction: "Bound validation curve"
  },
  {
    name: "Developer Agent",
    role: "Code",
    status: "Running",
    currentTask: "Patch categorical preprocessing",
    tokenBudget: "21.4k / 32k",
    toolBudget: "6 / 8",
    latency: "940 ms",
    lastAction: "Suggested patch"
  },
  {
    name: "Reviewer Agent",
    role: "Gate",
    status: "Running",
    currentTask: "Leakage and metric checks",
    tokenBudget: "11.7k / 32k",
    toolBudget: "4 / 8",
    latency: "760 ms",
    lastAction: "Approved metric definition"
  },
  {
    name: "Writer Agent",
    role: "Report",
    status: "Idle",
    currentTask: "Draft report once human gate passes",
    tokenBudget: "8.5k / 32k",
    toolBudget: "2 / 8",
    latency: "690 ms",
    lastAction: "Prepared report outline"
  }
];

export const agentTrace = [
  {
    time: "19:18:12",
    agent: "Analyst Agent",
    action: "Data Inspection",
    tool: "python_profile",
    output: "No target leakage columns detected",
    evidence: "data_profile.md",
    state: "Passed"
  },
  {
    time: "19:19:41",
    agent: "Developer Agent",
    action: "Model Training",
    tool: "lightgbm.train",
    output: "CV RMSE=0.12899",
    evidence: "metrics.json",
    state: "Running"
  },
  {
    time: "19:20:12",
    agent: "Reviewer Agent",
    action: "Validation Review",
    tool: "leakage_check",
    output: "Metric and split match config",
    evidence: "validation_report.md",
    state: "Passed"
  },
  {
    time: "19:22:02",
    agent: "Orchestrator Agent",
    action: "Human Gate",
    tool: "manual_gate.open",
    output: "Kaggle submission approval required",
    evidence: "approval.json",
    state: "Pending"
  }
];

export const toolSandbox = [
  { name: "Python Runner", state: "Ready", tone: "green" },
  { name: "File System", state: "Read Only", tone: "blue" },
  { name: "Kaggle API", state: "Disabled", tone: "amber" },
  { name: "GPU Runner", state: "Disabled", tone: "amber" },
  { name: "Report Generator", state: "Ready", tone: "green" }
] as const;
