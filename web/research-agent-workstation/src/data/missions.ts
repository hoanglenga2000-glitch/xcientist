import type { StatusTone } from "@/components/ui/status-badge";

export type MissionStatus = "Running" | "Review" | "Planning";

export const activeMission = {
  id: "house-prices",
  name: "House Prices Regression",
  question:
    "Can log-target transformation + LightGBM reduce RMSE compared to a linear baseline on House Prices?",
  objective:
    "Build a strong, reproducible baseline and validate whether log-target + GBDT is beneficial.",
  hypothesis:
    "Log1p(target) + LightGBM will significantly lower RMSE versus the linear regression baseline.",
  dataset: "house_prices v1.2.0",
  datasetDetail: "49,842 rows / 79 features",
  metric: "RMSE (CV 5-Fold)",
  baseline: "Linear Regression (log-target), CV 0.14532",
  expectedOutput: "RMSE improvement > 0.01 and stable across 5 folds.",
  currentStage: "Model Training",
  currentRun: "exp_20250606_192030",
  bestMetric: "0.12899",
  metricDelta: "-0.01633 (-11.23%)",
  decision: "Review validation results and leakage check. Approve for submission if integrity passes.",
  decisionStatus: "Pending Review",
  owner: "Research Admin",
  environment: "Local Sandbox",
  seed: 42,
  codeCommit: "a1f392c (main)",
  model: "LightGBM 4.3.0",
  validation: "KFold (5, shuffle=True)",
  stageProgress: "6 / 9"
};

export const missionBrief = [
  { label: "Research Question", value: activeMission.question, icon: "FileQuestion" },
  { label: "Objective", value: activeMission.objective, icon: "Target" },
  { label: "Hypothesis", value: activeMission.hypothesis, icon: "Lightbulb" },
  {
    label: "Dataset & Metric",
    value: `${activeMission.dataset} (${activeMission.metric})`,
    icon: "Database"
  },
  { label: "Current Decision", value: activeMission.decision, icon: "GitPullRequest" }
];

export const taskQueue: Array<{
  id: string;
  title: string;
  question: string;
  finding: string;
  stage: string;
  progress: number;
  status: MissionStatus;
  owner: string;
  priority: string;
  tone: StatusTone;
}> = [
  {
    id: "house-prices",
    title: "House Prices Regression",
    question: "Can log-target + LightGBM reduce RMSE vs linear baseline?",
    finding: "CV RMSE improved to 0.12899, leakage check pending.",
    stage: "Model Training",
    progress: 66,
    status: "Running",
    owner: "Developer Agent",
    priority: "High",
    tone: "blue"
  },
  {
    id: "titanic",
    title: "Titanic Survival",
    question: "Can we predict survival with high accuracy using tabular features?",
    finding: "CV Accuracy 0.84061 achieved, ready for report.",
    stage: "Validation Review",
    progress: 90,
    status: "Review",
    owner: "Analyst Agent",
    priority: "Medium",
    tone: "amber"
  },
  {
    id: "autokaggle",
    title: "AutoKaggle Reproduction",
    question: "Reproduce AutoKaggle top solution pipeline for House Prices.",
    finding: "Top solution pipeline analyzed. Reproduction plan ready.",
    stage: "Experiment Planning",
    progress: 18,
    status: "Planning",
    owner: "Orchestrator Agent",
    priority: "Medium",
    tone: "blue"
  }
];

export const stages = [
  "Task Understanding",
  "Literature Grounding",
  "Data Inspection",
  "Hypothesis Planning",
  "Experiment Design",
  "Code Generation",
  "Model Training",
  "Validation Review",
  "Report Writing"
].map((name, index) => ({
  id: `stage-${index + 1}`,
  name,
  state: index < 6 ? "Completed" : index === 6 ? "Running" : "Pending",
  progress: index < 6 ? 100 : index === 6 ? 66 : 0
}));

export const validationCurve = [
  { iteration: 0, fold1: 0.302, fold2: 0.288, fold3: 0.278, fold4: 0.292, fold5: 0.281, mean: 0.288 },
  { iteration: 250, fold1: 0.188, fold2: 0.174, fold3: 0.168, fold4: 0.179, fold5: 0.172, mean: 0.176 },
  { iteration: 500, fold1: 0.158, fold2: 0.146, fold3: 0.141, fold4: 0.149, fold5: 0.144, mean: 0.148 },
  { iteration: 1000, fold1: 0.139, fold2: 0.132, fold3: 0.129, fold4: 0.134, fold5: 0.131, mean: 0.133 },
  { iteration: 1500, fold1: 0.132, fold2: 0.129, fold3: 0.127, fold4: 0.130, fold5: 0.128, mean: 0.129 },
  { iteration: 2000, fold1: 0.130, fold2: 0.128, fold3: 0.127, fold4: 0.129, fold5: 0.128, mean: 0.129 }
];

export const trainingLog = [
  "[19:20:15] INFO Start demo training LightGBM-style baseline (log-target)",
  "[19:20:15] INFO Params: num_leaves=31, lr=0.05, n_estimators=2000",
  "[19:20:15] INFO Dataset: 49,842 rows, 79 features",
  "[19:20:15] INFO CV: KFold(5), seed=42",
  "[19:20:16] INFO Fold 1 start",
  "[19:20:41] INFO Iter 500 RMSE=0.14521",
  "[19:20:52] INFO Iter 1000 RMSE=0.13211",
  "[19:21:04] INFO Fold 5 RMSE=0.12910",
  "[19:21:04] INFO Mean CV RMSE = 0.12899 (Std 0.00521)",
  "[19:21:05] INFO Validation completed successfully."
];
