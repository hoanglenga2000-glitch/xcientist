export const reportOutline = [
  "Research Question",
  "Dataset & Provenance",
  "Baseline Method",
  "Hypothesis",
  "Experiment Design",
  "Validation Results",
  "Limitations",
  "Conclusion"
];

export const reportSections = [
  {
    title: "Validation Result",
    body:
      "Log-target LightGBM achieved a mean CV RMSE of 0.12899 across five folds, improving over the linear baseline by 0.01633.",
    evidence: ["metrics.json", "validation_curve.png"]
  },
  {
    title: "Reproducibility",
    body:
      "The run records dataset version, commit, random seed, metric definition, validation split and artifact manifest.",
    evidence: ["reproducibility_record.json", "manifest.json"]
  },
  {
    title: "Limitations",
    body:
      "Official Kaggle leaderboard submission remains disabled until a Kaggle token is configured and human approval is granted.",
    evidence: ["limitations.md", "manual_gate.json"]
  }
];
