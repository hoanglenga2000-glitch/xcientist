export const artifacts = [
  { name: "metrics.json", size: "2.1 KB", updated: "19:20:18", binding: "Bound" },
  { name: "train_log.txt", size: "18.7 KB", updated: "19:20:18", binding: "Bound" },
  { name: "validation_curve.png", size: "120.4 KB", updated: "19:20:18", binding: "Bound" },
  { name: "feature_importance.csv", size: "9.2 KB", updated: "19:20:17", binding: "Bound" },
  { name: "oof_predictions.csv", size: "1.6 MB", updated: "19:20:15", binding: "Bound" }
];

export const evidenceItems = [
  {
    id: "metrics",
    label: "metrics.json",
    type: "Metric Evidence",
    provenance: "Auto-collected by system",
    generated: "2025-06-06 19:20:12"
  },
  {
    id: "log",
    label: "train_log.txt",
    type: "Run Trace",
    provenance: "Python Runner",
    generated: "2025-06-06 19:21:05"
  },
  {
    id: "curve",
    label: "validation_curve.png",
    type: "Validation Figure",
    provenance: "Figure Generator",
    generated: "2025-06-06 19:21:08"
  }
];

export const claimRecord = {
  claim: "Log-target LightGBM improves RMSE over the linear baseline.",
  reviewer: "Analyst Agent",
  leakageCheck: "Passed",
  confidence: 0.86,
  risk: "Medium",
  status: "Pending Review"
};

export const reproducibility = [
  { key: "Dataset Version", value: "house_prices v1.2.0", status: "Locked" },
  { key: "Code Commit", value: "a1f392c (main)", status: "Locked" },
  { key: "Environment", value: "Python 3.11.9 / LightGBM 4.3.0", status: "Locked" },
  { key: "Random Seed", value: "42", status: "Locked" },
  { key: "Validation Split", value: "KFold (5, shuffle=True)", status: "Locked" },
  { key: "Metric Definition", value: "RMSE (root_mean_squared_error)", status: "Locked" },
  { key: "Artifact Manifest", value: "12 files", status: "Locked" }
];
