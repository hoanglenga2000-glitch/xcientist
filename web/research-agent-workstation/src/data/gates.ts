export const gateSummary = [
  { name: "Provenance", value: "100%", status: "Passed" },
  { name: "Reproducibility", value: "100%", status: "Passed" },
  { name: "Validity", value: "98.7%", status: "Passed" },
  { name: "Human Oversight", value: "Pending", status: "Pending" },
  { name: "Limitations", value: "5 / 5", status: "Disclosed" }
];

export const gateChecklist = [
  ["Data Source Verified", "data_sources.json", "Passed", "Dr. A. Patel"],
  ["Data Lineage Captured", "lineage_graph.json", "Passed", "Dr. A. Patel"],
  ["Environment Captured", "environment.yml", "Passed", "Dr. M. Chen"],
  ["Code Snapshot", "code_commit.txt", "Passed", "Dr. M. Chen"],
  ["Random Seeds Fixed", "seeds.json", "Passed", "Dr. M. Chen"],
  ["Train/Test Protocol", "splits.json", "Passed", "Dr. S. Iyer"],
  ["Cross-Validation", "cv_results.json", "Passed", "Dr. S. Iyer"],
  ["Scientific Review", "review_notes.md", "Pending", "Dr. K. Williams"],
  ["Kaggle Submission Approval", "kaggle_approval.json", "Pending", "Dr. K. Williams"],
  ["Known Limitations", "limitations.md", "Passed", "Dr. R. Singh"]
];

export const riskFactors = [
  { name: "Data Leakage", level: "High", status: "Mitigated", trend: "up" },
  { name: "Target Leakage", level: "High", status: "Mitigated", trend: "up" },
  { name: "Sampling Bias", level: "Medium", status: "Monitoring", trend: "flat" },
  { name: "Overfitting", level: "Low", status: "Controlled", trend: "down" },
  { name: "Metric Gaming", level: "Low", status: "Controlled", trend: "flat" }
];
