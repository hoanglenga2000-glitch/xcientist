export const experimentRuns = [
  {
    id: "exp_20250606_184210",
    name: "Linear Regression Baseline",
    model: "Linear Regression",
    hypothesis: "Log-target linear baseline",
    cv: 0.14532,
    holdout: 0.14841,
    std: 0.00411,
    improvement: "baseline",
    status: "Completed",
    evidence: 6
  },
  {
    id: "exp_20250606_190522",
    name: "XGBoost Log Target",
    model: "XGBoost",
    hypothesis: "Boosted trees capture nonlinear signals",
    cv: 0.13154,
    holdout: 0.13418,
    std: 0.00387,
    improvement: "+0.01378",
    status: "Completed",
    evidence: 9
  },
  {
    id: "exp_20250606_192030",
    name: "LightGBM Log Target",
    model: "LightGBM",
    hypothesis: "Leaf-wise GBDT improves RMSLE",
    cv: 0.12899,
    holdout: 0.13088,
    std: 0.00321,
    improvement: "+0.01633",
    status: "Best",
    evidence: 12
  },
  {
    id: "exp_20250606_193118",
    name: "Validation Review",
    model: "Reviewer Gate",
    hypothesis: "Check leakage and submission format",
    cv: 0.12899,
    holdout: 0.13088,
    std: 0.00321,
    improvement: "locked",
    status: "Review",
    evidence: 8
  }
];

export const featureImportance = [
  { feature: "OverallQual", value: 92 },
  { feature: "GrLivArea", value: 78 },
  { feature: "Neighborhood", value: 62 },
  { feature: "GarageCars", value: 49 },
  { feature: "TotalBsmtSF", value: 42 }
];
