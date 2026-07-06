# Reference Review

Status: draft started on 2026-06-14. This is a working research review; future updates should add more notebook-level evidence without copying code.

## Competition Facts

Competition:
- Kaggle: [Playground Series - Season 6, Episode 6](https://www.kaggle.com/competitions/playground-series-s6e6)
- Task: Predict `class` for stellar objects.
- Classes: `GALAXY`, `STAR`, `QSO`.
- Start date: 2026-06-01.
- Final submission deadline: 2026-06-30 23:59 UTC.
- Current entered team count observed through Kaggle CLI on 2026-06-14: `1585`.

Data:
- `train.csv`: training set with `class` target.
- `test.csv`: test set for prediction.
- `sample_submission.csv`: required submission format.
- The competition page says the data is inspired by the [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17), but feature distributions are close to, not identical to, the original.

Evaluation:
- Official metric: [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).
- This matters because the training target is imbalanced: `GALAXY` is about 65.38%, `QSO` about 20.29%, and `STAR` about 14.33%.
- Plain accuracy is not enough for model selection.

Submission format:

```csv
id,class
577347,STAR
577348,GALAXY
577349,STAR
```

## Competition Discussion Signals

Reviewed through Kaggle API topic listings and selected messages on 2026-06-14.

Observed discussion themes:
- `Formulae for spectral_type and galaxy_population`: competitors identified that `spectral_type` and `galaxy_population` appear derivable from simple thresholds on color-like features. This supports treating them as engineered categorical features and testing whether explicit numeric versions add value.
- `GPU Logistic Regression STACKER - Starter Pack`: strong public discussion around saving OOF/test predictions from base models and using multinomial logistic regression as a stacker. This aligns with an OOF-first ensemble plan.
- `Single Model or Ensemble?`, `Just another "Blending Topic"`, and `Battle of the CV OOF Warriors vs. the CSV Blenders`: public leaderboard movement appears strongly affected by blending/ensembling. We should use OOF-based blending, not blind CSV blending.
- `Binary Classification Chain`: one proposed route is decomposing the task into `GALAXY` vs not, then `QSO` vs `STAR`. This is plausible because the majority class is large and class boundaries may differ.
- `Sky Positions Features`: alpha/delta are angular coordinates. Raw values may be enough for trees, but sine/cosine transforms of angular coordinates should be tested because wrap-around can matter.
- `Feature Reduction Strategies`: RFE/RFECV and feature importance are reasonable tools, but feature selection must be evaluated by CV rather than public score.

Governance:
- We may summarize public ideas and implement our own experiments.
- We must not copy notebook code into the project.
- We must avoid leaderboard-only tuning and blind blending.

## Stellar Classification Method Notes

Relevant background:
- SDSS-style data uses photometric bands such as `u`, `g`, `r`, `i`, `z`; differences such as `u-g`, `g-r`, `r-i`, and `i-z` are standard color-index features in astronomical classification.
- Redshift and color indices are scientifically meaningful features for separating galaxies, quasars, and stars.
- The competition data is synthetic/playground data inspired by a real dataset, so external data usage must be treated cautiously and only under the competition rules.

References:
- [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)
- [Identifying galaxies, quasars, and stars with machine learning](https://www.aanda.org/articles/aa/full_html/2020/07/aa36770-19/aa36770-19.html)
- [Astronomical Point Source Classification through Machine Learning](https://cs229.stanford.edu/proj2013/Waisberg-Astronomical%20Point%20Source%20Classification%20through%20Machine%20Learning.pdf)
- [The Science of SDSS](https://classic.sdss.org/background/science.php)

## Official Method Documentation

Cross-validation and metric:
- [sklearn balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html): balanced accuracy averages recall across classes and is suitable for imbalanced classification.
- [sklearn StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html): preserves class percentages in each fold and should be the default CV split.

Model families:
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) and [LGBMClassifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html): strong gradient-boosted tree baseline for tabular data.
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html): for multiclass probability outputs, use `multi:softprob` rather than hard-label `multi:softmax`.
- [XGBoost GPU support](https://xgboost.readthedocs.io/en/release_1.5.0/gpu/): CUDA can accelerate tree construction and prediction if installed correctly on HPC.
- [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier): supports sklearn-style classification and native categorical handling.
- [CatBoost multiclassification losses and metrics](https://catboost.ai/docs/en/concepts/loss-functions-multiclassification): use multiclass settings and inspect per-class metrics.

Tuning, calibration, and ensemble:
- [Optuna TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html): TPE is appropriate for bounded hyperparameter search, but each study must record search space, seed, trial count, and CV objective.
- [Optuna create_study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html): use a persisted study for resumability where possible.
- [sklearn CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html): useful for calibrated probabilities, but balanced accuracy is based on final labels; calibration should be justified through OOF behavior, not assumed.
- [sklearn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html): calibration should use unbiased held-out predictions.
- [sklearn StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html): stackers should use cross-validated base predictions; our implementation should save explicit OOF arrays for auditability.

## Implications For Next Experiments

- Replace plain accuracy with balanced accuracy everywhere.
- Use StratifiedKFold because the target is imbalanced.
- Save OOF predictions for every serious model.
- Prioritize LightGBM, XGBoost, CatBoost, and OOF-based logistic stacker before larger neural work.
- Test angular transforms for `alpha` and `delta`.
- Test formula-derived categorical consistency and whether retaining or dropping derived categorical columns helps.
- Treat PyTorch MLP as one model family, not the default "advanced" choice.
- Do not submit until a candidate beats EXP000 under balanced-accuracy CV and has a submission report.
