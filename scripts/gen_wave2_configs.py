#!/usr/bin/env python3
"""Generate Wave-2 evolution configs for feasible tabular/text MLE-bench tasks.

Each config is hand-tuned from the REAL GPU file layout (probed) + the vendored
mle-bench grader metric. Only competitions that (a) have extracted trainable data
on the GPU under mlebench_raw_data/<id>/ and (b) are runnable with GBDT / TF-IDF /
dictionary baselines on a single A40 within the evolution timeout are included.

extra_notes is the contract the LLM codegen reads: it names the EXACT files,
the EXACT submission columns, and the EXACT CV metric to print as CV_SCORE.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "evolution"
RAW = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data"

CONFIGS: dict[str, dict] = {}

def add(stem, **cfg):
    cfg.setdefault("remote_data_dirname", cfg["task_name"])
    cfg.setdefault("gpu_data_dir", f"{RAW}/{cfg['task_name']}")
    cfg.setdefault("n_test", 0)
    cfg.setdefault("n_features", 0)
    cfg.setdefault("n_model_families", 2)
    CONFIGS[stem] = cfg


add("google_quest",
    task_name="google-quest-challenge",
    modality="text", task_type="regression", metric="spearman_mean",
    metric_direction="maximize", target_column="30 label columns", id_column="qa_id",
    n_train=6079, n_test=476, n_model_families=2,
    data_schema="train.csv: qa_id(id), question_title, question_body, answer, category, host (+ 6 user/url text cols), then 30 float target columns in [0,1] (question_* and answer_*). test.csv: qa_id + the same input text cols (NO targets). sample_submission.csv: qa_id + the 30 target columns.",
    extra_notes="Google QUEST Q&A Labeling. Files at data-dir: train.csv, test.csv, sample_submission.csv. Predict 30 quality float scores in [0,1] per (question,answer). Official metric = mean column-wise Spearman rank correlation (HIGHER better). Approach that fits the A40 in minutes: TF-IDF (word+char n-grams) on question_title+question_body+answer, plus one-hot of category/host, then a multi-output Ridge/ridge-per-target or lightgbm per target (30 targets). Report CV_SCORE = mean over the 30 targets of Spearman correlation on an OOF/holdout split (higher better). submission.csv MUST have columns EXACTLY matching sample_submission.csv (qa_id + 30 target names) for all test qa_id rows in order. metrics.json {\"cv_score\": <float>, \"metric\": \"spearman_mean\"}.")

add("essay_scoring2",
    task_name="learning-agency-lab-automated-essay-scoring-2",
    modality="text", task_type="regression", metric="quadratic_weighted_kappa",
    metric_direction="maximize", target_column="score", id_column="essay_id",
    n_train=17307, n_test=3,
    data_schema="train.csv: essay_id(id), full_text(str essay), score(int 1..6 target). test.csv: essay_id, full_text. sample_submission.csv: essay_id, score.",
    extra_notes="Automated Essay Scoring 2.0. Files: train.csv, test.csv, sample_submission.csv. Predict integer essay score 1..6. Official metric = Quadratic Weighted Kappa (HIGHER better). Fast strong baseline on A40: engineer text features (length, word/sentence counts, mean word len, punctuation, TF-IDF word+char n-grams truncated by SVD) then lightgbm/xgboost REGRESSION, round+clip predictions to [1,6] ints for QWK. Report CV_SCORE = QWK on K-fold OOF (higher better). submission.csv columns EXACTLY essay_id,score (int) for all test rows. metrics.json {\"cv_score\": <float>, \"metric\": \"quadratic_weighted_kappa\"}.")

add("lmsys_arena",
    task_name="lmsys-chatbot-arena",
    modality="text", task_type="classification", metric="multiclass_log_loss",
    metric_direction="minimize", target_column="winner_model_a/b/tie", id_column="id",
    n_train=57477, n_test=3, n_classes=3,
    data_schema="train.csv: id, model_a, model_b, prompt(list-json str), response_a(list-json str), response_b(list-json str), winner_model_a, winner_model_b, winner_tie (3 one-hot target cols). test.csv: id, prompt, response_a, response_b. sample_submission.csv: id, winner_model_a, winner_model_b, winner_tie (probabilities summing to 1).",
    extra_notes="LMSYS Chatbot Arena human-preference prediction. Files: train.csv, test.csv, sample_submission.csv. 3-class: which of response_a / response_b wins, or tie. Official metric = multi-class log loss (LOWER better). prompt/response_a/response_b are JSON-encoded lists of strings — join them to plain text first. Fast baseline: TF-IDF on prompt + response_a + response_b (and length/diff features between a and b), then a 3-class classifier (lightgbm multiclass or logistic regression) with predict_proba. Report CV_SCORE = multiclass log loss on OOF (lower better). submission.csv columns EXACTLY id,winner_model_a,winner_model_b,winner_tie (probabilities, each row sums to 1) for all test ids. metrics.json {\"cv_score\": <float>, \"metric\": \"multiclass_log_loss\"}.")

add("us_patent_pppm",
    task_name="us-patent-phrase-to-phrase-matching",
    modality="text", task_type="regression", metric="pearson",
    metric_direction="maximize", target_column="score", id_column="id",
    n_train=36473, n_test=36,
    data_schema="train.csv: id, anchor(phrase), target(phrase), context(CPC code e.g. A47), score(float in {0,0.25,0.5,0.75,1.0}). test.csv: id, anchor, target, context. sample_submission.csv: id, score.",
    extra_notes="U.S. Patent Phrase-to-Phrase Matching. Files: train.csv, test.csv, sample_submission.csv. Predict semantic similarity score in [0,1] between anchor and target phrases within a CPC context. Official metric = Pearson correlation (HIGHER better). NOTE the mle-bench test.csv here is tiny (36 rows) — still predict all of them. Baseline: TF-IDF char+word n-grams on anchor, target, and 'anchor [SEP] target [SEP] context'; features = cosine similarity, shared-token counts, plus one-hot of context section; lightgbm/ridge regression, clip to [0,1]. Report CV_SCORE = Pearson corr on K-fold OOF over TRAIN (higher better) since test is tiny. submission.csv columns EXACTLY id,score for all test rows. metrics.json {\"cv_score\": <float>, \"metric\": \"pearson\"}.")

add("jigsaw_toxic",
    task_name="jigsaw-toxic-comment-classification-challenge",
    modality="text", task_type="classification", metric="mean_column_roc_auc",
    metric_direction="maximize", target_column="6 toxicity labels", id_column="id",
    n_train=159571, n_test=153164, n_classes=6,
    data_schema="train.csv: id, comment_text, and 6 binary label cols (toxic, severe_toxic, obscene, threat, insult, identity_hate). test.csv: id, comment_text. test_labels.csv: id + 6 labels but rows with -1 are unscored. sample_submission.csv: id + 6 label probability cols.",
    extra_notes="Jigsaw Toxic Comment Classification. Files: train.csv, test.csv, sample_submission.csv (test_labels.csv exists but the grader uses only scored rows). Multi-label: predict probability of each of 6 toxicity types. Official metric = mean column-wise ROC AUC over the 6 labels (HIGHER better). Fast strong baseline: TF-IDF (word 1-2gram + char 3-5gram, sublinear tf, max_features ~50k) then one logistic regression per label (predict_proba). Report CV_SCORE = mean over 6 labels of ROC AUC on OOF (higher better). submission.csv columns EXACTLY id,toxic,severe_toxic,obscene,threat,insult,identity_hate (probabilities) for all test ids in order. metrics.json {\"cv_score\": <float>, \"metric\": \"mean_column_roc_auc\"}.")

add("text_norm_en",
    task_name="text-normalization-challenge-english-language",
    modality="text", task_type="classification", metric="token_accuracy",
    metric_direction="maximize", target_column="after", id_column="id",
    n_train=9918441, n_test=1088565,
    data_schema="en_train.csv: sentence_id, token_id, class, before, after. en_test.csv: sentence_id, token_id, before (id for submission = 'sentence_id_token_id'). en_sample_submission.csv: id, after.",
    extra_notes="Text Normalization (English). Files: en_train.csv (9.9M token rows), en_test.csv, en_sample_submission.csv. For each token predict its normalized 'after' string. Official metric = per-token accuracy (HIGHER better). Strong cheap baseline that finishes on A40: build a memorization dictionary from en_train mapping before->most_frequent(after); at predict time, output the dict value if 'before' seen else identity (after=before). ~most tokens are class=PLAIN where after==before, so identity+dict is a very strong baseline. Report CV_SCORE = token accuracy on a held-out slice of en_train (higher better). submission.csv columns EXACTLY id,after where id = f'{sentence_id}_{token_id}' for every en_test row in order. metrics.json {\"cv_score\": <float>, \"metric\": \"token_accuracy\"}.")

add("text_norm_ru",
    task_name="text-normalization-challenge-russian-language",
    modality="text", task_type="classification", metric="token_accuracy",
    metric_direction="maximize", target_column="after", id_column="id",
    n_train=10574516, n_test=1074564,
    data_schema="ru_train.csv: sentence_id, token_id, class, before, after. ru_test.csv: sentence_id, token_id, before. ru_sample_submission.csv: id, after.",
    extra_notes="Text Normalization (Russian). Files: ru_train.csv (10.6M token rows), ru_test.csv, ru_sample_submission.csv. Same task as English. Official metric = per-token accuracy (HIGHER better). Baseline: before->most_frequent(after) dictionary from ru_train, fall back to identity (after=before) for unseen tokens. Read CSVs as utf-8. Report CV_SCORE = token accuracy on a held-out slice of ru_train. submission.csv columns EXACTLY id,after where id = f'{sentence_id}_{token_id}' for every ru_test row in order. metrics.json {\"cv_score\": <float>, \"metric\": \"token_accuracy\"}.")

add("stanford_covid_vaccine",
    task_name="stanford-covid-vaccine",
    modality="tabular", task_type="regression", metric="mcrmse",
    metric_direction="minimize", target_column="5 reactivity targets", id_column="id_seqpos",
    n_train=2400, n_test=3634,
    data_schema="train.json (records: id, sequence(RNA str A/C/G/U), structure, predicted_loop_type, seq_length, seq_scored, and per-position target arrays reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C). test.json: id, sequence, structure, predicted_loop_type, seq_length. sample_submission.csv: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C.",
    extra_notes="OpenVaccine COVID mRNA degradation. Files: train.json, test.json, sample_submission.csv (read train/test with pandas.read_json lines-or-records). Predict 5 per-base degradation/reactivity floats. Official metric = MCRMSE (mean columnwise RMSE, LOWER better). Fast tabular baseline: explode each sequence into per-position rows; features = one-hot of sequence char + structure char + predicted_loop_type char at the position, plus neighbor chars (window +-2) and positional index; lightgbm regressor per target (5 targets). Only the first seq_scored positions are scored in train. Report CV_SCORE = MCRMSE on OOF over the 5 targets (lower better). submission.csv columns EXACTLY id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C for every test id_seqpos (id + '_' + position) in sample order. metrics.json {\"cv_score\": <float>, \"metric\": \"mcrmse\"}.")

add("random_acts_pizza",
    task_name="random-acts-of-pizza",
    modality="text", task_type="classification", metric="roc_auc",
    metric_direction="maximize", target_column="requester_received_pizza", id_column="request_id",
    n_train=4040, n_test=1631,
    data_schema="train.json (list of records incl. request_id, request_text_edit_aware, request_title, requester_* numeric account features, unix_timestamp_of_request, and boolean target requester_received_pizza). test.json: same minus target. sampleSubmission.csv: request_id, requester_received_pizza (probability).",
    extra_notes="Random Acts of Pizza. Files: train.json, test.json, sampleSubmission.csv. Predict probability the request got a pizza. Official metric = ROC AUC (HIGHER better). Read json with pandas.read_json. Baseline: TF-IDF on request_title + request_text_edit_aware, plus the numeric requester_* account-age/activity features present in BOTH train and test (ignore _at_retrieval leakage cols that are train-only), then logistic regression / lightgbm predict_proba. Report CV_SCORE = ROC AUC on OOF (higher better). submission.csv columns EXACTLY request_id,requester_received_pizza (probability) for all test request_ids. metrics.json {\"cv_score\": <float>, \"metric\": \"roc_auc\"}.")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, cfg in CONFIGS.items():
        path = OUT / f"{stem}.json"
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(f"{stem}.json  <- {cfg['task_name']}  [{cfg['metric']} {cfg['metric_direction']}]")
    print(f"wrote {len(written)} Wave-2 configs to {OUT}:")
    for w in written:
        print("  " + w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

