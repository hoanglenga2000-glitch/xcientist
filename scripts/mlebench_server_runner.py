#!/usr/bin/env python3
"""
MLE-Bench 75-Competition Autonomous Runner — Server-Side
========================================================
Self-contained training script deployed to GPU server.
No external dependencies beyond standard ML stack.

Usage: python3 mlebench_server_runner.py [--resume CHECKPOINT] [--single COMP_ID] [--start-from N]
"""
import pandas as pd, numpy as np, json, sys, os, time, gc, subprocess, traceback
from pathlib import Path
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────
HOME = Path.home()
BASE_DIR = HOME / "jinghw" / "scripts" / "gpu_tra"
DATA_DIR = BASE_DIR / "mlebench_data"
RESULTS_DIR = BASE_DIR / "mlebench_results"
CHECKPOINT_PATH = BASE_DIR / "mlebench_75_checkpoint.json"
LOG_PATH = BASE_DIR / "mlebench_75_runner.log"

for d in [DATA_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── All 75 MLE-Bench Competitions ───────────────────────────────────────
COMPETITIONS = [
    # Lite (22 competitions)
    ("aerial-cactus-identification", "lite", "image_classification"),
    ("aptos2019-blindness-detection", "lite", "image_classification"),
    ("denoising-dirty-documents", "lite", "image_classification"),
    ("detecting-insults-in-social-commentary", "lite", "nlp"),
    ("dog-breed-identification", "lite", "image_classification"),
    ("dogs-vs-cats-redux-kernels-edition", "lite", "image_classification"),
    ("histopathologic-cancer-detection", "lite", "image_classification"),
    ("jigsaw-toxic-comment-classification-challenge", "lite", "nlp"),
    ("leaf-classification", "lite", "image_classification"),
    ("mlsp-2013-birds", "lite", "audio"),
    ("new-york-city-taxi-fare-prediction", "lite", "tabular"),
    ("nomad2018-predict-transparent-conductors", "lite", "tabular"),
    ("plant-pathology-2020-fgvc7", "lite", "image_classification"),
    ("random-acts-of-pizza", "lite", "nlp"),
    ("ranzcr-clip-catheter-line-classification", "lite", "image_classification"),
    ("siim-isic-melanoma-classification", "lite", "image_classification"),
    ("spooky-author-identification", "lite", "nlp"),
    ("tabular-playground-series-dec-2021", "lite", "tabular"),
    ("tabular-playground-series-may-2022", "lite", "tabular"),
    ("text-normalization-challenge-english-language", "lite", "nlp"),
    ("text-normalization-challenge-russian-language", "lite", "nlp"),
    ("the-icml-2013-whale-challenge-right-whale-redux", "lite", "image_classification"),

    # Medium/High (53 competitions)
    ("3d-object-detection-for-autonomous-vehicles", "medium_high", "image_classification"),
    ("AI4Code", "medium_high", "nlp"),
    ("alaska2-image-steganalysis", "medium_high", "image_classification"),
    ("billion-word-imputation", "medium_high", "nlp"),
    ("bms-molecular-translation", "medium_high", "chemistry"),
    ("cassava-leaf-disease-classification", "medium_high", "image_classification"),
    ("cdiscount-image-classification-challenge", "medium_high", "image_classification"),
    ("chaii-hindi-and-tamil-question-answering", "medium_high", "nlp"),
    ("champs-scalar-coupling", "medium_high", "chemistry"),
    ("facebook-recruiting-iii-keyword-extraction", "medium_high", "nlp"),
    ("freesound-audio-tagging-2019", "medium_high", "audio"),
    ("google-quest-challenge", "medium_high", "nlp"),
    ("google-research-identify-contrails-reduce-global-warming", "medium_high", "image_classification"),
    ("h-and-m-personalized-fashion-recommendations", "medium_high", "tabular"),
    ("herbarium-2020-fgvc7", "medium_high", "image_classification"),
    ("herbarium-2021-fgvc8", "medium_high", "image_classification"),
    ("herbarium-2022-fgvc9", "medium_high", "image_classification"),
    ("hms-harmful-brain-activity-classification", "medium_high", "tabular"),
    ("hotel-id-2021-fgvc8", "medium_high", "image_classification"),
    ("hubmap-kidney-segmentation", "medium_high", "image_segmentation"),
    ("icecube-neutrinos-in-deep-ice", "medium_high", "physics"),
    ("imet-2020-fgvc7", "medium_high", "image_classification"),
    ("inaturalist-2019-fgvc6", "medium_high", "image_classification"),
    ("invasive-species-monitoring", "medium_high", "image_classification"),
    ("iwildcam-2019-fgvc6", "medium_high", "image_classification"),
    ("iwildcam-2020-fgvc7", "medium_high", "image_classification"),
    ("jigsaw-unintended-bias-in-toxicity-classification", "medium_high", "nlp"),
    ("kuzushiji-recognition", "medium_high", "image_classification"),
    ("learning-agency-lab-automated-essay-scoring-2", "medium_high", "nlp"),
    ("lmsys-chatbot-arena", "medium_high", "nlp"),
    ("ml2021spring-hw2", "medium_high", "tabular"),
    ("movie-review-sentiment-analysis-kernels-only", "medium_high", "nlp"),
    ("multi-modal-gesture-recognition", "medium_high", "tabular"),
    ("nfl-player-contact-detection", "medium_high", "tabular"),
    ("osic-pulmonary-fibrosis-progression", "medium_high", "tabular"),
    ("paddy-disease-classification", "medium_high", "image_classification"),
    ("petfinder-pawpularity-score", "medium_high", "tabular"),
    ("plant-pathology-2021-fgvc8", "medium_high", "image_classification"),
    ("plant-seedlings-classification", "medium_high", "image_classification"),
    ("playground-series-s3e18", "medium_high", "tabular"),
    ("predict-volcanic-eruptions-ingv-oe", "medium_high", "tabular"),
    ("rsna-2022-cervical-spine-fracture-detection", "medium_high", "image_classification"),
    ("rsna-breast-cancer-detection", "medium_high", "image_classification"),
    ("rsna-miccai-brain-tumor-radiogenomic-classification", "medium_high", "image_classification"),
    ("seti-breakthrough-listen", "medium_high", "tabular"),
    ("siim-covid19-detection", "medium_high", "image_classification"),
    ("smartphone-decimeter-2022", "medium_high", "tabular"),
    ("spaceship-titanic", "medium_high", "tabular"),
    ("stanford-covid-vaccine", "medium_high", "nlp"),
    ("statoil-iceberg-classifier-challenge", "medium_high", "image_classification"),
    ("tensorflow-speech-recognition-challenge", "medium_high", "audio"),
    ("tensorflow2-question-answering", "medium_high", "nlp"),
    ("tgs-salt-identification-challenge", "medium_high", "image_segmentation"),
    ("tweet-sentiment-extraction", "medium_high", "nlp"),
    ("us-patent-phrase-to-phrase-matching", "medium_high", "nlp"),
    ("uw-madison-gi-tract-image-segmentation", "medium_high", "image_segmentation"),
    ("ventilator-pressure-prediction", "medium_high", "tabular"),
    ("vesuvius-challenge-ink-detection", "medium_high", "image_segmentation"),
    ("vinbigdata-chest-xray-abnormalities-detection", "medium_high", "image_classification"),
    ("whale-categorization-playground", "medium_high", "image_classification"),
]

# ─── Kaggle Slug Mapping ─────────────────────────────────────────────────
KAGGLE_SLUGS = {
    "aerial-cactus-identification": "aerial-cactus-identification",
    "aptos2019-blindness-detection": "aptos2019-blindness-detection",
    "denoising-dirty-documents": "denoising-dirty-documents",
    "detecting-insults-in-social-commentary": "detecting-insults-in-social-commentary",
    "dog-breed-identification": "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition": "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection": "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge": "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification": "leaf-classification",
    "mlsp-2013-birds": "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction": "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors": "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7": "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza": "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification": "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification": "siim-isic-melanoma-classification",
    "spooky-author-identification": "spooky-author-identification",
    "tabular-playground-series-dec-2021": "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022": "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language": "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language": "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux": "the-icml-2013-whale-challenge-right-whale-redux",
    "3d-object-detection-for-autonomous-vehicles": "3d-object-detection-for-autonomous-vehicles",
    "AI4Code": "AI4Code",
    "alaska2-image-steganalysis": "alaska2-image-steganalysis",
    "billion-word-imputation": "billion-word-imputation",
    "bms-molecular-translation": "bms-molecular-translation",
    "cassava-leaf-disease-classification": "cassava-leaf-disease-classification",
    "cdiscount-image-classification-challenge": "cdiscount-image-classification-challenge",
    "chaii-hindi-and-tamil-question-answering": "chaii-hindi-tamil-question-answering",
    "champs-scalar-coupling": "champs-scalar-coupling",
    "facebook-recruiting-iii-keyword-extraction": "facebook-recruiting-iii-keyword-extraction",
    "freesound-audio-tagging-2019": "freesound-audio-tagging-2019",
    "google-quest-challenge": "google-quest-challenge",
    "google-research-identify-contrails-reduce-global-warming": "google-research-identify-contrails-reduce-global-warming",
    "h-and-m-personalized-fashion-recommendations": "h-and-m-personalized-fashion-recommendations",
    "herbarium-2020-fgvc7": "herbarium-2020-fgvc7",
    "herbarium-2021-fgvc8": "herbarium-2021-fgvc8",
    "herbarium-2022-fgvc9": "herbarium-2022-fgvc9",
    "hms-harmful-brain-activity-classification": "hms-harmful-brain-activity-classification",
    "hotel-id-2021-fgvc8": "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation": "hubmap-kidney-segmentation",
    "icecube-neutrinos-in-deep-ice": "icecube-neutrinos-in-deep-ice",
    "imet-2020-fgvc7": "imet-2020-fgvc7",
    "inaturalist-2019-fgvc6": "inaturalist-2019-fgvc6",
    "invasive-species-monitoring": "invasive-species-monitoring",
    "iwildcam-2019-fgvc6": "iwildcam-2019-fgvc6",
    "iwildcam-2020-fgvc7": "iwildcam-2020-fgvc7",
    "jigsaw-unintended-bias-in-toxicity-classification": "jigsaw-unintended-bias-in-toxicity-classification",
    "kuzushiji-recognition": "kuzushiji-recognition",
    "learning-agency-lab-automated-essay-scoring-2": "learning-agency-lab-automated-essay-scoring-2",
    "lmsys-chatbot-arena": "lmsys-chatbot-arena",
    "ml2021spring-hw2": "ml2021spring-hw2",
    "movie-review-sentiment-analysis-kernels-only": "movie-review-sentiment-analysis-kernels-only",
    "multi-modal-gesture-recognition": "multi-modal-gesture-recognition",
    "nfl-player-contact-detection": "nfl-player-contact-detection",
    "osic-pulmonary-fibrosis-progression": "osic-pulmonary-fibrosis-progression",
    "paddy-disease-classification": "paddy-disease-classification",
    "petfinder-pawpularity-score": "petfinder-pawpularity-score",
    "plant-pathology-2021-fgvc8": "plant-pathology-2021-fgvc8",
    "plant-seedlings-classification": "plant-seedlings-classification",
    "playground-series-s3e18": "playground-series-s3e18",
    "predict-volcanic-eruptions-ingv-oe": "predict-volcanic-eruptions-ingv-oe",
    "rsna-2022-cervical-spine-fracture-detection": "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection": "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification": "rsna-miccai-brain-tumor-radiogenomic-classification",
    "seti-breakthrough-listen": "seti-breakthrough-listen",
    "siim-covid19-detection": "siim-covid19-detection",
    "smartphone-decimeter-2022": "smartphone-decimeter-2022",
    "spaceship-titanic": "spaceship-titanic",
    "stanford-covid-vaccine": "stanford-covid-vaccine",
    "statoil-iceberg-classifier-challenge": "statoil-iceberg-classifier-challenge",
    "tensorflow-speech-recognition-challenge": "tensorflow-speech-recognition-challenge",
    "tensorflow2-question-answering": "tensorflow2-question-answering",
    "tgs-salt-identification-challenge": "tgs-salt-identification-challenge",
    "tweet-sentiment-extraction": "tweet-sentiment-extraction",
    "us-patent-phrase-to-phrase-matching": "us-patent-phrase-to-phrase-matching",
    "uw-madison-gi-tract-image-segmentation": "uw-madison-gi-tract-image-segmentation",
    "ventilator-pressure-prediction": "ventilator-pressure-prediction",
    "vesuvius-challenge-ink-detection": "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection": "vinbigdata-chest-xray-abnormalities-detection",
    "whale-categorization-playground": "whale-categorization-playground",
}


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except:
        pass


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


def save_checkpoint(results):
    ckpt = {
        "updated_at": datetime.now().isoformat(),
        "results": results,
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def extract_archive(arc_path):
    """Extract archive using Python stdlib (no external tools needed)."""
    import zipfile, tarfile
    arc_str = str(arc_path)
    out_dir = str(arc_path.parent)
    try:
        if arc_str.endswith('.zip'):
            with zipfile.ZipFile(arc_str, 'r') as zf:
                zf.extractall(out_dir)
            return True
        elif arc_str.endswith('.tar.gz') or arc_str.endswith('.tgz'):
            with tarfile.open(arc_str, 'r:gz') as tf:
                tf.extractall(out_dir)
            return True
        elif arc_str.endswith('.tar'):
            with tarfile.open(arc_str, 'r:') as tf:
                tf.extractall(out_dir)
            return True
        elif arc_str.endswith('.7z'):
            result = subprocess.run(["7z", "x", arc_str, f"-o{out_dir}", "-y"],
                                    capture_output=True, timeout=120)
            return result.returncode == 0
    except Exception as e:
        log(f"    Extract error for {arc_path.name}: {e}")
        return False


def download_data(comp_id):
    """Download Kaggle competition data. Uses Python stdlib for extraction."""
    slug = KAGGLE_SLUGS.get(comp_id, comp_id)
    comp_dir = DATA_DIR / comp_id
    comp_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(comp_dir.glob("*.csv"))
    if csv_files:
        log(f"  [{comp_id}] Data exists ({len(csv_files)} csv files)")
        return True

    log(f"  [{comp_id}] Downloading from Kaggle: {slug}...")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", slug, "-p", str(comp_dir)],
            capture_output=True, text=True, timeout=300
        )
        log(f"  Download stdout: {result.stdout[:200]}")
        if result.returncode != 0:
            log(f"  Download stderr: {result.stderr[:200]}")

        # Extract all archives using Python stdlib
        arc_patterns = ["*.zip", "*.7z", "*.tar.gz", "*.tar", "*.tgz"]
        for pattern in arc_patterns:
            for arc in list(comp_dir.glob(pattern)):
                if extract_archive(arc):
                    arc.unlink()
                    log(f"    Extracted: {arc.name}")

        csv_files = list(comp_dir.glob("*.csv"))
        if csv_files:
            log(f"  [{comp_id}] Downloaded: {[f.name for f in csv_files[:5]]}")
            return True
        else:
            # Check subdirectories for CSVs
            all_csvs = list(comp_dir.rglob("*.csv"))
            log(f"  [{comp_id}] CSVs found (with subdirs): {len(all_csvs)}")
            return len(all_csvs) > 0
    except Exception as e:
        log(f"  [{comp_id}] Download error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# TABULAR TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def train_tabular(comp_id, gpu_id=0, seed=42):
    """Autonomous tabular ML: auto-detect target, preprocess, ensemble, predict."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import accuracy_score, mean_squared_error

    comp_dir = DATA_DIR / comp_id

    # Auto-find CSV files
    train_file = None
    test_file = None
    for f in sorted(comp_dir.glob("*.csv")):
        name = f.stem.lower()
        if "train" in name:
            train_file = f
        elif "test" in name or "sample" in name:
            test_file = f
    if train_file is None:
        csvs = sorted(comp_dir.glob("*.csv"))
        if len(csvs) >= 2:
            train_file = csvs[0]
            test_file = csvs[1]
        elif len(csvs) == 1:
            train_file = csvs[0]
    if train_file is None:
        log(f"  [{comp_id}] No CSV found in {comp_dir}")
        return None

    log(f"  [{comp_id}] Train: {train_file.name}")

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file) if test_file and test_file != train_file else None
    log(f"  [{comp_id}] Train shape: {train.shape}, Test: {test.shape if test is not None else 'None'}")

    # Auto-detect target column (last column with <50% unique values)
    target_col = None
    skip_cols = {"id", "Id", "ID", "ImageId", "PassengerId", "image_id",
                 "img_id", "filename", "file_name", "path", "url"}
    for col in reversed(list(train.columns)):
        if col.lower() not in {s.lower() for s in skip_cols}:
            n_unique = train[col].nunique()
            if n_unique < len(train) * 0.5 and n_unique > 1:
                target_col = col
                break
    if target_col is None:
        target_col = train.columns[-1]
    log(f"  [{comp_id}] Target: '{target_col}' (nunique={train[target_col].nunique()})")

    # Separate target
    y = train[target_col].copy()
    X = train.drop(columns=[target_col])

    # Remove ID columns
    id_cols = [c for c in X.columns if c.lower() in
               {"id", "passengerid", "imageid", "img_id", "image_id", "filename"}]
    common_ids = [c for c in id_cols if c in X.columns]
    X = X.drop(columns=common_ids, errors="ignore")
    if test is not None:
        test_ids_in_test = [c for c in common_ids if c in test.columns]
        test = test.drop(columns=test_ids_in_test, errors="ignore")

    # Split numeric/categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute numeric
    X_num = pd.DataFrame(index=X.index)
    if numeric_cols:
        imp = SimpleImputer(strategy="median")
        X_num = pd.DataFrame(imp.fit_transform(X[numeric_cols]), columns=numeric_cols, index=X.index)

    # Encode categorical
    X_cat = pd.DataFrame(index=X.index)
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        all_vals = list(X[c].astype(str).fillna("MISSING").values)
        if test is not None and c in test.columns:
            all_vals += list(test[c].astype(str).fillna("MISSING").values)
        le.fit(all_vals)
        X_cat[c] = le.transform(X[c].astype(str).fillna("MISSING").values)
        encoders[c] = le

    X_processed = pd.concat([X_num, X_cat], axis=1)
    if X_processed.empty:
        # Fallback: use only numeric
        X_processed = X_num

    # Process test
    test_processed = None
    if test is not None:
        test_num = pd.DataFrame(index=test.index)
        if numeric_cols:
            test_num_cols = [c for c in numeric_cols if c in test.columns]
            if test_num_cols:
                test_num = pd.DataFrame(imp.transform(test[test_num_cols]), columns=test_num_cols, index=test.index)
        test_cat = pd.DataFrame(index=test.index)
        for c, le in encoders.items():
            if c in test.columns:
                test_cat[c] = le.transform(test[c].astype(str).fillna("MISSING").values)
        test_processed = pd.concat([test_num, test_cat], axis=1)
        # Align columns
        for col in X_processed.columns:
            if col not in test_processed.columns:
                test_processed[col] = 0
        test_processed = test_processed[X_processed.columns]

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_processed), columns=X_processed.columns)
    if test_processed is not None:
        test_scaled = pd.DataFrame(scaler.transform(test_processed), columns=test_processed.columns)
    else:
        test_scaled = None

    X_np = X_scaled.values.astype(np.float32)
    y_np = y.values
    X_test_np = test_scaled.values.astype(np.float32) if test_scaled is not None else None

    # Determine classification vs regression
    is_clf = y.dtype in [np.int64, np.int32, int, bool] or y.nunique() <= 30
    if is_clf and y.nunique() > 100:
        is_clf = False

    if is_clf:
        from sklearn.preprocessing import LabelEncoder as LE2
        y_le = LE2()
        y_np = y_le.fit_transform(y_np.astype(str))

    n_folds = min(5, y.nunique()) if is_clf else 5
    if len(y_np) < 100:
        n_folds = min(3, max(1, len(y_np) // 20))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed) if is_clf else \
         KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Train ensemble
    oof_preds = []
    test_preds = []
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_np, y_np)):
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        fold_preds = []

        # CatBoost
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            if is_clf:
                cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                        task_type="GPU", devices=[gpu_id],
                                        random_seed=seed+fold, verbose=0, early_stopping_rounds=50)
                cb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                p = cb.predict_proba(X_va)
                fold_preds.append(p[:, 1] if p.shape[1] > 1 else cb.predict(X_va).astype(float))
                if X_test_np is not None:
                    tp = cb.predict_proba(X_test_np)
                    test_preds.append(tp[:, 1] if tp.shape[1] > 1 else cb.predict(X_test_np).astype(float))
            else:
                cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                       task_type="GPU", devices=[gpu_id],
                                       random_seed=seed+fold, verbose=0, early_stopping_rounds=50)
                cb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                fold_preds.append(cb.predict(X_va))
                if X_test_np is not None:
                    test_preds.append(cb.predict(X_test_np))
        except Exception as e:
            log(f"  Fold {fold+1} CatBoost error: {e}")

        # LightGBM
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            if is_clf:
                lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                     random_state=seed+fold, verbose=-1)
                lgb.fit(X_tr, y_tr)
                p = lgb.predict_proba(X_va)
                fold_preds.append(p[:, 1] if p.shape[1] > 1 else lgb.predict(X_va).astype(float))
                if X_test_np is not None:
                    tp = lgb.predict_proba(X_test_np)
                    test_preds.append(tp[:, 1] if tp.shape[1] > 1 else lgb.predict(X_test_np).astype(float))
            else:
                lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                    random_state=seed+fold, verbose=-1)
                lgb.fit(X_tr, y_tr)
                fold_preds.append(lgb.predict(X_va))
                if X_test_np is not None:
                    test_preds.append(lgb.predict(X_test_np))
        except Exception as e:
            log(f"  Fold {fold+1} LightGBM error: {e}")

        if not fold_preds:
            log(f"  Fold {fold+1} - no models succeeded!")
            continue

        fold_avg = np.mean(fold_preds, axis=0)
        oof_preds.append((va_idx, fold_avg))

        if is_clf:
            score = accuracy_score(y_va, (fold_avg > 0.5).astype(int) if fold_avg.ndim == 1 else fold_avg.argmax(axis=1))
        else:
            score = np.sqrt(mean_squared_error(y_va, fold_avg))
        scores.append(score)
        log(f"  Fold {fold+1}: score={score:.5f}")

    if not scores:
        log(f"  [{comp_id}] No successful folds!")
        return None

    # Compute OOF
    oof = np.zeros(len(y_np))
    for idx, pred in oof_preds:
        oof[idx] = pred

    # Compute test predictions
    if test_preds:
        test_pred = np.mean(test_preds, axis=0)
    else:
        test_pred = oof

    # Generate submission CSV
    if test is not None:
        sub = pd.DataFrame({"id": range(len(test_pred)), "prediction": test_pred})
    else:
        sub = pd.DataFrame({"id": range(len(oof)), "prediction": oof})

    # Save
    out_dir = RESULTS_DIR / comp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_dir / f"submission_seed{seed}.csv", index=False)

    result = {
        "comp_id": comp_id,
        "seed": seed,
        "n_samples": int(len(y_np)),
        "n_features": int(X_scaled.shape[1]),
        "is_classification": is_clf,
        "n_folds": n_folds,
        "oof_score_mean": float(np.mean(scores)),
        "oof_score_std": float(np.std(scores)) if len(scores) > 1 else 0.0,
        "folds_completed": len(scores),
        "gpu_id": gpu_id,
        "timestamp": datetime.now().isoformat(),
    }
    json.dump(result, open(out_dir / f"result_seed{seed}.json", "w"), indent=2)

    log(f"  [{comp_id}] RESULT: oof={result['oof_score_mean']:.5f} ± {result['oof_score_std']:.5f}, folds={len(scores)}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CV TRAINING ENGINE (PyTorch ResNet50)
# ═══════════════════════════════════════════════════════════════════════════

def train_cv(comp_id, gpu_id=0, epochs=10, batch_size=32):
    """PyTorch ResNet50 fine-tuning for image classification."""
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  [{comp_id}] Device: {device}")

    comp_dir = DATA_DIR / comp_id

    # Find image data
    img_col = None
    label_cols = []
    df = None

    for csv_file in sorted(comp_dir.glob("*.csv")):
        try:
            df_tmp = pd.read_csv(csv_file)
            for c in df_tmp.columns:
                if c.lower() in {"image", "img", "filename", "file", "path", "image_path", "file_path"}:
                    if df_tmp[c].astype(str).str.contains(r'\.(jpg|png|jpeg|tif|dcm)', case=False, regex=True).any():
                        img_col = c
                        break
            if img_col:
                label_cols = [c for c in df_tmp.columns if c != img_col]
                df = df_tmp
                log(f"  [{comp_id}] Found: {csv_file.name}, img={img_col}, labels={label_cols[:5]}")
                break
        except:
            continue

    if df is None:
        # Try to find image directories
        log(f"  [{comp_id}] No CSV with image paths, searching directories...")
        for ext in [".jpg", ".png", ".jpeg"]:
            imgs = list(comp_dir.rglob(f"*{ext}"))
            if imgs:
                log(f"  [{comp_id}] Found {len(imgs)} images")
                df = pd.DataFrame({"path": [str(f) for f in imgs]})
                img_col = "path"
                label_cols = []
                break

    if df is None:
        log(f"  [{comp_id}] No image data found, falling back to tabular")
        return train_tabular(comp_id, gpu_id)

    num_labels = len(label_cols) if label_cols else 1

    class ImageDS(Dataset):
        def __init__(self, dataframe, transform=None, is_test=False):
            self.df = dataframe
            self.transform = transform
            self.is_test = is_test

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            p = Path(str(row[img_col]))
            if not p.is_absolute():
                p = comp_dir / p
            try:
                img = Image.open(p).convert("RGB")
            except:
                img = Image.new("RGB", (224, 224))
            if self.transform:
                img = self.transform(img)
            if self.is_test or not label_cols:
                return img, idx
            labels = [float(row[c]) if not pd.isna(row[c]) else 0.0 for c in label_cols]
            return img, torch.tensor(labels, dtype=torch.float32)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    from sklearn.model_selection import train_test_split
    tr_df, va_df = train_test_split(df, test_size=0.2, random_state=42)
    tr_ds = ImageDS(tr_df, train_tf)
    va_ds = ImageDS(va_df, val_tf)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss() if num_labels > 1 else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss = float("inf")
    best_path = f"/tmp/{comp_id}_best.pth"
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for imgs, labels in tr_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for imgs, labels in va_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                va_loss += criterion(model(imgs), labels).item()

        scheduler.step()
        if va_loss < best_loss:
            best_loss = va_loss
            torch.save(model.state_dict(), best_path)

        if epoch % 3 == 0:
            log(f"  Epoch {epoch+1}/{epochs}: tr={tr_loss/len(tr_dl):.4f}, va={va_loss/len(va_dl):.4f}")

    model.load_state_dict(torch.load(best_path))

    # Generate predictions on full dataset
    full_ds = ImageDS(df, val_tf, is_test=not bool(label_cols))
    full_dl = DataLoader(full_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _ in full_dl:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds.append(outputs.cpu().numpy())
    all_preds = np.concatenate(preds)

    # Save
    out_dir = RESULTS_DIR / comp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": range(len(all_preds)), "prediction": all_preds[:, 0] if all_preds.ndim > 1 else all_preds})
    sub.to_csv(out_dir / "submission_cv.csv", index=False)

    result = {
        "comp_id": comp_id,
        "model": "resnet50",
        "epochs": epochs,
        "num_labels": num_labels,
        "n_samples": len(df),
        "best_val_loss": float(best_loss),
        "gpu_id": gpu_id,
        "timestamp": datetime.now().isoformat(),
    }
    json.dump(result, open(out_dir / "result_cv.json", "w"), indent=2)

    log(f"  [{comp_id}] CV RESULT: val_loss={best_loss:.5f}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_all(start_from=0, single=None, max_seeds=3):
    global_start = datetime.now()
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get("completed", []) if checkpoint else [])
    failed = set(checkpoint.get("failed", []) if checkpoint else [])

    log(f"{'='*60}")
    log(f"MLE-Bench 75 Autonomous Runner STARTING")
    log(f"Server: AI-X86_NVIDIA GPU Server (87739), 2x A40 49GB")
    log(f"Data: {DATA_DIR}  |  Results: {RESULTS_DIR}")
    log(f"Already completed: {len(completed)}  |  Failed: {len(failed)}")
    log(f"{'='*60}")

    competition_list = COMPETITIONS
    if single:
        competition_list = [(c[0], c[1], c[2]) for c in COMPETITIONS if c[0] == single]
        if not competition_list:
            log(f"ERROR: Competition '{single}' not found!")
            return

    for i, (comp_id, tier, ctype) in enumerate(competition_list):
        if i < start_from:
            continue
        if comp_id in completed:
            log(f"[{i+1}/{len(competition_list)}] {comp_id} - SKIP (already completed)")
            continue

        gpu_id = 0 if i % 2 == 0 else 1
        log(f"\n{'#'*60}")
        log(f"[{i+1}/{len(competition_list)}] {comp_id} ({tier}, {ctype}) - GPU {gpu_id}")
        log(f"Elapsed: {(datetime.now() - global_start).total_seconds()/3600:.1f}h")
        log(f"{'#'*60}")

        t_start = time.time()

        try:
            # Step 1: Download data
            if not download_data(comp_id):
                log(f"  [{comp_id}] FAILED: data download")
                failed.add(comp_id)
                save_checkpoint({"completed": list(completed), "failed": list(failed)})
                continue

            # Step 2: Train
            best = None
            for seed_idx in range(max_seeds):
                s = 42 + seed_idx
                log(f"  Seed {seed_idx+1}/{max_seeds} (seed={s})...")

                try:
                    if ctype == "tabular":
                        res = train_tabular(comp_id, gpu_id, s)
                    elif ctype in ("image_classification", "image_segmentation"):
                        res = train_cv(comp_id, gpu_id, epochs=10)
                    else:
                        res = train_tabular(comp_id, gpu_id, s)

                    if res and res.get("oof_score_mean") is not None:
                        score = res["oof_score_mean"]
                        if best is None or score < best["oof_score_mean"]:
                            best = res
                        log(f"  Seed {seed_idx+1}: score={score:.5f}")
                except Exception as e:
                    log(f"  Seed {seed_idx+1} ERROR: {e}")
                    traceback.print_exc()

            if best:
                log(f"  [{comp_id}] BEST: score={best['oof_score_mean']:.5f}")
                completed.add(comp_id)
                if comp_id in failed:
                    failed.discard(comp_id)
            else:
                log(f"  [{comp_id}] No successful training runs")
                # Try minimal fallback approach
                try:
                    log(f"  [{comp_id}] Attempting fallback training...")
                    res = train_tabular(comp_id, gpu_id, 42)
                    if res:
                        completed.add(comp_id)
                        log(f"  [{comp_id}] Fallback succeeded: score={res['oof_score_mean']:.5f}")
                    else:
                        failed.add(comp_id)
                except:
                    failed.add(comp_id)

            elapsed = time.time() - t_start
            log(f"  [{comp_id}] Time: {elapsed/60:.1f}min")

        except Exception as e:
            log(f"  [{comp_id}] CRASHED: {e}")
            traceback.print_exc()
            failed.add(comp_id)

        # Save checkpoint every competition
        save_checkpoint({"completed": list(completed), "failed": list(failed)})

    # Final report
    total_time = (datetime.now() - global_start).total_seconds() / 3600
    log(f"\n{'='*60}")
    log(f"MLE-Bench 75 Autonomous Runner - COMPLETE")
    log(f"Completed: {len(completed)}/{len(competition_list)}")
    log(f"Failed: {len(failed)}")
    log(f"Total time: {total_time:.1f}h")
    log(f"{'='*60}")

    # Save final report
    report = {
        "schema": "mlebench_autonomous_results.v1",
        "generated_at": datetime.now().isoformat(),
        "total": len(competition_list),
        "completed": len(completed),
        "failed": len(failed),
        "completed_list": sorted(completed),
        "failed_list": sorted(failed),
        "total_time_hours": total_time,
    }
    json.dump(report, open(BASE_DIR / "mlebench_75_final_report.json", "w"), indent=2)
    log(f"Report saved: {BASE_DIR / 'mlebench_75_final_report.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--single", type=str, help="Run single competition")
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=3)
    parser.add_argument("--test", action="store_true", help="Run spaceship-titanic as quick test")
    args = parser.parse_args()

    if args.test:
        run_all(start_from=0, single="spaceship-titanic", max_seeds=1)
    else:
        run_all(start_from=args.start_from, single=args.single, max_seeds=args.max_seeds)
