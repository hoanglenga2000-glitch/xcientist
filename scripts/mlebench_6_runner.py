#!/usr/bin/env python3
"""
MLE-Bench 6-Competition Accessible Runner - Server Side
Runs only the 6 competitions that are accessible via Kaggle API
"""
import pandas as pd, numpy as np, json, sys, os, time, gc, subprocess, traceback
from pathlib import Path
from datetime import datetime

HOME = Path.home()
BASE_DIR = HOME / "jinghw" / "scripts" / "gpu_tra"
DATA_DIR = BASE_DIR / "mlebench_data"
RESULTS_DIR = BASE_DIR / "mlebench_results"
CHECKPOINT_PATH = BASE_DIR / "mlebench_6_checkpoint.json"
LOG_PATH = BASE_DIR / "mlebench_6_runner.log"

for d in [DATA_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Only the 6 competitions we confirmed as accessible
COMPETITIONS = [
    ("spaceship-titanic", "medium_high", "tabular", "spaceship-titanic"),
    ("dog-breed-identification", "lite", "image_classification", "dog-breed-identification"),
    ("tabular-playground-series-dec-2021", "lite", "tabular", "tabular-playground-series-dec-2021"),
    ("tabular-playground-series-may-2022", "lite", "tabular", "tabular-playground-series-may-2022"),
    ("lmsys-chatbot-arena", "medium_high", "nlp", "lmsys-chatbot-arena"),
    ("multi-modal-gesture-recognition", "medium_high", "tabular", "multi-modal-gesture-recognition"),
]

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f: return json.load(f)
    return None

def save_checkpoint(results):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"updated_at": datetime.now().isoformat(), "results": results}, f, indent=2)

def download_data(comp_id, kaggle_slug):
    comp_dir = DATA_DIR / comp_id
    comp_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list(comp_dir.glob("*.csv"))
    if csv_files:
        log(f"  [{comp_id}] Data exists ({len(csv_files)} csvs)")
        return True
    log(f"  [{comp_id}] Downloading {kaggle_slug}...")
    try:
        r = subprocess.run(["kaggle", "competitions", "download", "-c", kaggle_slug, "-p", str(comp_dir)],
                          capture_output=True, text=True, timeout=300)
        import zipfile
        for z in list(comp_dir.glob("*.zip")):
            with zipfile.ZipFile(str(z)) as zf: zf.extractall(str(comp_dir))
            z.unlink()
            log(f"    Extracted: {z.name}")
        csvs = list(comp_dir.glob("*.csv")) + list(comp_dir.rglob("*.csv"))
        return len(csvs) > 0
    except Exception as e:
        log(f"  [{comp_id}] Error: {e}")
        return False

def train_tabular(comp_id, gpu_id=0, seed=42):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import accuracy_score, mean_squared_error

    comp_dir = DATA_DIR / comp_id
    train_file = None; test_file = None
    for f in sorted(comp_dir.glob("*.csv")):
        n = f.stem.lower()
        if "train" in n: train_file = f
        elif "test" in n: test_file = f
    if train_file is None:
        csvs = sorted(comp_dir.glob("*.csv"))
        train_file = csvs[0] if csvs else None
        test_file = csvs[1] if len(csvs) > 1 else None
    if train_file is None:
        log(f"  [{comp_id}] No CSV found")
        return None

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file) if test_file and test_file != train_file else None
    log(f"  [{comp_id}] Train: {train.shape}")

    target_col = None
    skip = {"id", "Id", "ID", "ImageId", "PassengerId"}
    for col in reversed(list(train.columns)):
        if col.lower() not in {s.lower() for s in skip}:
            if train[col].nunique() < len(train) * 0.5 and train[col].nunique() > 1:
                target_col = col; break
    if target_col is None: target_col = train.columns[-1]
    log(f"  [{comp_id}] Target: '{target_col}'")

    y = train[target_col].copy()
    X = train.drop(columns=[target_col])
    id_cols = [c for c in X.columns if c.lower() in {"id","passengerid","imageid","img_id","filename"}]
    X = X.drop(columns=[c for c in id_cols if c in X.columns], errors="ignore")
    if test is not None:
        test = test.drop(columns=[c for c in id_cols if c in test.columns], errors="ignore")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    imp = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(imp.fit_transform(X[num_cols]), columns=num_cols, index=X.index) if num_cols else pd.DataFrame(index=X.index)
    encs = {}
    X_cat = pd.DataFrame(index=X.index)
    for c in cat_cols:
        le = LabelEncoder()
        vals = list(X[c].astype(str).fillna("MISSING"))
        if test is not None and c in test.columns: vals += list(test[c].astype(str).fillna("MISSING"))
        le.fit(vals)
        X_cat[c] = le.transform(X[c].astype(str).fillna("MISSING"))
        encs[c] = le
    Xp = pd.concat([X_num, X_cat], axis=1)

    test_p = None
    if test is not None:
        tn = pd.DataFrame(imp.transform(test[[c for c in num_cols if c in test.columns]]), columns=[c for c in num_cols if c in test.columns], index=test.index) if num_cols else pd.DataFrame(index=test.index)
        tc = pd.DataFrame(index=test.index)
        for c, le in encs.items():
            if c in test.columns: tc[c] = le.transform(test[c].astype(str).fillna("MISSING"))
        test_p = pd.concat([tn, tc], axis=1)
        for col in Xp.columns:
            if col not in test_p.columns: test_p[col] = 0
        test_p = test_p[Xp.columns]

    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(Xp), columns=Xp.columns)
    Xt = pd.DataFrame(scaler.transform(test_p), columns=test_p.columns) if test_p is not None else None

    Xn = Xs.values.astype(np.float32)
    yn = y.values
    Xtn = Xt.values.astype(np.float32) if Xt is not None else None

    is_clf = y.dtype in [np.int64, np.int32, int, bool] or y.nunique() <= 30
    if is_clf and y.nunique() > 100: is_clf = False
    if is_clf:
        yle = LabelEncoder()
        yn = yle.fit_transform(yn.astype(str))
    nf = min(5, y.nunique()) if is_clf else 5
    nf = max(2, min(nf, len(yn)//20))
    cv = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed) if is_clf else KFold(n_splits=nf, shuffle=True, random_state=seed)

    oof_p = []; test_ps = []; scores = []; fold_preds_all = []
    for fold, (ti, vi) in enumerate(cv.split(Xn, yn)):
        xtr, xva = Xn[ti], Xn[vi]; ytr, yva = yn[ti], yn[vi]
        fps = []
        # CatBoost
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            if is_clf:
                cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, task_type="GPU", devices=[gpu_id], random_seed=seed+fold, verbose=0, early_stopping_rounds=50)
                cb.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
                p = cb.predict_proba(xva); fps.append(p[:,1] if p.shape[1]>1 else cb.predict(xva).astype(float))
                if Xtn is not None: tp = cb.predict_proba(Xtn); test_ps.append(tp[:,1] if tp.shape[1]>1 else cb.predict(Xtn).astype(float))
            else:
                cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, task_type="GPU", devices=[gpu_id], random_seed=seed+fold, verbose=0, early_stopping_rounds=50)
                cb.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
                fps.append(cb.predict(xva))
                if Xtn is not None: test_ps.append(cb.predict(Xtn))
        except Exception as e: log(f"  CB error f{fold}: {e}")

        # LightGBM
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            if is_clf:
                lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed+fold, verbose=-1)
                lgb.fit(xtr, ytr); p = lgb.predict_proba(xva)
                fps.append(p[:,1] if p.shape[1]>1 else lgb.predict(xva).astype(float))
                if Xtn is not None: tp = lgb.predict_proba(Xtn); test_ps.append(tp[:,1] if tp.shape[1]>1 else lgb.predict(Xtn).astype(float))
            else:
                lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed+fold, verbose=-1)
                lgb.fit(xtr, ytr); fps.append(lgb.predict(xva))
                if Xtn is not None: test_ps.append(lgb.predict(Xtn))
        except Exception as e: log(f"  LGB error f{fold}: {e}")

        if not fps: continue
        fa = np.mean(fps, axis=0); oof_p.append((vi, fa))
        sc = accuracy_score(yva, (fa>0.5).astype(int) if fa.ndim==1 else fa.argmax(axis=1)) if is_clf else np.sqrt(mean_squared_error(yva, fa))
        scores.append(sc); log(f"  Fold {fold+1}: {sc:.5f}")

    if not scores: return None
    oof = np.zeros(len(yn))
    for idx,p in oof_p: oof[idx] = p
    tp_final = np.mean(test_ps, axis=0) if test_ps else oof

    if test is not None:
        sub = pd.DataFrame({"id": range(len(tp_final)), "prediction": tp_final})
    else:
        sub = pd.DataFrame({"id": range(len(oof)), "prediction": oof})

    od = RESULTS_DIR / comp_id; od.mkdir(parents=True, exist_ok=True)
    sub.to_csv(od / f"submission_s{seed}.csv", index=False)
    result = {"comp_id":comp_id, "seed":seed, "n_samples":int(len(yn)), "n_features":Xp.shape[1], "is_clf":is_clf, "n_folds":nf, "oof_mean":float(np.mean(scores)), "oof_std":float(np.std(scores)) if len(scores)>1 else 0.0, "folds_ok":len(scores), "gpu":gpu_id}
    json.dump(result, open(od / f"result_s{seed}.json","w"), indent=2)
    log(f"  [{comp_id}] RESULT: {result['oof_mean']:.5f} +/- {result['oof_std']:.5f}")
    return result

def run_all():
    global_start = datetime.now()
    ckpt = load_checkpoint()
    completed = set(ckpt.get("completed",[]) if ckpt else [])
    log(f"{'='*60}")
    log(f"MLE-Bench Accessible-6 Runner — {len(COMPETITIONS)} competitions")
    log(f"{'='*60}")

    for i,(comp_id, tier, ctype, slug) in enumerate(COMPETITIONS):
        if comp_id in completed:
            log(f"[{i+1}/6] {comp_id} - SKIP"); continue
        gpu = 0 if i%2==0 else 1
        log(f"\n{'#'*60}\n[{i+1}/6] {comp_id} ({tier}, {ctype}) GPU={gpu}\n{'#'*60}")
        t0 = time.time()
        try:
            if not download_data(comp_id, slug):
                log(f"  [{comp_id}] Data FAIL"); continue
            best = None
            for si in range(3):
                s = 42+si
                try:
                    r = train_tabular(comp_id, gpu, s)
                    if r and (best is None or r["oof_mean"] < best["oof_mean"]): best = r
                except Exception as e: log(f"  Seed {si+1} ERR: {e}")
            if best:
                log(f"  [{comp_id}] BEST: {best['oof_mean']:.5f}")
                completed.add(comp_id)
            else:
                log(f"  [{comp_id}] ALL SEEDS FAILED")
        except Exception as e:
            log(f"  [{comp_id}] CRASH: {e}"); traceback.print_exc()
        log(f"  [{comp_id}] Time: {(time.time()-t0)/60:.1f}min")
        save_checkpoint({"completed": list(completed)})

    elapsed = (datetime.now()-global_start).total_seconds()/3600
    log(f"\n{'='*60}\nDone: {len(completed)}/6 in {elapsed:.1f}h\n{'='*60}")
    json.dump({"completed":sorted(completed),"total_hours":elapsed}, open(BASE_DIR/"mlebench_6_report.json","w"), indent=2)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--single", type=str)
    args = p.parse_args()
    if args.single:
        for cid,tier,ctype,slug in COMPETITIONS:
            if cid == args.single:
                download_data(cid, slug)
                train_tabular(cid, 0, 42)
    else:
        run_all()
