#!/usr/bin/env python3
"""MLE-Bench Fix Runner — handles the 3 failed competitions from the 6-comp run.

1. dog-breed-identification: Image classification with pretrained ResNet50
2. lmsys-chatbot-arena: NLP with TF-IDF + LightGBM
3. multi-modal-gesture-recognition: Tabular with larger timeout
"""
import os, sys, json, time, subprocess, zipfile, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path.home() / "jinghw" / "scripts" / "gpu_tra" / "mlebench_data"
RESULTS_DIR = Path.home() / "jinghw" / "scripts" / "gpu_tra" / "mlebench_results"
CHECKPOINT_FILE = Path.home() / "jinghw" / "scripts" / "gpu_tra" / "mlebench_6_fix_checkpoint.json"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"completed": []}


def save_checkpoint(results):
    CHECKPOINT_FILE.write_text(json.dumps(results, indent=2))


def download_kaggle_zip(slug, data_dir, timeout=600):
    """Download competition data via kaggle CLI with retry."""
    zip_path = data_dir / f"{slug}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 100000:
        log(f"  [{slug}] Zip exists ({zip_path.stat().st_size/1024/1024:.0f}MB)")
        return zip_path

    # Remove partial downloads
    for f in data_dir.glob(f"{slug}*"):
        f.unlink()

    log(f"  [{slug}] Downloading (timeout={timeout}s)...")
    try:
        result = subprocess.run([
            "kaggle", "competitions", "download", "-c", slug,
            "-p", str(data_dir)
        ], capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            log(f"  [{slug}] Download FAILED: {result.stderr.strip()}")
            return None
        log(f"  [{slug}] Download OK")
    except subprocess.TimeoutExpired:
        log(f"  [{slug}] Download TIMEOUT after {timeout}s")
        return None

    zip_path = data_dir / f"{slug}.zip"
    if zip_path.exists():
        return zip_path
    return None


def extract_zip(zip_path, extract_dir):
    """Extract zip if directory doesn't have CSVs."""
    csvs = list(extract_dir.glob("*.csv"))
    if csvs:
        return True
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        log(f"    Extracted: {zip_path.name}")
        return True
    except Exception as e:
        log(f"    Extract error: {e}")
        return False


# ============================================================
# Task 1: dog-breed-identification (Image Classification)
# ============================================================
def train_dog_breed(seed=42):
    """Train image classifier for dog breed identification."""
    comp_id = "dog-breed-identification"
    comp_dir = DATA_DIR / comp_id
    os.makedirs(comp_dir, exist_ok=True)

    zip_path = download_kaggle_zip(comp_id, comp_dir, timeout=300)
    if not zip_path:
        log(f"  [{comp_id}] No zip, trying to use existing data")
        # Check if images are already extracted
        train_img_dir = comp_dir / "train"
        if not train_img_dir.exists():
            return None

    # Extract if needed
    extract_zip(zip_path, comp_dir)

    train_img_dir = comp_dir / "train"
    test_img_dir = comp_dir / "test"
    labels_csv = comp_dir / "labels.csv"

    if not train_img_dir.exists() or not labels_csv.exists():
        log(f"  [{comp_id}] Missing data: train_dir={train_img_dir.exists()} labels={labels_csv.exists()}")
        return None

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, models
        from PIL import Image
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import StratifiedKFold

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"  [{comp_id}] Device: {device}")

        # Load labels
        labels_df = pd.read_csv(labels_csv)
        label_encoder = LabelEncoder()
        labels_df['label_idx'] = label_encoder.fit_transform(labels_df['breed'])
        n_classes = len(label_encoder.classes_)
        log(f"  [{comp_id}] Classes: {n_classes}, Images: {len(labels_df)}")

        # Dataset
        class DogDataset(Dataset):
            def __init__(self, df, img_dir, transform=None):
                self.df = df.reset_index(drop=True)
                self.img_dir = img_dir
                self.transform = transform

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                img_path = self.img_dir / f"{row['id']}.jpg"
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception:
                    img = Image.new('RGB', (224, 224), (128, 128, 128))
                if self.transform:
                    img = self.transform(img)
                return img, row['label_idx']

        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Model
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
        model = model.to(device)

        # 3-fold CV
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        fold_scores = []
        test_preds_all = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(labels_df, labels_df['label_idx'])):
            log(f"  [{comp_id}] Fold {fold+1}/3")

            train_df = labels_df.iloc[train_idx]
            val_df = labels_df.iloc[val_idx]

            train_ds = DogDataset(train_df, train_img_dir, train_transform)
            val_ds = DogDataset(val_df, train_img_dir, test_transform)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

            # Reset model
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

            best_acc = 0
            for epoch in range(10):
                model.train()
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(imgs), labels)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                acc = correct / total
                if acc > best_acc:
                    best_acc = acc
                log(f"    Epoch {epoch+1}: val_acc={acc:.4f}")

            fold_scores.append(best_acc)

            # Predict on test
            if test_img_dir.exists():
                test_files = sorted(test_img_dir.glob("*.jpg"))
                test_ds = DogDataset(pd.DataFrame({'id': [f.stem for f in test_files], 'label_idx': 0}), test_img_dir, test_transform)
                test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
                fold_preds = []
                model.eval()
                with torch.no_grad():
                    for imgs, _ in test_loader:
                        imgs = imgs.to(device)
                        outputs = model(imgs)
                        fold_preds.append(outputs.softmax(1).cpu().numpy())
                test_preds_all.append(np.concatenate(fold_preds))

        test_preds = np.mean(test_preds_all, axis=0) if test_preds_all else None
        cv_mean = np.mean(fold_scores)

        # Generate submission
        sub = None
        if test_preds is not None and test_img_dir.exists():
            test_files = sorted(test_img_dir.glob("*.jpg"))
            pred_classes = label_encoder.inverse_transform(test_preds.argmax(axis=1))
            sub = pd.DataFrame({"id": [f.stem for f in test_files], "breed": pred_classes})
            # Check sample format
            sample_path = comp_dir / "sample_submission.csv"
            if sample_path.exists():
                sample = pd.read_csv(sample_path)
                # Map to correct format
                sub.columns = sample.columns[:2]

        od = RESULTS_DIR / comp_id
        os.makedirs(od, exist_ok=True)
        if sub is not None:
            sub.to_csv(od / f"submission_s{seed}.csv", index=False)
        json.dump({"comp_id": comp_id, "seed": seed, "cv_mean": cv_mean, "fold_scores": fold_scores},
                  open(od / f"result_s{seed}.json", "w"), indent=2)

        log(f"  [{comp_id}] RESULT: CV={cv_mean:.4f}")
        return {"cv_mean": cv_mean, "submission": sub}

    except ImportError as e:
        log(f"  [{comp_id}] Import error: {e}")
        return None
    except Exception as e:
        log(f"  [{comp_id}] Error: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================
# Task 2: lmsys-chatbot-arena (NLP)
# ============================================================
def train_lmsys(seed=42):
    """Train NLP model for LMSYS Chatbot Arena."""
    comp_id = "lmsys-chatbot-arena"
    comp_dir = DATA_DIR / comp_id
    os.makedirs(comp_dir, exist_ok=True)

    zip_path = download_kaggle_zip(comp_id, comp_dir, timeout=300)
    if not zip_path:
        return None

    extract_zip(zip_path, comp_dir)

    train_file = comp_dir / "train.csv"
    test_file = comp_dir / "test.csv"

    if not train_file.exists():
        log(f"  [{comp_id}] No train.csv")
        return None

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score
        from lightgbm import LGBMClassifier

        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file) if test_file.exists() else None

        log(f"  [{comp_id}] Train: {train.shape}")

        # Identify target
        target_col = None
        skip = {"id", "model_a", "model_b", "prompt", "response_a", "response_b"}
        for col in reversed(list(train.columns)):
            if col not in skip and train[col].nunique() < len(train) * 0.3 and train[col].nunique() > 1:
                target_col = col
                break
        if target_col is None:
            target_col = "winner_tie"  # known from earlier run
        log(f"  [{comp_id}] Target: '{target_col}'")

        # Text columns
        text_cols = ["prompt", "response_a", "response_b"]
        available_text = [c for c in text_cols if c in train.columns]

        # TF-IDF encoding
        log(f"  [{comp_id}] TF-IDF on {len(available_text)} text columns...")
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', sublinear_tf=True)

        all_text = []
        for c in available_text:
            all_text.extend(train[c].fillna("").astype(str).tolist())
            if test is not None and c in test.columns:
                all_text.extend(test[c].fillna("").astype(str).tolist())

        tfidf.fit(all_text)

        train_feats = []
        test_feats = []
        for c in available_text:
            train_vec = tfidf.transform(train[c].fillna("").astype(str))
            test_vec = tfidf.transform(test[c].fillna("").astype(str)) if test is not None else None
            train_feats.append(pd.DataFrame(train_vec.toarray(), columns=[f"{c}_tfidf_{i}" for i in range(train_vec.shape[1])]))
            if test is not None:
                test_feats.append(pd.DataFrame(test_vec.toarray(), columns=[f"{c}_tfidf_{i}" for i in range(test_vec.shape[1])]))

        X = pd.concat(train_feats, axis=1)
        Xt = pd.concat(test_feats, axis=1) if test_feats else None

        y = train[target_col]
        yle = LabelEncoder()
        y_enc = yle.fit_transform(y.astype(str))

        log(f"  [{comp_id}] Features: {X.shape}")

        # LightGBM with CV
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        scores = []
        test_ps = []

        for fold, (ti, vi) in enumerate(cv.split(X, y_enc)):
            xtr, xva = X.iloc[ti], X.iloc[vi]
            ytr, yva = y_enc[ti], y_enc[vi]

            lgb = LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=8, num_leaves=64,
                                 random_state=seed+fold, verbose=-1, n_jobs=-1)
            lgb.fit(xtr, ytr)

            p = lgb.predict_proba(xva)
            pred = p.argmax(axis=1) if p.shape[1] > 1 else (p[:, 1] > 0.5).astype(int)
            sc = accuracy_score(yva, pred)
            scores.append(sc)
            log(f"  Fold {fold+1}: {sc:.4f}")

            if Xt is not None:
                tp = lgb.predict_proba(Xt)
                test_ps.append(tp)

        cv_mean = np.mean(scores)
        log(f"  [{comp_id}] CV: {cv_mean:.4f}")

        # Generate submission
        sub = None
        if Xt is not None and test_ps:
            tp_final = np.mean(test_ps, axis=0)
            if tp_final.shape[1] == 1:
                preds = (tp_final[:, 0] > 0.5).astype(int)
            else:
                preds = tp_final.argmax(axis=1)
            pred_labels = yle.inverse_transform(preds)

            od = RESULTS_DIR / comp_id
            os.makedirs(od, exist_ok=True)
            # Use test.csv IDs
            if test is not None and 'id' in test.columns:
                sub = pd.DataFrame({"id": test['id'], "winner_tie": pred_labels})
            else:
                sub = pd.DataFrame({"id": range(len(preds)), "winner_tie": pred_labels})
            sub.to_csv(od / f"submission_s{seed}.csv", index=False)
            json.dump({"comp_id": comp_id, "seed": seed, "cv_mean": cv_mean, "fold_scores": scores},
                      open(od / f"result_s{seed}.json", "w"), indent=2)

        log(f"  [{comp_id}] DONE: CV={cv_mean:.4f}")
        return {"cv_mean": cv_mean, "submission": sub}

    except Exception as e:
        log(f"  [{comp_id}] Error: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================
# Task 3: multi-modal-gesture-recognition (Tabular, big data)
# ============================================================
def train_multimodal(seed=42):
    """Re-download and train tabular model for multi-modal-gesture-recognition."""
    comp_id = "multi-modal-gesture-recognition"
    comp_dir = DATA_DIR / comp_id
    os.makedirs(comp_dir, exist_ok=True)

    # Remove partial download
    partial = comp_dir / "multi-modal-gesture-recognition.zip"
    if partial.exists() and partial.stat().st_size < 100000000:
        partial.unlink()
        log(f"  [{comp_id}] Removed partial download")

    zip_path = download_kaggle_zip(comp_id, comp_dir, timeout=900)
    if not zip_path:
        return None

    extract_zip(zip_path, comp_dir)

    # Use the existing tabular training pipeline
    from mlebench_6_runner import train_tabular
    return train_tabular(comp_id, comp_dir, "tabular", gpu_id=1, seed=seed)


# ============================================================
# Main
# ============================================================
def run_all():
    ckpt = load_checkpoint()
    completed = set(ckpt.get("completed", []))

    log("=" * 60)
    log("MLE-Bench Fix Runner — 3 failed competitions")
    log("=" * 60)

    fixes = [
        ("dog-breed-identification", train_dog_breed),
        ("lmsys-chatbot-arena", train_lmsys),
        ("multi-modal-gesture-recognition", train_multimodal),
    ]

    for comp_id, train_fn in fixes:
        if comp_id in completed:
            log(f"\nSKIP {comp_id} — already done")
            continue

        log(f"\n{'#'*60}")
        log(f"FIX: {comp_id}")
        log(f"{'#'*60}")

        result = None
        for seed in [42, 43, 44]:
            log(f"  Seed {seed}...")
            result = train_fn(seed=seed)
            if result is not None:
                break  # Use first successful seed

        if result is not None:
            completed.add(comp_id)
            save_checkpoint({"completed": list(completed)})
            log(f"  [{comp_id}] COMPLETED ✓")
        else:
            log(f"  [{comp_id}] FAILED ✗ — all seeds")

    log(f"\n{'='*60}")
    log(f"Done. Completed: {len(completed)}/3 fixes")
    log(f"{'='*60}")


if __name__ == "__main__":
    run_all()
