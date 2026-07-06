"""Prepare new MLE-Bench competition datasets on the server via Kaggle API.
Self-contained — no mlebench dependency. Run on server.

Usage: /opt/miniconda3/bin/python prepare_new_competitions.py
"""
import os, sys, subprocess, zipfile, json, io, requests
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HOME = Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra')
RAW = HOME / 'mlebench_raw_data'
PREPARED = HOME / 'mlebench_prepared'
RAW.mkdir(exist_ok=True, parents=True)

# Hardcoded from mle-bench leaf-classification classes.py
LEAF_CLASSES = [
    "Acer_Capillipes", "Acer_Circinatum", "Acer_Mono", "Acer_Opalus",
    "Acer_Palmatum", "Acer_Pictum", "Acer_Platanoids", "Acer_Rubrum",
    "Acer_Rufinerve", "Acer_Saccharinum", "Alnus_Cordata", "Alnus_Maximowiczii",
    "Alnus_Rubra", "Alnus_Sieboldiana", "Alnus_Viridis", "Arundinaria_Simonii",
    "Betula_Austrosinensis", "Betula_Pendula", "Callicarpa_Bodinieri",
    "Castanea_Sativa", "Celtis_Koraiensis", "Cercis_Siliquastrum",
    "Cornus_Chinensis", "Cornus_Controversa", "Cornus_Macrophylla",
    "Cotinus_Coggygria", "Crataegus_Monogyna", "Cytisus_Battandieri",
    "Eucalyptus_Glaucescens", "Eucalyptus_Neglecta", "Eucalyptus_Urnigera",
    "Fagus_Sylvatica", "Ginkgo_Biloba", "Ilex_Aquifolium", "Ilex_Cornuta",
    "Liquidambar_Styraciflua", "Liriodendron_Tulipifera",
    "Lithocarpus_Cleistocarpus", "Lithocarpus_Edulis", "Magnolia_Heptapeta",
    "Magnolia_Salicifolia", "Morus_Nigra", "Olea_Europaea", "Phildelphus",
    "Populus_Adenopoda", "Populus_Grandidentata", "Populus_Nigra",
    "Prunus_Avium", "Prunus_X_Shmittii", "Pterocarya_Stenoptera",
    "Quercus_Afares", "Quercus_Agrifolia", "Quercus_Alnifolia",
    "Quercus_Brantii", "Quercus_Canariensis", "Quercus_Castaneifolia",
    "Quercus_Cerris", "Quercus_Chrysolepis", "Quercus_Coccifera",
    "Quercus_Coccinea", "Quercus_Crassifolia", "Quercus_Crassipes",
    "Quercus_Dolicholepis", "Quercus_Ellipsoidalis", "Quercus_Greggii",
    "Quercus_Hartwissiana", "Quercus_Ilex", "Quercus_Imbricaria",
    "Quercus_Infectoria_sub", "Quercus_Kewensis", "Quercus_Nigra",
    "Quercus_Palustris", "Quercus_Phellos", "Quercus_Phillyraeoides",
    "Quercus_Pontica", "Quercus_Pubescens", "Quercus_Pyrenaica",
    "Quercus_Rhysophylla", "Quercus_Rubra", "Quercus_Semecarpifolia",
    "Quercus_Shumardii", "Quercus_Suber", "Quercus_Texana",
    "Quercus_Trojana", "Quercus_Variabilis", "Quercus_Vulcanica",
    "Quercus_x_Hispanica", "Quercus_x_Turneri", "Rhododendron_x_Russellianum",
    "Salix_Fragilis", "Salix_Intergra", "Sorbus_Aria", "Tilia_Oliveri",
    "Tilia_Platyphyllos", "Tilia_Tomentosa", "Ulmus_Bergmanniana",
    "Viburnum_Tinus", "Viburnum_x_Rhytidophylloides", "Zelkova_Serrata",
]

KAGGLE_JSON = os.path.expanduser('~/.kaggle/kaggle.json')


def _get_kaggle_session():
    """Create authenticated requests session for Kaggle API."""
    with open(KAGGLE_JSON) as f:
        creds = json.load(f)
    session = requests.Session()
    session.auth = (creds['username'], creds['key'])
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    return session


def _accept_rules(session, competition_name):
    """Accept competition rules via Kaggle API."""
    url = f'https://www.kaggle.com/api/v1/competitions/{competition_name}/rules/accept'
    try:
        resp = session.post(url, timeout=30)
        if resp.status_code == 200:
            print(f"    Rules accepted for {competition_name}")
            return True
        elif resp.status_code == 403:
            print(f"    Rules accept 403 for {competition_name} — may need manual acceptance")
            return False
        else:
            print(f"    Rules accept {resp.status_code} for {competition_name}")
            return resp.ok
    except Exception as e:
        print(f"    Rules accept error: {e}")
        return False


def _download_kaggle_files(session, competition_name, comp_dir):
    """Download competition data files via Kaggle API."""
    # Get file list
    url = f'https://www.kaggle.com/api/v1/competitions/{competition_name}/data/download'
    resp = session.get(url, stream=True, timeout=600)
    if resp.status_code == 403:
        print(f"    Download 403 Forbidden — rules not accepted?")
        return False
    if resp.status_code != 200:
        print(f"    Download {resp.status_code}: {resp.text[:200]}")
        return False

    content_type = resp.headers.get('Content-Type', '')
    content_disp = resp.headers.get('Content-Disposition', '')
    if 'application/zip' in content_type or '.zip' in content_disp:
        zf_path = comp_dir / f'{competition_name}.zip'
        with open(zf_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"    Downloaded: {zf_path.name}")
        with zipfile.ZipFile(zf_path, 'r') as z:
            z.extractall(comp_dir)
        print(f"    Extracted to {comp_dir}")
        return True
    else:
        print(f"    Unexpected content type: {content_type}")
        return False


def download_kaggle(competition_name):
    comp_dir = RAW / competition_name
    if comp_dir.exists() and list(comp_dir.glob('*')):
        print(f"  {competition_name}: already downloaded, skipping")
        return comp_dir
    comp_dir.mkdir(exist_ok=True, parents=True)

    session = _get_kaggle_session()
    _accept_rules(session, competition_name)

    print(f"  {competition_name}: downloading via API...")
    for attempt in range(3):
        if _download_kaggle_files(session, competition_name, comp_dir):
            # Extract any downloaded zips
            for zf in sorted(comp_dir.glob('*.zip')):
                with zipfile.ZipFile(zf, 'r') as z:
                    z.extractall(comp_dir)
            print(f"  {competition_name}: done")
            return comp_dir
        print(f"    Attempt {attempt+1} failed, retrying in 5s...")
        import time; time.sleep(5)

    raise RuntimeError(f"Failed to download {competition_name}")


def find_csv(comp_dir, pattern):
    """Find a CSV file, extracting from zip if needed."""
    csv_path = comp_dir / pattern
    if csv_path.exists():
        return csv_path
    for zf in sorted(comp_dir.glob('*.zip')):
        with zipfile.ZipFile(zf, 'r') as z:
            names = z.namelist()
            matches = [n for n in names if pattern.replace('.csv', '') in n.lower() and n.endswith('.csv')]
            if matches:
                z.extract(matches[0], comp_dir)
                return comp_dir / matches[0]
    return None


# ================================================================
# Competition preparers
# ================================================================

def prepare_playground_s3e18():
    """Multi-label classification, 32 features, targets: EC1-EC6 (only EC1,EC2 scored)."""
    name = 'playground-series-s3e18'
    out_dir = PREPARED / name
    if out_dir.exists() and (out_dir / 'train.csv').exists():
        print(f"  {name}: already prepared")
        return
    out_dir.mkdir(exist_ok=True, parents=True)

    comp_dir = download_kaggle(name)
    train = pd.read_csv(find_csv(comp_dir, 'train.csv'))
    new_train, new_test = train_test_split(train, test_size=0.1, random_state=0)

    target_cols = ['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']
    new_train.to_csv(out_dir / 'train.csv', index=False)
    test_x = new_test.drop(columns=[c for c in target_cols if c in new_test.columns])
    test_x.to_csv(out_dir / 'test.csv', index=False)
    new_test.to_csv(out_dir / 'test_private.csv', index=False)

    sub = test_x[['id']].copy()
    sub['EC1'] = 0.5
    sub['EC2'] = 0.5
    sub.to_csv(out_dir / 'sample_submission.csv', index=False)
    print(f"  {name}: train={new_train.shape}, test={new_test.shape}")


def prepare_leaf_classification():
    """99-class classification, 192 pre-extracted features (margin/shape/texture)."""
    name = 'leaf-classification'
    out_dir = PREPARED / name
    if out_dir.exists() and (out_dir / 'train.csv').exists():
        print(f"  {name}: already prepared")
        return
    out_dir.mkdir(exist_ok=True, parents=True)

    comp_dir = download_kaggle(name)
    train_csv = find_csv(comp_dir, 'train.csv')
    train = pd.read_csv(train_csv)
    new_train, new_test = train_test_split(train, test_size=0.1, random_state=0)

    target_col = 'species'
    new_train.to_csv(out_dir / 'train.csv', index=False)
    test_x = new_test.drop(columns=[target_col])
    test_x.to_csv(out_dir / 'test.csv', index=False)
    new_test.to_csv(out_dir / 'test_private.csv', index=False)

    sub = test_x[['id']].copy()
    for cls in LEAF_CLASSES:
        sub[cls] = 1.0 / len(LEAF_CLASSES)
    sub.to_csv(out_dir / 'sample_submission.csv', index=False)
    print(f"  {name}: train={new_train.shape}, test={new_test.shape}, {len(LEAF_CLASSES)} classes")


def prepare_taxi_fare():
    """Regression, 55M rows → subsample to ~1M for GPU training."""
    name = 'new-york-city-taxi-fare-prediction'
    out_dir = PREPARED / name
    if out_dir.exists() and (out_dir / 'train.csv').exists():
        print(f"  {name}: already prepared")
        return
    out_dir.mkdir(exist_ok=True, parents=True)

    comp_dir = download_kaggle(name)
    train_csv = find_csv(comp_dir, 'train.csv')
    print(f"    Reading CSV...")
    train = pd.read_csv(train_csv)
    print(f"    Loaded {len(train):,} rows, {train.columns.tolist()}")

    # Subsample: full dataset is too large for 5-fold 5-seed GPU training
    MAX_TRAIN = 1_000_000
    if len(train) > MAX_TRAIN:
        train = train.sample(n=MAX_TRAIN, random_state=42)
        print(f"    Subsampled to {len(train):,} rows")

    new_train, new_test = train_test_split(train, test_size=0.1, random_state=0)
    target_col = 'fare_amount'

    new_train.to_csv(out_dir / 'train.csv', index=False)
    test_x = new_test.drop(columns=[target_col])
    test_x.to_csv(out_dir / 'test.csv', index=False)
    new_test.to_csv(out_dir / 'test_private.csv', index=False)

    sub = test_x[['key']].copy()
    sub['fare_amount'] = 11.35
    sub.to_csv(out_dir / 'sample_submission.csv', index=False)
    print(f"  {name}: train={new_train.shape}, test={new_test.shape}")


def prepare_nomad2018():
    """Multi-target regression (formation_energy, bandgap), CSV only."""
    name = 'nomad2018-predict-transparent-conductors'
    out_dir = PREPARED / name
    if out_dir.exists() and (out_dir / 'train.csv').exists():
        print(f"  {name}: already prepared")
        return
    out_dir.mkdir(exist_ok=True, parents=True)

    comp_dir = download_kaggle(name)
    train_csv = find_csv(comp_dir, 'train.csv')
    train = pd.read_csv(train_csv)
    target_cols = ['formation_energy_ev_natom', 'bandgap_energy_ev']

    new_train, new_test = train_test_split(train, test_size=0.1, random_state=0)

    new_train.to_csv(out_dir / 'train.csv', index=False)
    test_x = new_test.drop(columns=target_cols)
    test_x.to_csv(out_dir / 'test.csv', index=False)
    new_test.to_csv(out_dir / 'test_private.csv', index=False)

    sub = pd.DataFrame({
        'id': new_test['id'],
        'formation_energy_ev_natom': 0.1779,
        'bandgap_energy_ev': 1.8892,
    })
    sub.to_csv(out_dir / 'sample_submission.csv', index=False)
    print(f"  {name}: train={new_train.shape}, test={new_test.shape}")


# ================================================================
if __name__ == '__main__':
    print("=== Preparing New Competition Datasets ===\n")
    prepare_playground_s3e18()
    prepare_leaf_classification()
    prepare_taxi_fare()
    prepare_nomad2018()
    print("\n=== Done ===")
    for d in sorted(PREPARED.iterdir()):
        if d.is_dir():
            files = list(d.glob('*.csv'))
            sizes = [f.stat().st_size for f in files]
            total_mb = sum(sizes) / 1024**2
            print(f"  {d.name}: {len(files)} files, {total_mb:.1f} MB")
