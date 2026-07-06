"""Prepare 3 new competitions from already-downloaded zips in /tmp.
Skips Kaggle API download — zips already obtained via kaggle CLI.
All data stays in /hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/ (NOT home dir, NOT local).
"""
import os, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HOME = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
RAW = HOME / "mlebench_raw_data"
PREPARED = HOME / "mlebench_prepared"
RAW.mkdir(exist_ok=True, parents=True)
PREPARED.mkdir(exist_ok=True, parents=True)

LEAF_CLASSES = [
    "Acer_Capillipes","Acer_Circinatum","Acer_Mono","Acer_Opalus",
    "Acer_Palmatum","Acer_Pictum","Acer_Platanoids","Acer_Rubrum",
    "Acer_Rufinerve","Acer_Saccharinum","Alnus_Cordata","Alnus_Maximowiczii",
    "Alnus_Rubra","Alnus_Sieboldiana","Alnus_Viridis","Arundinaria_Simonii",
    "Betula_Austrosinensis","Betula_Pendula","Callicarpa_Bodinieri",
    "Castanea_Sativa","Celtis_Koraiensis","Cercis_Siliquastrum",
    "Cornus_Chinensis","Cornus_Controversa","Cornus_Macrophylla",
    "Cotinus_Coggygria","Crataegus_Monogyna","Cytisus_Battandieri",
    "Eucalyptus_Glaucescens","Eucalyptus_Neglecta","Eucalyptus_Urnigera",
    "Fagus_Sylvatica","Ginkgo_Biloba","Ilex_Aquifolium","Ilex_Cornuta",
    "Liquidambar_Styraciflua","Liriodendron_Tulipifera",
    "Lithocarpus_Cleistocarpus","Lithocarpus_Edulis","Magnolia_Heptapeta",
    "Magnolia_Salicifolia","Morus_Nigra","Olea_Europaea","Phildelphus",
    "Populus_Adenopoda","Populus_Grandidentata","Populus_Nigra",
    "Prunus_Avium","Prunus_X_Shmittii","Pterocarya_Stenoptera",
    "Quercus_Afares","Quercus_Agrifolia","Quercus_Alnifolia",
    "Quercus_Brantii","Quercus_Canariensis","Quercus_Castaneifolia",
    "Quercus_Cerris","Quercus_Chrysolepis","Quercus_Coccifera",
    "Quercus_Coccinea","Quercus_Crassifolia","Quercus_Crassipes",
    "Quercus_Dolicholepis","Quercus_Ellipsoidalis","Quercus_Greggii",
    "Quercus_Hartwissiana","Quercus_Ilex","Quercus_Imbricaria",
    "Quercus_Infectoria_sub","Quercus_Kewensis","Quercus_Nigra",
    "Quercus_Palustris","Quercus_Phellos","Quercus_Phillyraeoides",
    "Quercus_Pontica","Quercus_Pubescens","Quercus_Pyrenaica",
    "Quercus_Rhysophylla","Quercus_Rubra","Quercus_Semecarpifolia",
    "Quercus_Shumardii","Quercus_Suber","Quercus_Texana",
    "Quercus_Trojana","Quercus_Variabilis","Quercus_Vulcanica",
    "Quercus_x_Hispanica","Quercus_x_Turneri","Rhododendron_x_Russellianum",
    "Salix_Fragilis","Salix_Intergra","Sorbus_Aria","Tilia_Oliveri",
    "Tilia_Platyphyllos","Tilia_Tomentosa","Ulmus_Bergmanniana",
    "Viburnum_Tinus","Viburnum_x_Rhytidophylloides","Zelkova_Serrata",
]


def prep_from_zip(name, target_cols, zip_path):
    """Extract zip from /tmp, copy to RAW, train_test_split to PREPARED."""
    out_dir = PREPARED / name
    if out_dir.exists() and (out_dir / "train.csv").exists():
        print(f"  {name}: already prepared, skip")
        return

    raw_dir = RAW / name
    raw_dir.mkdir(exist_ok=True, parents=True)

    # Copy zip from /tmp to RAW (if not already there)
    dest_zip = raw_dir / f"{name}.zip"
    if not dest_zip.exists():
        shutil.copy2(zip_path, dest_zip)
        print(f"  {name}: copied zip ({dest_zip.stat().st_size/1024**2:.1f}MB)")

    # Extract
    with zipfile.ZipFile(dest_zip, "r") as z:
        z.extractall(raw_dir)
    csv_files = list(raw_dir.glob("*.csv"))
    print(f"  {name}: extracted, CSV files: {[f.name for f in csv_files]}")

    # Find train CSV
    train_csv = None
    for f in csv_files:
        if "train" in f.name.lower():
            train_csv = f
            break
    if train_csv is None and csv_files:
        train_csv = csv_files[0]

    if train_csv is None:
        print(f"  {name}: NO CSV FOUND")
        return

    train = pd.read_csv(train_csv)
    print(f"  {name}: loaded {train.shape}, cols={list(train.columns)[:10]}...")

    # Taxi: subsample to 1M
    MAX_TRAIN = 1_000_000
    if len(train) > MAX_TRAIN:
        train = train.sample(n=MAX_TRAIN, random_state=42)
        print(f"  {name}: subsampled to {len(train):,}")

    new_train, new_test = train_test_split(train, test_size=0.1, random_state=0)
    out_dir.mkdir(exist_ok=True, parents=True)

    new_train.to_csv(out_dir / "train.csv", index=False)
    test_drop = new_test.drop(columns=[c for c in target_cols if c in new_test.columns])
    test_drop.to_csv(out_dir / "test.csv", index=False)
    new_test.to_csv(out_dir / "test_private.csv", index=False)

    # Sample submission
    id_col = "id" if "id" in test_drop.columns else test_drop.columns[0]
    sub = test_drop[[id_col]].copy()
    if isinstance(target_cols, list):
        for tc in target_cols:
            sub[tc] = 0.5
    else:
        sub[target_cols] = 0.5
    sub.to_csv(out_dir / "sample_submission.csv", index=False)
    print(f"  {name}: DONE train={new_train.shape} test={new_test.shape}")


# ================================================================
print("=== Preparing from existing /tmp zips ===\n")

prep_from_zip("leaf-classification", ["species"],
              "/tmp/kaggle_dl_leaf-classification/leaf-classification.zip")

prep_from_zip("new-york-city-taxi-fare-prediction", ["fare_amount"],
              "/tmp/kaggle_dl_new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction.zip")

prep_from_zip("nomad2018-predict-transparent-conductors",
              ["formation_energy_ev_natom", "bandgap_energy_ev"],
              "/tmp/kaggle_dl_nomad2018-predict-transparent-conductors/nomad2018-predict-transparent-conductors.zip")

print("\n=== Summary ===")
for d in sorted(PREPARED.iterdir()):
    if d.is_dir():
        files = list(d.glob("*.csv"))
        total = sum(f.stat().st_size for f in files) / 1024**2
        print(f"  {d.name}: {len(files)} files, {total:.1f} MB")
print("Done!")
