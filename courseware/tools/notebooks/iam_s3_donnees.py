"""Dataset of *IA pour la médecine*, séance 3: `maladie_cardiaque.csv`.

Source: the **Statlog (Heart)** dataset, UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/145/statlog+heart (file `heart.dat`,
270 patients, 13 attributes, no missing values). Licence: CC BY 4.0
(UCI). The same data is OpenML `heart-statlog` (used by the English source
TP, session3/tp1.ipynb of github.com/racousin/DFGSM2-IA-S2) and Kaggle
`ritwikb3/heart-disease-statlog`; the owner asked for clear feature names,
so the columns are renamed in French below. The integer codes are kept as
they are (the TP discusses integer-encoded categories); the target
`maladie_cardiaque` is 1 = présente (UCI class 2), 0 = absente (class 1).

Writes `ai_for_medicine/ia-medecine/donnees/maladie_cardiaque.csv` (`aim_repo`)
and nothing else:

    uv run --with pandas python courseware/tools/notebooks/iam_s3_donnees.py
"""
from __future__ import annotations

import io
import pathlib
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aim_repo  # noqa: E402

SOURCE = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
          "statlog/heart/heart.dat")
OUT = aim_repo.out_dir("ia-medecine", "donnees") / "maladie_cardiaque.csv"

# UCI attribute order -> French name (UCI documentation of heart.dat)
COLUMNS = [
    "age",                        # 1  âge (années)
    "sexe",                       # 2  1 = homme, 0 = femme
    "type_douleur_thoracique",    # 3  1 angor typique, 2 angor atypique,
                                  #    3 douleur non angineuse, 4 asymptomatique
    "pression_arterielle_repos",  # 4  mmHg
    "cholesterol",                # 5  cholestérol sérique, mg/dL
    "glycemie_jeun_sup_120",      # 6  glycémie à jeun > 120 mg/dL (1 = oui)
    "ecg_repos",                  # 7  0 normal, 1 anomalie ST-T, 2 HVG
    "frequence_cardiaque_max",    # 8  fréquence cardiaque maximale à l'effort
    "angor_effort",               # 9  angor induit par l'effort (1 = oui)
    "depression_st",              # 10 sous-décalage ST à l'effort (oldpeak), mm
    "pente_st",                   # 11 pente du segment ST : 1 ascendante,
                                  #    2 plate, 3 descendante
    "nb_vaisseaux_colores",       # 12 vaisseaux colorés en fluoroscopie (0-3)
    "thalassemie",                # 13 scintigraphie au thallium : 3 normal,
                                  #    6 défaut fixe, 7 défaut réversible
    "maladie_cardiaque",          # cible : 1 présente, 0 absente
]


def main() -> None:
    with urllib.request.urlopen(SOURCE, timeout=60) as resp:
        raw = resp.read().decode("ascii")
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=COLUMNS)
    if df.shape != (270, 14) or df.isna().any().any():
        raise ValueError(f"unexpected heart.dat content: shape {df.shape}")
    if set(df["maladie_cardiaque"]) != {1, 2}:
        raise ValueError("target is not coded 1/2")
    df["maladie_cardiaque"] = (df["maladie_cardiaque"] == 2).astype(int)
    for col in COLUMNS:
        if col != "depression_st":
            df[col] = df[col].astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, df.shape, "présente:", df["maladie_cardiaque"].sum())


if __name__ == "__main__":
    main()
