"""Datasets of the course *Python pour la médecine* (`content/python-medecine`).

Writes every file the notebooks read into
`ai_for_modecine/python-medecine/donnees/` (`aim_repo.out_dir`), which Colab
fetches from GitHub (`aim_repo.raw`: racousin/ai_for_modecine, branch main).

Two real sources, downloaded and trimmed:

- `insuffisance_cardiaque.csv`: UCI *Heart failure clinical records* (id 519,
  Chicco & Jurman 2020, CC BY 4.0), 299 patients, columns translated to French.
- `covid_hospitalisations.csv.gz`: Santé publique France, *Données
  hospitalières relatives à l'épidémie de COVID-19* (data.gouv.fr, Licence
  Ouverte 2.0), both sexes only (`sexe == 0`), one row per department per day.

Three synthetic files, seeded, so every run writes the same bytes:

- `sejours_hopital_brut.csv`: 600 hospital stays with the faults a real export
  has (mixed date formats, units, spellings, duplicates, gaps), for cleaning.
- `examens_biologiques.csv`: lab results in long format, joined to the stays
  by `id_patient`, for merge / pivot.
- `constantes_vitales.csv`: 48 hours of hourly vitals for 12 ICU patients, for
  time series plots.

    uv run --with pandas --with numpy --with requests python courseware/tools/notebooks/pym_datasets.py
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import requests

import aim_repo

OUT = aim_repo.out_dir("python-medecine", "donnees")

HF_URL = "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip"
COVID_URL = "https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7"

HF_COLUMNS = {
    "age": "age",
    "anaemia": "anemie",
    "creatinine_phosphokinase": "cpk",
    "diabetes": "diabete",
    "ejection_fraction": "fraction_ejection",
    "high_blood_pressure": "hypertension",
    "platelets": "plaquettes",
    "serum_creatinine": "creatinine",
    "serum_sodium": "sodium",
    "sex": "sexe_homme",
    "smoking": "fumeur",
    "time": "suivi_jours",
    "DEATH_EVENT": "deces",
}

SERVICES = ["Cardiologie", "Pneumologie", "Neurologie", "Gériatrie",
            "Chirurgie digestive", "Endocrinologie"]
# The same service as an export from three different software versions.
SERVICE_VARIANTS = {
    "Cardiologie": ["Cardiologie", "cardiologie", "CARDIO", "Cardio "],
    "Pneumologie": ["Pneumologie", "pneumo", "PNEUMOLOGIE"],
    "Neurologie": ["Neurologie", "neuro", "Neurologie "],
    "Gériatrie": ["Gériatrie", "Geriatrie", "gériatrie"],
    "Chirurgie digestive": ["Chirurgie digestive", "chir. digestive", "CHIR DIG"],
    "Endocrinologie": ["Endocrinologie", "endocrino", "Endocrinologie"],
}
DIAGNOSTICS = {  # ICD-10 codes typical of each service
    "Cardiologie": ["I50.0", "I21.4", "I48.9", "I10"],
    "Pneumologie": ["J44.1", "J18.9", "J45.9"],
    "Neurologie": ["I63.9", "G40.9", "G35"],
    "Gériatrie": ["R54", "S72.0", "F03"],
    "Chirurgie digestive": ["K35.8", "K80.2", "K56.6"],
    "Endocrinologie": ["E11.9", "E10.9", "E05.0"],
}


def heart_failure() -> pd.DataFrame:
    raw = requests.get(HF_URL, timeout=60)
    raw.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(raw.content)) as z:
        df = pd.read_csv(z.open("heart_failure_clinical_records_dataset.csv"))
    df = df.rename(columns=HF_COLUMNS)[list(HF_COLUMNS.values())]
    df["age"] = df["age"].round().astype(int)
    df["plaquettes"] = (df["plaquettes"] / 1000).round(1)  # 10^3 / µL
    df.insert(0, "id_patient", [f"IC{i:03d}" for i in range(1, len(df) + 1)])
    return df


def covid() -> pd.DataFrame:
    raw = requests.get(COVID_URL, timeout=120)
    raw.raise_for_status()
    df = pd.read_csv(io.BytesIO(raw.content), sep=";", dtype={"dep": str})
    df = df[df["sexe"] == 0]
    df = df[["dep", "jour", "hosp", "rea", "rad", "dc"]]
    return df.sort_values(["dep", "jour"]).reset_index(drop=True)


def sejours(rng: np.random.Generator, n: int = 600) -> pd.DataFrame:
    rows = []
    for i in range(n):
        service = SERVICES[rng.integers(len(SERVICES))]
        age = int(np.clip(rng.normal(68 if service == "Gériatrie" else 58, 16), 18, 99))
        sexe = rng.choice(["F", "M"])
        taille = rng.normal(163 if sexe == "F" else 176, 7)
        imc = np.clip(rng.normal(26.5, 4.5), 16, 45)
        poids = imc * (taille / 100) ** 2
        entree = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
        duree = int(max(1, rng.gamma(2.0, 3.5 if service != "Gériatrie" else 6)))
        sortie = entree + pd.Timedelta(days=duree)
        glyc = rng.normal(1.05 if service != "Endocrinologie" else 1.8, 0.25)  # g/L
        rows.append({
            "id_sejour": f"S{i + 1:04d}",
            "id_patient": f"P{rng.integers(1, 451):04d}",
            "service": rng.choice(SERVICE_VARIANTS[service]),
            "date_entree": entree,
            "date_sortie": sortie,
            "age": age,
            "sexe": sexe,
            "poids_kg": round(poids, 1),
            "taille_cm": round(taille, 0),
            "glycemie": round(glyc, 2),
            "diagnostic_cim10": rng.choice(DIAGNOSTICS[service]),
            "deces": int(rng.random() < (0.02 + 0.0015 * max(0, age - 60))),
        })
    df = pd.DataFrame(rows)

    # The faults of a real export, each on a known share of rows.
    def pick(frac):
        return rng.random(len(df)) < frac

    fmt = pd.Series(df["date_entree"].dt.strftime("%Y-%m-%d"))
    alt = pick(0.25)
    fmt[alt] = df.loc[alt, "date_entree"].dt.strftime("%d/%m/%Y")
    df["date_entree"] = fmt
    df["date_sortie"] = df["date_sortie"].dt.strftime("%Y-%m-%d")
    sexe = df["sexe"].astype(object)
    sexe[pick(0.08)] = "Homme"
    sexe[pick(0.05)] = "f"
    df["sexe"] = np.where(df["sexe"].eq("F") & sexe.eq("Homme"), "Femme", sexe)
    taille = df["taille_cm"].astype(object)
    m = pick(0.06)
    taille[m] = (df.loc[m, "taille_cm"] / 100).round(2)  # entered in metres
    df["taille_cm"] = taille
    glyc = df["glycemie"].astype(float)
    mm = pick(0.15)
    glyc[mm] = (glyc[mm] * 5.551).round(1)  # mmol/L instead of g/L
    df["glycemie"] = glyc
    for col, frac in [("poids_kg", 0.07), ("glycemie", 0.10), ("taille_cm", 0.04)]:
        df.loc[pick(frac), col] = np.nan
    df.loc[pick(0.01), "age"] = 999  # a "unknown" sentinel
    dup = df.sample(12, random_state=1)
    df = pd.concat([df, dup]).sample(frac=1, random_state=2).reset_index(drop=True)
    return df


def examens(rng: np.random.Generator, sejours_df: pd.DataFrame) -> pd.DataFrame:
    tests = {  # name, unit, mean, sd
        "hemoglobine": ("g/dL", 13.2, 1.8),
        "creatinine": ("µmol/L", 85, 30),
        "potassium": ("mmol/L", 4.2, 0.5),
        "crp": ("mg/L", 18, 25),
    }
    rows = []
    for pid in sorted(sejours_df["id_patient"].unique()):
        for name, (unit, mu, sd) in tests.items():
            if rng.random() < 0.12:
                continue  # not measured
            val = rng.normal(mu, sd)
            if name == "crp":
                val = abs(rng.lognormal(np.log(8), 1.0))
            rows.append({"id_patient": pid, "examen": name,
                         "valeur": round(max(val, 0.1), 1), "unite": unit})
    return pd.DataFrame(rows)


def vitals(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for p in range(1, 13):
        septic = p in (3, 7, 11)  # three patients deteriorate after hour 20
        fc, pas, spo2, temp = rng.normal(82, 8), rng.normal(125, 12), 97.0, 37.0
        for h in range(48):
            drift = max(0, h - 20) / 28 if septic else 0
            rows.append({
                "id_patient": f"R{p:02d}",
                "heure": h,
                "frequence_cardiaque": round(fc + 30 * drift + rng.normal(0, 4)),
                "pression_systolique": round(pas - 35 * drift + rng.normal(0, 6)),
                "spo2": round(min(100, spo2 - 6 * drift + rng.normal(0, 1)), 0),
                "temperature": round(temp + 1.8 * drift + rng.normal(0, 0.2), 1),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    heart_failure().to_csv(OUT / "insuffisance_cardiaque.csv", index=False)
    covid().to_csv(OUT / "covid_hospitalisations.csv.gz", index=False,
                   compression={"method": "gzip", "mtime": 0})
    s = sejours(rng)
    s.to_csv(OUT / "sejours_hopital_brut.csv", index=False)
    examens(rng, s).to_csv(OUT / "examens_biologiques.csv", index=False)
    vitals(rng).to_csv(OUT / "constantes_vitales.csv", index=False)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:32s} {f.stat().st_size / 1e3:8.1f} kB")


if __name__ == "__main__":
    main()
