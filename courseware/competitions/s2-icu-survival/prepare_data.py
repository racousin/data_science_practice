#!/usr/bin/env python3
"""Build the Critical Care Survival split for the Session 2 preprocessing challenge.

Source: SUPPORT2, 9,105 seriously ill hospitalised adults in 5 US medical
centres, 1989-1994 (Knaus et al. 1995), published by Vanderbilt Biostatistics
at https://hbiostat.org/data/repo/support2csv.zip. No key needed.

The outcome is NOT the recorded one. `teacher_secret.py` (gitignored, teacher
only) draws a 60-day mortality from a hidden logistic model over the
correctly preprocessed clinical variables, and produces the served frame the
way a hospital export looks: five sites with their own units and spellings,
placeholders, a decimal slip, a "not done" here and there, and a charges column
that only exists for discharged patients. So the public source cannot be
joined to recover the label, and each preprocessing step in EXPERTISE.md is
worth a measurable, controlled amount of AUC.

Public    -> data/train.csv.gz, data/test.csv.gz, data/sample_submission.csv.gz,
             data/EXPERTISE.md, data/DICTIONARY.md (copies of the package's)
Private   -> data/labels_train.csv, data/labels_test.csv   (ENV folder only)
Benchmark -> data/benchmark_submission.csv.gz (== sample_submission.csv.gz)

Steps, each of which fails the build rather than degrade:

1. **Download** SUPPORT2 into `data/raw/` (cached; a rerun is offline).
2. **Outcome + export** from `teacher_secret` (fixed seed).
3. **Split** 70/30, stratified on the outcome; `charges` blanked on the test
   side (billing closes at discharge; test patients are still admitted).
   Opaque salted ids (`../_dataset_ids.py`), rows shuffled.
4. **Assertions**: headers in the declared order, no SUPPORT2 outcome column
   shipped, rates within 1.5 points, ids disjoint, every trap present.
5. **Benchmark**: the numeric columns as pandas reads them, median-imputed and
   standardised, all rows; scored by `env.py` itself. Its score is what
   `config.py` pins as `benchmark_expected_score`.
6. **Dictionary** rendered from the shipped train file; docs copied to data/.
7. **Leak canary**: LightGBM on the raw shipped columns. Far above the expert
   line means a label leaked.

    export MLARENA_ID_SALT=$(cat ../.id-salt)   # competitions/.id-salt, never committed
    uv run --with scikit-learn==1.8.0 --with pandas --with lightgbm \\
        python prepare_data.py [--skip-canary]
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedShuffleSplit

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _dataset_ids import shuffle_and_label  # noqa: E402

try:
    import teacher_secret as secret  # noqa: E402
except ImportError as exc:
    raise SystemExit(
        "teacher_secret.py is missing: it is the hidden outcome model and it is not "
        "in the public repository (see the platform repo, tests-dummy/catalog/icu_survival/)."
    ) from exc
from dictionary import ID_COLUMN, SERVED, TARGET, render as render_dictionary  # noqa: E402

SUPPORT2_URL = "https://hbiostat.org/data/repo/support2csv.zip"
DATASET = "s2-icu-survival"
SEED = secret.SEED
TEST_SIZE = 0.3
MAX_RATE_GAP = 0.015
RATE_BOUNDS = (0.40, 0.55)
CANARY_MAX_AUC = 0.955
GZ = {"method": "gzip", "compresslevel": 6, "mtime": 0}
# never shipped: the recorded outcomes, follow-up, derived scores, other costs
DROPPED = ["death", "hospdead", "d.time", "slos", "sfdm2", "adlsc", "dnr", "dnrday", "prg2m",
           "prg6m", "surv2m", "surv6m", "sps", "aps", "avtisst", "totcst", "totmcst", "num.co"]


# --------------------------------------------------------------------------- #
# 1. download
def download() -> pd.DataFrame:
    cache = DATA / "raw" / "support2.csv"
    if not cache.exists():
        print(f"  downloading {SUPPORT2_URL}")
        blob = urlopen(SUPPORT2_URL).read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(zf.read(name))
    raw = pd.read_csv(cache)
    if raw.shape != (9105, 47):
        raise SystemExit(f"SUPPORT2 should be 9105 x 47, got {raw.shape}")
    return raw


# --------------------------------------------------------------------------- #
# scorer + benchmark
def load_env():
    stage = Path(tempfile.mkdtemp(prefix="icu-env-"))
    shutil.copy2(HERE / "env.py", stage / "env.py")
    for name in ("labels_train.csv", "labels_test.csv"):
        shutil.copy2(DATA / name, stage / name)
    spec = importlib.util.spec_from_file_location("icu_env", stage / "env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Env(is_evaluation=True)


def benchmark_features(train: pd.DataFrame, test: pd.DataFrame):
    """No domain knowledge: the numeric columns as pandas reads them, median
    imputed and standardised, both fitted on the train rows."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    features = [c for c in train.columns if c not in (ID_COLUMN, TARGET)]
    num_tr = [c for c in features if is_numeric_dtype(train[c])]
    num_te = [c for c in features if is_numeric_dtype(test[c])]
    if num_tr != num_te:
        raise SystemExit(f"numeric columns differ between train and test: "
                         f"{sorted(set(num_tr) ^ set(num_te))}")
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    a = scaler.fit_transform(imputer.fit_transform(train[num_tr]))
    b = scaler.transform(imputer.transform(test[num_tr]))
    return pd.concat([
        pd.DataFrame(a, columns=num_tr).assign(**{ID_COLUMN: train[ID_COLUMN].to_numpy()}),
        pd.DataFrame(b, columns=num_tr).assign(**{ID_COLUMN: test[ID_COLUMN].to_numpy()}),
    ], ignore_index=True)[[ID_COLUMN, *num_tr]], num_tr


# --------------------------------------------------------------------------- #
# 7. leak canary
def leak_canary(train: pd.DataFrame, test: pd.DataFrame, y_test: np.ndarray) -> float:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    features = [c for c in train.columns if c not in (ID_COLUMN, TARGET)]

    def frame(df):
        out = {}
        for i, c in enumerate(features):
            if is_numeric_dtype(train[c]):
                out[f"f{i}"] = df[c].astype(float)
            else:
                cats = sorted(train[c].dropna().astype(str).unique())
                values = df[c].astype(str).where(df[c].notna())
                out[f"f{i}"] = pd.Categorical(values.where(values.isin(cats)), categories=cats)
        return pd.DataFrame(out, index=df.index)
    X_tr, X_te, y_tr = frame(train), frame(test), train[TARGET].to_numpy()
    val = np.random.default_rng(0).random(len(X_tr)) < 0.15
    model = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=15,
                               min_child_samples=40, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
    model.fit(X_tr[~val], y_tr[~val], eval_set=[(X_tr[val], y_tr[val])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
    auc = float(roc_auc_score(y_test, model.predict_proba(X_te)[:, 1]))
    print(f"  leak canary: LightGBM on {len(features)} raw columns, "
          f"{model.best_iteration_} trees -> test AUC {auc:.4f}")
    return auc


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-canary", action="store_true", help="skip step 7 (LightGBM)")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    print("1. download …")
    raw = download()

    print("2. outcome + export …")
    y, p, b0, _phi = secret.draw_target(raw, secret.K, secret.RATE, SEED)
    served = secret.serve(raw, y, SEED)
    if list(served.columns) != SERVED + [TARGET]:
        raise SystemExit("served frame does not match dictionary.SERVED")
    print(f"  k={secret.K} b0={b0:.3f} rate={float(y.mean()):.4f}")

    print("3. split, ids …")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_idx, te_idx = next(splitter.split(np.zeros(len(served)), served[TARGET]))
    served = served.copy()
    served["_p"] = p.to_numpy()

    def ship(idx, split_name, prefix):
        part = served.iloc[np.sort(idx)].reset_index(drop=True)
        X, yy = part[SERVED + ["_p"]], part[TARGET]
        return shuffle_and_label(X, yy, dataset=DATASET, split=split_name, prefix=prefix)
    X_tr, y_tr = ship(tr_idx, "train", "tr")
    X_te, y_te = ship(te_idx, "test", "te")
    # billing closes at discharge: the test patients are still admitted
    X_te["charges"] = np.nan
    p_te = X_te.pop("_p").to_numpy()
    X_tr = X_tr.drop(columns="_p")

    print("4. write + assertions …")
    train = X_tr.assign(**{TARGET: y_tr.to_numpy()})
    test = X_te
    train.to_csv(DATA / "train.csv.gz", index=False, compression=GZ)
    test.to_csv(DATA / "test.csv.gz", index=False, compression=GZ)
    pd.DataFrame({ID_COLUMN: X_tr[ID_COLUMN], TARGET: y_tr}).to_csv(DATA / "labels_train.csv", index=False)
    pd.DataFrame({ID_COLUMN: X_te[ID_COLUMN], TARGET: y_te}).to_csv(DATA / "labels_test.csv", index=False)
    train = pd.read_csv(DATA / "train.csv.gz", low_memory=False)
    test = pd.read_csv(DATA / "test.csv.gz", low_memory=False)
    if list(train.columns) != [ID_COLUMN, *SERVED, TARGET] or list(test.columns) != [ID_COLUMN, *SERVED]:
        raise SystemExit("shipped headers differ from id + SERVED (+ target)")
    leaked = (set(train.columns) | set(test.columns)) & set(DROPPED)
    if leaked:
        raise SystemExit(f"dropped columns in a shipped file: {sorted(leaked)}")
    rate_tr, rate_te = float(y_tr.mean()), float(y_te.mean())
    if abs(rate_tr - rate_te) > MAX_RATE_GAP or not (RATE_BOUNDS[0] < rate_tr < RATE_BOUNDS[1]):
        raise SystemExit(f"target rates: train {rate_tr:.4f}, test {rate_te:.4f}")
    if set(train[ID_COLUMN]) & set(test[ID_COLUMN]) or not train[ID_COLUMN].is_unique:
        raise SystemExit("ids overlap or repeat")
    if test["charges"].notna().any() or train["charges"].notna().mean() < 0.95:
        raise SystemExit("charges must be empty in test and filled in train")
    if is_numeric_dtype(train["glucose"]) or is_numeric_dtype(test["glucose"]):
        raise SystemExit("glucose should read as text ('not done' present)")
    if sorted(train["site"].unique()) != secret.SITES:
        raise SystemExit("sites")
    for name, cond in (("age 999", (train["age"] == 999).any()), ("temp 0.0", (train["temp"] == 0).any()),
                       ("sod x10", (train["sod"] > 500).any()), ("sex spellings", train["sex"].nunique() > 2)):
        if not cond:
            raise SystemExit(f"trap missing from train: {name}")
    print(f"  train {len(train)} rows, dead {rate_tr:.4f} | test {len(test)} rows, dead {rate_te:.4f}")

    print("5. benchmark …")
    for name in ("sample_submission.csv.gz", "benchmark_submission.csv.gz"):
        (DATA / name).unlink(missing_ok=True)
    bench, numeric = benchmark_features(train, test)
    bench.to_csv(DATA / "benchmark_submission.csv.gz", index=False, float_format="%.6f", compression=GZ)
    shutil.copy2(DATA / "benchmark_submission.csv.gz", DATA / "sample_submission.csv.gz")
    result = load_env().evaluate(str(DATA / "benchmark_submission.csv.gz"))["agent_results"][0]
    if result.get("is_agent_code_error"):
        raise SystemExit(f"env.py rejected the benchmark: {result['agent_code_error_message']}")
    print(f"  benchmark ({len(numeric)} numeric columns as read): {result['info_message']}")

    print("6. dictionary …")
    render_dictionary(train, HERE / "DICTIONARY.md")
    for doc in ("EXPERTISE.md", "DICTIONARY.md"):
        shutil.copy2(HERE / doc, DATA / doc)

    from sklearn.metrics import roc_auc_score
    oracle = float(roc_auc_score(y_te.to_numpy(), p_te))
    print(f"  oracle AUC on test (true probabilities): {oracle:.4f}")
    canary = None
    if args.skip_canary:
        print("7. leak canary SKIPPED (--skip-canary): not a releasable build")
    else:
        print("7. leak canary …")
        canary = leak_canary(train, test, y_te.to_numpy())
        if canary > CANARY_MAX_AUC:
            raise SystemExit(f"leak canary {canary:.4f} > {CANARY_MAX_AUC}")
    summary = {"rows_train": len(train), "rows_test": len(test), "rate_train": round(rate_tr, 6),
               "rate_test": round(rate_te, 6), "benchmark_score": result["score"],
               "benchmark_metrics": result["metrics_detail"], "oracle_auc_test": round(oracle, 6),
               "canary_auc": None if canary is None else round(canary, 6), "k": secret.K, "b0": round(b0, 4)}
    (DATA / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  set benchmark_expected_score = {result['score']}")
    print(f"wrote {DATA}")


if __name__ == "__main__":
    main()
