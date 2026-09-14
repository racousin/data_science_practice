#!/usr/bin/env python3
"""Build the DPE energy-label split for the Session 2 preprocessing challenge.

Source: ADEME, "DPE Logements existants (depuis juillet 2021)", data-fair
dataset `meg-83tjwtg8dyz4vv7h1dqe`, Licence Ouverte 2.0. Pulled through the
public API; nothing here needs a key.

Public    -> data/train.csv.gz, data/test.csv.gz, data/sample_submission.csv.gz,
             data/EXPERTISE.md, data/DICTIONNAIRE.md (copies of the package's)
Private   -> data/labels_train.csv, data/labels_test.csv   (ENV folder only)
Benchmark -> data/benchmark_submission.csv.gz (== sample_submission.csv.gz)

Steps, each of which fails the build rather than degrade:

1. **Pull.** Per département, `qs` = département + DPE date in
   2024-07-01..2025-12-31 + `type_batiment` maison|appartement, `sort=_rand`
   (the API's default order is not random), paginated with `after`. Every page
   is cached under `data/raw/`, so a rerun with a complete cache is offline and
   byte-for-byte reproducible even though ADEME updates the dataset weekly.
2. **Dedupe.** A DPE whose number appears in any other DPE's
   `numero_dpe_remplace` has been superseded. Checked against the WHOLE
   dataset through the API (a replacement may be outside the window, the
   département or the sample), and within the sample.
3. **Sample** 12,500 dwellings per département, in `_rand` order: 100,000.
4. **Split** with `GroupShuffleSplit` 70/30. Group = the building: the
   connected rows that share a building DPE (`numero_dpe_immeuble_associe`),
   a BAN address (`identifiant_ban` or `adresse_ban`) or an RNB building id
   (`id_rnb`). Flats generated from one building DPE are copies and must sit
   on one side, and so must the copies of a building diagnosed twice (two
   building DPEs at one address, neither marked as replacing the other:
   1,004 addresses in the sample). Opaque salted ids (`../_dataset_ids.py`),
   rows shuffled. Every surface column of a row is multiplied by one random
   factor in [0.98, 1.02], rounded to 0.1 m2. Street addresses written into
   the free-text descriptions (a heating substation "du 82 rue de ...") are
   replaced by `[adresse retirée]`.
5. **Drop audit** (`columns.py`): every source column is kept or dropped, or
   the build stops. Then the assertions: no building DPE, BAN address or RNB
   id spans both sides, no superseded DPE kept, no dropped column in a shipped
   file, no street address left in a shipped text column, train and test
   target rates within 0.5 point.
6. **Write** the files and the benchmark: the numeric columns as pandas reads
   them, median-imputed and standardized, scored by `env.py` itself.
7. **Leak canary.** LightGBM on the shipped raw columns lands near 0.93; above
   0.97 means a calculated column slipped through, and the build fails.

The split seed, the jitter and the id permutation all derive from
`MLARENA_ID_SALT`, which is not in the repository (see `_dataset_ids.py`).

    export MLARENA_ID_SALT=...     # competitions/.id-salt
    uv run --with scikit-learn==1.8.0 --with pandas --with lightgbm \\
        python prepare_data.py
"""
from __future__ import annotations

import argparse
import gzip
import re
import http.client
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
DATASET = "s2-dpe-energy-label"

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _dataset_ids import _digest, shuffle_and_label  # noqa: E402
from columns import JITTERED_SURFACES, KEPT, DROPPED, classify  # noqa: E402

API = "https://data.ademe.fr/data-fair/api/v1/datasets/meg-83tjwtg8dyz4vv7h1dqe"
# One département per climate zone.
DEPARTEMENTS = {"59": "H1a", "67": "H1b", "69": "H1c", "35": "H2a",
                "44": "H2b", "33": "H2c", "84": "H2d", "13": "H3"}
WINDOW = ("2024-07-01", "2025-12-31")
TYPES = ("maison", "appartement")
PER_DEPARTEMENT = 12_500
PAGE_SIZE = 2_500
TEST_SIZE = 0.3
TARGET = "classe_efg"
POSITIVE_LABELS = {"E", "F", "G"}
JITTER = 0.02
MAX_RATE_GAP = 0.005
CANARY_MAX_AUC = 0.97
SUPERSEDED_BATCH = 400          # ids per `numero_dpe_remplace_in` lookup (URL length)
ID_COLUMN = "id"
# A building = the rows connected through any of these (all dropped columns).
BUILDING_KEYS = ("numero_dpe_immeuble_associe", "identifiant_ban", "adresse_ban", "id_rnb")
# A street address inside a free-text description: "82 rue de la Bottière".
STREET_WORDS = r"(?:rue|avenue|av\.|boulevard|bd|all[ée]e|impasse|chemin|place|route|cours|quai|square|passage|r[ée]sidence)"
ADDRESS_IN_TEXT = re.compile(
    rf"\b\d{{1,4}}(?:\s*(?:bis|ter))?\s*,?\s+{STREET_WORDS}\b[^.,;:<>()\n]*", re.IGNORECASE)
# What must not survive the scrub: a street word followed by a capitalised name.
ADDRESS_LEFT = re.compile(
    rf"(?i:\b\d{{1,4}}(?:\s*(?:bis|ter))?\s*,?\s+{STREET_WORDS}\b)"
    r"|\b(?i:rue|avenue|boulevard|impasse|all[ée]e|quai)\s+(?i:de\s+la\s+|du\s+|des\s+|de\s+l'|d')?[A-ZÉÈ]")
ADDRESS_REPLACEMENT = "[adresse retirée]"

GZ = {"method": "gzip", "mtime": 0}      # reproducible gzip bytes


# --------------------------------------------------------------------------- #
# 1. pull (cached)
# --------------------------------------------------------------------------- #
def _get_json(url: str) -> dict:
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                return json.load(resp)
        # URLError, timeouts and resets are OSError; a truncated body is an
        # HTTPException (IncompleteRead); a garbled one a JSONDecodeError.
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            last = exc
            print(f"    retry {attempt + 1}/6 after {type(exc).__name__}: {exc}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"API request failed 6 times: {url[:200]}…\n  last error: {last}")


def _read_gz_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _write_gz_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    tmp.replace(path)                     # a killed run never leaves half a page


def first_page_url(dep: str) -> str:
    qs = (f'code_departement_ban:"{dep}" AND date_etablissement_dpe:[{WINDOW[0]} TO {WINDOW[1]}]'
          f" AND type_batiment:({' OR '.join(TYPES)})")
    return f"{API}/lines?" + urllib.parse.urlencode(
        {"qs": qs, "sort": "_rand", "size": PAGE_SIZE})


def superseded_by_api(ids: list[str]) -> list[dict]:
    """Every DPE in the whole dataset whose `numero_dpe_remplace` is one of `ids`."""
    hits = []
    for i in range(0, len(ids), SUPERSEDED_BATCH):
        batch = ids[i:i + SUPERSEDED_BATCH]
        url = f"{API}/lines?" + urllib.parse.urlencode({
            "numero_dpe_remplace_in": ",".join(batch), "size": 10_000,
            "select": "numero_dpe,numero_dpe_remplace,date_etablissement_dpe"})
        payload = _get_json(url)
        if payload["total"] > len(payload["results"]):
            raise SystemExit(f"supersession lookup truncated: {payload['total']} hits")
        hits.extend(payload["results"])
    return hits


def pull_departement(dep: str) -> pd.DataFrame:
    """Pages in `_rand` order until 12,500 non-superseded DPEs, all cached."""
    folder = RAW / "pull" / dep
    rows: list[dict] = []
    superseded: set[str] = set()
    url, page = first_page_url(dep), 0
    while True:
        path = folder / f"page_{page:03d}.json.gz"
        if path.exists():
            cached = _read_gz_json(path)
            if cached["url"] != url:
                raise SystemExit(f"{path}: cached for a different request; delete data/raw/ to re-pull")
        else:
            print(f"  {dep}: fetching page {page}", flush=True)
            payload = _get_json(url)
            results = payload["results"]
            ids = [r["numero_dpe"] for r in results]
            # `total` is only returned on the first page (data-fair omits it with `after`)
            cached = {"url": url, "next": payload.get("next"),
                      "total": payload["total"] if page == 0 else None,
                      "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "results": results, "superseded_hits": superseded_by_api(ids)}
            _write_gz_json(path, cached)
        if page == 0:
            pool_total = cached["total"]
        rows.extend(cached["results"])
        superseded.update(h["numero_dpe_remplace"] for h in cached["superseded_hits"])
        kept = sum(1 for r in rows if r["numero_dpe"] not in superseded)
        if kept >= PER_DEPARTEMENT:
            break
        if not cached["next"] or not cached["results"]:
            raise SystemExit(f"{dep}: API exhausted at {kept} kept rows (< {PER_DEPARTEMENT})")
        url, page = cached["next"], page + 1

    df = pd.DataFrame(rows)
    df["_dep_requested"] = dep                  # the cache folder, not the row's claim
    df["_superseded_api"] = df["numero_dpe"].isin(superseded)
    df["_pool_total"] = pool_total
    return df


def load_schema() -> list[str]:
    path = RAW / "schema.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_get_json(f"{API}/schema"), ensure_ascii=False, indent=1))
    return [f["key"] for f in json.loads(path.read_text())]


def pull() -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(pull_departement, dep): dep for dep in DEPARTEMENTS}
        for future in as_completed(futures):     # the first dead worker stops the pull
            error = future.exception()
            if error is not None:
                pool.shutdown(cancel_futures=True)
                raise SystemExit(f"pull of département {futures[future]} failed: "
                                 f"{type(error).__name__}: {error}")
        by_dep = {dep: future.result() for future, dep in futures.items()}
        frames = [by_dep[dep] for dep in DEPARTEMENTS]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 2-3. validate, dedupe, sample
# --------------------------------------------------------------------------- #
def validate_pool(df: pd.DataFrame) -> None:
    bad_dep = df.groupby("_dep_requested")["code_departement_ban"].apply(
        lambda s: sorted(set(s.astype(str)) - {s.name}))
    if bad_dep.map(len).any():
        raise SystemExit(f"rows outside their requested département: {bad_dep[bad_dep.map(len) > 0]}")
    dates = df["date_etablissement_dpe"]
    if dates.isna().any() or (dates < WINDOW[0]).any() or (dates > WINDOW[1]).any():
        raise SystemExit(f"DPE dates outside {WINDOW}: {dates.min()}..{dates.max()}")
    if not df["type_batiment"].isin(TYPES).all():
        raise SystemExit(f"unexpected type_batiment: {df['type_batiment'].unique()}")
    if not df["etiquette_dpe"].isin(list("ABCDEFG")).all():
        raise SystemExit(f"unexpected etiquette_dpe: {df['etiquette_dpe'].unique()}")
    for dep, part in df.groupby("_dep_requested"):
        if not part["_rand"].is_monotonic_increasing:
            raise SystemExit(f"{dep}: pages are not in _rand order")


def dedupe_and_sample(pool: pd.DataFrame) -> tuple[pd.DataFrame, set[str], dict]:
    if not pool["numero_dpe"].is_unique:
        raise SystemExit("the pull returned a numero_dpe twice")
    in_sample = set(pool["numero_dpe_remplace"].dropna()) & set(pool["numero_dpe"])
    superseded = set(pool.loc[pool["_superseded_api"], "numero_dpe"]) | in_sample
    stats = {"pulled": len(pool), "superseded_api": int(pool["_superseded_api"].sum()),
             "superseded_in_sample": len(in_sample),
             "replaces_an_older_dpe": float(pool["numero_dpe_remplace"].notna().mean())}
    fresh = pool[~pool["numero_dpe"].isin(superseded)]
    sample = (fresh.groupby("_dep_requested", sort=False, group_keys=False)
                   .head(PER_DEPARTEMENT).reset_index(drop=True))
    counts = sample["_dep_requested"].value_counts()
    if set(counts.index) != set(DEPARTEMENTS) or (counts != PER_DEPARTEMENT).any():
        raise SystemExit(f"per-département sample sizes wrong: {counts.to_dict()}")
    return sample, superseded, stats


# --------------------------------------------------------------------------- #
# 4. split, jitter, ids
# --------------------------------------------------------------------------- #
def salted_seed(*parts: str) -> int:
    return int.from_bytes(_digest(DATASET, *parts, size=4), "big")


def jitter_surfaces(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(salted_seed("surface-jitter"))
    factor = rng.uniform(1 - JITTER, 1 + JITTER, size=len(df))
    out = df.copy()
    for col in JITTERED_SURFACES:
        values = pd.to_numeric(out[col], errors="raise").astype(float)
        out[col] = np.round(values.to_numpy() * factor, 1)
    return out


def scrub_addresses(df: pd.DataFrame) -> pd.DataFrame:
    """Replace street addresses inside the kept free-text descriptions.

    A few diagnosticians write the address of a collective heating substation
    into the description ("hors volume chauffé en sous station du 82 rue de la
    ..."): a direct join to the public base, which every address column is
    dropped to prevent. Only the address is replaced; the rest of the text
    stays as ADEME publishes it.
    """
    out = df.copy()
    for col in [c for c in KEPT if c.startswith("description_") and c in out.columns]:
        values = out[col]
        text = values.dropna().astype(str)
        hits = text[text.str.contains(ADDRESS_IN_TEXT)]
        if hits.empty:
            continue
        snippets = sorted({m.group(0).strip() for v in hits for m in ADDRESS_IN_TEXT.finditer(v)})
        out.loc[hits.index, col] = hits.str.replace(ADDRESS_IN_TEXT, ADDRESS_REPLACEMENT, regex=True)
        print(f"  {col}: street address replaced in {len(hits)} rows ({len(snippets)} distinct)")
    return out


def building_groups(sample: pd.DataFrame) -> pd.Series:
    """Connected components of rows sharing any BUILDING_KEYS value.

    Labelled by the smallest `numero_dpe` of the component, so the labels (and
    therefore GroupShuffleSplit's draw) depend on the rows, not on their order.
    """
    parent = np.arange(len(sample))

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:                 # path compression
            parent[i], i = root, parent[i]
        return root

    for key in BUILDING_KEYS:
        first: dict[str, int] = {}
        for i, value in enumerate(sample[key].to_numpy()):
            if not isinstance(value, str) or not value.strip():
                continue
            j = first.setdefault(value, i)
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(sample))])
    numbers = pd.Series(sample["numero_dpe"].to_numpy())
    label = numbers.groupby(roots).transform("min")
    return pd.Series(label.to_numpy(), index=sample.index, name="_group")


def split(sample: pd.DataFrame):
    from sklearn.model_selection import GroupShuffleSplit

    groups = building_groups(sample)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=salted_seed("group-split"))
    tr_idx, te_idx = next(splitter.split(sample, groups=groups))
    return tr_idx, te_idx, groups


# --------------------------------------------------------------------------- #
# 6. benchmark + env
# --------------------------------------------------------------------------- #
def load_env():
    """Stage env.py + its private labels the way the platform lays out the env folder."""
    stage = Path(tempfile.mkdtemp(prefix="dpe-env-"))
    shutil.copy2(HERE / "env.py", stage / "env.py")
    for name in ("labels_train.csv", "labels_test.csv"):
        shutil.copy2(DATA / name, stage / name)
    spec = importlib.util.spec_from_file_location("dpe_env", stage / "env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Env(is_evaluation=True)


def benchmark_features(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """No domain knowledge: the numeric columns as pandas reads them, median
    imputed and standardized, both fitted on the train rows."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    features = [c for c in train.columns if c not in (ID_COLUMN, TARGET)]
    num_tr = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    num_te = [c for c in features if pd.api.types.is_numeric_dtype(test[c])]
    if num_tr != num_te:
        raise SystemExit(f"numeric columns differ between train and test: "
                         f"{sorted(set(num_tr) ^ set(num_te))}")
    # a column with no value at all in train has no median: it cannot be imputed
    names = [c for c in num_tr if train[c].notna().any()]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    a = scaler.fit_transform(imputer.fit_transform(train[names]))
    b = scaler.transform(imputer.transform(test[names]))
    return pd.concat([
        pd.DataFrame(a, columns=names).assign(**{ID_COLUMN: train[ID_COLUMN].to_numpy()}),
        pd.DataFrame(b, columns=names).assign(**{ID_COLUMN: test[ID_COLUMN].to_numpy()}),
    ], ignore_index=True)[[ID_COLUMN, *names]], len(num_tr)


# --------------------------------------------------------------------------- #
# 7. leak canary
# --------------------------------------------------------------------------- #
def leak_canary(train: pd.DataFrame, test: pd.DataFrame, y_test: np.ndarray) -> float:
    """LightGBM on the shipped raw columns (text as categories). The ceiling a
    boosted model reaches without any calculated column; far above it means
    a calculated column slipped through."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    features = [c for c in train.columns if c not in (ID_COLUMN, TARGET)]

    def frame(df):
        out = {}
        for i, c in enumerate(features):
            if pd.api.types.is_numeric_dtype(train[c]):
                out[f"f{i}"] = df[c].astype(float)
            else:
                cats = sorted(train[c].dropna().astype(str).unique())
                values = df[c].astype(str).where(df[c].notna())
                out[f"f{i}"] = pd.Categorical(values.where(values.isin(cats)), categories=cats)
        return pd.DataFrame(out, index=df.index)

    X_tr, X_te = frame(train), frame(test)
    y_tr = train[TARGET].to_numpy()
    val = np.random.default_rng(0).random(len(X_tr)) < 0.15
    model = lgb.LGBMClassifier(
        n_estimators=4000, learning_rate=0.03, num_leaves=63, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=0, verbose=-1)
    model.fit(X_tr[~val], y_tr[~val], eval_set=[(X_tr[val], y_tr[val])],
              eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    auc = float(roc_auc_score(y_test, model.predict_proba(X_te)[:, 1]))
    print(f"  leak canary: LightGBM on {len(features)} raw columns, "
          f"{model.best_iteration_} trees -> test AUC {auc:.4f}")
    return auc


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pull-only", action="store_true",
                    help="fill the data/raw/ cache and stop")
    ap.add_argument("--skip-canary", action="store_true",
                    help="skip step 7 (LightGBM); for iterating on docs only")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    print("1. pull (data/raw/ cache) …")
    schema = load_schema()
    pool = pull()
    validate_pool(pool)
    print(f"  pulled {len(pool)} rows; pool sizes "
          f"{pool.groupby('_dep_requested')['_pool_total'].first().to_dict()}")
    if args.pull_only:
        return

    print("2-3. dedupe + sample …")
    sample, superseded, stats = dedupe_and_sample(pool)
    print(f"  {stats}")
    print(f"  sample: {len(sample)} rows, {PER_DEPARTEMENT} per département")

    print("5a. drop audit …")
    source_columns = [c for c in pool.columns if not c.startswith("_") or c in DROPPED]
    source_columns = [c for c in source_columns if c not in ("_dep_requested",)]
    verdict = classify(sorted(set(schema) | set(source_columns)))
    n_kept = sum(v == "kept" for v in verdict.values())
    print(f"  {len(verdict)} source columns: {n_kept} kept, {len(verdict) - n_kept} dropped")
    never_filled = [c for c in KEPT if c not in sample.columns]   # absent from every row pulled
    sample = pd.concat([sample, pd.DataFrame(np.nan, index=sample.index, columns=never_filled)],
                       axis=1).copy()
    print(f"  kept but never filled in this sample: {never_filled}")

    print("4. split, jitter, ids …")
    sample[TARGET] = sample["etiquette_dpe"].isin(POSITIVE_LABELS).astype(int)
    sample = jitter_surfaces(sample)
    sample = scrub_addresses(sample)
    tr_idx, te_idx, groups = split(sample)
    sample["_group"] = groups
    helpers = ["numero_dpe", "_group", "_dep_requested", *BUILDING_KEYS]

    def ship(idx, split_name, prefix):
        part = sample.iloc[idx].reset_index(drop=True)
        X = part[KEPT + helpers]
        y = part[TARGET]
        X, y = shuffle_and_label(X, y, dataset=DATASET, split=split_name, prefix=prefix)
        return X, y

    X_tr, y_tr = ship(tr_idx, "train", "tr")
    X_te, y_te = ship(te_idx, "test", "te")

    print("5b. assertions …")
    spans = set(X_tr["_group"]) & set(X_te["_group"])
    if spans:
        raise SystemExit(f"{len(spans)} groups span train and test, e.g. {sorted(spans)[:3]}")
    for key in BUILDING_KEYS:           # checked on the keys themselves, not on the labels
        shared = set(X_tr[key].dropna()) & set(X_te[key].dropna())
        if shared:
            raise SystemExit(f"{len(shared)} values of {key} are on both sides of the split")
    kept_superseded = (set(X_tr["numero_dpe"]) | set(X_te["numero_dpe"])) & superseded
    if kept_superseded:
        raise SystemExit(f"superseded DPEs kept: {sorted(kept_superseded)[:5]}")
    rate_tr, rate_te = float(y_tr.mean()), float(y_te.mean())
    if abs(rate_tr - rate_te) > MAX_RATE_GAP:
        raise SystemExit(f"target rates differ by more than {MAX_RATE_GAP:.3f}: "
                         f"train {rate_tr:.4f}, test {rate_te:.4f}")
    if set(X_tr[ID_COLUMN]) & set(X_te[ID_COLUMN]):
        raise SystemExit("train and test ids overlap")
    n_groups = groups.nunique()
    multi = groups.value_counts()
    print(f"  groups: {n_groups} for {len(sample)} rows; largest {int(multi.max())}; "
          f"{int((multi > 1).sum())} groups with >1 row covering {int(multi[multi > 1].sum())} rows")
    print(f"  train {len(X_tr)} rows, E/F/G {rate_tr:.4f} | test {len(X_te)} rows, E/F/G {rate_te:.4f}")
    by_dep = pd.concat([X_tr.assign(y=y_tr.to_numpy()), X_te.assign(y=y_te.to_numpy())]
                       ).groupby("_dep_requested")["y"].mean().round(3).to_dict()
    print(f"  E/F/G by département: {by_dep}")

    print("6. write …")
    train = X_tr.drop(columns=helpers).assign(**{TARGET: y_tr.to_numpy()})
    test = X_te.drop(columns=helpers)
    train.to_csv(DATA / "train.csv.gz", index=False, compression=GZ)
    test.to_csv(DATA / "test.csv.gz", index=False, compression=GZ)
    pd.DataFrame({ID_COLUMN: X_tr[ID_COLUMN], TARGET: y_tr}).to_csv(DATA / "labels_train.csv", index=False)
    pd.DataFrame({ID_COLUMN: X_te[ID_COLUMN], TARGET: y_te}).to_csv(DATA / "labels_test.csv", index=False)

    # Read back what ships: the assertions and the benchmark see exactly the
    # files a student downloads.
    train = pd.read_csv(DATA / "train.csv.gz", low_memory=False)
    test = pd.read_csv(DATA / "test.csv.gz", low_memory=False)
    expected_train = [ID_COLUMN, *KEPT, TARGET]
    if list(train.columns) != expected_train or list(test.columns) != expected_train[:-1]:
        raise SystemExit("shipped headers differ from id + KEPT (+ target)")
    leaked = (set(train.columns) | set(test.columns)) & set(DROPPED)
    if leaked:
        raise SystemExit(f"dropped columns in a shipped file: {sorted(leaked)}")
    for name, frame in (("train", train), ("test", test)):
        for col in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[col]):
                continue
            left = frame[col].dropna().astype(str).str.contains(ADDRESS_LEFT)
            if left.any():
                raise SystemExit(f"{name}.csv.gz {col}: a street address survived the scrub "
                                 f"in {int(left.sum())} rows")
    # Not a leak, a measurement: identical diagnoses in different buildings
    # (a housing programme built and diagnosed from one template) are allowed
    # on both sides; this is how many test rows have one.
    same = [c for c in KEPT if c not in JITTERED_SURFACES]
    h_tr = pd.util.hash_pandas_object(train[same].astype(str), index=False)
    h_te = pd.util.hash_pandas_object(test[same].astype(str), index=False)
    print(f"  test rows with an identical non-surface twin in train (other building): "
          f"{h_te.isin(set(h_tr)).mean():.4f}")
    for name in ("sample_submission.csv.gz", "benchmark_submission.csv.gz"):
        (DATA / name).unlink(missing_ok=True)

    bench, n_numeric = benchmark_features(train, test)
    bench.to_csv(DATA / "benchmark_submission.csv.gz", index=False,
                 float_format="%.6f", compression=GZ)
    shutil.copy2(DATA / "benchmark_submission.csv.gz", DATA / "sample_submission.csv.gz")
    print(f"  benchmark: {n_numeric} numeric columns as read, "
          f"{bench.shape[1] - 1} after dropping empty ones")

    result = load_env().evaluate(str(DATA / "benchmark_submission.csv.gz"))["agent_results"][0]
    if result.get("is_agent_code_error"):
        raise SystemExit(f"env.py rejected the benchmark: {result['agent_code_error_message']}")
    print(f"  benchmark scored by env.py: {result['info_message']}")
    print(f"  metrics_detail: {result['metrics_detail']}")

    import dictionary
    dictionary.render(train, HERE / "DICTIONNAIRE.md")
    for doc in ("EXPERTISE.md", "DICTIONNAIRE.md"):
        shutil.copy2(HERE / doc, DATA / doc)
    print("  DICTIONNAIRE.md rendered; EXPERTISE.md + DICTIONNAIRE.md copied to data/")

    if args.skip_canary:
        print("7. leak canary SKIPPED (--skip-canary): not a releasable build")
    else:
        print("7. leak canary …")
        auc = leak_canary(train, test, y_te.to_numpy())
        if auc > CANARY_MAX_AUC:
            raise SystemExit(f"leak canary {auc:.4f} > {CANARY_MAX_AUC}: a calculated "
                             f"column is in the shipped data")

    summary = {"rows_train": len(train), "rows_test": len(test),
               "rate_train": round(rate_tr, 6), "rate_test": round(rate_te, 6),
               "benchmark_score": result["score"], **stats}
    (DATA / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  set benchmark_expected_score = {result['score']}")
    print(f"wrote {DATA}")


if __name__ == "__main__":
    main()
