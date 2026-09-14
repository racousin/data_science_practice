#!/usr/bin/env python3
"""Local tests for the critical-care survival challenge — the scorer, the split, the notebook.

    cd courseware/competitions
    uv run --with scikit-learn==1.8.0 --with pandas --with pytest \\
        --with nbformat --with nbclient --with ipykernel \\
        pytest s2-icu-survival/test_icu_survival.py -v

Needs `data/` (run prepare_data.py first): every test but the config ones
reads the shipped split, and the starter-notebook test copies the public
files into its working directory. Kept out of `test_challenges.py` because
that suite's generic checks assume an `id,prediction` submission, and this
challenge's submission is a feature matrix.

What is covered, and why:

* the declared `benchmark_expected_score` is what `env.py` returns for the
  package's own benchmark file (skipped, with a message, while the score is
  still None), and `metrics_detail` carries exactly the keys `metrics_schema`
  declares (the executor enforces equal-mapping at run time);
* the split: disjoint ids prefixed tr_/te_, labels aligned with the shipped
  files, shipped headers are exactly the served columns of the design in
  order (`dead` last in train, absent from test), no dropped SUPPORT2 column
  anywhere, target rates of train and test within 1.5 points and both in the
  documented 40-55% band;
* the traps the data is served with, each one measurable: `charges` empty in
  test and filled in train (the post-outcome leak), `glucose` read as text
  ('not done'), dirty `sex` spellings, sites A-E, creatinine in µmol/L at sites D/E,
  temperature in °F at site C, the 999 age placeholder, the sodium decimal
  slip, the 0.0 temperature placeholder;
* every rejection rule of the scorer, each with a message naming the offender
  (plus a UTF-8 BOM accepted, and a fanned-out file stopped before parsing);
* a train subset of exactly 4,000 rows is accepted, 3,999 is not;
* the two warnings: a label copied into a train feature trips the gap flag, and
  unscaled features trip the non-convergence flag;
* `evaluate()` on the benchmark file takes well under 10 s — memory is not a
  concern at ~9,100 rows (DPE's 100k x 250 RSS probe has no counterpart here);
* the starter notebook runs end to end offline against `data/`, and the
  submission it writes is accepted by the scorer. Its download and submit cells
  are neutralised: a test never touches the live platform.
"""
from __future__ import annotations

import gzip
import importlib.util
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"
sys.path.insert(0, str(PKG))
from dictionary import ID_COLUMN, SERVED, TARGET  # noqa: E402

# The served list, whichever way dictionary.py spells it (with or without the
# id and the target): the shipped headers are id + FEATURES (+ target).
FEATURES = [c for c in SERVED if c not in (ID_COLUMN, TARGET)]
# SUPPORT2 columns that never ship (DESIGN.md): outcomes, follow-up, the
# SUPPORT model's own scores, and the cost columns.
DROPPED_SUPPORT2 = [
    "death", "hospdead", "d.time", "slos", "sfdm2", "adlsc", "dnr", "dnrday",
    "prg2m", "prg6m", "surv2m", "surv6m", "sps", "aps", "avtisst", "totcst",
    "totmcst",
]
MIN_TRAIN_ROWS = 4_000
GZ = {"method": "gzip", "compresslevel": 1, "mtime": 0}
EVALUATE_SECONDS_LIMIT = 10.0


def load_config() -> dict:
    ns: dict = {}
    exec((PKG / "config.py").read_text(), ns)
    return ns["CONFIG"]


def stage_env(tmp: Path) -> Path:
    """Lay out env.py + its private files the way the platform does."""
    cfg = load_config()
    shutil.copy2(PKG / "env.py", tmp / "env.py")
    for rel in cfg["private_files"]:
        shutil.copy2(DATA / rel, tmp / rel)
    return tmp / "env.py"


def load_env(tmp: Path):
    path = stage_env(tmp)
    spec = importlib.util.spec_from_file_location(f"icu_env_{tmp.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Env(is_evaluation=True)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not (DATA / "labels_test.csv").exists():
        pytest.fail("data/ is missing: run prepare_data.py first")
    return load_env(tmp_path_factory.mktemp("envstage"))


@pytest.fixture(scope="module")
def bench(cfg):
    return pd.read_csv(PKG / cfg["benchmark_file"], dtype={ID_COLUMN: str})


@pytest.fixture(scope="module")
def labels():
    return (pd.read_csv(DATA / "labels_train.csv", dtype={ID_COLUMN: str}),
            pd.read_csv(DATA / "labels_test.csv", dtype={ID_COLUMN: str}))


@pytest.fixture(scope="module")
def shipped():
    """(train, test) exactly as a student reads them: pandas' own dtypes."""
    if not (DATA / "train.csv.gz").exists():
        pytest.fail("data/ is missing: run prepare_data.py first")
    return (pd.read_csv(DATA / "train.csv.gz", low_memory=False, dtype={ID_COLUMN: str}),
            pd.read_csv(DATA / "test.csv.gz", low_memory=False, dtype={ID_COLUMN: str}))


@pytest.fixture(scope="module")
def both(shipped):
    train, test = shipped
    return pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)


def evaluate(env, frame: pd.DataFrame, path: Path) -> dict:
    frame.to_csv(path, index=False, compression=GZ)
    return env.evaluate(str(path))["agent_results"][0]


def assert_rejected(result: dict, *needles: str) -> None:
    assert result.get("is_agent_code_error"), f"accepted: {result.get('info_message')}"
    assert result["score"] == 0.0
    message = result["agent_code_error_message"]
    for needle in needles:
        assert needle in message, f"{needle!r} not in message: {message}"


# --------------------------------------------------------------------------- #
# package
# --------------------------------------------------------------------------- #
def test_package_files_present(cfg):
    for name in ("config.py", "env.py", "overview.md", "prepare_data.py", "dictionary.py",
                 "EXPERTISE.md", "DICTIONARY.md", "starter.ipynb"):
        assert (PKG / name).is_file(), f"{name} missing"
    for rel in cfg["public_files"] + cfg["private_files"]:
        assert (DATA / rel).is_file() or (PKG / rel).is_file(), \
            f"{rel} missing — run prepare_data.py"
    assert (PKG / cfg["benchmark_file"]).is_file()
    assert (DATA / "sample_submission.csv.gz").read_bytes() == \
        (PKG / cfg["benchmark_file"]).read_bytes(), "sample_submission is the benchmark's file"


def test_config_contract(cfg):
    assert cfg["name"] == "MLP S2 — Critical Care Survival"
    assert cfg["kernel_version"] == "file_v1"
    assert cfg["module_slug"] == "s2-data-preprocessing"
    assert cfg["label"] == cfg["dataset_label"] == "Critical Care Survival"
    assert cfg["submission_filename"] == "submission.csv.gz"
    assert cfg["max_upload_size_bytes"] == 50 * 1024 * 1024
    assert cfg["engine_id"] == 27
    assert cfg["metric"] == "auc" and cfg["metric2"] == "auc_train"
    assert (cfg["deployment_nb_constraint_run"], cfg["deployment_nb_initial_score_run"]) == (1, 1)
    assert cfg["is_public_initial"] is False
    assert len(cfg["dataset_description"]) <= 500
    assert cfg["public_files"] == ["train.csv.gz", "test.csv.gz", "sample_submission.csv.gz",
                                   "EXPERTISE.pdf", "DICTIONARY.pdf"]
    assert cfg["private_files"] == ["labels_train.csv", "labels_test.csv"]
    # pinned after prepare_data.py / the ladder; None reads as absent for the builder
    for key in ("benchmark_expected_score", "pass_threshold", "expert_expected_score"):
        assert key in cfg
        assert cfg[key] is None or isinstance(cfg[key], float), f"{key}: {cfg[key]!r}"
    if cfg["pass_threshold"] is not None:
        assert cfg["expert_expected_score"] is not None, "a bar needs the expert score"
        assert cfg["pass_threshold"] <= cfg["expert_expected_score"]
    public = set(cfg["public_files"])
    assert not public & {"labels_train.csv", "labels_test.csv", "env.py"}
    assert not any("teacher" in f for f in public), "the expert pipeline must never be public"
    assert not any(f.startswith("y_test") for f in cfg["private_files"]), (
        "a private file named y_test.csv would be auto-extracted into agent_template")


def test_env_constants_match_the_design():
    spec = importlib.util.spec_from_file_location("icu_env_constants", PKG / "env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TARGET_COLUMN == TARGET == "dead"
    assert module.ID_COLUMN == ID_COLUMN == "id"
    assert module.MIN_TRAIN_ROWS == MIN_TRAIN_ROWS
    assert module.MAX_FEATURES == 300
    assert module.SUBMISSION == "submission.csv.gz"
    assert module.METRIC_KEYS == ("auc", "auc_train", "n_features", "n_train_rows", "converged")


def test_metrics_schema_is_valid(cfg):
    repo = PKG.parents[4]            # .../reinforcement_learning_challenge
    sys.path.insert(0, str(repo / "modelmanager"))
    try:
        from modelmanager.metrics_schema import validate_metrics_schema
    except ImportError:
        pytest.skip("modelmanager not importable from here")
    assert validate_metrics_schema(cfg["metrics_schema"]) == []


def test_benchmark_hits_declared_score(cfg, env):
    result = env.evaluate(str(PKG / cfg["benchmark_file"]))["agent_results"][0]
    assert not result.get("is_agent_code_error"), result.get("agent_code_error_message")
    declared = {d["key"] for d in cfg["metrics_schema"] if d["source"] == "env"}
    assert set(result["metrics_detail"]) == declared
    assert result["metrics_detail"]["n_train_rows"] == len(pd.read_csv(DATA / "labels_train.csv"))
    # deterministic: the same file, the same number
    again = env.evaluate(str(PKG / cfg["benchmark_file"]))["agent_results"][0]
    assert again["score"] == result["score"] and again["score2"] == result["score2"]
    if cfg["benchmark_expected_score"] is None:
        pytest.skip(f"benchmark_expected_score is not pinned yet: env.py scores the "
                    f"benchmark file {result['score']} — set that in config.py")
    assert abs(result["score"] - cfg["benchmark_expected_score"]) <= cfg["benchmark_score_tol"], (
        f"env.py scored {result['score']}, config declares {cfg['benchmark_expected_score']}")


def test_error_rows_still_carry_the_declared_keys(cfg, env, tmp_path):
    bad = tmp_path / "submission.csv.gz"
    bad.write_text("id,x\n")
    result = env.evaluate(str(bad))["agent_results"][0]
    declared = {d["key"] for d in cfg["metrics_schema"] if d["source"] == "env"}
    assert set(result["metrics_detail"]) == declared


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def test_split_is_disjoint_complete_and_clean(shipped, labels):
    train, test = shipped
    ytr, yte = labels
    assert list(train.columns) == [ID_COLUMN, *FEATURES, TARGET]
    assert list(test.columns) == [ID_COLUMN, *FEATURES]
    assert not (set(train.columns) | set(test.columns)) & set(DROPPED_SUPPORT2)
    assert train[ID_COLUMN].is_unique and test[ID_COLUMN].is_unique
    assert not set(train[ID_COLUMN]) & set(test[ID_COLUMN])
    assert train[ID_COLUMN].str.fullmatch(r"tr_[0-9a-f]{12}").all()
    assert test[ID_COLUMN].str.fullmatch(r"te_[0-9a-f]{12}").all()
    assert list(train[ID_COLUMN]) == list(ytr[ID_COLUMN])
    assert list(test[ID_COLUMN]) == list(yte[ID_COLUMN])
    assert list(ytr.columns) == list(yte.columns) == [ID_COLUMN, TARGET]
    assert (train[TARGET].to_numpy() == ytr[TARGET].to_numpy()).all()
    assert set(train[TARGET].unique()) == {0, 1}
    # the SUPPORT2 cohort is 9,105 patients; a ~70/30 split
    assert 9_000 <= len(train) + len(test) <= 9_105
    assert 0.25 <= len(test) / (len(train) + len(test)) <= 0.35
    assert len(train) >= MIN_TRAIN_ROWS
    # ~45-47% positives by design, and the split does not shift the rate
    assert 0.40 <= ytr[TARGET].mean() <= 0.55
    assert 0.40 <= yte[TARGET].mean() <= 0.55
    assert abs(ytr[TARGET].mean() - yte[TARGET].mean()) <= 0.015


def test_no_dropped_support2_column_leaks_into_dictionary():
    assert not set(SERVED) & set(DROPPED_SUPPORT2)
    assert FEATURES[0] == "site" and FEATURES[-1] == "charges"
    assert len(FEATURES) == 31


# --------------------------------------------------------------------------- #
# the traps, as served
# --------------------------------------------------------------------------- #
def test_charges_are_empty_in_test_and_filled_in_train(shipped):
    """Billed at discharge: known for the historical cohort, unknown for the
    patients still admitted — the post-outcome column a model must not use."""
    train, test = shipped
    assert test["charges"].isna().all(), "charges must be 100% NaN in test.csv.gz"
    assert train["charges"].notna().mean() > 0.9, (
        f"charges filled for only {train['charges'].notna().mean():.1%} of train rows")


def test_glucose_is_read_as_text(shipped):
    """The chemistry panel's "not available" token must survive pd.read_csv.
    pandas' default na_values already swallow 'n/a', 'N/A', 'NA', 'null',
    'None', 'nan' and '' (they come back as NaN and the column stays float64),
    so the token prepare_data.py serves has to be one that is not on that list
    — the trap is the dtype, not the spelling."""
    for name, frame in zip(("train", "test"), shipped):
        col = frame["glucose"]
        assert not pd.api.types.is_numeric_dtype(col), (
            f"{name}: glucose is numeric — no text token survived pd.read_csv "
            f"(pandas reads 'n/a' as NaN by default)")
        text = col[pd.to_numeric(col, errors="coerce").isna() & col.notna()]
        assert len(text) > 0, f"{name}: no text value in glucose"
        tokens = sorted(text.astype(str).unique())
        assert len(tokens) == 1, f"{name}: one 'not available' token expected, got {tokens}"
        assert pd.to_numeric(col, errors="coerce").notna().mean() > 0.3, (
            f"{name}: glucose should mostly be numbers")
    print(f"\n  glucose text token: {tokens}")


def test_sex_has_dirty_spellings(both):
    spellings = set(both["sex"].dropna().unique())
    assert len(spellings) > 2, f"sex has only {sorted(spellings)}"
    lowered = {s.strip().lower() for s in spellings}
    assert lowered <= {"male", "female", "m", "f"}, f"unexpected sex value(s): {sorted(spellings)}"


def test_site_takes_exactly_a_to_e(both):
    assert set(both["site"].dropna().unique()) == set("ABCDE")
    assert both["site"].notna().all()


def test_creatinine_is_in_umol_per_l_at_sites_d_and_e(both):
    crea = pd.to_numeric(both["crea"], errors="coerce")
    de = crea[both["site"].isin(["D", "E"])].median()
    abc = crea[both["site"].isin(["A", "B", "C"])].median()
    assert de > 30 * abc, f"creatinine median D/E {de} vs A-C {abc}: not x88.4 apart"


def test_temperature_is_in_fahrenheit_at_site_c(both):
    temp = pd.to_numeric(both["temp"], errors="coerce")
    at_c = temp[(both["site"] == "C") & (temp != 0.0)].median()
    elsewhere = temp[(both["site"] != "C") & (temp != 0.0)].median()
    assert at_c > 90, f"site C temperature median {at_c} is not °F"
    assert 30 < elsewhere < 45, f"other sites' temperature median {elsewhere} is not °C"


def test_age_carries_the_999_placeholder(both):
    age = pd.to_numeric(both["age"], errors="coerce")
    assert (age == 999).any()
    assert (age == 999).mean() < 0.02, "the 999 placeholder is rare (~0.4%)"
    assert age[age != 999].max() < 110


def test_sodium_carries_the_decimal_slip(both):
    sod = pd.to_numeric(both["sod"], errors="coerce")
    assert (sod > 500).any()
    assert (sod > 500).mean() < 0.02, "the x10 slip is rare (~0.6%)"
    assert 130 < sod[sod <= 500].median() < 145


def test_temperature_carries_the_zero_placeholder(both):
    temp = pd.to_numeric(both["temp"], errors="coerce")
    assert (temp == 0.0).any()
    assert (temp == 0.0).mean() < 0.02, "the 0.0 placeholder is rare (~0.5%)"


# --------------------------------------------------------------------------- #
# rejections
# --------------------------------------------------------------------------- #
def test_rejects_a_plain_csv(env, bench, tmp_path):
    path = tmp_path / "submission.csv.gz"
    bench.head(50).to_csv(path, index=False, compression=None)   # pandas would infer gzip from .gz
    assert_rejected(env.evaluate(str(path))["agent_results"][0], "not gzip")


def test_rejects_a_missing_test_id(env, bench, labels, tmp_path):
    victim = labels[1][ID_COLUMN].iloc[123]
    assert_rejected(evaluate(env, bench[bench[ID_COLUMN] != victim], tmp_path / "s.csv.gz"),
                    victim)


def test_rejects_an_unknown_id(env, bench, tmp_path):
    extra = bench.head(1).assign(id="te_000000000000")
    assert_rejected(evaluate(env, pd.concat([bench, extra]), tmp_path / "s.csv.gz"),
                    "te_000000000000")


def test_rejects_a_duplicate_id(env, bench, tmp_path):
    dup = bench.iloc[[500]]
    assert_rejected(evaluate(env, pd.concat([bench, dup]), tmp_path / "s.csv.gz"),
                    dup[ID_COLUMN].iloc[0])


def test_rejects_an_empty_id(env, bench, tmp_path):
    frame = bench.copy()
    frame.loc[7, ID_COLUMN] = None
    assert_rejected(evaluate(env, frame, tmp_path / "s.csv.gz"), "Empty 'id'", "line 9")


def test_rejects_a_text_column(env, bench, tmp_path):
    frame = bench.copy()
    col = frame.columns[3]
    frame[col] = frame[col].astype(object)
    frame.loc[42, col] = "Lung Cancer"
    assert_rejected(evaluate(env, frame, tmp_path / "s.csv.gz"), repr(col), "Lung Cancer",
                    frame.loc[42, ID_COLUMN])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite(env, bench, tmp_path, bad):
    frame = bench.copy()
    col = frame.columns[2]
    frame.loc[999, col] = bad
    assert_rejected(evaluate(env, frame, tmp_path / "s.csv.gz"), repr(col),
                    frame.loc[999, ID_COLUMN])


def test_target_column_left_in_names_train_csv(env, bench, labels, tmp_path):
    """The classic slip: train.csv.gz's `dead` kept as a feature is NaN on the
    test rows, and the message says which file it came from."""
    y = labels[0].set_index(ID_COLUMN)[TARGET]
    frame = bench.copy()
    frame[TARGET] = frame[ID_COLUMN].map(y)          # NaN on every test row
    assert_rejected(evaluate(env, frame, tmp_path / "s.csv.gz"), repr(TARGET),
                    "target column of train.csv.gz")


def test_rejects_too_few_train_rows_and_accepts_the_minimum(env, bench, labels, tmp_path):
    train_ids = labels[0][ID_COLUMN]
    is_test = bench[ID_COLUMN].str.startswith("te_")
    keep = set(train_ids.iloc[:MIN_TRAIN_ROWS])
    subset = bench[is_test | bench[ID_COLUMN].isin(keep)]
    ok = evaluate(env, subset, tmp_path / "ok.csv.gz")
    assert not ok.get("is_agent_code_error"), ok.get("agent_code_error_message")
    assert ok["metrics_detail"]["n_train_rows"] == MIN_TRAIN_ROWS
    keep.discard(train_ids.iloc[0])
    too_few = bench[is_test | bench[ID_COLUMN].isin(keep)]
    assert_rejected(evaluate(env, too_few, tmp_path / "few.csv.gz"),
                    str(MIN_TRAIN_ROWS - 1), str(MIN_TRAIN_ROWS))


def test_rejects_no_feature_column(env, bench, tmp_path):
    assert_rejected(evaluate(env, bench[[ID_COLUMN]], tmp_path / "s.csv.gz"),
                    "No feature columns")


def test_rejects_more_than_300_features(env, bench, tmp_path):
    frame = bench[[ID_COLUMN]].copy()
    extra = pd.DataFrame(np.zeros((len(frame), 301)), columns=[f"f{i}" for i in range(301)])
    assert_rejected(evaluate(env, pd.concat([frame, extra], axis=1), tmp_path / "s.csv.gz"),
                    "301 feature columns", "300")


def test_rejects_duplicate_column_names(env, bench, tmp_path):
    path = tmp_path / "s.csv.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("id,a,a\n" + "".join(f"{i},1,2\n" for i in bench[ID_COLUMN]))
    assert_rejected(env.evaluate(str(path))["agent_results"][0], "'a' appears twice")


def test_rejects_the_index_written_as_a_column(env, bench, tmp_path):
    path = tmp_path / "s.csv.gz"
    bench.head(100).to_csv(path, compression=GZ)            # index=True
    assert_rejected(env.evaluate(str(path))["agent_results"][0], "index=False")


def test_missing_test_ids_are_named_in_file_order(env, bench, labels, tmp_path):
    first, later = labels[1][ID_COLUMN].iloc[10], labels[1][ID_COLUMN].iloc[2_000]
    result = evaluate(env, bench[~bench[ID_COLUMN].isin({first, later})], tmp_path / "s.csv.gz")
    assert_rejected(result, first, "2 of")


def test_accepts_a_utf8_bom(env, bench, tmp_path):
    path = tmp_path / "s.csv.gz"
    path.write_bytes(gzip.compress(bench.to_csv(index=False).encode("utf-8-sig")))
    result = env.evaluate(str(path))["agent_results"][0]
    assert not result.get("is_agent_code_error"), result.get("agent_code_error_message")


def test_rejects_a_semicolon_separated_file(env, bench, tmp_path):
    path = tmp_path / "s.csv.gz"
    bench.head(100).to_csv(path, index=False, sep=";", compression=GZ)
    assert_rejected(env.evaluate(str(path))["agent_results"][0], "separated by ';'")


def test_rejects_far_more_rows_than_ids_before_parsing(env, bench, labels, tmp_path):
    """A fanned-out join must be a message, not an OOM-killed pod."""
    n_ids = len(labels[0]) + len(labels[1])
    max_rows = int(n_ids * 1.1)                      # env.ROW_SLACK
    path = tmp_path / "s.csv.gz"
    with gzip.open(path, "wt", compresslevel=1) as fh:
        fh.write("id,x\n")
        row = f"{bench[ID_COLUMN].iloc[0]},0\n"
        for _ in range(max_rows // 1_000 + 2):
            fh.write(row * 1_000)
    assert_rejected(env.evaluate(str(path))["agent_results"][0], f"more than {max_rows} rows",
                    "a join that fanned out")


def test_rejects_missing_id_column_and_header_only(env, bench, tmp_path):
    assert_rejected(evaluate(env, bench.drop(columns=[ID_COLUMN]).head(10), tmp_path / "a.csv.gz"),
                    "Missing the 'id' column")
    assert_rejected(evaluate(env, bench.head(0), tmp_path / "b.csv.gz"), "no rows")


# --------------------------------------------------------------------------- #
# the two warnings
# --------------------------------------------------------------------------- #
def test_label_in_a_train_feature_trips_the_gap_flag(env, bench, labels, tmp_path):
    y = pd.concat([labels[0], labels[1].assign(**{TARGET: 0})]).set_index(ID_COLUMN)[TARGET]
    frame = bench.copy()
    frame["leak"] = frame[ID_COLUMN].map(y).astype(float)    # the label on train, 0 on test
    result = evaluate(env, frame, tmp_path / "s.csv.gz")
    assert not result.get("is_agent_code_error"), result.get("agent_code_error_message")
    detail = result["metrics_detail"]
    assert detail["auc_train"] - detail["auc"] > 0.05
    assert "encodes the target on the train rows" in result["info_message"]


def test_unscaled_features_trip_the_convergence_flag(env, bench, tmp_path):
    frame = bench.copy()
    for i, col in enumerate(frame.columns[1:6]):
        frame[col] = frame[col] * 10 ** (i + 3)
    result = evaluate(env, frame, tmp_path / "s.csv.gz")
    assert not result.get("is_agent_code_error"), result.get("agent_code_error_message")
    assert result["metrics_detail"]["converged"] == 0
    assert "did not converge" in result["info_message"] and "scale your" in result["info_message"]


def test_benchmark_converges_and_carries_the_gap_warning(cfg, env):
    """The benchmark is scaled, so lbfgs converges. It is also naive: it keeps
    `charges`, which exists only for the (discharged) train patients, so its
    train AUC sits far above its test AUC and the scorer says so. That warning
    on the benchmark's own row is a designed lesson, not a defect."""
    result = env.evaluate(str(PKG / cfg["benchmark_file"]))["agent_results"][0]
    assert result["metrics_detail"]["converged"] == 1
    assert "did not converge" not in result["info_message"]
    assert "above test AUC" in result["info_message"]
    assert result["metrics_detail"]["auc_train"] - result["metrics_detail"]["auc"] > 0.05


# --------------------------------------------------------------------------- #
# time
# --------------------------------------------------------------------------- #
def test_evaluate_on_the_benchmark_is_fast(cfg, env):
    """~9,100 rows: the scorer is a few hundred milliseconds, and memory is
    not a concern (DPE's 100k x 250 RSS probe has no counterpart here)."""
    t0 = time.perf_counter()
    result = env.evaluate(str(PKG / cfg["benchmark_file"]))["agent_results"][0]
    seconds = time.perf_counter() - t0
    assert not result.get("is_agent_code_error"), result.get("agent_code_error_message")
    print(f"\n  benchmark evaluate(): {seconds:.2f} s, {result['metrics_detail']['n_features']} features")
    assert seconds < EVALUATE_SECONDS_LIMIT


# --------------------------------------------------------------------------- #
# the starter notebook
# --------------------------------------------------------------------------- #
def test_starter_notebook_runs_offline_and_is_accepted(env, tmp_path):
    """Needs `data/`: the public files are copied next to the notebook, and its
    download / submit / leaderboard / pip cells are replaced by a comment."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(str(PKG / "starter.ipynb"), as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert 'pip install -q mlarena-sdk' in code and "pip install mlarena\n" not in code
    assert 'files=["submission.csv.gz"]' in code
    assert "mlk_user_..." in code and "mlk_user_4" not in code

    for name in ("train.csv.gz", "test.csv.gz", "sample_submission.csv.gz",
                 "EXPERTISE.pdf", "DICTIONARY.pdf"):
        src = DATA / name if (DATA / name).is_file() else PKG / name
        shutil.copy2(src, tmp_path / name)
    neutralised = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if (src.lstrip().startswith("!pip") or "download_dataset(" in src
                or "client.submit(" in src or "client.leaderboard(" in src):
            cell.source = "# neutralised for the offline test: no network, no live platform"
            neutralised += 1
    assert neutralised == 4, f"expected 4 online cells (pip, download, submit, leaderboard), got {neutralised}"
    nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()

    produced = tmp_path / "submission.csv.gz"
    assert produced.is_file(), "the notebook did not write submission.csv.gz"
    result = env.evaluate(str(produced))["agent_results"][0]
    assert not result.get("is_agent_code_error"), result.get("agent_code_error_message")
    print(f"\n  starter notebook submission: {result['info_message']}")
