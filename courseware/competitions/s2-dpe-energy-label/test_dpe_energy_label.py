#!/usr/bin/env python3
"""Local tests for the DPE energy-label challenge — the scorer, the split, the notebook.

    cd courseware/competitions
    uv run --with scikit-learn==1.8.0 --with pandas --with pytest \\
        --with nbformat --with nbclient --with ipykernel \\
        pytest s2-dpe-energy-label/test_dpe_energy_label.py -v

Needs `data/` (run prepare_data.py first). Kept out of `test_challenges.py`
because that suite's generic checks assume an `id,prediction` submission, and
this challenge's submission is a feature matrix.

What is covered, and why:

* the declared `benchmark_expected_score` is what `env.py` returns for the
  package's own benchmark file, and `metrics_detail` carries exactly the keys
  `metrics_schema` declares (the executor enforces equal-mapping at run time);
* the split: disjoint ids, labels aligned, shipped headers are `id` + the kept
  columns (+ target), no dropped column anywhere, 12,500 rows per département,
  target rates within 0.5 point;
* the building grouping connects rows through every building key, and no
  street address survives in a shipped text column;
* every rejection rule of the scorer, each with a message naming the offender
  (plus a UTF-8 BOM accepted, and a fanned-out file stopped before parsing);
* a train subset of exactly 20,000 rows is accepted, 19,999 is not;
* the two warnings: a label copied into a train feature trips the gap flag, and
  unscaled features trip the non-convergence flag;
* peak RSS of `evaluate()` on a dense 100,000 x 250 submission, in a fresh
  process, stays well under the 3Gi engine memory (the default 1024Mi is not
  enough — see config.py);
* the starter notebook runs end to end offline against `data/`, and the
  submission it writes is accepted by the scorer. Its download and submit cells
  are neutralised: a test never touches the live platform.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"
sys.path.insert(0, str(PKG))
from columns import DROPPED, KEPT  # noqa: E402

TARGET = "classe_efg"
GZ = {"method": "gzip", "compresslevel": 1, "mtime": 0}
RSS_LIMIT_BYTES = 3 * 1024 ** 3


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
    spec = importlib.util.spec_from_file_location(f"dpe_env_{tmp.name}", path)
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
    return pd.read_csv(PKG / cfg["benchmark_file"], dtype={"id": str})


@pytest.fixture(scope="module")
def labels():
    return (pd.read_csv(DATA / "labels_train.csv", dtype={"id": str}),
            pd.read_csv(DATA / "labels_test.csv", dtype={"id": str}))


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
    for name in ("config.py", "env.py", "overview.md", "prepare_data.py", "columns.py",
                 "EXPERTISE.md", "DICTIONNAIRE.md", "starter.ipynb"):
        assert (PKG / name).is_file(), f"{name} missing"
    for rel in cfg["public_files"] + cfg["private_files"]:
        assert (DATA / rel).is_file(), f"data/{rel} missing — run prepare_data.py"
    assert (PKG / cfg["benchmark_file"]).is_file()
    assert (DATA / "sample_submission.csv.gz").read_bytes() == \
        (PKG / cfg["benchmark_file"]).read_bytes(), "sample_submission is the benchmark's file"


def test_config_contract(cfg):
    assert cfg["kernel_version"] == "file_v1"
    assert cfg["module_slug"] == "s2-data-preprocessing"
    assert cfg["submission_filename"] == "submission.csv.gz"
    assert cfg["max_upload_size_bytes"] == 200 * 1024 * 1024
    assert cfg["metric"] == "auc" and cfg["metric2"] == "auc_train"
    assert (cfg["deployment_nb_constraint_run"], cfg["deployment_nb_initial_score_run"]) == (1, 1)
    assert cfg["is_public_initial"] is False
    assert isinstance(cfg["benchmark_expected_score"], float), "benchmark score not pinned"
    public = set(cfg["public_files"])
    assert not public & {"labels_train.csv", "labels_test.csv", "env.py"}
    assert not any("teacher" in f for f in public), "the expert pipeline must never be public"
    assert not any(f.startswith("y_test") for f in cfg["private_files"]), (
        "a private file named y_test.csv would be auto-extracted into agent_template")


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
    assert abs(result["score"] - cfg["benchmark_expected_score"]) <= cfg["benchmark_score_tol"], (
        f"env.py scored {result['score']}, config declares {cfg['benchmark_expected_score']}")
    declared = {d["key"] for d in cfg["metrics_schema"] if d["source"] == "env"}
    assert set(result["metrics_detail"]) == declared
    assert result["metrics_detail"]["n_train_rows"] == len(pd.read_csv(DATA / "labels_train.csv"))
    # deterministic: the same file, the same number
    again = env.evaluate(str(PKG / cfg["benchmark_file"]))["agent_results"][0]
    assert again["score"] == result["score"] and again["score2"] == result["score2"]


def test_error_rows_still_carry_the_declared_keys(cfg, env, tmp_path):
    bad = tmp_path / "submission.csv.gz"
    bad.write_text("id,x\n")
    result = env.evaluate(str(bad))["agent_results"][0]
    declared = {d["key"] for d in cfg["metrics_schema"] if d["source"] == "env"}
    assert set(result["metrics_detail"]) == declared


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def test_split_is_disjoint_complete_and_clean(labels):
    train = pd.read_csv(DATA / "train.csv.gz", low_memory=False, dtype={"id": str})
    test = pd.read_csv(DATA / "test.csv.gz", low_memory=False, dtype={"id": str})
    ytr, yte = labels
    assert list(train.columns) == ["id", *KEPT, TARGET]
    assert list(test.columns) == ["id", *KEPT]
    assert not (set(train.columns) | set(test.columns)) & set(DROPPED)
    assert train["id"].is_unique and test["id"].is_unique
    assert not set(train["id"]) & set(test["id"])
    assert train["id"].str.startswith("tr_").all() and test["id"].str.startswith("te_").all()
    assert list(train["id"]) == list(ytr["id"])
    assert list(test["id"]) == list(yte["id"])
    assert (train[TARGET].to_numpy() == ytr[TARGET].to_numpy()).all()
    assert len(train) + len(test) == 100_000
    both = pd.concat([train["code_departement_ban"], test["code_departement_ban"]])
    assert both.value_counts().to_dict() == {d: 12_500 for d in (59, 67, 69, 35, 44, 33, 84, 13)}
    assert abs(ytr[TARGET].mean() - yte[TARGET].mean()) <= 0.005


def test_no_street_address_in_shipped_text():
    """Addresses typed into descriptions are replaced (prepare_data.scrub_addresses)."""
    import prepare_data
    for name in ("train.csv.gz", "test.csv.gz"):
        frame = pd.read_csv(DATA / name, low_memory=False, dtype={"id": str})
        for col in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[col]):
                continue
            left = frame[col].dropna().astype(str).str.contains(prepare_data.ADDRESS_LEFT)
            assert not left.any(), f"{name} {col}: street address in {int(left.sum())} rows"
    # the patterns do catch the addresses the raw data holds
    raw = "hors volume chauffé en sous station du 82 rue de la Bottière.Réseau isolé"
    assert prepare_data.ADDRESS_LEFT.search(raw)
    scrubbed = prepare_data.ADDRESS_IN_TEXT.sub(prepare_data.ADDRESS_REPLACEMENT, raw)
    assert scrubbed == "hors volume chauffé en sous station du [adresse retirée].Réseau isolé"
    assert not prepare_data.ADDRESS_LEFT.search(scrubbed)


def test_building_groups_connect_every_building_key():
    import prepare_data
    rows = pd.DataFrame({
        "numero_dpe": ["d1", "d2", "d3", "d4", "d5", "d6"],
        # d1-d2 one building DPE; d2-d3 one address; d3-d4 one RNB id; d5 alone;
        # d6 shares only an address label
        "numero_dpe_immeuble_associe": ["b1", "b1", None, None, None, None],
        "identifiant_ban": ["a1", "a2", "a2", "a3", "a4", "a5"],
        "adresse_ban": ["x1", "x2", "x3", "x4", "x5", "x1"],
        "id_rnb": [None, None, "r1", "r1", None, None],
    })
    groups = prepare_data.building_groups(rows)
    assert groups.tolist() == ["d1", "d1", "d1", "d1", "d5", "d1"]


# --------------------------------------------------------------------------- #
# rejections
# --------------------------------------------------------------------------- #
def test_rejects_a_plain_csv(env, bench, tmp_path):
    path = tmp_path / "submission.csv.gz"
    bench.head(50).to_csv(path, index=False, compression=None)   # pandas would infer gzip from .gz
    assert_rejected(env.evaluate(str(path))["agent_results"][0], "not gzip")


def test_rejects_a_missing_test_id(env, bench, labels, tmp_path):
    victim = labels[1]["id"].iloc[123]
    assert_rejected(evaluate(env, bench[bench["id"] != victim], tmp_path / "s.csv.gz"), victim)


def test_rejects_an_unknown_id(env, bench, tmp_path):
    extra = bench.head(1).assign(id="te_000000000000")
    assert_rejected(evaluate(env, pd.concat([bench, extra]), tmp_path / "s.csv.gz"),
                    "te_000000000000")


def test_rejects_a_duplicate_id(env, bench, tmp_path):
    dup = bench.iloc[[5000]]
    assert_rejected(evaluate(env, pd.concat([bench, dup]), tmp_path / "s.csv.gz"),
                    dup["id"].iloc[0])


def test_rejects_an_empty_id(env, bench, tmp_path):
    frame = bench.copy()
    frame.loc[7, "id"] = None
    assert_rejected(evaluate(env, frame, tmp_path / "s.csv.gz"), "Empty 'id'", "line 9")


def test_rejects_a_text_column(env, bench, tmp_path):
    frame = bench.copy()
    col = frame.columns[3]
    frame[col] = frame[col].astype(object)
    frame.loc[42, col] = "Gaz naturel"
    assert_rejected(evaluate(env, frame, tmp_path / "s.csv.gz"), repr(col), "Gaz naturel",
                    frame.loc[42, "id"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite(env, bench, tmp_path, bad):
    frame = bench.copy()
    col = frame.columns[2]
    frame.loc[999, col] = bad
    assert_rejected(evaluate(env, frame, tmp_path / "s.csv.gz"), repr(col), frame.loc[999, "id"])


def test_rejects_too_few_train_rows_and_accepts_the_minimum(env, bench, labels, tmp_path):
    train_ids = labels[0]["id"]
    is_test = bench["id"].str.startswith("te_")
    keep = set(train_ids.iloc[:20_000])
    subset = bench[is_test | bench["id"].isin(keep)]
    ok = evaluate(env, subset, tmp_path / "ok.csv.gz")
    assert not ok.get("is_agent_code_error"), ok.get("agent_code_error_message")
    assert ok["metrics_detail"]["n_train_rows"] == 20_000
    keep.discard(train_ids.iloc[0])
    too_few = bench[is_test | bench["id"].isin(keep)]
    assert_rejected(evaluate(env, too_few, tmp_path / "few.csv.gz"), "19999", "20000")


def test_rejects_no_feature_column(env, bench, tmp_path):
    assert_rejected(evaluate(env, bench[["id"]], tmp_path / "s.csv.gz"), "No feature columns")


def test_rejects_more_than_300_features(env, bench, tmp_path):
    frame = bench[["id"]].copy()
    extra = pd.DataFrame(np.zeros((len(frame), 301)), columns=[f"f{i}" for i in range(301)])
    assert_rejected(evaluate(env, pd.concat([frame, extra], axis=1), tmp_path / "s.csv.gz"),
                    "301 feature columns", "300")


def test_rejects_duplicate_column_names(env, bench, tmp_path):
    path = tmp_path / "s.csv.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("id,a,a\n" + "".join(f"{i},1,2\n" for i in bench["id"]))
    assert_rejected(env.evaluate(str(path))["agent_results"][0], "'a' appears twice")


def test_rejects_the_index_written_as_a_column(env, bench, tmp_path):
    path = tmp_path / "s.csv.gz"
    bench.head(100).to_csv(path, compression=GZ)            # index=True
    assert_rejected(env.evaluate(str(path))["agent_results"][0], "index=False")


def test_missing_test_ids_are_named_in_file_order(env, bench, labels, tmp_path):
    first, later = labels[1]["id"].iloc[10], labels[1]["id"].iloc[20_000]
    result = evaluate(env, bench[~bench["id"].isin({first, later})], tmp_path / "s.csv.gz")
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


def test_rejects_far_more_rows_than_ids_before_parsing(env, bench, tmp_path):
    """A fanned-out join must be a message, not an OOM-killed pod."""
    path = tmp_path / "s.csv.gz"
    with gzip.open(path, "wt", compresslevel=1) as fh:
        fh.write("id,x\n")
        row = f"{bench['id'].iloc[0]},0\n"
        for _ in range(12):
            fh.write(row * 10_000)
    assert_rejected(env.evaluate(str(path))["agent_results"][0], "more than 110000 rows",
                    "a join that fanned out")


def test_rejects_missing_id_column_and_header_only(env, bench, tmp_path):
    assert_rejected(evaluate(env, bench.drop(columns=["id"]).head(10), tmp_path / "a.csv.gz"),
                    "Missing the 'id' column")
    assert_rejected(evaluate(env, bench.head(0), tmp_path / "b.csv.gz"), "no rows")


# --------------------------------------------------------------------------- #
# the two warnings
# --------------------------------------------------------------------------- #
def test_label_in_a_train_feature_trips_the_gap_flag(env, bench, labels, tmp_path):
    y = pd.concat([labels[0], labels[1].assign(**{TARGET: 0})]).set_index("id")[TARGET]
    frame = bench.copy()
    frame["leak"] = frame["id"].map(y).astype(float)    # the label on train, 0 on test
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


def test_benchmark_converges_without_warnings(cfg, env):
    result = env.evaluate(str(PKG / cfg["benchmark_file"]))["agent_results"][0]
    assert result["metrics_detail"]["converged"] == 1
    assert "WARNING" not in result["info_message"]


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
def test_peak_rss_on_a_dense_100k_by_250_submission(tmp_path, labels):
    """In a fresh process, the way the executor would call it."""
    ids = pd.concat([labels[0]["id"], labels[1]["id"]], ignore_index=True)
    rng = np.random.default_rng(0)
    dense = pd.DataFrame(rng.standard_normal((len(ids), 250)),
                         columns=[f"f{i:03d}" for i in range(250)])
    dense.insert(0, "id", ids)
    path = tmp_path / "submission.csv.gz"
    dense.to_csv(path, index=False, float_format="%.6f", compression=GZ)
    del dense
    stage = tmp_path / "env"
    stage.mkdir()
    stage_env(stage)
    probe = textwrap.dedent(f"""
        import importlib.util, json, resource, sys, time
        spec = importlib.util.spec_from_file_location("env", {str(stage / 'env.py')!r})
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        env = m.Env(is_evaluation=True)
        t0 = time.perf_counter()
        r = env.evaluate({str(path)!r})["agent_results"][0]
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = peak if sys.platform == "darwin" else peak * 1024     # bytes
        print(json.dumps({{"peak": peak, "seconds": time.perf_counter() - t0,
                          "error": r.get("agent_code_error_message"),
                          "n_features": r["metrics_detail"]["n_features"]}}))
    """)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    report = json.loads(out.stdout.strip().splitlines()[-1])
    size_mb = path.stat().st_size / 1e6
    print(f"\n  dense 100k x 250 ({size_mb:.0f} MB gz): peak RSS {report['peak'] / 1e9:.2f} GB, "
          f"evaluate {report['seconds']:.1f} s")
    assert report["error"] is None
    assert report["n_features"] == 250
    assert report["peak"] < RSS_LIMIT_BYTES * 0.6, (
        f"peak RSS {report['peak'] / 1e9:.2f} GB is not well under 3Gi")


# --------------------------------------------------------------------------- #
# the starter notebook
# --------------------------------------------------------------------------- #
def test_starter_notebook_runs_offline_and_is_accepted(env, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(str(PKG / "starter.ipynb"), as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert 'pip install -q mlarena-sdk' in code and "pip install mlarena\n" not in code
    assert 'files=["submission.csv.gz"]' in code
    assert "mlk_user_..." in code and "mlk_user_4" not in code

    for name in ("train.csv.gz", "test.csv.gz", "sample_submission.csv.gz",
                 "EXPERTISE.md", "DICTIONNAIRE.md"):
        shutil.copy2(DATA / name, tmp_path / name)
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
