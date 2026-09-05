#!/usr/bin/env python3
"""Tests for the Session 2, 3 and 4 challenge packages and their notebooks.

    uv run --with pytest --with pandas --with scikit-learn --with seaborn \
        --with torch --with nbformat --with nbclient --with ipykernel \
        pytest courseware/competitions/test_challenges.py -v

`torch` is needed from Session 4 on; without it the Session 4 notebook tests
skip and everything else still runs.

What is covered, and why each one exists:

* the declared `benchmark_expected_score` is what `env.py` actually returns for
  the package's own benchmark file — the number quoted in the overview and in
  `config.py` is therefore reachable, not aspirational;
* the scorer rejects every malformed submission shape it claims to reject,
  rather than silently imputing;
* the train/test split is disjoint and the private labels cover the test ids
  exactly;
* R2 = 0 really is the predict-the-mean model and always-0 really is F1 = 0 —
  the two claims the overviews lead with;
* each **worked notebook runs end to end** and the submission it writes scores
  what it should — the real contract, since that notebook is what a student
  runs. Sessions 2 and 3 pin it to the declared benchmark exactly; Session 4
  cannot, because its notebooks train a torch model and a float pinned to 1e-6
  would not survive a different BLAS, so those assert
  `notebook_expected_min_score` instead;
* each **guided notebook contains no code**, which is the point of it;
* the pandas/seaborn pre-flight notebook and the two Session 4 warm-ups run
  with **no credentials at all** -- they are what a student opens before they
  have an account;
* every notebook is in sync with the builder that generates them.

Notebook execution needs a real key (the notebook downloads its own data):

    export MLARENA_USER_API_KEY=mlk_user_...

Without it the notebook-execution tests skip; everything else still runs.
The submit cell is neutralised during the test — a test run must not put rows on
a live leaderboard. `--submit-for-real` opts into it.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NOTEBOOKS = REPO / "website" / "public" / "modules" / "python-ai-engineering" / "challenges"

PACKAGES = ["s2-bike-demand", "s2-bank-marketing",
            "s3-diabetes-progression", "s3-credit-risk",
            "s4-california-housing", "s4-forest-cover"]
REGRESSION = {"s2-bike-demand", "s3-diabetes-progression", "s4-california-housing"}
# Multi-class, so the binary-classifier assertions (F1 = 0 for always-0, a
# `prediction` of 0/1) do not apply to it.
MULTICLASS = {"s4-forest-cover"}

# Notebooks a student can open before they have an ML-Arena account. They must
# not mention the SDK or a key.
CREDENTIAL_FREE = ["aie-s0-pandas-seaborn.ipynb",
                   "aie-s4-optimization-warmup.ipynb",
                   "aie-s4-cpu-gpu-benchmark.ipynb"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_config(pkg: str) -> dict:
    ns: dict = {}
    exec((HERE / pkg / "config.py").read_text(), ns)
    return ns["CONFIG"]


def load_env(pkg: str, tmp: Path):
    """Stage env.py + its private files the way the platform lays out the env
    folder, then import it. Catches 'env.py reads a file that was never listed
    in private_files'."""
    cfg = load_config(pkg)
    shutil.copy(HERE / pkg / "env.py", tmp / "env.py")
    for rel in cfg.get("private_files", []):
        shutil.copy(HERE / pkg / "data" / rel, tmp / rel)
    spec = importlib.util.spec_from_file_location(f"env_{pkg.replace('-', '_')}",
                                                  tmp / "env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Env(is_evaluation=True)


def score(pkg: str, submission: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        env = load_env(pkg, Path(td))
        return env.evaluate(str(submission))["agent_results"][0]


def write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("id,prediction\n" + "\n".join(rows) + "\n")
    return path


@pytest.fixture(scope="module")
def user_key():
    key = os.environ.get("MLARENA_USER_API_KEY")
    if not key:
        pytest.skip("MLARENA_USER_API_KEY not set — notebook execution skipped")
    return key


# --------------------------------------------------------------------------- #
# package integrity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_files_present(pkg):
    for name in ("config.py", "env.py", "overview.md", "prepare_data.py"):
        assert (HERE / pkg / name).is_file(), f"{pkg}/{name} missing"
    cfg = load_config(pkg)
    for rel in cfg["public_files"] + cfg["private_files"]:
        assert (HERE / pkg / "data" / rel).is_file(), (
            f"{pkg}/data/{rel} missing — run {pkg}/prepare_data.py")


@pytest.mark.parametrize("pkg", PACKAGES)
def test_metrics_schema_is_valid(pkg):
    """The same validator the backend runs at authoring time."""
    sys.path.insert(0, str(REPO.parent.parent / "modelmanager"))
    try:
        from modelmanager.metrics_schema import validate_metrics_schema
    except ImportError:
        pytest.skip("modelmanager not importable from here")
    cfg = load_config(pkg)
    assert validate_metrics_schema(cfg["metrics_schema"]) == []


@pytest.mark.parametrize("pkg", PACKAGES)
def test_env_metrics_detail_matches_schema(pkg):
    """The executor enforces equal-mapping between metrics_detail and the
    schema's env-sourced keys; a mismatch fails the run, not the upload."""
    cfg = load_config(pkg)
    declared = {d["key"] for d in cfg["metrics_schema"] if d["source"] == "env"}
    result = score(pkg, HERE / pkg / cfg["benchmark_file"])
    assert set(result["metrics_detail"]) == declared


@pytest.mark.parametrize("pkg", PACKAGES)
def test_benchmark_hits_declared_score(pkg):
    """The number in config.py and in the overview is the number env.py gives."""
    cfg = load_config(pkg)
    result = score(pkg, HERE / pkg / cfg["benchmark_file"])
    assert abs(result["score"] - cfg["benchmark_expected_score"]) <= cfg["benchmark_score_tol"], (
        f"{pkg}: env.py scored {result['score']}, config declares "
        f"{cfg['benchmark_expected_score']}")
    assert not result.get("is_agent_code_error")


@pytest.mark.parametrize("pkg", PACKAGES)
def test_split_is_disjoint_and_complete(pkg):
    import pandas as pd
    d = HERE / pkg / "data"
    tr = pd.read_csv(d / "X_train.csv")
    te = pd.read_csv(d / "X_test.csv")
    ytr = pd.read_csv(d / "y_train.csv")
    yte = pd.read_csv(d / "y_test.csv")

    assert set(tr["id"]) & set(te["id"]) == set(), "train and test ids overlap"
    assert list(tr["id"]) == list(ytr["id"]), "X_train / y_train ids misaligned"
    assert list(te["id"]) == list(yte["id"]), "X_test / y_test ids misaligned"
    assert tr["id"].is_unique and te["id"].is_unique
    assert list(tr.columns) == list(te.columns), "train and test schemas differ"
    # The held-back labels must not be reachable from anything published.
    assert "prediction" not in tr.columns and "prediction" not in te.columns


@pytest.mark.parametrize("pkg", PACKAGES)
def test_rejects_malformed_submissions(pkg, tmp_path):
    import pandas as pd
    cfg = load_config(pkg)
    good = pd.read_csv(HERE / pkg / cfg["benchmark_file"])
    ids = good["id"].tolist()
    val = str(good["prediction"].iloc[0])

    cases = {
        "missing a row": [f"{i},{val}" for i in ids[:-1]],
        "an unknown id": [f"{i},{val}" for i in ids] + [f"zz_99999,{val}"],
        "a duplicate id": [f"{i},{val}" for i in ids] + [f"{ids[0]},{val}"],
        "an empty id": [f",{val}"] + [f"{i},{val}" for i in ids[1:]],
        "a header only": [],
    }
    if pkg in MULTICLASS:                           # 7 classes, coded 1-7
        cases["a probability"] = [f"{ids[0]},0.73"] + [f"{i},{val}" for i in ids[1:]]
        cases["an out-of-range class"] = [f"{ids[0]},9"] + [f"{i},{val}" for i in ids[1:]]
        # The whole column shifted down by one — the silent failure the
        # 1-based coding invites. It must be caught at upload, not scored.
        cases["a 0-based label"] = [f"{ids[0]},0"] + [f"{i},{val}" for i in ids[1:]]
    elif cfg["metric"] == "f1":                     # binary: classes only
        cases["a probability"] = [f"{ids[0]},0.73"] + [f"{i},{val}" for i in ids[1:]]
        cases["an out-of-range class"] = [f"{ids[0]},2"] + [f"{i},{val}" for i in ids[1:]]
    else:                                           # regressor: finite floats
        cases["a non-numeric value"] = [f"{ids[0]},abc"] + [f"{i},{val}" for i in ids[1:]]
        cases["a NaN"] = [f"{ids[0]},nan"] + [f"{i},{val}" for i in ids[1:]]

    for label, rows in cases.items():
        path = write_csv(tmp_path / "sub.csv", rows)
        result = score(pkg, path)
        assert result.get("is_agent_code_error"), f"{pkg}: accepted {label}"
        assert result["score"] == 0.0
        assert result["agent_code_error_message"], f"{pkg}: no message for {label}"


def test_regressor_accepts_negative_predictions(tmp_path):
    """Documented behaviour: counts cannot be negative, but a linear model
    produces them and the scorer must score rather than reject — otherwise the
    baseline notebook's own submission would be refused."""
    import pandas as pd
    cfg = load_config("s2-bike-demand")
    good = pd.read_csv(HERE / "s2-bike-demand" / cfg["benchmark_file"])
    assert (good["prediction"] < 0).any(), "fixture no longer exercises the case"
    result = score("s2-bike-demand", HERE / "s2-bike-demand" / cfg["benchmark_file"])
    assert not result.get("is_agent_code_error")
    assert result["metrics_detail"]["n_negative"] > 0


@pytest.mark.parametrize("pkg", sorted(REGRESSION))
def test_r2_zero_is_the_mean_model(pkg, tmp_path):
    """The overview tells students R2 = 0 is exactly predict-the-training-mean.
    That claim should be true of the scorer, not just of the textbook."""
    import pandas as pd
    d = HERE / pkg / "data"
    yte = pd.read_csv(d / "y_test.csv")
    mean_of_test = yte["prediction"].mean()
    path = write_csv(tmp_path / "sub.csv",
                     [f"{i},{mean_of_test}" for i in yte["id"]])
    result = score(pkg, path)
    assert abs(result["score"]) < 1e-9, "constant-mean predictor is not R2 = 0"


@pytest.mark.parametrize("pkg", ["s2-bank-marketing", "s3-credit-risk"])
def test_always_zero_is_f1_zero(pkg, tmp_path):
    """Likewise for the classifiers' headline claim."""
    import pandas as pd
    d = HERE / pkg / "data"
    yte = pd.read_csv(d / "y_test.csv")
    path = write_csv(tmp_path / "sub.csv", [f"{i},0" for i in yte["id"]])
    result = score(pkg, path)
    assert result["metrics_detail"]["f1"] == 0.0
    assert abs(result["metrics_detail"]["accuracy"] - (1 - yte["prediction"].mean())) < 1e-6


# --------------------------------------------------------------------------- #
# notebooks
# --------------------------------------------------------------------------- #
def test_notebooks_match_their_builder(tmp_path):
    """The .ipynb files are committed for Colab but generated from
    build_notebooks.py. Drift means someone edited the wrong one."""
    before = {p.name: p.read_text() for p in NOTEBOOKS.glob("*.ipynb")}
    subprocess.run([sys.executable, str(REPO / "courseware" / "tools" / "build_notebooks.py")],
                   check=True, capture_output=True)
    after = {p.name: p.read_text() for p in NOTEBOOKS.glob("*.ipynb")}
    assert before == after, "notebooks are out of sync with build_notebooks.py"


def test_notebook_code_cells_compile():
    """Catches generator bugs that only surface when a cell is executed --
    an escaped newline landing inside an f-string, say. Execution catches these
    too, but only with an API key; this runs everywhere."""
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        nb = json.loads(path.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if not src.strip():
                continue
            # `!pip ...` is IPython syntax, not Python.
            src = "\n".join(("#" + ln if ln.lstrip().startswith("!") else ln)
                             for ln in src.splitlines())
            try:
                compile(src, f"{path.name}:cell{i}", "exec")
            except SyntaxError as exc:
                raise AssertionError(f"{path.name} cell {i} does not compile: {exc}")


@pytest.mark.parametrize("notebook", ["aie-s2-bank-marketing.ipynb",
                                     "aie-s3-credit-risk.ipynb",
                                     "aie-s4-forest-cover.ipynb"])
def test_guide_notebook_has_no_code(notebook):
    """The guided notebooks guide in English and ship empty cells on purpose."""
    nb = json.loads((NOTEBOOKS / notebook).read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "guide notebook has no cells for the student to fill"
    for i, cell in enumerate(code_cells):
        body = "".join(cell["source"]).strip()
        assert body == "", f"guide notebook code cell {i} is not empty: {body[:60]!r}"


@pytest.mark.parametrize("notebook", ["aie-s2-bike-demand.ipynb",
                                     "aie-s3-diabetes-progression.ipynb",
                                     "aie-s4-california-housing.ipynb"])
def test_worked_notebook_has_no_leftover_placeholder_key(notebook):
    nb = json.loads((NOTEBOOKS / notebook).read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "mlk_user_..." in src, "the placeholder students replace is gone"
    assert "mlk_user_4" not in src and "mlk_creator" not in src and "mlk_teacher" not in src, \
        "a real API key leaked into the committed notebook"


@pytest.mark.parametrize("notebook", CREDENTIAL_FREE)
def test_credential_free_notebooks_need_no_credentials(notebook):
    """The pre-flight and the two Session 4 warm-ups are what a student runs
    before they have an ML-Arena account. None may reference the SDK or a key."""
    nb = json.loads((NOTEBOOKS / notebook).read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for forbidden in ("mlarena", "API_KEY", "mlk_", "download_dataset", "client."):
        assert forbidden not in src, (
            f"{notebook} references {forbidden!r}; it must run standalone")


def test_warmup_notebooks_do_not_install_from_git():
    """The Session 4 warm-ups are readapted from a workshop whose notebooks
    open with `pip install git+https://github.com/...`. That dependency is the
    thing the readaptation removed — the helpers are inlined instead — and it
    must not creep back."""
    for notebook in ("aie-s4-optimization-warmup.ipynb",
                     "aie-s4-cpu-gpu-benchmark.ipynb"):
        nb = json.loads((NOTEBOOKS / notebook).read_text())
        src = "".join("".join(c["source"]) for c in nb["cells"])
        assert "git+http" not in src, f"{notebook} installs from a git URL"
        assert "aiforscience" not in src, (
            f"{notebook} imports the workshop package; inline the helper instead")


@pytest.mark.parametrize("notebook", ["aie-s4-optimization-warmup.ipynb",
                                      "aie-s4-cpu-gpu-benchmark.ipynb"])
def test_warmup_notebooks_run(notebook, tmp_path):
    """Executed with no key and no network. The CPU/GPU one must also survive
    having no GPU, which is the case on every machine that runs this suite."""
    pytest.importorskip("torch")
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(str(NOTEBOOKS / notebook), as_version=4)
    nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()


def test_preflight_notebook_runs_standalone(tmp_path):
    """Executed with no key and no network beyond seaborn's bundled dataset."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(str(NOTEBOOKS / "aie-s0-pandas-seaborn.ipynb"), as_version=4)
    nbclient.NotebookClient(nb, timeout=1200, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()
    assert (tmp_path / "submission.csv").is_file(), (
        "the pre-flight notebook should end by writing a submission.csv, since "
        "that is the artefact every later challenge asks for")


@pytest.mark.parametrize("pkg,notebook,expect_key,tol", [
    ("s2-bike-demand", "aie-s2-bike-demand.ipynb", "r2", 1e-6),
    ("s3-diabetes-progression", "aie-s3-diabetes-progression.ipynb", "r2", 1e-6),
    ("s4-california-housing", "aie-s4-california-housing.ipynb", "r2", 1e-6),
])
def test_worked_notebook_runs_and_scores_the_baseline(pkg, notebook, expect_key, tol,
                                                      user_key, tmp_path):
    """Execute the student-facing notebook for real — download, EDA, fit,
    predict, write submission.csv — then score that file with the package's own
    env.py. This is the test that the quoted baseline is what the notebook
    actually produces.

    Sessions 2 and 3 assert equality with the declared benchmark, because their
    notebooks fit deterministic sklearn models. Session 4's trains a network, so
    it asserts `notebook_expected_min_score` — a floor with real headroom
    (measured 0.776 against a floor of 0.72) rather than a float that would
    break on a machine with a different BLAS."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    if pkg.startswith("s4-"):
        pytest.importorskip("torch")

    nb = nbformat.read(str(NOTEBOOKS / notebook), as_version=4)
    submit_for_real = os.environ.get("MLARENA_SUBMIT_FOR_REAL") == "1"

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        src = src.replace('API_KEY = "mlk_user_..."   # <- paste yours here',
                          f'API_KEY = {user_key!r}')
        if src.strip().startswith("!pip install"):
            src = "# pip install neutralised: the test env already has the SDK"
        if not submit_for_real and ("client.submit(" in src or "client.leaderboard(" in src):
            src = "# submit/leaderboard neutralised for the test run"
        cell.source = src

    client = nbclient.NotebookClient(nb, timeout=1200, kernel_name="python3",
                                     resources={"metadata": {"path": str(tmp_path)}})
    client.execute()

    produced = tmp_path / "submission.csv"
    assert produced.is_file(), "the notebook did not write submission.csv"

    cfg = load_config(pkg)
    result = score(pkg, produced)
    assert not result.get("is_agent_code_error"), result.get("agent_code_error_message")

    floor = cfg.get("notebook_expected_min_score")
    if floor is not None:
        assert result["score"] >= floor, (
            f"notebook scored {result['score']}, below the floor {floor} the "
            f"package declares. The MLP is not learning — check the scaler.")
        assert result["score"] > cfg["benchmark_expected_score"], (
            f"notebook scored {result['score']}, which does not beat the "
            f"challenge's own linear benchmark "
            f"({cfg['benchmark_expected_score']}). That is the entire claim of "
            f"the session.")
    else:
        assert abs(result["score"] - cfg["benchmark_expected_score"]) <= tol, (
            f"notebook scored {result['score']}, the challenge advertises "
            f"{cfg['benchmark_expected_score']}")
