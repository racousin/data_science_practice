#!/usr/bin/env python3
"""Tests for the MLP S1 multi-source store-sales challenge package.

    uv run --with pytest --with pandas --with scikit-learn --with openpyxl \\
        pytest courseware/competitions/test_mlp_s1_store_sales.py -q

Needs data/ built first, by `prepare_data.py` (network) and then
`reference_solution.py` (offline). What is covered:

* the benchmark scores exactly `benchmark_expected_score`, clears
  `pass_threshold`, and is what `reference_solution.py` computes;
* the constant submissions score the values quoted in config.py, and the
  "first push" one does not clear the bar;
* every malformed shape the overview says is rejected is rejected, with a
  message, including the malformed file the live checks upload;
* the private files: `y_test.csv` is the target byte for byte, and the target
  covers the Neighborhood_Market items exactly;
* the overview carries the live challenge id from the lockfile (no
  placeholder left) and none of the measured ladder;
* `build_competitions.py` lists the package under its course only and writes
  its own bar at attach.

Kept apart from test_challenges.py, whose helpers assume `id,prediction`
submissions and the X/y file roles of python-ai-engineering.
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PKG = "mlp-s1-store-sales"
PKG_DIR = HERE / PKG
DATA = PKG_DIR / "data"
TOOLS = HERE.parent / "tools"

# Measured with env.py on the shipped data (reference_solution.py prints the
# same MAEs). The constant of "Your first push" is CityMart's mean.
MINIMAL_SCORE = -26.369329
FOUR_STORE_MEAN_SCORE = -23.175889
N_ITEMS = 409


def load_config() -> dict:
    ns: dict = {}
    exec((PKG_DIR / "config.py").read_text(), ns)
    return ns["CONFIG"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """env.py + its private files staged the way the platform lays out the env
    folder, so a file env.py reads but config.py does not list fails here."""
    stage = tmp_path_factory.mktemp("envstage")
    shutil.copy(PKG_DIR / "env.py", stage / "env.py")
    for rel in load_config()["private_files"]:
        shutil.copy(DATA / rel, stage / rel)
    return load_module("env_mlp_s1_store_sales", stage / "env.py").Env(
        is_evaluation=True)


@pytest.fixture(scope="module")
def truth():
    import pandas as pd
    return pd.read_csv(DATA / "neighborhood_market_target.csv")


def result(env, path) -> dict:
    return env.evaluate(str(path))["agent_results"][0]


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# --------------------------------------------------------------------------- #
# package integrity
# --------------------------------------------------------------------------- #
def test_package_files_present(cfg):
    for name in ("config.py", "env.py", "overview.md", "prepare_data.py",
                 "reference_solution.py"):
        assert (PKG_DIR / name).is_file(), f"{name} missing"
    for rel in cfg["public_files"] + cfg["private_files"]:
        assert (DATA / rel).is_file(), f"data/{rel} missing — run prepare_data.py"
    for rel in (cfg["benchmark_file"], "data/minimal_submission.csv",
                "data/malformed_submission.csv"):
        assert (PKG_DIR / rel).is_file(), f"{rel} missing — run reference_solution.py"


def test_config_declares_the_contract(cfg):
    assert cfg["name"] == "MLP S1 — Multi-Source Store Sales"
    assert cfg["kernel_version"] == "file_v1"
    assert cfg["module_slug"] == "s1-data-collection"
    assert (cfg["metric"], cfg["metric2"]) == ("neg_mae", "rmse")
    assert cfg["pass_threshold"] == -20.0
    assert cfg["benchmark_score_tol"] == 1e-6
    assert cfg["is_public_initial"] is False
    assert "submission_filename" not in cfg  # the platform's submission.csv


def test_y_test_is_the_target_byte_for_byte():
    """The platform needs y_test.csv to start the challenge and to check
    uploads; env.py reads the other name. They must never disagree."""
    assert (DATA / "y_test.csv").read_bytes() == \
        (DATA / "neighborhood_market_target.csv").read_bytes()


def test_target_covers_the_test_items_exactly(truth):
    import pandas as pd
    test = pd.read_csv(DATA / "Neighborhood_Market_data.csv")
    assert len(truth) == N_ITEMS and truth["item_code"].is_unique
    assert set(truth["item_code"]) == set(test["item_code"])
    assert "quantity_sold" not in test.columns


def test_train_and_test_items_are_disjoint():
    ref = load_module("mlp_reference_solution",
                      PKG_DIR / "reference_solution.py")
    train, test = ref.read_files()
    assert set(train.index) & set(test.index) == set()


def test_metrics_schema(cfg, env):
    declared = {d["key"] for d in cfg["metrics_schema"] if d["source"] == "env"}
    got = result(env, PKG_DIR / cfg["benchmark_file"])
    assert set(got["metrics_detail"]) == declared
    sys.path.insert(0, str(HERE.parent.parent.parent.parent / "modelmanager"))
    try:
        from modelmanager.metrics_schema import validate_metrics_schema
    except ImportError:
        pytest.skip("modelmanager not importable from here")
    assert validate_metrics_schema(cfg["metrics_schema"]) == []


# --------------------------------------------------------------------------- #
# scores
# --------------------------------------------------------------------------- #
def test_benchmark_hits_declared_score(cfg, env):
    got = result(env, PKG_DIR / cfg["benchmark_file"])
    assert not got.get("is_agent_code_error"), got.get("agent_code_error_message")
    assert abs(got["score"] - cfg["benchmark_expected_score"]) <= \
        cfg["benchmark_score_tol"]
    assert got["metrics_detail"]["n_rows"] == N_ITEMS


def test_benchmark_clears_the_pass_threshold(cfg, env):
    got = result(env, PKG_DIR / cfg["benchmark_file"])
    assert got["score"] >= cfg["pass_threshold"]


def test_benchmark_is_what_the_reference_solution_computes():
    import pandas as pd
    ref = load_module("mlp_reference_solution",
                      PKG_DIR / "reference_solution.py")
    train, test = ref.read_files()
    train, test = ref.add_sources(train), ref.add_sources(test)
    cv, pred = ref.fit(train, test,
                       ref.FILE_COLS + ref.API_COLS + ref.SCRAPED_COLS)
    assert abs(cv - ref.REFERENCE_CV_MAE) <= 1e-6
    shipped = pd.read_csv(DATA / "benchmark_submission.csv")
    assert list(shipped["item_code"]) == list(test.index)
    assert max(abs(shipped["quantity_sold"] - pred)) < 1e-9


def test_minimal_submission_scores_the_measured_value(cfg, env, truth):
    import pandas as pd
    got = result(env, DATA / "minimal_submission.csv")
    assert not got.get("is_agent_code_error"), got.get("agent_code_error_message")
    assert got["score"] == MINIMAL_SCORE
    assert got["score"] < cfg["pass_threshold"], "the first push should not pass"
    city_mean = pd.read_csv(DATA / "CityMart_data.csv")["quantity_sold"].mean()
    expected = -(truth["quantity_sold"] - city_mean).abs().mean()
    assert abs(got["score"] - expected) < 1e-6


def test_four_store_mean_scores_the_measured_value(cfg, env, truth, tmp_path):
    ref = load_module("mlp_reference_solution",
                      PKG_DIR / "reference_solution.py")
    train, _ = ref.read_files()
    const = float(train["quantity_sold"].mean())
    rows = "".join(f"{i},{const!r}\n" for i in truth["item_code"])
    got = result(env, write(tmp_path / "s.csv", "item_code,quantity_sold\n" + rows))
    assert got["score"] == FOUR_STORE_MEAN_SCORE
    assert got["score"] < cfg["pass_threshold"]


def test_neg_mae_and_rmse_are_what_they_say(env, truth, tmp_path):
    const = 180.0
    rows = "".join(f"{i},{const}\n" for i in truth["item_code"])
    got = result(env, write(tmp_path / "s.csv", "item_code,quantity_sold\n" + rows))
    err = truth["quantity_sold"] - const
    assert abs(got["score"] + err.abs().mean()) < 1e-6
    assert abs(got["metrics_detail"]["rmse"] - math.sqrt((err ** 2).mean())) < 1e-6
    assert got["score"] == got["metrics_detail"]["neg_mae"]
    assert got["score2"] == got["metrics_detail"]["rmse"]


def test_order_extra_columns_and_negatives_are_scored(env, truth, tmp_path):
    codes = list(truth["item_code"])[::-1]
    rows = "".join(f"x,{i},-1.5\n" for i in codes)
    got = result(env, write(tmp_path / "s.csv",
                            "note,item_code,quantity_sold\n" + rows))
    assert not got.get("is_agent_code_error"), got.get("agent_code_error_message")
    assert abs(got["score"] + (truth["quantity_sold"] + 1.5).abs().mean()) < 1e-6


# --------------------------------------------------------------------------- #
# rejections
# --------------------------------------------------------------------------- #
def _cases(truth) -> dict[str, str]:
    ids = list(truth["item_code"])
    body = [f"{i},200" for i in ids]
    header = "item_code,quantity_sold\n"

    def csv(lines, head=header):
        return head + "".join(f"{line}\n" for line in lines)

    return {
        "empty file": "",
        "header only": header,
        "missing quantity_sold column": csv(ids, head="item_code\n"),
        "wrong column names": csv(body, head="id,prediction\n"),
        "missing ids": csv(body[:-3]),
        "unknown id": csv(body + ["P9999,200"]),
        "a training-store id": csv(body + ["P0001,200"]),
        "duplicate id": csv(body + [body[0]]),
        "empty id": csv([",200"] + body[1:]),
        "non-numeric value": csv([f"{ids[0]},abc"] + body[1:]),
        "empty value": csv([f"{ids[0]},"] + body[1:]),
        "NaN": csv([f"{ids[0]},nan"] + body[1:]),
        "infinity": csv([f"{ids[0]},inf"] + body[1:]),
    }


CASE_LABELS = [
    "empty file", "header only", "missing quantity_sold column",
    "wrong column names", "missing ids", "unknown id", "a training-store id",
    "duplicate id", "empty id", "non-numeric value", "empty value", "NaN",
    "infinity"]


@pytest.mark.parametrize("label", CASE_LABELS)
def test_malformed_submission_is_rejected(label, env, truth, tmp_path):
    got = result(env, write(tmp_path / "s.csv", _cases(truth)[label]))
    assert got.get("is_agent_code_error"), f"accepted: {label}"
    assert got["score"] == 0.0
    assert got["agent_code_error_message"], f"no message for {label}"


def test_every_case_is_parametrized(truth):
    assert sorted(_cases(truth)) == sorted(CASE_LABELS)


def test_shipped_malformed_file_is_rejected_for_missing_items(env):
    got = result(env, DATA / "malformed_submission.csv")
    assert got.get("is_agent_code_error")
    assert got["agent_code_error_message"].startswith(
        "submission.csv is missing 204 of 409")


# --------------------------------------------------------------------------- #
# overview
# --------------------------------------------------------------------------- #
def test_overview_placeholders_and_links():
    text = (PKG_DIR / "overview.md").read_text()
    # The page carries the live id (competitions/.mlarena-state.json), not a
    # placeholder: students copy these snippets as they are.
    state = json.loads((PKG_DIR.parent / ".mlarena-state.json").read_text())
    cid = state["https://ml-arena.com"]["competitions"]["mlp-s1-store-sales"]["id"]
    assert "CHALLENGE_ID" not in text
    assert f'client.submit(challenge_id={cid}, files=["submission.csv"])' in text
    assert f'client.download_dataset({cid}, ".")' in text
    assert f"client.leaderboard({cid})" in text
    assert ("colab.research.google.com/github/racousin/data_science_practice/"
            "blob/main/website/public/modules/ms2a-machine-learning-practice/"
            "challenges/mlp-s1-store-sales.ipynb") in text
    assert "https://www.raphaelcousin.com/module4/api-doc" in text
    for table in ("retail.stores", "retail.data_dictionary"):
        assert table in text


def test_overview_prints_no_ladder_and_no_secret(cfg):
    text = (PKG_DIR / "overview.md").read_text()
    for number in ("2.11", "2.15", "3.51", "17.05", "19.89", "23.18",
                   "26.37", str(cfg["benchmark_expected_score"])):
        assert number not in text, f"overview prints the ladder value {number}"
    for secret_marker in ("postgresql+psycopg://", "sbx_", "mlk_creator_",
                          "mlk_teacher_"):
        assert secret_marker not in text


def test_overview_code_lines_fit_76_characters():
    in_code, long_lines = False, []
    for line in (PKG_DIR / "overview.md").read_text().splitlines():
        if line.startswith("```"):
            in_code = not in_code
        elif in_code and len(line) > 76:
            long_lines.append(line)
    assert long_lines == []


# --------------------------------------------------------------------------- #
# build_competitions.py
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bc():
    return load_module("build_competitions_under_test",
                       TOOLS / "build_competitions.py")


def test_package_belongs_to_its_course_only(bc):
    assert PKG in bc.PACKAGES_BY_COURSE["ms2a-machine-learning-practice"]
    assert PKG not in bc.PACKAGES_BY_COURSE["python-ai-engineering"]
    assert bc.select_packages("ms2a-machine-learning-practice", [PKG]) == [PKG]
    with pytest.raises(SystemExit):
        bc.select_packages("python-ai-engineering", [PKG])


def test_attach_writes_the_declared_bar(bc):
    ours = bc.load_config(PKG)
    assert bc.pass_threshold(ours) == -20.0
    for pkg in bc.PACKAGES_BY_COURSE["python-ai-engineering"]:
        if not (bc.PACKAGES_DIR / pkg / "config.py").is_file():
            continue
        theirs = bc.load_config(pkg)
        assert "pass_threshold" not in theirs
        assert bc.pass_threshold(theirs) == theirs["benchmark_expected_score"]


def test_a_bar_above_the_benchmark_is_refused(bc):
    broken = dict(bc.load_config(PKG), pass_threshold=0.0)
    with pytest.raises(SystemExit, match="above the benchmark"):
        bc.validate_config(broken)
