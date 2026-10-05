#!/usr/bin/env python3
"""Local tests for the Pet Segmentation challenge — the format, the metric, the scorer.

    cd courseware/competitions
    uv run --with numpy --with pytest pytest mlp-s7-pet-segmentation/test_pet_segmentation.py -v

Needs `data/` (run prepare_data.py, then reference_solution.py for the
benchmark test); the format and metric tests on synthetic masks need nothing.

What is covered, and why:

* RLE round-trips on random masks, an empty and a full mask, and the first
  and last pixel — the off-by-one edges of a 1-based format;
* the metric's fixed points: the truth resubmitted scores 1, all background
  scores IoU(background) / 2, and border pixels change nothing whatever the
  prediction says there;
* every rejection of the format, each through env.py as a
  ParticipantSubmissionError naming the problem;
* the declared benchmark_expected_score is what env.py returns for the
  benchmark file, and metrics_detail carries exactly the schema keys.
"""
import csv
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"
sys.path.insert(0, str(PKG))

import seg_metric as sm  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = load("config", PKG / "config.py").CONFIG


class ParticipantSubmissionError(Exception):
    pass


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not (DATA / "y_test.csv").exists():
        pytest.skip("data/ not built: run prepare_data.py")
    stage = tmp_path_factory.mktemp("env")
    shutil.copy(PKG / "env.py", stage)
    for rel in CONFIG["private_files"]:
        src = DATA / rel if (DATA / rel).exists() else PKG / rel
        shutil.copy(src, stage)
    sys.path.insert(0, str(stage))
    module = load("pet_env", stage / "env.py")
    module.Env.ParticipantSubmissionError = ParticipantSubmissionError  # the worker injects it
    yield module.Env()
    sys.path.remove(str(stage))


def write_csv(path, rows, header=("image_id", "mask_rle")):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


# ------------------------------------------------------------------ format

@pytest.mark.parametrize("seed", range(5))
def test_rle_round_trip(seed):
    mask = np.random.RandomState(seed).rand(sm.SIZE, sm.SIZE) > 0.6
    assert (sm.rle_decode(sm.rle_encode(mask)) == mask).all()


def test_rle_touching_runs_accepted():
    """An encoder that does not merge adjacent runs still writes a valid mask."""
    assert (sm.rle_decode("10 5 15 3") == sm.rle_decode("10 8")).all()


def test_rle_edges():
    empty = np.zeros((sm.SIZE, sm.SIZE), bool)
    assert sm.rle_encode(empty) == ""
    assert not sm.rle_decode("").any()
    full = ~empty
    assert sm.rle_encode(full) == f"1 {sm.SIZE * sm.SIZE}"
    corners = empty.copy()
    corners[0, 0] = corners[-1, -1] = True
    assert sm.rle_encode(corners) == f"1 1 {sm.SIZE * sm.SIZE} 1"
    labels = np.full((sm.SIZE, sm.SIZE), sm.BORDER, np.uint8)   # labels: only 1 counts as pet
    assert sm.rle_encode(labels) == ""


@pytest.mark.parametrize("rle, needle", [
    ("1", "odd number"),
    ("1 a", "integers"),
    ("5 0", "length 0"),
    ("10 5 12 3", "overlaps"),
    (f"{sm.SIZE * sm.SIZE} 2", "past pixel"),
])
def test_rle_rejects(rle, needle):
    with pytest.raises(ValueError, match=needle):
        sm.rle_decode(rle)


# ------------------------------------------------------------------ metric

def test_metric_fixed_points():
    rng = np.random.RandomState(0)
    true = (rng.rand(4, sm.SIZE, sm.SIZE) > 0.5).astype(np.uint8)
    true[:, :, :8] = sm.BORDER
    perfect = sm.score_masks(true, true == sm.PET)
    assert perfect["miou"] == 1 and perfect["pixel_accuracy"] == 1
    nothing = sm.score_masks(true, np.zeros(true.shape, bool))
    assert nothing["iou_pet"] == 0
    assert nothing["miou"] == pytest.approx(nothing["iou_background"] / 2)
    # whatever is predicted on the border changes nothing
    noisy = (true == sm.PET) | ((true == sm.BORDER) & (rng.rand(*true.shape) > 0.5))
    assert sm.score_masks(true, noisy) == perfect


def test_metric_is_pooled_over_images():
    """IoU over all pixels together, not a mean of per-image IoUs."""
    true = np.zeros((2, sm.SIZE, sm.SIZE), np.uint8)
    true[0, :64] = sm.PET            # a big pet, found
    true[1, :1, :4] = sm.PET         # 4 pixels, missed
    pred = true == sm.PET
    pred[1] = False
    tp, fn = 64 * sm.SIZE, 4
    assert sm.score_masks(true, pred)["iou_pet"] == pytest.approx(tp / (tp + fn))


# ------------------------------------------------------------------ env.py

def test_ground_truth_shape(env):
    assert len(env.ground_truth) == 1500
    m = next(iter(env.ground_truth.values()))
    assert m.shape == (sm.SIZE, sm.SIZE) and set(np.unique(m)) <= {0, 1, 2}


def test_truth_resubmitted_scores_one(env, tmp_path):
    rows = [(i, sm.rle_encode(m == sm.PET)) for i, m in env.ground_truth.items()]
    out = env.evaluate(str(write_csv(tmp_path / "s.csv", rows)))["agent_results"][0]
    assert out["score"] == 1.0


def test_sample_submission(env):
    out = env.evaluate(str(DATA / "sample_submission.csv"))["agent_results"][0]
    assert out["metrics_detail"]["iou_pet"] == 0
    assert 0.3 < out["score"] < 0.36


def _rows(env):
    return [(i, "") for i in env.ground_truth]


@pytest.mark.parametrize("mutate, needle", [
    (lambda rows: rows[1:], "missing"),
    (lambda rows: rows + [("te_nope", "")], "unknown"),
    (lambda rows: rows + [rows[0]], "twice"),
    (lambda rows: [(rows[0][0], "3 x")] + rows[1:], "line 2"),
    (lambda rows: [(rows[0][0], "1 999999")] + rows[1:], "past pixel"),
])
def test_rejections(env, tmp_path, mutate, needle):
    path = write_csv(tmp_path / "s.csv", mutate(_rows(env)))
    with pytest.raises(ParticipantSubmissionError, match=needle):
        env.evaluate(str(path))


def test_wrong_header(env, tmp_path):
    path = write_csv(tmp_path / "s.csv", _rows(env), header=("id", "rle"))
    with pytest.raises(ParticipantSubmissionError, match="header"):
        env.evaluate(str(path))


def test_benchmark_score(env):
    bench = PKG / CONFIG["benchmark_file"]
    if not bench.exists():
        pytest.skip("run reference_solution.py")
    out = env.evaluate(str(bench))["agent_results"][0]
    assert out["score"] == pytest.approx(CONFIG["benchmark_expected_score"], abs=CONFIG["benchmark_score_tol"])
    assert set(out["metrics_detail"]) == {d["key"] for d in CONFIG["metrics"]}
