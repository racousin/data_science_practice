#!/usr/bin/env python3
"""Local tests for the Aquarium Detection challenge — the format, the metric, the scorer.

    cd courseware/competitions
    uv run --with numpy --with pytest pytest mlp-s6-aquarium-detection/test_aquarium_detection.py -v

Needs `data/` (run prepare_data.py, then reference_solution.py for the
benchmark test); the metric tests on synthetic boxes need nothing.

What is covered, and why:

* the metric's fixed points: the ground truth resubmitted scores the 0.995
  ceiling Ultralytics reports for a perfect model, no detection scores 0, a
  wrong class scores 0, and a box shifted by a fifth of its width keeps mAP50
  and loses the stricter thresholds;
* a false positive ranked below every true positive costs nothing, and ranked
  above them halves precision at full recall;
* every rejection of the format, each through env.py as a
  ParticipantSubmissionError naming the line;
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

import detection_metric as dm  # noqa: E402


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
    module = load("aquarium_env", stage / "env.py")
    instance = module.Env(is_evaluation=True)
    instance.ParticipantSubmissionError = ParticipantSubmissionError
    return instance


def gt_as_predictions(gt, transform=lambda b: b):
    return {k: np.column_stack([b[:, 0], np.ones(len(b)), transform(b[:, 1:])]).reshape(-1, 6)
            for k, b in gt.items()}


GT = {"a": np.array([[0, 0.3, 0.3, 0.2, 0.2], [4, 0.7, 0.7, 0.2, 0.3]]),
      "b": np.array([[0, 0.5, 0.5, 0.4, 0.4]])}


def test_perfect_is_the_ultralytics_ceiling():
    r = dm.evaluate(GT, gt_as_predictions(GT))
    assert r["map50_95"] == pytest.approx(0.995) and r["map50"] == pytest.approx(0.995)


def test_nothing_and_wrong_class_score_zero():
    assert dm.evaluate(GT, {})["map50_95"] == 0
    wrong = {k: v.copy() for k, v in gt_as_predictions(GT).items()}
    for v in wrong.values():
        v[:, 0] = 6 - v[:, 0]
    assert dm.evaluate(GT, wrong)["map50_95"] == 0


def test_shift_keeps_map50_loses_strict_thresholds():
    def shift(b):
        b = b.copy()
        b[:, 0] += 0.2 * b[:, 2]
        return b
    r = dm.evaluate(GT, gt_as_predictions(GT, shift))
    assert r["map50"] == pytest.approx(0.995)
    assert 0.2 < r["map50_95"] < 0.6


def test_zero_size_prediction_is_accepted_and_never_matches():
    gt = {"a": np.array([[0, 0.5, 0.5, 0.2, 0.2]])}
    pred = {"a": np.array([[0, 0.9, 0.5, 0.5, 0.0, 0.2], [0, 0.5, 0.5, 0.5, 0.2, 0.2]])}
    assert dm.evaluate(gt, pred)["map50_95"] == pytest.approx(0.995 / 2)


def test_unreached_recall_earns_nothing():
    # One of two objects found, with precision 1: AP is about half, not 3/4.
    gt = {"a": np.array([[0, 0.3, 0.3, 0.2, 0.2], [0, 0.7, 0.7, 0.2, 0.2]])}
    pred = {"a": np.array([[0, 0.9, 0.3, 0.3, 0.2, 0.2]])}
    assert dm.evaluate(gt, pred)["map50_95"] == pytest.approx(0.5, abs=0.01)


def test_duplicate_is_a_false_positive():
    gt = {"a": np.array([[0, 0.5, 0.5, 0.2, 0.2]])}
    true = [0, 0.5, 0.5, 0.5, 0.2, 0.2]
    dup_low = {"a": np.array([true, [0, 0.4, 0.5, 0.5, 0.2, 0.2]])}
    # Ranked after full recall, a false positive costs nothing (the envelope).
    assert dm.evaluate(gt, dup_low)["map50_95"] == pytest.approx(0.995)
    # Ranked first, it halves precision at full recall.
    offset_high = {"a": np.array([true, [0, 0.9, 0.62, 0.5, 0.2, 0.2]])}   # IoU < 0.5, ranked first
    assert dm.evaluate(gt, offset_high)["map50_95"] == pytest.approx(0.995 / 2)


@pytest.mark.parametrize("row, message", [
    ("{id},0 0.5 0.5 0.5 0.1", "not a multiple of 6"),
    ("{id},0 0.5 0.5 0.5 0.1 x", "'x' is not a number"),
    ("{id},7 0.5 0.5 0.5 0.1 0.1", "class_id must be an integer"),
    ("{id},1.5 0.5 0.5 0.5 0.1 0.1", "class_id must be an integer"),
    ("{id},0 1.5 0.5 0.5 0.1 0.1", "confidence must be in [0, 1]"),
    ("{id},0 0.5 0.5 0.5 -0.1 0.1", "width and height must be >= 0"),
    ("{id},0 nan 0.5 0.5 0.1 0.1", "NaN or infinity"),
    ("{id},", "every test image exactly once"),          # the other ids missing
])
def test_rejections_name_the_problem(env, tmp_path, row, message):
    first = next(iter(env.ground_truth))
    path = tmp_path / "submission.csv"
    path.write_text("image_id,prediction_string\n" + row.format(id=first) + "\n")
    with pytest.raises(ParticipantSubmissionError, match=message.replace("[", r"\[").replace("]", r"\]")):
        env.evaluate(str(path))


def test_header_and_duplicates_rejected(env, tmp_path):
    path = tmp_path / "submission.csv"
    path.write_text("id,boxes\n")
    with pytest.raises(ParticipantSubmissionError, match="header"):
        env.evaluate(str(path))
    first = next(iter(env.ground_truth))
    path.write_text(f"image_id,prediction_string\n{first},\n{first},\n")
    with pytest.raises(ParticipantSubmissionError, match="listed twice"):
        env.evaluate(str(path))


def test_too_many_detections_rejected(env, tmp_path):
    first = next(iter(env.ground_truth))
    boxes = [(0, 0.5, 0.5, 0.5, 0.1, 0.1)] * (dm.MAX_DETECTIONS_PER_IMAGE + 1)
    path = tmp_path / "submission.csv"
    path.write_text(f"image_id,prediction_string\n{first},{dm.to_prediction_string(boxes)}\n")
    with pytest.raises(ParticipantSubmissionError, match="at most 300"):
        env.evaluate(str(path))


def test_sample_submission_scores_zero(env):
    r = env.evaluate(str(DATA / "sample_submission.csv"))["agent_results"][0]
    assert r["score"] == 0


def test_benchmark_is_pinned_and_detail_matches_schema(env):
    bench = PKG / CONFIG["benchmark_file"]
    if not bench.exists():
        pytest.skip("no benchmark: run reference_solution.py")
    r = env.evaluate(str(bench))["agent_results"][0]
    assert r["score"] == pytest.approx(CONFIG["benchmark_expected_score"],
                                       abs=CONFIG["benchmark_score_tol"])
    assert set(r["metrics_detail"]) == {m["key"] for m in CONFIG["metrics_schema"]}


def test_test_set_covers_every_class():
    if not (DATA / "y_test.csv").exists():
        pytest.skip("data/ not built")
    gt = dm.read_boxes(DATA / "y_test.csv", with_confidence=False)
    assert len(gt) == 190
    assert set(np.concatenate([b[:, 0] for b in gt.values()]).astype(int)) == set(range(7))
    with open(DATA / "sample_submission.csv") as fh:
        assert [r["image_id"] for r in csv.DictReader(fh)] == list(gt)
