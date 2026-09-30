"""Aquarium Detection — the submission format and the metric.

This file is handed out byte for byte and is what the leaderboard imports, so
a score computed locally with it is computed by the same code.

Format — one row per test image, every test image exactly once:

    image_id,prediction_string
    te_3f0c...,0 0.91 0.412 0.530 0.120 0.085 4 0.33 0.702 0.201 0.310 0.144
    te_81aa...,

`prediction_string` is a space-separated list of detections, six numbers each:
`class_id confidence x_center y_center width height`, the box normalised to
[0, 1] by the image's width and height (the YOLO label convention). An empty
string means no detection. The ground truth uses the same layout without the
confidence (five numbers per box).

Metric — mAP@[.5:.95], the COCO / Ultralytics box metric (`metrics.box.map`):
for each class, detections are sorted by confidence and matched to ground-truth
boxes of the same image and class, greedily by IoU, at each of ten IoU
thresholds 0.50, 0.55, ..., 0.95; AP is the area under the precision envelope,
sampled at 101 recall points; the score is the mean over the ten thresholds and
over the classes present in the ground truth. It follows
`ultralytics.utils.metrics.ap_per_class` and `DetectionValidator.match_predictions`
(Ultralytics 8.4). A predicted box of zero width or height is accepted and
never matches.
"""
import csv
import math

import numpy as np

CLASS_NAMES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
MAX_DETECTIONS_PER_IMAGE = 300   # Ultralytics' default max_det
COLUMNS = ["image_id", "prediction_string"]


class SubmissionError(ValueError):
    """The submission file is malformed; the message names the line."""


def to_prediction_string(boxes) -> str:
    """`[(class_id, confidence, xc, yc, w, h), ...]` -> the submission cell."""
    return " ".join(f"{int(c)} {p:.6f} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                    for c, p, x, y, w, h in boxes)


def _parse(cell: str, width: int, where: str) -> np.ndarray:
    """A prediction_string (width 6) or a ground-truth string (width 5)."""
    tokens = cell.split()
    if len(tokens) % width:
        raise SubmissionError(f"{where}: {len(tokens)} numbers, not a multiple of {width} "
                              f"(class_id{' confidence' if width == 6 else ''} x_center "
                              f"y_center width height per box)")
    try:
        values = [float(t) for t in tokens]
    except ValueError:
        bad = next(t for t in tokens if not _is_float(t))
        raise SubmissionError(f"{where}: {bad!r} is not a number") from None
    boxes = np.array(values, dtype=float).reshape(-1, width)
    if not np.isfinite(boxes).all():
        raise SubmissionError(f"{where}: NaN or infinity in the boxes")
    cls = boxes[:, 0]
    if ((cls != np.round(cls)) | (cls < 0) | (cls >= len(CLASS_NAMES))).any():
        raise SubmissionError(f"{where}: class_id must be an integer in 0..{len(CLASS_NAMES) - 1}")
    if width == 6 and ((boxes[:, 1] < 0) | (boxes[:, 1] > 1)).any():
        raise SubmissionError(f"{where}: confidence must be in [0, 1]")
    wh = boxes[:, -2:]
    # A detector can emit a degenerate box, and rounding can make one; it only
    # costs its own false positive. A labelled box must have an area.
    if width == 6 and (wh < 0).any():
        raise SubmissionError(f"{where}: width and height must be >= 0")
    if width == 5 and (wh <= 0).any():
        raise SubmissionError(f"{where}: width and height must be > 0")
    return boxes


def _is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def read_boxes(path: str, *, with_confidence: bool, image_ids=None) -> dict:
    """Read a submission (with_confidence=True) or a ground-truth file.

    Returns {image_id: array (n, 6) or (n, 5)}. When `image_ids` is given, the
    file must list exactly those images, each once.
    """
    width = 6 if with_confidence else 5
    out = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None or [h.strip() for h in header] != COLUMNS:
            raise SubmissionError(f"line 1: the header must be {','.join(COLUMNS)}, got {header}")
        for line, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise SubmissionError(f"line {line}: 2 fields expected, got {len(row)}")
            image_id = row[0].strip()
            if image_id in out:
                raise SubmissionError(f"line {line}: image {image_id} listed twice")
            boxes = _parse(row[1], width, f"line {line} ({image_id})")
            if with_confidence and len(boxes) > MAX_DETECTIONS_PER_IMAGE:
                raise SubmissionError(f"line {line} ({image_id}): {len(boxes)} detections, "
                                      f"at most {MAX_DETECTIONS_PER_IMAGE} per image")
            out[image_id] = boxes
    if image_ids is not None:
        missing = set(image_ids) - set(out)
        extra = set(out) - set(image_ids)
        if missing or extra:
            raise SubmissionError(
                f"the file must list every test image exactly once: {len(missing)} missing"
                f"{' (e.g. ' + sorted(missing)[0] + ')' if missing else ''}, {len(extra)} unknown"
                f"{' (e.g. ' + sorted(extra)[0] + ')' if extra else ''}")
    return out


def _xyxy(b: np.ndarray) -> np.ndarray:
    xc, yc, w, h = b.T
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a), len(b)) of xywh boxes. IoU is unchanged by scaling
    each axis, so normalised coordinates give the pixel-space value."""
    a, b = _xyxy(a), _xyxy(b)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _match(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(n_pred, 10) bool: is each detection a true positive at each threshold."""
    correct = np.zeros((len(pred), len(IOU_THRESHOLDS)), dtype=bool)
    if len(gt) == 0 or len(pred) == 0:
        return correct
    iou = box_iou(gt[:, 1:], pred[:, 2:]) * (gt[:, :1] == pred[None, :, 0])
    for t, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.argwhere(iou >= threshold)          # (label, detection) pairs
        if len(matches) > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1], t] = True
    return correct


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    # Precision falls to 0 just past the highest recall reached, so recall the
    # model never reached earns nothing (`ultralytics.utils.metrics.compute_ap`).
    mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def evaluate(ground_truth: dict, predictions: dict) -> dict:
    """mAP@[.5:.95], mAP@.5 and AP@[.5:.95] per class.

    `ground_truth`: {image_id: (n, 5) class xc yc w h};
    `predictions`: {image_id: (m, 6) class conf xc yc w h}. An image missing
    from `predictions` has no detections.
    """
    tp, conf, pred_cls, target_cls = [], [], [], []
    for image_id, gt in ground_truth.items():
        pred = predictions.get(image_id, np.zeros((0, 6)))
        tp.append(_match(gt, pred))
        conf.append(pred[:, 1])
        pred_cls.append(pred[:, 0])
        target_cls.append(gt[:, 0])
    tp, conf = np.concatenate(tp), np.concatenate(conf)
    pred_cls, target_cls = np.concatenate(pred_cls), np.concatenate(target_cls)

    order = np.argsort(-conf, kind="stable")
    tp, pred_cls = tp[order], pred_cls[order]
    classes, n_labels = np.unique(target_cls, return_counts=True)
    ap = np.zeros((len(classes), len(IOU_THRESHOLDS)))
    for k, (c, n_l) in enumerate(zip(classes, n_labels)):
        hits = tp[pred_cls == c]
        if len(hits) == 0:
            continue
        tpc = hits.cumsum(0)
        fpc = (~hits).cumsum(0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc)
        for t in range(len(IOU_THRESHOLDS)):
            ap[k, t] = _ap(recall[:, t], precision[:, t])

    result = {"map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean())}
    for k, c in enumerate(classes):
        result[f"ap_{CLASS_NAMES[int(c)]}"] = float(ap[k].mean())
    result["n_detections"] = int(len(conf))
    assert all(math.isfinite(v) for v in result.values())
    return result


def score_file(submission_path: str, ground_truth_path: str) -> dict:
    """Score a submission file against a ground-truth file of the same layout."""
    gt = read_boxes(ground_truth_path, with_confidence=False)
    pred = read_boxes(submission_path, with_confidence=True, image_ids=gt.keys())
    return evaluate(gt, pred)
