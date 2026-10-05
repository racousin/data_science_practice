"""Pet Segmentation: the submission format and the leaderboard's metric.

This file is handed out with the data, byte for byte, and env.py imports it,
so a score computed locally with it is the leaderboard's score.

Masks
-----
Every image is SIZE x SIZE (128 x 128). A label mask holds one value per pixel:

    0  background
    1  pet
    2  border: the thin band drawn around the pet, not scored

Submission
----------
`submission.csv`, one row per test image:

    image_id,mask_rle
    te_016ee383598d,1153 9 1280 13 1407 16
    te_0185dd1595bf,

`mask_rle` is the run-length encoding of the **pet** pixels (your predicted
1s): the mask flattened row by row (`mask.flatten()`, C order), then
"start length" pairs, starts counted from 1, runs in increasing order and not
overlapping. An empty string means no pet pixel. `rle_encode` writes it.

Score
-----
Mean IoU over the two classes, background and pet, computed over every scored
pixel of every test image together (border pixels skipped):

    IoU(c) = |pred == c and true == c| / |pred == c or true == c|
    score  = (IoU(background) + IoU(pet)) / 2
"""
import csv

import numpy as np

SIZE = 128
BACKGROUND, PET, BORDER = 0, 1, 2
COLUMNS = ["image_id", "mask_rle"]


class SubmissionError(ValueError):
    """The file breaks the format; the message names the line."""


def rle_encode(mask) -> str:
    """Pet pixels of a (SIZE, SIZE) mask (bool, or labels where 1 = pet) -> RLE string."""
    flat = (np.asarray(mask) == PET).flatten().astype(np.int8)
    if flat.size != SIZE * SIZE:
        raise ValueError(f"mask has {flat.size} pixels, expected {SIZE}x{SIZE}")
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flat, [0]])))
    starts, ends = edges[0::2], edges[1::2]
    return " ".join(f"{s + 1} {e - s}" for s, e in zip(starts, ends))


def rle_decode(rle: str) -> np.ndarray:
    """RLE string -> (SIZE, SIZE) bool mask. Raises ValueError on a malformed string."""
    mask = np.zeros(SIZE * SIZE, dtype=bool)
    tokens = rle.split()
    if len(tokens) % 2:
        raise ValueError("odd number of values: expected 'start length' pairs")
    try:
        nums = [int(t) for t in tokens]
    except ValueError:
        raise ValueError("values must be integers") from None
    previous_end = 0
    for start, length in zip(nums[0::2], nums[1::2]):
        if length < 1:
            raise ValueError(f"run at {start} has length {length}")
        if start <= previous_end:  # a run may start right after the previous one
            raise ValueError(f"run at {start} overlaps the previous one "
                             "(runs must be in increasing order)")
        end = start - 1 + length
        if end > SIZE * SIZE:
            raise ValueError(f"run at {start} ends past pixel {SIZE * SIZE}")
        mask[start - 1:end] = True
        previous_end = end
    return mask.reshape(SIZE, SIZE)


def read_rle_csv(path, image_ids=None, column="mask_rle") -> dict:
    """{image_id: (SIZE, SIZE) bool mask} from a CSV in the submission layout.

    With `image_ids`, the file must hold exactly those ids, once each.
    """
    masks = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None or [h.strip() for h in header] != ["image_id", column]:
            raise SubmissionError(f"header must be 'image_id,{column}', got {header}")
        for line, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 2:
                raise SubmissionError(f"line {line}: expected 2 fields, got {len(row)}")
            iid, rle = row[0].strip(), row[1]
            if iid in masks:
                raise SubmissionError(f"line {line}: image_id {iid!r} appears twice")
            try:
                masks[iid] = rle_decode(rle)
            except ValueError as exc:
                raise SubmissionError(f"line {line} ({iid}): {exc}") from None
    if image_ids is not None:
        expected = set(image_ids)
        missing = expected - masks.keys()
        extra = masks.keys() - expected
        if missing:
            raise SubmissionError(f"{len(missing)} test image(s) missing, e.g. {sorted(missing)[0]!r}")
        if extra:
            raise SubmissionError(f"{len(extra)} unknown image_id(s), e.g. {sorted(extra)[0]!r}")
    return masks


def confusion(true_labels, pred_pet) -> np.ndarray:
    """2x2 counts [true class, predicted class] over the scored (non-border) pixels."""
    true_labels = np.asarray(true_labels)
    pred_pet = np.asarray(pred_pet).astype(bool)
    keep = true_labels != BORDER
    t = (true_labels[keep] == PET).astype(np.int64)
    p = pred_pet[keep].astype(np.int64)
    return np.bincount(2 * t + p, minlength=4).reshape(2, 2)


def scores(cm) -> dict:
    """The leaderboard's numbers from a 2x2 confusion matrix (summed over images)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
    iou_pet = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fn + fp, 1)
    return {
        "miou": (iou_pet + iou_bg) / 2,
        "iou_pet": iou_pet,
        "iou_background": iou_bg,
        "dice_pet": 2 * tp / max(2 * tp + fp + fn, 1),
        "pixel_accuracy": (tp + tn) / max(cm.sum(), 1),
    }


def evaluate(true_labels: dict, pred_pet: dict) -> dict:
    """true_labels: {id: (SIZE, SIZE) labels 0/1/2}; pred_pet: {id: bool mask}."""
    cm = sum(confusion(true_labels[i], pred_pet[i]) for i in true_labels)
    return scores(cm)


def score_masks(true_labels, pred_pet) -> dict:
    """Score a stack of validation masks: arrays (N, SIZE, SIZE), labels 0/1/2 vs bool."""
    return scores(confusion(true_labels, pred_pet))
