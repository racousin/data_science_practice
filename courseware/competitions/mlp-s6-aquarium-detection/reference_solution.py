#!/usr/bin/env python3
"""The benchmark: YOLOv8n fine-tuned on train.zip, the starter notebook's recipe.

    uv run --with ultralytics python reference_solution.py [--epochs 30] [--device mps]

Holds out 15% of the training images (seed 0), fine-tunes the COCO-pretrained
YOLOv8n on the rest, then

  1. prints detection_metric.evaluate beside Ultralytics' own `val` on the
     held-out images. They agree to about 0.01: identical matches and AP
     (ap_per_class gives the same number on these predictions), but `val`
     letterboxes the images its own way, so its boxes differ slightly, and
  2. predicts the test images and writes data/benchmark_submission.csv.

A torch training run is not bit-reproducible across machines, so the
benchmark is this file, not this script: re-pin benchmark_expected_score in
config.py with `python ../localtest.py mlp-s6-aquarium-detection` after a rerun.
"""
import argparse
import csv
import random
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detection_metric import CLASS_NAMES, COLUMNS, evaluate, read_boxes, to_prediction_string  # noqa: E402

DATA = HERE / "data"
WORK = DATA / "_work"


def detect(model, paths, device):
    """{image_id: [(class, conf, xc, yc, w, h), ...]} at the mAP operating point."""
    out = {}
    for i in range(0, len(paths), 16):
        for p, r in zip(paths[i:i + 16], model.predict([str(p) for p in paths[i:i + 16]],
                                                        conf=0.001, device=device, verbose=False)):
            b = r.boxes
            out[p.stem] = list(zip(b.cls.tolist(), b.conf.tolist(), *b.xywhn.T.tolist()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="mps", help="training device")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true",
                      help="continue an interrupted run from data/_work/runs/ref/weights/last.pt")
    mode.add_argument("--skip-train", action="store_true",
                      help="reuse data/_work/runs/ref/weights/best.pt")
    args = ap.parse_args()
    # Inference always on CPU: on Apple MPS, Ultralytics 8.4 scored the same
    # best.pt 0.29 mAP50-95 where CPU scored 0.37 (2026-09-30).
    infer = "cpu"

    from ultralytics import YOLO
    import numpy as np

    fresh = not (args.resume or args.skip_train)
    if fresh:
        shutil.rmtree(WORK, ignore_errors=True)
        zipfile.ZipFile(DATA / "train.zip").extractall(WORK / "all")
        zipfile.ZipFile(DATA / "test_images.zip").extractall(WORK / "test")
    ids = sorted(p.stem for p in (WORK / "all/images").glob("*.jpg"))
    random.Random(0).shuffle(ids)
    n_val = round(0.15 * len(ids))
    if fresh:
        for split, part in (("val", ids[:n_val]), ("train", ids[n_val:])):
            for sub in ("images", "labels"):
                (WORK / split / sub).mkdir(parents=True)
            for iid in part:
                shutil.copy(WORK / f"all/images/{iid}.jpg", WORK / f"{split}/images/")
                shutil.copy(WORK / f"all/labels/{iid}.txt", WORK / f"{split}/labels/")
        (WORK / "data.yaml").write_text(
            f"path: {WORK}\ntrain: train/images\nval: val/images\nnames: {CLASS_NAMES}\n")
        YOLO("yolov8n.pt").train(data=str(WORK / "data.yaml"), epochs=args.epochs, imgsz=640,
                                 batch=16, seed=0, device=args.device,
                                 project=str(WORK / "runs"), name="ref", plots=False,
                                 verbose=False)
    elif args.resume:
        YOLO(str(WORK / "runs/ref/weights/last.pt")).train(resume=True)
    best = YOLO(str(WORK / "runs/ref/weights/best.pt"))

    # 1. The same held-out images through Ultralytics' val and through ours.
    ultra = best.val(data=str(WORK / "data.yaml"), device=infer, plots=False, verbose=False)
    gt = {iid: np.array([[float(v) for v in line.split()] for line in
                         (WORK / f"val/labels/{iid}.txt").read_text().splitlines() if line.strip()]
                        ).reshape(-1, 5) for iid in ids[:n_val]}
    val_pred = detect(best, sorted((WORK / "val/images").glob("*.jpg")), infer)
    ours = evaluate(gt, {k: np.array(v).reshape(-1, 6) for k, v in val_pred.items()})
    print(f"held-out {n_val} images  ultralytics val: mAP50-95 {ultra.box.map:.4f} "
          f"mAP50 {ultra.box.map50:.4f} | detection_metric: mAP50-95 {ours['map50_95']:.4f} "
          f"mAP50 {ours['map50']:.4f}")

    # 2. The benchmark submission.
    test_pred = detect(best, sorted((WORK / "test/images").glob("*.jpg")), infer)
    with open(DATA / "benchmark_submission.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for iid in sorted(test_pred):
            w.writerow([iid, to_prediction_string(test_pred[iid])])
    read_boxes(DATA / "benchmark_submission.csv", with_confidence=True)   # well-formed
    shutil.copy(WORK / "runs/ref/weights/best.pt", DATA / "benchmark_best.pt")
    print("wrote data/benchmark_submission.csv")


if __name__ == "__main__":
    main()
