#!/usr/bin/env python3
"""Make data/benchmark_submission.csv: the starter notebook's first submission.

    uv run --with torch --with numpy --with pandas --with pillow \
        python reference_solution.py [--finetune]

Runs the notebook's own model and training code (`MODEL_SRC`, `TRAIN_SRC`,
`SCRATCH_RUN` in courseware/tools/notebooks/mlp_s7_segmentation.py) on the
same 85 / 15 split, then predicts the test images. `--finetune` also runs
Section 5 (needs segmentation-models-pytorch) and writes
data/finetuned_submission.csv, to check the gap the notebook promises.

A torch run is reproducible on one machine, not across machines, so the
benchmark is this file, scored exactly; a student's rerun lands near it.
"""
import argparse
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools" / "notebooks"))
import seg_metric as sm  # noqa: E402
from mlp_s7_segmentation import (FINETUNE_EPOCHS, FINETUNE_LR, MODEL_SRC,  # noqa: E402
                                 SCRATCH_RUN, TRAIN_SRC)


def load(zip_name, folder):
    root = DATA / "_unzipped" / zip_name
    if not root.exists():
        zipfile.ZipFile(DATA / zip_name).extractall(root)
    ids = sorted(p.stem for p in (root / "images").glob("*.jpg"))
    X = np.stack([np.array(Image.open(root / "images" / f"{i}.jpg")) for i in ids])
    M = (np.stack([np.array(Image.open(root / "masks" / f"{i}.png")) for i in ids])
         if folder == "train" else None)
    return ids, X, M


def write(ids, pred, path):
    pd.DataFrame({"image_id": ids, "mask_rle": [sm.rle_encode(m) for m in pred]}).to_csv(path, index=False)
    print(f"wrote {path.relative_to(HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)
    _, X_all, M_all = load("train.zip", "train")
    test_ids, X_test, _ = load("test_images.zip", "test")
    idx = np.random.RandomState(0).permutation(len(X_all))
    n_val = len(idx) * 15 // 100
    ns = {"np": np, "pd": pd, "torch": torch, "nn": nn, "F": F, "time": time, "sm": sm,
          "device": device, "X_val": X_all[idx[:n_val]], "M_val": M_all[idx[:n_val]],
          "X_tr": X_all[idx[n_val:]], "M_tr": M_all[idx[n_val:]]}
    exec(MODEL_SRC, ns)
    exec(TRAIN_SRC, ns)
    model_expr, epochs, lr = SCRATCH_RUN
    print(f"device {device}; {model_expr}, {epochs} epochs, lr {lr}")
    model = eval(model_expr, ns)
    ns["train"](model, epochs=epochs, lr=lr)
    write(test_ids, ns["predict"](model, X_test), DATA / "benchmark_submission.csv")

    if args.finetune:
        import segmentation_models_pytorch as smp
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        pretrained = smp.Unet("resnet18", encoder_weights="imagenet", classes=2)
        ns["train"](pretrained, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR, mean=mean, std=std)
        write(test_ids, ns["predict"](pretrained, X_test, mean, std), DATA / "finetuned_submission.csv")


if __name__ == "__main__":
    main()
