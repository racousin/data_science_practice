#!/usr/bin/env python3
"""Build data/ for mlp-s6-aquarium-detection.

    export MLARENA_ID_SALT=...        # see ../_dataset_ids.py
    python prepare_data.py

Source: the Roboflow "Aquarium Combined" dataset, v2 (CC BY 4.0), 638 images
from two US aquariums, 7 classes, YOLO labels — the file Day 3 TP2 of
ai_for_sciences downloads, sha256-pinned below.

Split: the source's `train` (448 images) ships with its labels; its `valid`
and `test` (127 + 63) become the private test set of 190 images, so the
leaderboard scores 1,491 boxes rather than the source test split's 521.
Every image is resized to 640 px on its long side (the size YOLO trains at;
labels are normalised, so they do not change) and renamed to a salted opaque id: the source file names
carry Roboflow hashes that identify the image.

Writes:
    data/train.zip            public   images/<id>.jpg + labels/<id>.txt (YOLO)
    data/test_images.zip      public   images/<id>.jpg
    data/classes.csv          public   class_id,name
    data/sample_submission.csv public  every test id, no detections (scores 0)
    data/y_test.csv           private  image_id,prediction_string ground truth
                                       ("class xc yc w h" per box)
"""
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _dataset_ids import _digest  # noqa: E402
from detection_metric import CLASS_NAMES, COLUMNS  # noqa: E402

DATA = HERE / "data"
SOURCE_URL = "https://www.raphaelcousin.com/modules/sandbox/aquarium_yolo.zip"
SOURCE_SHA256 = "6672c419fc8a56e03a51781e3178034f2948aa95e80584eff5e60f8d5e05f01d"
DATASET = "aquarium-v2"
LONG_SIDE = 640
JPEG_QUALITY = 90
ZIP_DATE = (2026, 9, 30, 0, 0, 0)   # fixed, so the archives are byte-reproducible


def source() -> zipfile.ZipFile:
    cache = DATA / "_source.zip"
    if not cache.exists():
        DATA.mkdir(exist_ok=True)
        print(f"downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, cache)
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f"{cache}: sha256 {digest}, expected {SOURCE_SHA256}")
    return zipfile.ZipFile(cache)


def image_id(split: str, stem: str) -> str:
    prefix = "tr" if split == "train" else "te"
    return f"{prefix}_{_digest(DATASET, split, stem, size=6).hex()}"


def resized_jpeg(raw: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    scale = LONG_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY)   # no EXIF carried over
    return out.getvalue()


def read_labels(src: zipfile.ZipFile, name: str) -> list[list[str]]:
    """The YOLO rows of one label file, minus zero-area boxes.

    Two source test labels are points (width = height = 0: IMG_2423, IMG_2570,
    both "shark"). No detection can overlap a point, so they would only cap
    recall; they are annotation errors and are dropped, loudly.
    """
    rows = [line.split() for line in src.read(name).decode().splitlines() if line.strip()]
    for r in rows:
        if len(r) != 5 or not 0 <= int(r[0]) < len(CLASS_NAMES):
            raise SystemExit(f"{name}: unexpected label line {r}")
    kept = [r for r in rows if float(r[3]) > 0 and float(r[4]) > 0]
    for r in rows:
        if r not in kept:
            print(f"  dropped zero-area box in {name}: {' '.join(r)}")
    return kept


def split_items(src: zipfile.ZipFile, split: str):
    """(stem, image member, label member) of one source split, sorted."""
    images = sorted(n for n in src.namelist()
                    if n.startswith(f"{split}/images/") and n.endswith(".jpg"))
    for name in images:
        stem = Path(name).stem
        yield stem, name, f"{split}/labels/{stem}.txt"


def write(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_STORED if name.endswith(".jpg") else zipfile.ZIP_DEFLATED
    zf.writestr(info, data)


def main() -> None:
    src = source()
    DATA.mkdir(exist_ok=True)

    n_train_boxes = n_train_images = 0
    with zipfile.ZipFile(DATA / "train.zip", "w") as zf:
        for stem, img, lbl in split_items(src, "train"):
            iid = image_id("train", stem)
            labels = read_labels(src, lbl)
            n_train_boxes += len(labels)
            n_train_images += 1
            write(zf, f"images/{iid}.jpg", resized_jpeg(src.read(img)))
            write(zf, f"labels/{iid}.txt", "".join(" ".join(r) + "\n" for r in labels).encode())

    test = []
    for split in ("valid", "test"):
        for stem, img, lbl in split_items(src, split):
            test.append((image_id(split, stem), img, read_labels(src, lbl)))
    test.sort()                       # by opaque id: the source order is not recoverable
    ids = [t[0] for t in test]
    if len(set(ids)) != len(ids) or set(ids) & {Path(n).stem for n in zipfile.ZipFile(DATA / "train.zip").namelist()}:
        raise SystemExit("id collision")
    with zipfile.ZipFile(DATA / "test_images.zip", "w") as zf:
        for iid, img, _ in test:
            write(zf, f"images/{iid}.jpg", resized_jpeg(src.read(img)))

    with open(DATA / "y_test.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for iid, _, labels in test:
            w.writerow([iid, " ".join(" ".join(r) for r in labels)])
    with open(DATA / "sample_submission.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for iid in ids:
            w.writerow([iid, ""])
    with open(DATA / "classes.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["class_id", "name"])
        w.writerows(enumerate(CLASS_NAMES))

    test_classes = {int(r[0]) for _, _, labels in test for r in labels}
    if test_classes != set(range(len(CLASS_NAMES))):
        raise SystemExit(f"test set misses classes {set(range(len(CLASS_NAMES))) - test_classes}")
    n_test_boxes = sum(len(t[2]) for t in test)
    print(f"train: {n_train_boxes} boxes on {n_train_images} images; test: {n_test_boxes} boxes on {len(test)} images")
    for f in ("train.zip", "test_images.zip", "y_test.csv"):
        print(f"  {f}: {(DATA / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
