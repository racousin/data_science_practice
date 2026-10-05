#!/usr/bin/env python3
"""Build data/ for mlp-s7-pet-segmentation.

    export MLARENA_ID_SALT=...        # see ../_dataset_ids.py
    uv run --with numpy --with pillow python prepare_data.py

Source: the Oxford-IIIT Pet dataset (Parkhi et al., CVPR 2012, CC BY-SA 4.0),
7,390 photos of 37 cat and dog breeds, each with a pixel trimap (pet,
background, border). The two archives are the ones torchvision's
`OxfordIIITPet` downloads, sha256-pinned below.

Split: the official trainval and test lists are pooled (7,349 images; the 41
photos outside both lists are left out) and re-split by a salted hash of the
file name: 1,500 images become the private test set, the rest ship with their
masks. Every image and trimap is resized to 128 x 128 (the image squashed with
Lanczos, the trimap with nearest neighbour, so a mask holds labels only) and
renamed to a salted opaque id: the source file names carry the breed.

Writes:
    data/train.zip              public   images/<id>.jpg + masks/<id>.png (0/1/2)
    data/test_images.zip        public   images/<id>.jpg
    data/sample_submission.csv  public   every test id, empty mask (scores 0.33)
    data/y_test.csv             private  image_id,mask_rle  (pet pixels)
    data/y_test_border.csv      private  image_id,border_rle (unscored band)
"""
import csv
import hashlib
import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _dataset_ids import _digest  # noqa: E402
from seg_metric import BACKGROUND, BORDER, COLUMNS, PET, SIZE, rle_encode  # noqa: E402

DATA = HERE / "data"
SOURCE = DATA / "_source"
ARCHIVES = {
    "images.tar.gz": "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d",
    "annotations.tar.gz": "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91",
}
URL = "https://thor.robots.ox.ac.uk/datasets/pets/"
DATASET = "oxford-pets"
N_TEST = 1500
JPEG_QUALITY = 90
ZIP_DATE = (2026, 10, 5, 0, 0, 0)   # fixed, so the archives are byte-reproducible
# trimap values in the source: 1 pet, 2 background, 3 border
TRIMAP_TO_LABEL = np.array([0, PET, BACKGROUND, BORDER], dtype=np.uint8)


def sources() -> dict:
    SOURCE.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, sha in ARCHIVES.items():
        path = SOURCE / name
        if not path.exists():
            print(f"downloading {URL}{name}")
            urllib.request.urlretrieve(URL + name, path)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            raise SystemExit(f"{path}: sha256 {h.hexdigest()}, expected {sha}")
        out[name] = tarfile.open(path)
    return out


def stems(annotations: tarfile.TarFile) -> list[str]:
    names = []
    for split in ("trainval", "test"):
        text = annotations.extractfile(f"annotations/{split}.txt").read().decode()
        names += [line.split()[0] for line in text.splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise SystemExit("a stem is in both official lists")
    return sorted(names)


def image_id(prefix: str, stem: str) -> str:
    return f"{prefix}_{_digest(DATASET, 'id', stem, size=6).hex()}"


def load_pair(images: dict, annotations: tarfile.TarFile, stem: str):
    img = Image.open(io.BytesIO(images[stem])).convert("RGB")
    img = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    trimap = Image.open(annotations.extractfile(f"annotations/trimaps/{stem}.png"))
    trimap = np.asarray(trimap.resize((SIZE, SIZE), Image.Resampling.NEAREST))
    if not set(np.unique(trimap)) <= {1, 2, 3}:
        raise SystemExit(f"{stem}: unexpected trimap values {np.unique(trimap)}")
    return img, TRIMAP_TO_LABEL[trimap]


def jpeg(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def png(labels: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(labels, mode="L").save(out, format="PNG", optimize=True)
    return out.getvalue()


def write(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_STORED if name.endswith((".jpg", ".png")) else zipfile.ZIP_DEFLATED
    zf.writestr(info, data)


def main() -> None:
    src = sources()
    annotations = src["annotations.tar.gz"]
    all_stems = stems(annotations)
    images = {}
    for member in src["images.tar.gz"]:
        if member.isfile() and member.name.endswith(".jpg"):
            stem = Path(member.name).stem
            if stem in all_stems:
                images[stem] = src["images.tar.gz"].extractfile(member).read()
    if set(images) != set(all_stems):
        raise SystemExit(f"{len(set(all_stems) - set(images))} listed images have no photo")

    # salted order: which images are private is not recoverable from this file
    order = sorted(all_stems, key=lambda s: _digest(DATASET, "split", s, size=8))
    test_stems, train_stems = order[:N_TEST], order[N_TEST:]

    DATA.mkdir(exist_ok=True)
    with zipfile.ZipFile(DATA / "train.zip", "w") as zf:
        for stem in sorted(train_stems, key=lambda s: image_id("tr", s)):
            img, labels = load_pair(images, annotations, stem)
            iid = image_id("tr", stem)
            write(zf, f"images/{iid}.jpg", jpeg(img))
            write(zf, f"masks/{iid}.png", png(labels))

    test = sorted((image_id("te", s), s) for s in test_stems)
    ids = [iid for iid, _ in test]
    if len(set(ids)) != N_TEST or len({image_id("tr", s) for s in train_stems}) != len(train_stems):
        raise SystemExit("id collision")
    pet_fraction = []
    with zipfile.ZipFile(DATA / "test_images.zip", "w") as zf, \
            open(DATA / "y_test.csv", "w", newline="") as fy, \
            open(DATA / "y_test_border.csv", "w", newline="") as fb:
        wy, wb = csv.writer(fy, lineterminator="\n"), csv.writer(fb, lineterminator="\n")
        wy.writerow(COLUMNS)
        wb.writerow(["image_id", "border_rle"])
        for iid, stem in test:
            img, labels = load_pair(images, annotations, stem)
            write(zf, f"images/{iid}.jpg", jpeg(img))
            wy.writerow([iid, rle_encode(labels == PET)])
            wb.writerow([iid, rle_encode(labels == BORDER)])
            scored = labels != BORDER
            pet_fraction.append((labels[scored] == PET).mean())
    with open(DATA / "sample_submission.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for iid in ids:
            w.writerow([iid, ""])

    print(f"train: {len(train_stems)} images; test: {N_TEST} images, "
          f"pet = {np.mean(pet_fraction):.1%} of the scored pixels")
    for f in ("train.zip", "test_images.zip", "y_test.csv", "y_test_border.csv"):
        print(f"  {f}: {(DATA / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
