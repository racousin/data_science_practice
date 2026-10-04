"""Computed figures for `s6-computer-vision-2/self-supervised-learning.md` of
`ms2a-machine-learning-practice` (Session 7 — Computer Vision 2).

DINOv2 (Oquab et al., 2023) is run from its released weights
(github.com/facebookresearch/dinov2, Apache-2.0, through `torch.hub`) on
photographs from Wikimedia Commons, credited in `PHOTOS` (all CC BY-SA, so
the derived figures are CC BY-SA 4.0 too; the credit line is printed).

1. `ssl-multicrop.png`    — the DINO multi-crop: two global and four local
                            augmented views of one photo.
2. `dinov2-pca.png`       — patch features of DINOv2 ViT-B/14 with registers:
                            the foreground is seeded where the [CLS] attends
                            (60% of the mass) and grown by two-means, coloured by the first three principal
                            components of all foreground patches.
3. `dinov2-matching.png`  — mutual nearest-neighbour patch matches between
                            two running dogs of different breeds.
4. `dinov2-few-labels.png` and the numbers the lesson quotes — frozen features
   + logistic regression on CIFAR-10, k labelled images per class, DINOv2
   ViT-S/14 against an ImageNet-supervised ResNet-50 of the same size.
   Test: 2,000 fixed test images. Train pool: the first 500 images per class
   of the training split; k-shot subsets drawn with a fixed generator, three
   draws per k. Every number is cached in build/figures-cache/s6-dinov2.json
   (`--rerun` to measure again) and asserted against the lesson below.

Re-run with:

    uv run --no-project --with torch --with torchvision --with matplotlib \
        --with numpy --with scikit-learn --with pillow \
        python courseware/tools/figures/ms2a_s6_dinov2.py
"""

from __future__ import annotations

import html
import io
import json
import os
import pathlib
import re
import sys
import urllib.parse
import urllib.request

# DINOv2 resizes its position embeddings with an antialiased bicubic that MPS
# does not implement; that one op runs on the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

COURSEWARE = pathlib.Path(__file__).resolve().parents[2]
OUT = COURSEWARE / "content" / "ms2a-machine-learning-practice" / "assets" / "cv"
DATA = COURSEWARE / "build" / "data"
CACHE = COURSEWARE / "build" / "figures-cache" / "s6-dinov2.json"
PHOTO_CACHE = COURSEWARE / "build" / "figures-cache" / "s6-photos"
RERUN = "--rerun" in sys.argv
UA = {"User-Agent": "ms2a-courseware/1.0 (course figures; racousin)"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK_2, MUTED, GRID, SPINE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# key -> Commons file name
PHOTOS = {
    "dalmatian": "Dalmatian fetching a stick.jpg",
    "retriever": "Dülmen, Hausdülmen, Golden Retriever -- 2022 -- 5945.jpg",
    "cat": "Calico cat, - Assisi, Italy.jpg",
    "waxwing": "Cedar waxwing in pokeweed (10132).jpg",
    "monarch": "Black-naped Monarch 0A2A8267.jpg",
    "greyhound": "Greyhound Racing 2 amk.jpg",
}
PCA_PHOTOS = ("dalmatian", "retriever", "cat", "waxwing", "monarch")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SHOTS = (1, 2, 5, 10, 25, 100, 500)
DRAWS = 3


def commons(title: str) -> tuple[Image.Image, str]:
    """The photo at 1280 px and its credit, kept in build/ so a re-run does not
    hit Wikimedia's rate limit."""
    key = re.sub(r"[^A-Za-z0-9]+", "_", title)
    jpg, meta_path = PHOTO_CACHE / f"{key}.jpg", PHOTO_CACHE / f"{key}.txt"
    if jpg.exists() and meta_path.exists():
        return Image.open(jpg).convert("RGB"), meta_path.read_text()
    q = urllib.parse.urlencode({
        "action": "query", "prop": "imageinfo", "format": "json",
        "iiprop": "url|extmetadata", "iiurlwidth": 1280, "titles": f"File:{title}"})
    with urllib.request.urlopen(urllib.request.Request(
            f"https://commons.wikimedia.org/w/api.php?{q}", headers=UA)) as r:
        info = next(iter(json.load(r)["query"]["pages"].values()))["imageinfo"][0]
    meta = info["extmetadata"]
    artist = html.unescape(re.sub(r"<[^>]+>", "", meta["Artist"]["value"])).strip()
    licence = meta["LicenseShortName"]["value"]
    with urllib.request.urlopen(urllib.request.Request(info["thumburl"], headers=UA)) as r:
        img = Image.open(io.BytesIO(r.read())).convert("RGB")
    PHOTO_CACHE.mkdir(parents=True, exist_ok=True)
    img.save(jpg, quality=95)
    meta_path.write_text(f"{artist} ({licence})")
    return img, f"{artist} ({licence})"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
        "axes.edgecolor": SPINE, "axes.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "text.color": INK_2,
    })


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote assets/cv/{name}")


def center_square(img: Image.Image, size: int) -> Image.Image:
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)


# ------------------------------------------------------------------ figures

def fig_multicrop(img: Image.Image) -> None:
    color = [T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
             T.RandomGrayscale(p=0.2)]
    glob = T.Compose([T.RandomResizedCrop(224, scale=(0.4, 1.0)),
                      T.RandomHorizontalFlip(), *color,
                      T.RandomApply([T.GaussianBlur(23, (0.1, 2.0))], p=0.5)])
    loc = T.Compose([T.RandomResizedCrop(96, scale=(0.05, 0.4)),
                     T.RandomHorizontalFlip(), *color])
    torch.manual_seed(3)
    views = [glob(img) for _ in range(2)] + [loc(img) for _ in range(4)]
    fig = plt.figure(figsize=(12, 3.4))
    gs = fig.add_gridspec(2, 7, width_ratios=[2.2, 2, 2, 1, 1, 0.01, 0.01],
                          wspace=0.08, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0]); ax.imshow(img); ax.set_title("original", color=INK_2)
    for i in range(2):
        a = fig.add_subplot(gs[:, 1 + i]); a.imshow(views[i])
        a.set_title(f"global view {i + 1}\n224 × 224", color=BLUE)
    for i in range(4):
        a = fig.add_subplot(gs[i // 2, 3 + i % 2]); a.imshow(views[2 + i])
        if i < 2:
            a.set_title("local\n96 × 96" if i == 0 else " \n ", color=ORANGE)
    for a in fig.axes:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    save(fig, "ssl-multicrop.png")


def fit14(img: Image.Image, width: int) -> Image.Image:
    """Resize to `width` keeping the aspect ratio, both sides multiples of 14."""
    h = max(14, round(img.height * width / img.width / 14) * 14)
    return img.resize((width, h), Image.BICUBIC)


@torch.no_grad()
def patch_features(model, img: Image.Image, keep: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Normalised patch tokens (H/14 * W/14, D) of an image whose sides are
    multiples of 14, and its foreground: the patches holding `keep` of the
    [CLS] token's attention in the last block, averaged over heads — the
    thresholding of the DINO paper's Fig. 4, grown by two-means."""
    seen = {}
    hook = model.blocks[-1].attn.register_forward_pre_hook(
        lambda mod, args: seen.update(x=args[0], mod=mod))
    x = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])(img)
    out = model.forward_features(x[None].to(DEVICE))
    hook.remove()
    mod, h = seen["mod"], seen["x"]
    B, N, C = h.shape
    qkv = mod.qkv(h).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
    attn = ((qkv[0] * mod.scale) @ qkv[1].transpose(-2, -1)).softmax(-1)
    first_patch = 1 + model.num_register_tokens
    a = attn[0, :, 0, first_patch:].mean(0).float().cpu().numpy()
    order = np.argsort(a)[::-1]
    cum = np.cumsum(a[order]) / a.sum()
    fg = np.zeros(len(a), bool)
    fg[order[: np.searchsorted(cum, keep) + 1]] = True
    feats = out["x_norm_patchtokens"][0].float().cpu().numpy()
    # The attention marks the most telling patches only (the head, mostly).
    # Grow it into the whole object: two-means on the cosine geometry of the
    # patch tokens, seeded with the attended patches as the foreground.
    u = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    for _ in range(10):
        cf, cb = u[fg].mean(0), u[~fg].mean(0)
        fg = u @ cf > u @ cb
    return feats, fg


def fig_pca(model, photos: dict[str, Image.Image]) -> None:
    size = 448
    keys = list(photos)
    g = size // 14
    feats, fg = {}, {}
    for k in keys:
        feats[k], fg[k] = patch_features(model, center_square(photos[k], size))
    rgb_pca = PCA(3).fit(np.concatenate([feats[k][fg[k]] for k in keys]))
    comps = {k: rgb_pca.transform(feats[k][fg[k]]) for k in keys}
    lo = np.percentile(np.concatenate(list(comps.values())), 1, axis=0)
    hi = np.percentile(np.concatenate(list(comps.values())), 99, axis=0)
    fig, axes = plt.subplots(2, len(keys), figsize=(2.3 * len(keys), 4.7))
    for j, k in enumerate(keys):
        canvas = np.zeros((g * g, 3))
        canvas[fg[k]] = np.clip((comps[k] - lo) / (hi - lo), 0, 1)
        axes[0, j].imshow(center_square(photos[k], size))
        axes[1, j].imshow(canvas.reshape(g, g, 3), interpolation="nearest")
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    save(fig, "dinov2-pca.png")


def fig_matching(model, a_img: Image.Image, b_img: Image.Image) -> None:
    a_img, b_img = fit14(a_img, 560), fit14(b_img, 560)
    (fa, fga), (fb, fgb) = patch_features(model, a_img), patch_features(model, b_img)
    ga, gb = a_img.width // 14, b_img.width // 14
    fa_n = fa / np.linalg.norm(fa, axis=1, keepdims=True)
    fb_n = fb / np.linalg.norm(fb, axis=1, keepdims=True)
    sim = fa_n @ fb_n.T
    ab, ba = sim.argmax(1), sim.argmax(0)
    mutual = [i for i in range(len(ab)) if ba[ab[i]] == i]
    mutual = [i for i in mutual if fga[i] and fgb[ab[i]]]
    mutual.sort(key=lambda i: -sim[i, ab[i]])
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(mutual[:120], size=min(24, len(mutual)), replace=False))
    A, B = np.asarray(a_img), np.asarray(b_img)
    H = max(A.shape[0], B.shape[0])
    pad = lambda X: np.pad(X, ((0, H - X.shape[0]), (0, 0), (0, 0)), constant_values=255)
    fig, ax = plt.subplots(figsize=(12, 12 * H / (A.shape[1] + B.shape[1] + 20)))
    ax.imshow(np.concatenate([pad(A), np.full((H, 20, 3), 255, np.uint8), pad(B)], axis=1))
    cmap = plt.get_cmap("turbo")
    for n, i in enumerate(pick):
        j = ab[i]
        ya, xa = divmod(i, ga); yb, xb = divmod(j, gb)
        pa = (xa * 14 + 7, ya * 14 + 7)
        pb = (A.shape[1] + 20 + xb * 14 + 7, yb * 14 + 7)
        c = cmap(n / max(1, len(pick) - 1))
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=c, lw=1.3)
        ax.scatter([pa[0], pb[0]], [pa[1], pb[1]], color=c, s=18, zorder=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "dinov2-matching.png")


# ------------------------------------------------------------ the experiment

@torch.no_grad()
def embed(model, ds, idx, fn) -> np.ndarray:
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(MEAN, STD)])
    out = []
    for s in range(0, len(idx), 128):
        x = torch.stack([tf(ds[i][0]) for i in idx[s:s + 128]]).to(DEVICE)
        out.append(fn(model, x).float().cpu().numpy())
    return np.concatenate(out)


def measure() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    train = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
    test = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
    ytr_all = np.array(train.targets)
    pool = np.concatenate([np.flatnonzero(ytr_all == c)[:500] for c in range(10)])
    test_idx = np.arange(2000)
    ytr, yte = ytr_all[pool], np.array(test.targets)[test_idx]

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE).eval()
    rn = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    rn.fc = torch.nn.Identity()
    rn = rn.to(DEVICE).eval()
    nets = {
        "dinov2_vits14": (dino, lambda m, x: m(x)),            # [CLS], 384
        "resnet50_supervised": (rn, lambda m, x: m(x)),        # pooled, 2048
    }
    params = {k: sum(p.numel() for p in m.parameters()) for k, (m, _) in nets.items()}
    res = {"params": params, "shots": list(SHOTS), "acc": {}, "dim": {}}
    for name, (m, fn) in nets.items():
        Xtr, Xte = embed(m, train, pool, fn), embed(m, test, test_idx, fn)
        res["dim"][name] = int(Xtr.shape[1])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        accs = {}
        for k in SHOTS:
            draws = []
            for d in range(DRAWS if k < 500 else 1):
                rng = np.random.default_rng(100 * k + d)
                sub = np.concatenate([rng.choice(np.flatnonzero(ytr == c), k, replace=False)
                                      for c in range(10)])
                clf = LogisticRegression(C=0.1, max_iter=3000).fit(Xtr[sub], ytr[sub])
                draws.append(float((clf.predict(Xte) == yte).mean()))
            accs[str(k)] = float(np.mean(draws))
            print(f"{name:22s} k={k:3d}  acc={accs[str(k)]:.3f}")
        res["acc"][name] = accs
    return res


def fig_few_labels(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name, label, c in (("dinov2_vits14", "DINOv2 ViT-S/14 — no labels in pre-training", BLUE),
                           ("resnet50_supervised", "ResNet-50 — ImageNet labels", ORANGE)):
        y = [100 * res["acc"][name][str(k)] for k in SHOTS]
        ax.plot(SHOTS, y, marker="o", color=c, label=label)
        ax.annotate(f"{y[-1]:.1f}", (SHOTS[-1], y[-1]), textcoords="offset points",
                    xytext=(6, -3), color=c)
        ax.annotate(f"{y[0]:.1f}", (SHOTS[0], y[0]), textcoords="offset points",
                    xytext=(6, -10), color=c)
    ax.set_xscale("log"); ax.set_xticks(SHOTS); ax.set_xticklabels(SHOTS)
    ax.set_xlabel("labelled images per class (frozen features + logistic regression)")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_ylim(0, 100); ax.legend(loc="lower right")
    save(fig, "dinov2-few-labels.png")


def main() -> None:
    style()
    photos, credits = {}, []
    for k, title in PHOTOS.items():
        photos[k], who = commons(title)
        credits.append(f"[{who.split(' (')[0]}](https://commons.wikimedia.org/wiki/File:"
                       f"{urllib.parse.quote(title.replace(' ', '_'))})")
        print(f"photo {k}: {who}")

    fig_multicrop(center_square(photos["dalmatian"], 448))
    big = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(DEVICE).eval()
    fig_pca(big, {k: photos[k] for k in PCA_PHOTOS})
    fig_matching(big, photos["dalmatian"], photos["greyhound"])
    del big

    if CACHE.exists() and not RERUN:
        res = json.loads(CACHE.read_text())
    else:
        res = measure()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(res, indent=1))
    fig_few_labels(res)
    print(json.dumps(res, indent=1))
    print("\nphoto credit for the derived figures (CC BY-SA 4.0):\n"
          "*Photos: " + ", ".join(credits) + ", Wikimedia Commons, "
          "[CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).*")


if __name__ == "__main__":
    main()
