"""Pet Segmentation — the Colab starter of challenge `mlp-s7-pet-segmentation`
(Session 7 — Computer Vision 2, Lab 7 branch A).

`mlp-s7-pet-segmentation.ipynb`: open the Oxford-IIIT Pet data the challenge
hands out (128 x 128 photos, masks 0 background / 1 pet / 2 border), look at
it, train a small U-Net written from scratch and submit, then fine-tune an
ImageNet-pretrained U-Net (`segmentation_models_pytorch`, ResNet-18 encoder)
and submit again. About 10 minutes on a free Colab T4.

The from-scratch model and the training loop are module constants
(`MODEL_SRC`, `TRAIN_SRC`): the competition's `reference_solution.py` runs the
same text to make the benchmark file, so the benchmark is this notebook's
first submission.

The challenge id comes from the competitions lockfile, so the competition is
built before the notebook is written:

    uv run python courseware/tools/notebooks/mlp_s7_segmentation.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_notebooks as nb  # noqa: E402  (helpers only; never call its main)

md, code = nb.md, nb.code

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = (REPO / "website" / "public" / "modules"
       / "ms2a-machine-learning-practice" / "challenges")
NAME = "mlp-s7-pet-segmentation.ipynb"
PACKAGE = "mlp-s7-pet-segmentation"
LOCKFILE = REPO / "courseware" / "competitions" / ".mlarena-state.json"
BASE_URL = "https://ml-arena.com"
BADGE = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
COLAB = (f"https://colab.research.google.com/github/{nb.GITHUB}/blob/{nb.BRANCH}/"
         f"website/public/modules/ms2a-machine-learning-practice/challenges/{NAME}")

# ---------------------------------------------------------------- shared code
# Run verbatim by ../../competitions/mlp-s7-pet-segmentation/reference_solution.py.

MODEL_SRC = '''\
def block(c_in, c_out):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU: the U-Net's unit."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

class SmallUNet(nn.Module):
    def __init__(self, n_classes=2, c=16):
        super().__init__()
        self.enc1, self.enc2 = block(3, c), block(c, 2 * c)               # 128, 64
        self.enc3, self.enc4 = block(2 * c, 4 * c), block(4 * c, 8 * c)   # 32, 16
        self.bottom = block(8 * c, 16 * c)                                # 8 x 8
        self.up4, self.dec4 = nn.ConvTranspose2d(16 * c, 8 * c, 2, stride=2), block(16 * c, 8 * c)
        self.up3, self.dec3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2), block(8 * c, 4 * c)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2), block(4 * c, 2 * c)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2), block(2 * c, c)
        self.head = nn.Conv2d(c, n_classes, 1)                            # one score per class and pixel

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottom(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))    # skip connection: concatenate
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))   # the encoder's map of the same size
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)                               # (batch, 2, 128, 128) logits
'''

TRAIN_SRC = '''\
def to_input(images, mean=None, std=None):
    """uint8 (N, H, W, 3) -> float tensor (N, 3, H, W) in [0, 1], optionally normalised."""
    x = torch.as_tensor(images).permute(0, 3, 1, 2).float().contiguous() / 255
    if mean is not None:
        x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    return x

def train(model, epochs, lr, mean=None, std=None, batch_size=32):
    """Adam + cross-entropy, border pixels ignored, random horizontal flips.
    Keeps the weights of the epoch with the best validation mIoU."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(X_tr) // batch_size))
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)       # label 2 = border: no gradient there
    X, Y = to_input(X_tr, mean, std), torch.as_tensor(M_tr).long()
    gen = torch.Generator().manual_seed(0)
    history, best = [], (-1.0, None)
    for epoch in range(epochs):
        model.train()
        t0, losses = time.time(), []
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X) - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            x, y = X[idx], Y[idx]
            flip = torch.rand(len(idx), generator=gen) < 0.5   # augmentation: mirror half the batch
            x[flip], y[flip] = x[flip].flip(-1), y[flip].flip(-1)
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        val = sm.score_masks(M_val, predict(model, X_val, mean, std))
        history.append({"epoch": epoch + 1, "loss": np.mean(losses), **val})
        print(f"epoch {epoch + 1:2d}  loss {np.mean(losses):.3f}  val mIoU {val['miou']:.4f}"
              f"  (pet IoU {val['iou_pet']:.3f})  {time.time() - t0:.0f}s")
        if val["miou"] > best[0]:
            best = (val["miou"], {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return pd.DataFrame(history)

@torch.no_grad()
def predict(model, images, mean=None, std=None, batch_size=64):
    """uint8 images (N, H, W, 3) -> bool pet masks (N, H, W)."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        logits = model(to_input(images[i:i + batch_size], mean, std).to(device))
        out.append((logits.argmax(1) == 1).cpu().numpy())
    return np.concatenate(out)
'''

SCRATCH_RUN = "SmallUNet(c=16)", 15, 3e-3      # model, epochs, learning rate
FINETUNE_EPOCHS, FINETUNE_LR = 8, 1e-3


def challenge_id() -> int:
    state = json.loads(LOCKFILE.read_text())
    try:
        return state[BASE_URL]["competitions"][PACKAGE]["id"]
    except KeyError:
        raise SystemExit(f"{PACKAGE} is not in {LOCKFILE.name} for {BASE_URL}: "
                         "build the competition first (make competitions)") from None


def build(cid: int) -> dict:
    model, epochs, lr = SCRATCH_RUN
    cells = [
        md(
            "# Pet Segmentation — a mask for every pixel",
            "",
            f"{BADGE}({COLAB})",
            "",
            "**MS2A — Machine Learning Practice · Session 7, Computer Vision 2**",
            "",
            "Classification gives one label per image; **segmentation** gives one per *pixel*. Here: which",
            "pixels of a photo are the cat or the dog, and which are background —",
            f"[challenge {cid}](https://ml-arena.com/viewchallenge/{cid}) on ML-Arena, scored by **mean IoU**.",
            "",
            "| Section | What you do |",
            "|---|---|",
            "| 0–2 | Download the data, look at the images and the masks |",
            "| 3 | Hold out a validation set, write down the metric |",
            "| 4 | **A small U-Net from scratch** — first submission |",
            "| 5 | **Fine-tune a pretrained U-Net** (ImageNet ResNet-18 encoder) — second submission |",
            "",
            "> In Colab: **Runtime → Change runtime type → T4 GPU**, then run the cells top to bottom",
            "> (about 10 minutes). The only thing you must edit is your API token (your **Profile** page",
            "> on ml-arena.com).",
        ),
        md("## 0. Setup"),
        code(
            '!pip install -q "mlarena-sdk>=4.4" segmentation-models-pytorch',
            "",
            "import mlarena",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import random, sys, time, zipfile",
            "import torch, torch.nn as nn, torch.nn.functional as F",
            "from pathlib import Path",
            "from PIL import Image",
            "",
            'API_TOKEN = "mlk_user_REPLACE_ME"   # <-- paste your token (Profile page)',
            f"CHALLENGE_ID = {cid}",
            "",
            'client = mlarena.connect(api_key=API_TOKEN, base_url="https://ml-arena.com")',
            'device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"',
            "torch.manual_seed(0)",
            'print("device:", device)',
        ),
        md(
            "## 1. Get the data",
            "",
            "`train.zip` holds the training photos (`images/<id>.jpg`) and their masks (`masks/<id>.png`),",
            "`test_images.zip` the 1,500 photos you predict. `seg_metric.py` is the leaderboard's own",
            "code: the submission format and the metric. We import it to score locally.",
        ),
        code(
            'client.download_dataset(CHALLENGE_ID, "data/")',
            'zipfile.ZipFile("data/train.zip").extractall("data/train")',
            'zipfile.ZipFile("data/test_images.zip").extractall("data/test")',
            'sys.path.insert(0, "data")',
            "import seg_metric as sm",
            "",
            'train_ids = sorted(p.stem for p in Path("data/train/images").glob("*.jpg"))',
            'test_ids = sorted(p.stem for p in Path("data/test/images").glob("*.jpg"))',
            'X_all = np.stack([np.array(Image.open(f"data/train/images/{i}.jpg")) for i in train_ids])',
            'M_all = np.stack([np.array(Image.open(f"data/train/masks/{i}.png")) for i in train_ids])',
            'X_test = np.stack([np.array(Image.open(f"data/test/images/{i}.jpg")) for i in test_ids])',
            'print("images", X_all.shape, X_all.dtype, "| masks", M_all.shape, "values", np.unique(M_all))',
            'print("test images", X_test.shape)',
        ),
        md(
            "## 2. Look at it",
            "",
            "A mask has one value per pixel: **0** background, **1** pet, **2** a thin **border** band",
            "around the pet, where the annotators could not decide. The border is **not scored**, and we",
            "will not train on it either.",
        ),
        code(
            "colors = np.array([[0, 0, 0], [255, 140, 0], [255, 255, 255]], dtype=np.uint8)  # bg, pet, border",
            "",
            "def show(images, masks_list, titles, n=6):",
            '    """One column per image: the photo, then one row per mask set (labels 0/1/2 or bool)."""',
            "    rows = 1 + len(masks_list)",
            "    fig, axes = plt.subplots(rows, n, figsize=(2.2 * n, 2.2 * rows))",
            "    for j in range(n):",
            "        axes[0, j].imshow(images[j])",
            "        for r, masks in enumerate(masks_list, start=1):",
            "            axes[r, j].imshow(colors[masks[j].astype(np.uint8)])",
            "        for r in range(rows):",
            "            axes[r, j].axis(\"off\")",
            "    for r, t in enumerate([\"photo\"] + titles):",
            "        axes[r, 0].set_title(t, fontsize=9, loc=\"left\")",
            "    plt.tight_layout(); plt.show()",
            "",
            "pick = random.Random(0).sample(range(len(X_all)), 6)",
            'show(X_all[pick], [M_all[pick]], ["mask: black background, orange pet, white border"])',
        ),
        md("How much of an image is pet? This decides what a trivial answer scores."),
        code(
            "scored = M_all != 2",
            "pet_share = ((M_all == 1) & scored).sum((1, 2)) / scored.sum((1, 2))",
            'plt.hist(pet_share, bins=50); plt.xlabel("share of pet pixels per image"); plt.show()',
            'print(f"pet = {(M_all == 1).sum() / scored.sum():.1%} of the scored pixels")',
            'print("predict all background:", sm.score_masks(M_all, np.zeros(M_all.shape, bool)))',
        ),
        md(
            "## 3. Validation set and metric",
            "",
            "The test masks are private, so we keep 15 % of the training images aside. For each class *c*",
            "(background, pet), over all scored pixels of all images together:",
            "",
            "$$\\text{IoU}(c) = \\frac{|\\text{pred}=c \\;\\cap\\; \\text{true}=c|}{|\\text{pred}=c \\;\\cup\\; \\text{true}=c|}"
            "\\qquad \\text{score} = \\frac{\\text{IoU(background)} + \\text{IoU(pet)}}{2}$$",
            "",
            "Predicting *all background* already gets IoU(background) ≈ 0.66, so the floor is about 0.33,",
            "not 0. `sm.score_masks(true, pred)` computes it exactly as the leaderboard does.",
        ),
        code(
            "idx = np.random.RandomState(0).permutation(len(X_all))",
            "n_val = len(idx) * 15 // 100",
            "X_val, M_val = X_all[idx[:n_val]], M_all[idx[:n_val]]",
            "X_tr, M_tr = X_all[idx[n_val:]], M_all[idx[n_val:]]",
            'print(len(X_tr), "train,", len(X_val), "validation")',
        ),
        md(
            "## 4. A small U-Net from scratch",
            "",
            "The **encoder** halves the resolution four times (128 → 8) while widening the channels:",
            "it sees more and more context, at a coarser and coarser grid. The **decoder** goes back up",
            "with transposed convolutions, and at each scale **concatenates** the encoder's map of the same",
            "size — the *skip connections* that give back the fine detail the bottleneck lost.",
            "The last 1×1 convolution outputs 2 scores per pixel (background, pet).",
        ),
        code(MODEL_SRC.rstrip("\n"), "",
             f"model = {model}",
             'print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")',
             "print(model(torch.zeros(2, 3, 128, 128)).shape)"),
        md(
            "The loop: Adam with a one-cycle learning rate, **cross-entropy per pixel** with",
            "`ignore_index=2` so the border band gives no gradient, random horizontal flips as the only",
            "augmentation, and the weights of the best validation epoch kept.",
        ),
        code(TRAIN_SRC.rstrip("\n")),
        code(f"history = train(model, epochs={epochs}, lr={lr})",
             'history.plot(x="epoch", y=["loss", "miou"], secondary_y="miou", figsize=(7, 3)); plt.show()'),
        md("Validation predictions next to the truth. Where does it go wrong?"),
        code(
            "pred_val = predict(model, X_val)",
            "print(sm.score_masks(M_val, pred_val))",
            "show(X_val[:6], [M_val[:6], pred_val[:6]], [\"truth\", \"small U-Net\"])",
        ),
        md(
            "### 4.1 First submission",
            "",
            "One row per test image: `image_id,mask_rle`, the pet pixels run-length encoded",
            "(`sm.rle_encode`). The leaderboard decodes it with `sm.rle_decode`.",
        ),
        code(
            "def submit(pred, name):",
            '    """pred: bool masks (1500, 128, 128) in test_ids order. Writes submission.csv and submits it."""',
            '    pd.DataFrame({"image_id": test_ids, "mask_rle": [sm.rle_encode(m) for m in pred]})\\',
            '      .to_csv("submission.csv", index=False)',
            '    assert len(sm.read_rle_csv("submission.csv", image_ids=test_ids)) == len(test_ids)  # the format check',
            '    print(client.submit(challenge_id=CHALLENGE_ID, files=["submission.csv"], submission_name=name))',
            "",
            'submit(predict(model, X_test), "small-unet-scratch")',
        ),
        md(
            "## 5. Fine-tune a pretrained U-Net",
            "",
            "Same U-Net shape, but the encoder is a **ResNet-18 trained on ImageNet**: it already",
            "knows edges, fur and eyes. Only the decoder starts from random weights. Two things change:",
            "the inputs are normalised with ImageNet's mean and std (what the encoder saw in training),",
            "and the learning rate is lower, so the pretrained features are adjusted rather than erased.",
        ),
        code(
            "import segmentation_models_pytorch as smp",
            "",
            "IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)",
            'pretrained = smp.Unet("resnet18", encoder_weights="imagenet", classes=2)',
            'print(f"{sum(p.numel() for p in pretrained.parameters()) / 1e6:.1f} M parameters")',
            f"history_ft = train(pretrained, epochs={FINETUNE_EPOCHS}, lr={FINETUNE_LR},"
            " mean=IMAGENET_MEAN, std=IMAGENET_STD)",
        ),
        code(
            "pred_val_ft = predict(pretrained, X_val, IMAGENET_MEAN, IMAGENET_STD)",
            'print("scratch   ", sm.score_masks(M_val, pred_val))',
            'print("fine-tuned", sm.score_masks(M_val, pred_val_ft))',
            "show(X_val[:6], [M_val[:6], pred_val[:6], pred_val_ft[:6]], [\"truth\", \"small U-Net\", \"fine-tuned\"])",
        ),
        md(
            "**Questions**",
            "- The pretrained encoder never saw a segmentation mask. Why does it help this much, and with",
            "  how many epochs?",
            "- Which images does the fine-tuned model still get wrong? Small pets, pets cut by the frame,",
            "  cluttered backgrounds?",
            "",
            "Submit it:",
        ),
        code('submit(predict(pretrained, X_test, IMAGENET_MEAN, IMAGENET_STD), "unet-resnet18-finetuned")'),
        md(
            "## Going further",
            "",
            "Change one thing, compare the validation mIoU, submit only if it improved:",
            "- a bigger encoder: `smp.Unet(\"resnet34\", ...)`, or another architecture: `smp.DeepLabV3Plus`;",
            "- more epochs, or retraining on all the training images once the settings are chosen;",
            "- **test-time augmentation**: average the logits of the image and its mirror;",
            "- upsample the inputs to 256 × 256 for the pretrained encoder, and the logits back to 128;",
            "- a Dice loss added to the cross-entropy.",
        ),
    ]
    out = nb.notebook(cells)
    out["metadata"]["accelerator"] = "GPU"
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / NAME
    path.write_text(json.dumps(build(challenge_id()), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
