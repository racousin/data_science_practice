"""Third-party figures for `s6-computer-vision-2/self-supervised-learning.md`
of `ms2a-machine-learning-practice` (Session 7 — Computer Vision 2).

Every figure here is reused under its licence, unmodified except for
rendering to PNG, resizing, and — for the two DINO grids — laying the paper's
own panels side by side as the paper does. The credit line each licence
requires is in `CREDITS` and is printed at the end; the lesson carries it
under the image.

- Liu et al., "Self-supervised Learning: Generative or Contrastive" (2021),
  figure on Wikimedia Commons, CC BY 4.0.
- "Positive and negative pairs for contrastive learning", Wikimedia Commons
  user A potato hater, CC BY-SA 4.0.
- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers"
  (DINO, ICCV 2021), arXiv:2104.14294, CC BY 4.0: figures from the arXiv
  source; the attention-map banner from github.com/facebookresearch/dino
  (Apache-2.0).
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE, CVPR
  2022), arXiv:2111.06377, CC BY 4.0: figures from the arXiv source.

The DINOv2 paper is not under an open licence, so its figures are not reused:
`ms2a_s6_dinov2.py` computes the DINOv2 figures from the released weights.

Re-run with (needs `pdftoppm` and `rsvg-convert`: `brew install poppler librsvg`):

    uv run --no-project --with pillow python courseware/tools/figures/ms2a_s6_ssl_sources.py
"""

from __future__ import annotations

import io
import pathlib
import subprocess
import tarfile
import tempfile
import urllib.request

from PIL import Image, ImageDraw, ImageFont

OUT = (pathlib.Path(__file__).resolve().parents[2] / "content"
       / "ms2a-machine-learning-practice" / "assets" / "cv")
UA = {"User-Agent": "ms2a-courseware/1.0 (course figures; racousin)"}

COMMONS = "https://upload.wikimedia.org/wikipedia/commons/"
DINO_SRC = "https://arxiv.org/e-print/2104.14294"
MAE_SRC = "https://arxiv.org/e-print/2111.06377"
DINO_BANNER = ("https://raw.githubusercontent.com/facebookresearch/dino/"
               "main/.github/attention_maps.png")

DINO_CREDIT = ("*Figure: M. Caron et al., [Emerging Properties in Self-Supervised "
               "Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021, "
               "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*")
MAE_CREDIT = ("*Figure: K. He et al., [Masked Autoencoders Are Scalable Vision "
              "Learners](https://arxiv.org/abs/2111.06377), CVPR 2022, "
              "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*")
CREDITS = {
    "ssl-paradigms.png": (
        "*Figure: X. Liu et al., [Self-supervised Learning: Generative or "
        "Contrastive](https://arxiv.org/abs/2006.08218), via "
        "[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Supervised,_Unsupervised,_and_Self-Supervised_Learning.jpg), "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*"),
    "ssl-contrastive-pairs.png": (
        "*Figure: A potato hater, "
        "[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Positive_and_negative_pairs_for_contrastive_learning.png), "
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*"),
    "dino-attention-banner.png": (
        "*Figure: [facebookresearch/dino](https://github.com/facebookresearch/dino), "
        "Apache-2.0.*"),
    "dino-self-distillation.png": DINO_CREDIT,
    "dino-collapse.png": DINO_CREDIT,
    "dino-heads.png": DINO_CREDIT,
    "dino-vs-supervised.png": DINO_CREDIT,
    "mae-architecture.png": MAE_CREDIT,
    "mae-samples.png": MAE_CREDIT,
    "mae-masking-ratio.png": MAE_CREDIT,
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
        return r.read()


def save(img: Image.Image, name: str, width: int = 1600) -> None:
    img = img.convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    img.save(OUT / name, optimize=True)
    print(f"wrote assets/cv/{name} {img.size}")


def pdf_to_png(pdf: pathlib.Path, scale_dpi: int = 220) -> Image.Image:
    stem = pdf.with_suffix("")
    subprocess.run(["pdftoppm", "-png", "-r", str(scale_dpi), "-singlefile",
                    str(pdf), str(stem)], check=True)
    return Image.open(f"{stem}.png")


def grid(rows: list[list[Image.Image]], cell: int, gap: int,
         labels: list[str] | None = None) -> Image.Image:
    """Square cells, `gap` px of white between them, an optional row label."""
    pad = 230 if labels else 0
    w = pad + len(rows[0]) * cell + (len(rows[0]) - 1) * gap
    h = len(rows) * cell + (len(rows) - 1) * gap
    out = Image.new("RGB", (w, h), "white")
    font = ImageFont.load_default(size=30)
    draw = ImageDraw.Draw(out)
    for r, row in enumerate(rows):
        y = r * (cell + gap)
        if labels:
            draw.text((pad - 16, y + cell // 2), labels[r], fill="black",
                      font=font, anchor="rm")
        for c, im in enumerate(row):
            out.paste(im.convert("RGB").resize((cell, cell), Image.LANCZOS),
                      (pad + c * (cell + gap), y))
    return out


def unpack(url: str, into: pathlib.Path) -> pathlib.Path:
    into.mkdir()
    with tarfile.open(fileobj=io.BytesIO(fetch(url)), mode="r:gz") as tar:
        tar.extractall(into, filter="data")
    return into


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)

        save(Image.open(io.BytesIO(fetch(
            COMMONS + "e/e4/Supervised%2C_Unsupervised%2C_and_Self-Supervised_Learning.jpg"))),
            "ssl-paradigms.png")
        save(Image.open(io.BytesIO(fetch(
            COMMONS + "4/4e/Positive_and_negative_pairs_for_contrastive_learning.png"))),
            "ssl-contrastive-pairs.png", width=1000)
        save(Image.open(io.BytesIO(fetch(DINO_BANNER))), "dino-attention-banner.png")

        dino = unpack(DINO_SRC, tmp / "dino")
        # Fig. 2: self-distillation with no labels.
        save(pdf_to_png(dino / "test.pdf"), "dino-self-distillation.png", width=900)
        # Fig. 7 (ablations): collapse study, centering vs sharpening.
        save(pdf_to_png(dino / "figure_collapse.pdf"), "dino-collapse.png", width=1400)
        # Fig. 3: the six heads of the last layer, image | attention, four pairs.
        pairs = [("1935", "bnw-1935"), ("2642", "bnw-2642"),
                 ("3425", "bnw-3425"), ("2032", "bnw-2032")]
        rows = [[Image.open(dino / f"{p}.png") for pair in half for p in pair]
                for half in (pairs[:2], pairs[2:])]
        save(grid(rows, cell=360, gap=12), "dino-heads.png")
        # Fig. 4: one head's thresholded attention, supervised ViT vs DINO.
        sup = ["fig4-sup-857mask-head1", "fig4-sup-1425mask-head1",
               "fig4-sup-1339mask-head1", "fig4-sup-967mask-head2", "2963_sup"]
        dn = ["fig4-dino-857mask-head2", "fig4-dino-1425mask-head2",
              "fig4-dino-1339mask-head3", "fig4-dino-967mask-head0", "2963_dino_fig3"]
        save(grid([[Image.open(dino / f"{n}.png") for n in sup],
                   [Image.open(dino / f"{n}.png") for n in dn]],
                  cell=300, gap=10, labels=["Supervised", "DINO"]),
             "dino-vs-supervised.png")

        mae = unpack(MAE_SRC, tmp / "mae")
        save(pdf_to_png(mae / "fig" / "arch.pdf"), "mae-architecture.png", width=1400)
        save(pdf_to_png(mae / "fig" / "samples.pdf"), "mae-samples.png")
        ft = pdf_to_png(mae / "fig" / "ratio_ft.pdf")
        lin = pdf_to_png(mae / "fig" / "ratio_linear.pdf")
        both = Image.new("RGB", (max(ft.width, lin.width), ft.height + lin.height + 40), "white")
        both.paste(ft, (0, 0))
        both.paste(lin, (0, ft.height + 40))
        save(both, "mae-masking-ratio.png", width=1400)

    print("\ncredit lines for the lesson:")
    for name, credit in CREDITS.items():
        print(f"{name}: {credit}")


if __name__ == "__main__":
    main()
