"""Third-party figures used by the Session 4 lessons of
`ms2a-machine-learning-practice` (regularization, dropout, skip-connections).

They are the classic diagrams of *Dive into Deep Learning* — A. Zhang, Z. C.
Lipton, M. Li, A. J. Smola, https://d2l.ai — whose text and figures are
licensed CC BY-SA 4.0 (https://github.com/d2l-ai/d2l-en/blob/master/LICENSE).
The licence requires attribution, so every lesson that shows one of these
carries the credit line in `CREDIT` under the image. Nothing is modified: the
SVG from the book's repository is rendered to PNG at 2.5x, on white.

Re-run with (needs `rsvg-convert`, `brew install librsvg`):

    python3 courseware/tools/figures/ms2a_s4_d2l.py
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
import urllib.request

SOURCE = "https://raw.githubusercontent.com/d2l-ai/d2l-en/master/img/{}.svg"
OUT = (pathlib.Path(__file__).resolve().parents[2] / "content"
       / "ms2a-machine-learning-practice" / "assets" / "nn")

CREDIT = ("*Figure: A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, "
          "[Dive into Deep Learning](https://d2l.ai), "
          "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*")

# book file -> file in assets/nn/
FIGURES = {
    "capacity-vs-error": "d2l-capacity-vs-error.png",
    "dropout2": "d2l-dropout-before-and-after.png",
    "residual-block": "d2l-regular-vs-residual-block.png",
    "resnet-block": "d2l-resnet-block-and-projection.png",
    "densenet-block": "d2l-add-vs-concatenate.png",
    "functionclasses": "d2l-nested-function-classes.png",
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for name, target in FIGURES.items():
            svg = pathlib.Path(tmp) / f"{name}.svg"
            urllib.request.urlretrieve(SOURCE.format(name), svg)
            subprocess.run(["rsvg-convert", "-z", "2.5", "-b", "white",
                            "-o", str(OUT / target), str(svg)], check=True)
            print(f"wrote assets/nn/{target}")
    print("credit line for the lessons:\n" + CREDIT)


if __name__ == "__main__":
    main()
