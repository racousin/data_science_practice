"""Where the medical courses' Colab notebooks and datasets live.

Both French medical courses (`python-medecine`, `ia-medecine`) publish their
notebooks and data to the public repository github.com/racousin/ai_for_medicine,
checked out (gitignored) at the root of this repo as `ai_for_medicine/`:

    ai_for_medicine/
    ├── python-medecine/{notebooks,donnees,images}/
    └── ia-medecine/{notebooks,donnees}/

The notebook builders write into that checkout; Colab badges and data URLs
point at its `main` branch, so a notebook works once the checkout is pushed.
"""
from __future__ import annotations

import pathlib

GITHUB = "racousin/ai_for_medicine"
BRANCH = "main"
ROOT = pathlib.Path(__file__).resolve().parents[3] / "ai_for_medicine"


def out_dir(course: str, kind: str) -> pathlib.Path:
    """Local folder for `kind` (notebooks, donnees, images) of `course`."""
    return ROOT / course / kind


def colab(course: str, notebook: str) -> str:
    """Open-in-Colab URL of a notebook of `course`."""
    return (f"https://colab.research.google.com/github/{GITHUB}/blob/{BRANCH}/"
            f"{course}/notebooks/{notebook}")


def raw(course: str, path: str) -> str:
    """Raw download URL of a file under `course/` (e.g. `donnees/x.csv`)."""
    return f"https://raw.githubusercontent.com/{GITHUB}/{BRANCH}/{course}/{path}"
