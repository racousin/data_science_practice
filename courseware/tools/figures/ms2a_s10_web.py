"""Third-party figures used by the Session 10 lessons of
`ms2a-machine-learning-practice` (module `s11-docker-and-deployment`, the
Model Deployment Workshop).

Each figure is downloaded unmodified into `assets/deploy/`. An SVG is rendered
to PNG at 2x on white with `rsvg-convert`; the Docker WebP is re-encoded to PNG
with Pillow; a PNG is copied as is. Every lesson that shows one of them
carries the credit line from `FIGURES` under the image.

- load-balancer.png: HFWMan, Wikimedia Commons, CC BY-SA 4.0.
- onprem-iaas-paas-saas.png: P. Rosati and T. Lynn (2020), Wikimedia Commons,
  public domain (marked ineligible for copyright on its file page). Preferred
  over the CC0 `Cloud_computing_layers.svg` because it shows who manages each
  layer, which is the point of the slide.
- docker-architecture.png: Docker Docs (docker/docs, Apache-2.0).
- penguins.png, culmen-depth.png: artwork by @allison_horst (palmerpenguins).

Re-run from `courseware/` with (needs `rsvg-convert`, `brew install librsvg`):

    uv run --with pillow python tools/figures/ms2a_s10_web.py
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
import urllib.request

from PIL import Image

OUT = (pathlib.Path(__file__).resolve().parents[2] / "content"
       / "ms2a-machine-learning-practice" / "assets" / "deploy")

# Wikimedia refuses requests without a descriptive User-Agent.
USER_AGENT = "ml-arena-courseware-figures/1.0 (+https://ml-arena.com)"

CC_BY_SA = "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)"
PENGUIN_CREDIT = ("*Artwork by [@allison_horst]"
                  "(https://allisonhorst.github.io/palmerpenguins/).*")

# file in assets/deploy/ -> (source URL, credit line for the lesson)
FIGURES = {
    "load-balancer.png": (
        "https://upload.wikimedia.org/wikipedia/commons/1/1f/"
        "Load_Balancing_Cluster_%28NAT%29.svg",
        "*Figure: HFWMan, [Wikimedia Commons](https://commons.wikimedia.org/"
        f"wiki/File:Load_Balancing_Cluster_(NAT).svg), {CC_BY_SA}.*"),
    "onprem-iaas-paas-saas.png": (
        "https://upload.wikimedia.org/wikipedia/commons/f/ff/"
        "Comparison_of_on-premise%2C_IaaS%2C_PaaS%2C_and_SaaS.png",
        "*Figure: P. Rosati and T. Lynn (2020), [Wikimedia Commons]"
        "(https://commons.wikimedia.org/wiki/File:Comparison_of_on-premise,"
        "_IaaS,_PaaS,_and_SaaS.png), public domain.*"),
    "docker-architecture.png": (
        "https://docs.docker.com/get-started/images/docker-architecture.webp",
        "*Figure: [Docker Docs](https://docs.docker.com/get-started/"
        "docker-overview/), [Apache-2.0](https://github.com/docker/docs/"
        "blob/main/LICENSE).*"),
    "penguins.png": (
        "https://allisonhorst.github.io/palmerpenguins/reference/figures/"
        "lter_penguins.png",
        PENGUIN_CREDIT),
    "culmen-depth.png": (
        "https://allisonhorst.github.io/palmerpenguins/reference/figures/"
        "culmen_depth.png",
        PENGUIN_CREDIT),
}


def fetch(url: str, path: pathlib.Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, path.open("wb") as f:
        shutil.copyfileobj(response, f)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for target, (url, credit) in FIGURES.items():
            suffix = url.rsplit(".", 1)[1]
            source = pathlib.Path(tmp) / f"source.{suffix}"
            fetch(url, source)
            if suffix == "svg":
                subprocess.run(["rsvg-convert", "-z", "2", "-b", "white",
                                "-o", str(OUT / target), str(source)],
                               check=True)
            elif suffix == "webp":
                Image.open(source).save(OUT / target)
            elif suffix == "png":
                shutil.copyfile(source, OUT / target)
            else:
                raise ValueError(f"no converter for .{suffix}: {url}")
            print(f"wrote assets/deploy/{target}\n  {credit}")


if __name__ == "__main__":
    main()
