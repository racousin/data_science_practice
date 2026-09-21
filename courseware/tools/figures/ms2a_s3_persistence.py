"""Figures for the `ms2a-machine-learning-practice` lesson
`s3-tabular-models/saving-and-loading-models.md`.

Every image this script writes is committed under
`content/ms2a-machine-learning-practice/assets/tabular/` with the prefix
`persist-`. Re-run with:

    uv run --no-project --python 3.12 --with scikit-learn==1.8.0 \
        --with lightgbm==4.6.0 --with xgboost==3.2.0 --with skops \
        --with pandas --with matplotlib \
        python courseware/tools/figures/ms2a_s3_persistence.py

The pins are Lab 3's runtime 182 (Python 3.12, scikit-learn 1.8.0,
XGBoost 3.2.0, LightGBM 4.6.0): the lesson tells students to train with the
versions the agent is loaded with, so the figures are drawn with them too.

Pass `--only serve,state,skew,formats,versions,claims` to run a subset.
`versions` shells out to `uv` six times, one environment per scikit-learn
version, and takes a minute or two with a warm cache; the rest take seconds.

Three of the five figures are measured on California housing
(`fetch_california_housing`, cached by scikit-learn after the first download),
split 80/20 with `random_state=0`:

* `skew` — a scaler + 10-NN pipeline, served correctly and served with the
  scaler refitted on each request batch, over random batches and over
  batches of neighbouring districts.
* `formats` — a random forest, a LightGBM and an XGBoost model written in
  every format the lesson names: file size, load time, and whether the
  reloaded model predicts the same numbers.
* `versions` — a pipeline, a forest and a histogram booster saved under
  scikit-learn 1.4.2 (numpy 1.26), 1.6.1 and 1.8.0, then each loaded under
  all three.

Every number the lesson quotes is printed by this script and asserted here, so
a claim in the markdown that stops being true fails the run rather than the
reader. Load times depend on the machine and are asserted only to be small.

`serve` is a diagram; `state` is a diagram whose sizes are read off a real fit.
`claims` draws nothing: it checks what the lesson says in prose and in code —
the column-order error and its silent numpy twin, the one-row scaler, the
pickle that prints while loading, a binary LightGBM Booster's output, and the
`lambda` and `__main__` errors, each reproduced in a fresh process.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import statistics
import subprocess
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "content"
    / "ms2a-machine-learning-practice"
    / "assets"
    / "tabular"
)

# The same tokens as the other ms2a_s3_* scripts, so Session 3's authored
# figures read as one set.
INK = "#1f2933"
MUTED = "#6b7684"
RULE = "#dfe3e8"
ACCENT = "#2f6f9f"
ACCENT_BG = "#e3eef7"
WARM = "#c1553b"
WARM_BG = "#f6e1dc"
OK_BG = "#e6f0e4"
OK = "#4a7c43"
AMBER_BG = "#fbefd5"
AMBER = "#a86b12"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "DejaVu Sans",
        "font.size": 11.5,
        "legend.frameon": False,
    }
)


def save(fig, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    p = ASSETS / name
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ASSETS.parents[3])}")


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def california():
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=0)


def box(ax, x, y, w, h, face, edge=None, lw=1.2, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=face, edgecolor=edge or face,
                                linewidth=lw))


def arrow(ax, a, b, colour=INK, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=15,
                                 color=colour, lw=lw))


# --------------------------------------------------------------------------- #
# (1) fit once, predict elsewhere — a diagram
# --------------------------------------------------------------------------- #


def serve_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # training process
    box(ax, 0.00, 0.20, 0.29, 0.66, ACCENT_BG)
    ax.text(0.145, 0.80, "Training", ha="center", fontsize=14, color=ACCENT,
            weight="bold")
    ax.text(0.145, 0.735, "your notebook, Colab", ha="center", fontsize=10.5,
            color=MUTED)
    for i, line in enumerate(["the training data", "fit the pipeline",
                              "score it, once", "write the files"]):
        ax.text(0.03, 0.62 - i * 0.095, f"{i + 1}.  {line}", fontsize=11.5)

    # the files that cross
    box(ax, 0.375, 0.20, 0.25, 0.66, "white", edge=INK, lw=1.3)
    ax.text(0.50, 0.80, "The files", ha="center", fontsize=14, weight="bold")
    ax.text(0.50, 0.735, "all that crosses the gap", ha="center",
            fontsize=10.5, color=MUTED)
    files = [("model.joblib", "fitted pipeline"),
             ("features.py", "your own functions"),
             ("meta.json", "columns, versions, score"),
             ("agent.py", "loads them")]
    for i, (name, what) in enumerate(files):
        y0 = 0.60 - i * 0.10
        ax.text(0.395, y0, name, fontsize=11.5, family="DejaVu Sans Mono")
        ax.text(0.395, y0 - 0.042, what, fontsize=9.5, color=MUTED)

    # serving process
    box(ax, 0.71, 0.20, 0.29, 0.66, WARM_BG)
    ax.text(0.855, 0.80, "Serving", ha="center", fontsize=14, color=WARM,
            weight="bold")
    ax.text(0.855, 0.735, "another process, another machine", ha="center",
            fontsize=10.5, color=MUTED)
    for i, line in enumerate(["import agent.py", "Agent(): load, once",
                              "predict(request)", "… every 6 hours"]):
        ax.text(0.74, 0.62 - i * 0.095, f"{i + 1}.  {line}", fontsize=11.5)

    arrow(ax, (0.295, 0.53), (0.37, 0.53))
    arrow(ax, (0.63, 0.53), (0.705, 0.53))

    ax.text(0.0, 0.10, "Does not cross", fontsize=11.5, color=WARM,
            weight="bold")
    ax.text(0.0, 0.02, "the training data · anything defined in the "
                       "notebook · your library versions · your working "
                       "directory", fontsize=11.5, color=INK)
    ax.set_ylim(0, 0.9)
    save(fig, "persist-train-serve.png")


# --------------------------------------------------------------------------- #
# (2) what a fitted pipeline holds — sizes read off a real fit
# --------------------------------------------------------------------------- #


def fitted_state() -> None:
    import joblib

    X_tr, _, y_tr, _ = california()
    knn = make_pipeline(StandardScaler(), KNeighborsRegressor(10))
    knn.fit(X_tr, y_tr)
    ridge = make_pipeline(StandardScaler(), Ridge()).fit(X_tr, y_tr)
    with tempfile.TemporaryDirectory() as d:
        joblib.dump(knn, f"{d}/knn.joblib")
        joblib.dump(ridge, f"{d}/ridge.joblib")
        knn_bytes = os.path.getsize(f"{d}/knn.joblib")
        ridge_bytes = os.path.getsize(f"{d}/ridge.joblib")
    sc, nn, rg = knn[0], knn[-1], ridge[-1]
    print(f"    10-NN pipeline file {knn_bytes / 1e6:.2f} MB, "
          f"ridge pipeline file {ridge_bytes / 1e3:.1f} kB")
    print(f"    _fit_X {nn._fit_X.shape} {nn._fit_X.nbytes / 1e6:.2f} MB, "
          f"method {nn._fit_method}")
    print(f"    mean_ MedInc {sc.mean_[0]:.2f}, scale_ MedInc {sc.scale_[0]:.2f}")
    # the lesson quotes these
    assert nn._fit_X.shape == (16512, 8)
    assert nn._fit_method == "kd_tree"
    assert 2.4e6 < knn_bytes < 2.7e6, knn_bytes      # "2.5 MB"
    assert 1.4e3 < ridge_bytes < 1.8e3, ridge_bytes  # "1.6 kB"
    assert round(sc.mean_[0], 2) == 3.88 and round(sc.scale_[0], 2) == 1.91
    assert 1500 < knn_bytes / ridge_bytes < 1700, knn_bytes / ridge_bytes

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def step(x, w, title, rows, face, edge):
        box(ax, x, 0.22, w, 0.62, face)
        ax.text(x + 0.02, 0.76, title, fontsize=13, weight="bold", color=edge)
        for i, (attr, what) in enumerate(rows):
            y0 = 0.62 - i * 0.12
            ax.text(x + 0.02, y0, attr, fontsize=11.5,
                    family="DejaVu Sans Mono")
            ax.text(x + 0.02, y0 - 0.05, what, fontsize=9.5, color=MUTED)

    step(0.0, 0.40, "StandardScaler",
         [("mean_", f"(8,) — MedInc: {sc.mean_[0]:.2f}"),
          ("scale_", f"(8,) — MedInc: {sc.scale_[0]:.2f}"),
          ("feature_names_in_", "the 8 column names, in order")],
         ACCENT_BG, ACCENT)
    arrow(ax, (0.41, 0.53), (0.46, 0.53))
    step(0.47, 0.53, "KNeighborsRegressor(10)",
         [("_fit_X", f"{nn._fit_X.shape} — the scaled training set"),
          ("_y", f"({len(nn._y)},) — its targets"),
          ("_tree", "a KD-tree over the same rows, a second copy")],
         WARM_BG, WARM)
    ax.text(0.0, 0.10,
            f"On disk: {knn_bytes / 1e6:.1f} MB. The same pipeline with a "
            f"Ridge instead of the 10-NN: {ridge_bytes / 1e3:.1f} kB — "
            "coef_ is 8 numbers.",
            fontsize=11.5)
    ax.text(0.0, 0.02, "Every attribute ending in _ was written by fit(). "
                       "That, and nothing else, is what the file has to carry.",
            fontsize=10.5, color=MUTED)
    save(fig, "persist-fitted-state.png")


# --------------------------------------------------------------------------- #
# (3) the scaler refitted at serving time — the silent mistake
# --------------------------------------------------------------------------- #

BATCHES = [1, 3, 10, 45, 100, 300, 1000]


def skew() -> None:
    X_tr, X_te, y_tr, y_te = california()
    pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(10))
    pipe.fit(X_tr, y_tr)
    model = pipe[-1]
    right = rmse(y_te, pipe.predict(X_te))
    raw = rmse(y_te, model.predict(X_te.to_numpy()))
    mean = rmse(y_te, np.full(len(y_te), y_tr.mean()))

    def refit(Xs, ys, b):
        parts = [model.predict(StandardScaler().fit_transform(Xs.iloc[i:i + b]))
                 for i in range(0, len(Xs), b)]
        return rmse(ys, np.concatenate(parts))

    # neighbouring districts: sorted by latitude, then longitude
    order = np.lexsort((X_te["Longitude"].to_numpy(),
                        X_te["Latitude"].to_numpy()))
    region = [refit(X_te.iloc[order], y_te.iloc[order], b) for b in BATCHES]
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(X_te)) for _ in range(3)]
    rand = [float(np.mean([refit(X_te.iloc[p], y_te.iloc[p], b)
                           for p in perms])) for b in BATCHES]

    print(f"    pipeline {right:.3f}  estimator on raw {raw:.3f}  "
          f"predict the mean {mean:.3f}")
    for b, r, q in zip(BATCHES, region, rand):
        print(f"    batch {b:5d}: neighbouring {r:.3f}   random {q:.3f}")
    i45 = BATCHES.index(45)
    # the lesson quotes these
    assert round(right, 3) == 0.625, right
    assert round(raw, 2) == 1.17, raw
    assert round(mean, 2) == 1.14, mean
    assert round(region[i45], 2) == 1.04, region[i45]
    assert round(region[0], 2) == 1.47 and region[0] > mean
    assert round(rand[i45], 2) == 0.69, rand[i45]
    assert region[-1] > right + 0.2, "large regional batches should still lose"
    lost = (region[i45] - right) / (mean - right)
    print(f"    share of the model's gain lost at 45 neighbouring rows: {lost:.0%}")
    assert round(lost, 1) == 0.8, lost

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.plot(BATCHES, region, "o-", color=WARM, lw=2.2,
            label="scaler refitted on the batch — neighbouring districts")
    ax.plot(BATCHES, rand, "o--", color=WARM, lw=1.6, alpha=0.55,
            label="scaler refitted on the batch — random rows")
    ax.axhline(right, color=ACCENT, lw=2.2,
               label=f"the pipeline as fitted — {right:.3f}")
    ax.axhline(mean, color=MUTED, lw=1.4, ls=":",
               label=f"predict the training mean — {mean:.3f}")
    ax.annotate(f"45 rows: {region[i45]:.2f}", (45, region[i45]),
                xytext=(60, 1.23), fontsize=11, color=WARM,
                arrowprops=dict(arrowstyle="-", color=WARM, lw=1))
    ax.set_xscale("log")
    ax.set_xticks(BATCHES)
    ax.set_xticklabels([str(b) for b in BATCHES])
    ax.set_xlabel("rows per request (log scale)")
    ax.set_ylabel("test RMSE (×$100k)")
    ax.set_ylim(0.5, 1.55)
    ax.set_title("California housing, StandardScaler + 10-NN: "
                 "no error, no warning", loc="left", fontsize=12.5)
    tidy(ax)
    ax.legend(loc="upper right", fontsize=10)
    save(fig, "persist-serving-skew.png")


# --------------------------------------------------------------------------- #
# (4) formats: size, load time, and whether the numbers survive
# --------------------------------------------------------------------------- #


def formats() -> None:
    import joblib
    import lightgbm as lgb
    import skops.io as sio
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor

    X_tr, X_te, y_tr, _ = california()
    models = {
        "Random forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, n_jobs=1, random_state=0),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, random_state=0, verbose=-1),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            random_state=0),
    }

    def xgb_load(p):
        m = xgb.XGBRegressor()
        m.load_model(p)
        return m

    def skops_load(p):
        return sio.load(p, trusted=sio.get_untrusted_types(file=p))

    def timed(f, n=5):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            out = f()
            ts.append(time.perf_counter() - t0)
        return out, statistics.median(ts)

    rows = []   # (model, format, MB, load s, identical)
    untrusted = {}
    with tempfile.TemporaryDirectory() as d:
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            p0 = m.predict(X_te)
            fmts = [
                ("joblib", "joblib",
                 lambda p: joblib.dump(m, p), joblib.load),
                ("joblib, compress=3", "joblib",
                 lambda p: joblib.dump(m, p, compress=3), joblib.load),
                ("skops", "skops", lambda p: sio.dump(m, p), skops_load),
            ]
            if name == "LightGBM":
                fmts.append(("native .txt", "txt",
                             lambda p: m.booster_.save_model(p),
                             lambda p: lgb.Booster(model_file=p)))
            if name == "XGBoost":
                fmts.append(("native .json", "json",
                             lambda p: m.save_model(p), xgb_load))
                fmts.append(("native .ubj", "ubj",
                             lambda p: m.save_model(p), xgb_load))
            if name != "XGBoost":
                with tempfile.TemporaryDirectory() as t:
                    sio.dump(m, f"{t}/m.skops")
                    untrusted[name] = sio.get_untrusted_types(
                        file=f"{t}/m.skops")
            for label, ext, dump, load in fmts:
                p = f"{d}/{name[:3]}-{label[:9].strip()}.{ext}"
                dump(p)
                obj, t = timed(lambda: load(p))
                same = bool(np.array_equal(obj.predict(X_te), p0))
                rows.append((name, label, os.path.getsize(p) / 1e6, t, same))
                print(f"    {name:13s} {label:19s} "
                      f"{os.path.getsize(p) / 1e6:6.2f} MB "
                      f"load {t * 1e3:6.1f} ms  identical={same}")
        # a plain pickle is what joblib writes, minus its array handling
        rf = models["Random forest"]
        with open(f"{d}/rf.pkl", "wb") as f:
            pickle.dump(rf, f, protocol=5)
        pkl_mb = os.path.getsize(f"{d}/rf.pkl") / 1e6

    size = {(m, f): s for m, f, s, _, _ in rows}
    load = {(m, f): t for m, f, _, t, _ in rows}
    print(f"    skops untrusted: {untrusted}")
    rf = size[("Random forest", "joblib")]
    rf3 = size[("Random forest", "joblib, compress=3")]
    js, ub = size[("XGBoost", "native .json")], size[("XGBoost", "native .ubj")]
    assert len(X_te) == 4128
    assert round(rf / rf3) == 3, rf / rf3
    assert round(js / ub, 1) == 1.4, js / ub
    assert load[("XGBoost", "native .json")] > load[("XGBoost", "native .ubj")]
    assert untrusted["Random forest"] == ["sklearn.tree._tree.Tree"]
    assert len(untrusted["LightGBM"]) == 3, untrusted["LightGBM"]
    # the lesson quotes these
    assert all(r[4] for r in rows), "a reloaded model predicted differently"
    assert all(r[3] < 0.5 for r in rows), "a load took longer than 0.5 s"
    assert round(size[("Random forest", "joblib")]) == 24
    assert round(size[("Random forest", "joblib, compress=3")]) == 8
    assert abs(pkl_mb - size[("Random forest", "joblib")]) < 0.1
    assert round(size[("XGBoost", "native .json")], 1) == 3.2
    assert round(size[("XGBoost", "native .ubj")], 1) == 2.3
    assert round(size[("LightGBM", "native .txt")], 1) == 1.4

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True,
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
    labels = [f"{m} · {f}" for m, f, *_ in rows]
    y = np.arange(len(rows))[::-1]
    colours = [WARM if "native" in f else (AMBER if f == "skops" else ACCENT)
               for _, f, *_ in rows]
    a1.barh(y, [r[2] for r in rows], color=colours, height=0.62)
    for yy, r in zip(y, rows):
        a1.text(r[2] * 1.08, yy, f"{r[2]:.1f} MB" if r[2] >= 1
                else f"{r[2] * 1e3:.0f} kB", va="center", fontsize=9.5,
                color=MUTED)
    a1.set_xscale("log")
    a1.set_xlim(0.3, 90)
    ticks = [0.5, 1, 2, 5, 10, 20, 50]
    a1.set_xticks(ticks)
    a1.set_xticklabels([f"{t:g}" for t in ticks])
    a1.minorticks_off()
    a1.set_yticks(y)
    a1.set_yticklabels(labels, fontsize=10.5)
    a1.set_xlabel("file size, MB (log scale)")
    a1.set_title("Size", loc="left", fontsize=12.5)
    a2.barh(y, [r[3] * 1e3 for r in rows], color=colours, height=0.62)
    a2.set_xlabel("load time, ms (median of 5, one laptop)")
    a2.set_title("Load time", loc="left", fontsize=12.5)
    for a in (a1, a2):
        tidy(a)
        a.grid(axis="y", visible=False)
    for i in (3, 7):     # separate the three models
        for a in (a1, a2):
            a.axhline(y[i - 1] - 0.5, color=MUTED, lw=0.8)
    a2.legend(handles=[Rectangle((0, 0), 1, 1, color=c) for c in
                       (ACCENT, AMBER, WARM)],
              labels=["pickle, through joblib", "skops", "the library's own"],
              loc="center right", fontsize=10)
    fig.suptitle("California housing, 16,512 training rows. Every reloaded "
                 "model predicted the test set identically.",
                 x=0.01, ha="left", fontsize=11, color=MUTED, y=0.995)
    fig.tight_layout()
    save(fig, "persist-formats.png")


# --------------------------------------------------------------------------- #
# (5) versions: which files load under which scikit-learn
# --------------------------------------------------------------------------- #

VERSIONS = [("1.4.2", "numpy<2"), ("1.6.1", "numpy"), ("1.8.0", "numpy")]
KINDS = [("pipeline", "Scaler + Ridge pipeline"),
         ("rf", "Random forest"),
         ("hgb", "HistGradientBoosting")]

SAVE = """
import sys, joblib, numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
X, y = fetch_california_housing(return_X_y=True)
X, y = X[:4000], y[:4000]
tag = sys.argv[1]
models = {
  "pipeline": make_pipeline(StandardScaler(), Ridge()),
  "hgb": HistGradientBoostingRegressor(max_iter=50, random_state=0),
  "rf": RandomForestRegressor(n_estimators=20, min_samples_leaf=5,
                              random_state=0),
}
for k, m in models.items():
    m.fit(X, y)
    joblib.dump(m, f"{k}__{tag}.joblib")
    np.save(f"{k}__{tag}.npy", m.predict(X[:200]))
"""

LOAD = """
import glob, json, warnings, joblib, numpy as np
from sklearn.datasets import fetch_california_housing
X, _ = fetch_california_housing(return_X_y=True)
out = {}
for f in sorted(glob.glob("*.joblib")):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            p = joblib.load(f).predict(X[:200])
            ref = np.load(f.replace(".joblib", ".npy"))
            res = "same" if np.allclose(p, ref, rtol=0, atol=1e-9) else "diff"
            err = ""
        except Exception as e:
            res, err = "error", f"{type(e).__name__}: {e}"
        warned = any(x.category.__name__ == "InconsistentVersionWarning"
                     for x in w)
    out[f] = {"result": res, "warned": warned, "error": err}
print(json.dumps(out))
"""


def uv(version: str, numpy: str, script: str, *args, cwd: str) -> str:
    cmd = ["uv", "run", "--quiet", "--no-project", "--python", "3.12",
           "--with", f"scikit-learn=={version}", "--with", numpy,
           "python", "-c", script, *args]
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


def versions() -> None:
    import json

    with tempfile.TemporaryDirectory() as d:
        for v, npy in VERSIONS:
            uv(v, npy, SAVE, f"sk{v}", cwd=d)
        loaded = {v: json.loads(uv(v, npy, LOAD, cwd=d).strip().splitlines()[-1])
                  for v, npy in VERSIONS}

    # cell[kind][saved][loaded] -> "same" | "warn" | "error"
    cell, errors = {}, {}
    for k, _ in KINDS:
        cell[k] = {}
        for s, _ in VERSIONS:
            cell[k][s] = {}
            for l, _ in VERSIONS:
                r = loaded[l][f"{k}__sk{s}.joblib"]
                assert r["result"] != "diff", (k, s, l, "loaded but differs")
                state = ("error" if r["result"] == "error"
                         else "warn" if r["warned"] else "same")
                cell[k][s][l] = state
                if state == "error":
                    errors[(k, s, l)] = r["error"]
                print(f"    {k:9s} saved {s} loaded {l}: {state}")
    for key, e in errors.items():
        print(f"    {key}: {e[:150]}")

    # the lesson quotes these
    for k in ("pipeline", "rf"):
        for s, _ in VERSIONS:
            for l, _ in VERSIONS:
                assert cell[k][s][l] == ("same" if s == l else "warn"), (k, s, l)
    assert cell["hgb"]["1.4.2"]["1.6.1"] == "error"
    assert cell["hgb"]["1.4.2"]["1.8.0"] == "error"
    assert cell["hgb"]["1.6.1"]["1.4.2"] == "error"
    assert cell["hgb"]["1.8.0"]["1.4.2"] == "error"
    assert cell["hgb"]["1.6.1"]["1.8.0"] == "warn"
    assert "__pyx_unpickle_CyHalfSquaredError" in errors[("hgb", "1.4.2", "1.8.0")]
    assert "is not a known BitGenerator" in errors[("hgb", "1.8.0", "1.4.2")]

    face = {"same": OK_BG, "warn": AMBER_BG, "error": WARM}
    text = {"same": ("loads", OK), "warn": ("loads +\nwarning", AMBER),
            "error": ("fails", "white")}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    names = [f"{v}" + ("\nnumpy 1.26" if v == "1.4.2" else "\nnumpy 2")
             for v, _ in VERSIONS]
    for ax, (k, title) in zip(axes, KINDS):
        n = len(VERSIONS)
        for i, (s, _) in enumerate(VERSIONS):
            for j, (l, _) in enumerate(VERSIONS):
                st = cell[k][s][l]
                ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                       facecolor=face[st], edgecolor="white",
                                       lw=2.5))
                t, c = text[st]
                ax.text(j + 0.5, n - 0.5 - i, t, ha="center", va="center",
                        fontsize=10, color=c,
                        weight="bold" if st == "error" else "normal")
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels(names, fontsize=9.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels(names[::-1], fontsize=9.5)
        ax.xaxis.tick_top()
        ax.set_xlabel("loaded with scikit-learn", fontsize=10.5, color=MUTED)
        ax.set_title(title, fontsize=12.5, pad=38)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("saved with scikit-learn", fontsize=10.5, color=MUTED)
    fig.tight_layout()
    save(fig, "persist-versions.png")



# --------------------------------------------------------------------------- #
# (6) the claims the lesson makes in text and code, with no figure
# --------------------------------------------------------------------------- #

MAIN_DUMP = """
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

def add_ratios(df):
    return df.assign(RoomsPerOcc=df.AveRooms / df.AveOccup)

pipe = make_pipeline(FunctionTransformer(add_ratios), Ridge())
joblib.dump(pipe.fit(X, y), "model.joblib")
try:
    lam = make_pipeline(FunctionTransformer(lambda d: d), Ridge())
    joblib.dump(lam.fit(X, y), "lambda.joblib")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
"""

MAIN_LOAD = """
import joblib
try:
    joblib.load("model.joblib")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
"""


def claims() -> None:
    import contextlib
    import io
    import sys
    import warnings

    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor

    # "Check yourself" 1: a scaler fitted on one request
    one = StandardScaler().fit_transform(np.array([[8.3, 41.0, 6.98]]))
    assert str(one) == "[[0. 0. 0.]]", str(one)

    # the pickle that prints while it loads
    class Innocent:
        def __reduce__(self):
            return (print, ("this line ran inside pickle.loads",))

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pickle.loads(pickle.dumps(Innocent()))
    assert out.getvalue() == "this line ran inside pickle.loads\n"

    # column order: refused on a DataFrame, silent on an array
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    pipe = make_pipeline(StandardScaler(), Ridge()).fit(X, y)
    try:
        pipe.predict(X[X.columns[::-1]])
        raise AssertionError("reordered columns were accepted")
    except ValueError as e:
        assert "Feature names must be in the same order as they were in fit" \
            in str(e), e
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a = pipe.predict(X.to_numpy()[:5])
        b = pipe.predict(X.to_numpy()[:5, ::-1])
    assert any("X does not have valid feature names, but StandardScaler was "
               "fitted with feature names" in str(x.message) for x in w)
    print(f"    five rows {np.round(a, 2)} -> reversed {np.round(b, 2)}")
    assert list(np.round(a, 2)) == [4.13, 3.98, 3.68, 3.24, 2.41]
    assert list(np.round(b, 2)) == [96.48, 1445.31, 204.46, 245.16, 250.49]

    X_tr, X_te, y_tr, _ = california()

    # a binary LightGBM Booster predicts P(class 1)
    clf = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
    clf.fit(X_tr, (y_tr > 2).astype(int))
    with tempfile.TemporaryDirectory() as d:
        clf.booster_.save_model(f"{d}/clf.txt")
        booster = lgb.Booster(model_file=f"{d}/clf.txt")
        assert np.allclose(booster.predict(X_te),
                           clf.predict_proba(X_te)[:, 1])

    # a forest on several threads is not bit-reproducible, even unsaved
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                               n_jobs=-1, random_state=0).fit(X_tr, y_tr)
    diff = float(np.max(np.abs(rf.predict(X_te) - rf.predict(X_te))))
    print(f"    forest on n_jobs=-1, same rows twice: max diff {diff:.1e}")
    assert diff < 1e-12, diff

    # __main__ and lambda, in separate processes as the lesson describes
    with tempfile.TemporaryDirectory() as d:
        dumped = subprocess.run([sys.executable, "-c", MAIN_DUMP], cwd=d,
                                check=True, capture_output=True, text=True)
        loaded = subprocess.run([sys.executable, "-c", MAIN_LOAD], cwd=d,
                                check=True, capture_output=True, text=True)
    lam, main = dumped.stdout.strip(), loaded.stdout.strip()
    print(f"    {lam}\n    {main}")
    assert lam.startswith("PicklingError: Can't pickle <function <lambda>")
    assert lam.endswith("it's not found as __main__.<lambda>")
    assert main.startswith("AttributeError: Can't get attribute "
                           "'add_ratios' on <module '__main__'")

# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only",
                    default="serve,state,skew,formats,versions,claims")
    parts = set(ap.parse_args().only.split(","))
    print("generating saving-and-loading figures ->", ASSETS)
    if "serve" in parts:
        serve_diagram()
    if "state" in parts:
        fitted_state()
    if "skew" in parts:
        skew()
    if "formats" in parts:
        formats()
    if "versions" in parts:
        versions()
    if "claims" in parts:
        claims()
    print("done.")


if __name__ == "__main__":
    main()
