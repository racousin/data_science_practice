"""Figures for the `ms2a-machine-learning-practice` lesson
`s3-tabular-models/stacking-and-blending.md`.

Every image this script writes is committed under
`content/ms2a-machine-learning-practice/assets/tabular/` with the prefix
`stack-`. Re-run with:

    uv run --no-project --with matplotlib --with numpy --with scipy \
        --with scikit-learn \
        python courseware/tools/figures/ms2a_s3_stacking.py

Pass `--only ladder,diversity,split,leak,weights,pooling` to redraw a subset;
the four measured figures share one experiment, so asking for any of them runs
it. About twenty minutes at the defaults.

Four of the six figures are measured, not sketched, and they are measured on
**two** datasets, because the answer differs:

* `california` — `fetch_california_housing`, 20,640 rows, 8 numeric features,
  cached by scikit-learn after the first download, subsampled to `--rows`. A
  gradient booster dominates it, and the lesson's point is that stacking then
  buys nothing.
* `complementary` — generated here: a linear part six features wide plus two
  sharp three-way indicator interactions. No single family fits both halves,
  which is the case stacking exists for.

Six base models are fitted on each — a spline ridge, 10-NN, an MLP, a random
forest, extra trees and a histogram gradient booster — plus a 1-NN that exists
only for the leak figure, where its zero in-sample error is the whole point.
Everything is repeated over `--repeats` train/test splits.

Every number the lesson quotes is printed by this script and asserted here, so
a claim in the markdown that stops being true fails the run rather than the
reader. `stack-diversity.png` asserts an identity rather than a trend: for two
forecasts with errors e1, e2, the equal-weight average beats the better one
exactly when sigma2/sigma1 < sqrt(rho^2 + 3) - rho, with sigma_i the RMSE and
rho the *uncentred* correlation mean(e1 e2)/(sigma1 sigma2). The scatter is a
check of that frontier, not a fit to it.

`split` is the one diagram: it has no data in it.

Drawn against scikit-learn 1.8.0, numpy 2.4.6, scipy 1.17.1, matplotlib
3.10.9. Everything is seeded, so a re-run reproduces the committed PNGs.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.optimize import nnls
from scipy.stats import norm
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "content"
    / "ms2a-machine-learning-practice"
    / "assets"
    / "tabular"
)

# The same tokens as ms2a_s3_optuna.py and ms2a_s3_probabilistic.py, so
# Session 3's authored figures read as one set.
INK = "#1f2933"
MUTED = "#6b7684"
RULE = "#dfe3e8"
ACCENT = "#2f6f9f"
ACCENT_BG = "#e3eef7"
WARM = "#c1553b"
BLUES = ["#9cc3e0", "#5b93c0", "#2f6f9f", "#1d4a6d"]

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

DATASETS = ("california", "complementary")
TITLES = {
    "california": "California housing — a booster dominates",
    "complementary": "Complementary task — linear part + interactions",
}


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


# --------------------------------------------------------------------------- #
# the experiment every measured figure reads from
# --------------------------------------------------------------------------- #

NAMES = ["ridge", "10-NN", "MLP", "forest", "extra trees", "boosting"]
LEAK_NAMES = NAMES + ["1-NN"]


def base_models(seed: int) -> dict:
    """Six models of four different families, plus the memoriser. Each is
    configured sensibly rather than tuned: the lesson's subject is how they
    combine, and tuning them would put itself in the way of reading that."""
    return {
        "ridge": make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=6, degree=3),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        ),
        "10-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(10)),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=600,
                early_stopping=True, random_state=seed,
            ),
        ),
        "forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "extra trees": ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, random_state=seed
        ),
        "1-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(1)),
    }


def complementary(rows: int, seed: int):
    """Half of the signal is smooth and linear over six features, the other
    half is two sharp three-way indicators. A linear model cannot reach the
    second half; a tree approximates the first by steps."""
    rng = np.random.default_rng(1000 + seed)
    X = rng.normal(0, 1, (rows, 12))
    beta = np.array([3.0, -2.5, 2.0, -1.5, 1.2, 0.8])
    y = X[:, :6] @ beta
    y += 9.0 * ((X[:, 6] > 0.4) & (X[:, 7] > 0.4) & (X[:, 8] > 0.0))
    y += 7.0 * ((X[:, 9] < -0.5) & (X[:, 10] > 0.5))
    y += rng.normal(0, 1.0, rows)
    return X, y


def load(dataset: str, rows: int, seed: int):
    if dataset == "california":
        d = fetch_california_housing()
        rng = np.random.default_rng(0)
        idx = rng.choice(len(d.data), size=rows, replace=False)
        X, y = d.data[idx], d.target[idx]
    elif dataset == "complementary":
        X, y = complementary(rows, seed)
    else:
        raise ValueError(dataset)
    return train_test_split(X, y, test_size=0.25, random_state=seed)


def one_repeat(dataset: str, rows: int, seed: int) -> dict:
    """Fit every base model once per protocol and return what the figures
    need: out-of-fold, in-sample and test predictions, plus the blend split's
    meta-features."""
    X_tr, X_te, y_tr, y_te = load(dataset, rows, seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    oof = {k: np.zeros(len(y_tr)) for k in LEAK_NAMES}
    for fit_idx, out_idx in kf.split(X_tr):
        models = base_models(seed)
        for k in LEAK_NAMES:
            models[k].fit(X_tr[fit_idx], y_tr[fit_idx])
            oof[k][out_idx] = models[k].predict(X_tr[out_idx])

    models = base_models(seed)
    insample, test = {}, {}
    for k in LEAK_NAMES:
        models[k].fit(X_tr, y_tr)
        insample[k] = models[k].predict(X_tr)
        test[k] = models[k].predict(X_te)

    # the blending protocol: one 80/20 split of the training rows, used once
    ia, ib = train_test_split(np.arange(len(y_tr)), test_size=0.2,
                              random_state=seed)
    blend = base_models(seed)
    blend_meta, blend_test = {}, {}
    for k in LEAK_NAMES:
        blend[k].fit(X_tr[ia], y_tr[ia])
        blend_meta[k] = blend[k].predict(X_tr[ib])
        blend_test[k] = blend[k].predict(X_te)

    return dict(y_tr=y_tr, y_te=y_te, oof=oof, insample=insample, test=test,
                blend_idx=ib, blend_meta=blend_meta, blend_test=blend_test)


def matrix(d: dict, key: str, names) -> np.ndarray:
    return np.column_stack([d[key][k] for k in names])


def fit_nnls(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares with an intercept — the meta-model this
    lesson recommends. The last entry of the returned vector is the
    intercept."""
    w, _ = nnls(np.column_stack([Z, np.ones(len(Z))]), y)
    return w


def apply_nnls(w: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return Z @ w[:-1] + w[-1]


def run_experiment(rows: int, repeats: int) -> dict:
    out = {}
    for dataset in DATASETS:
        print(f"  {dataset}: {rows} rows x {repeats} splits")
        runs = []
        for seed in range(repeats):
            print(f"    repeat {seed + 1}/{repeats}")
            runs.append(one_repeat(dataset, rows, seed))
        out[dataset] = runs
    return out


# --------------------------------------------------------------------------- #
# (1) the ladder: best single, equal average, blend, stack
# --------------------------------------------------------------------------- #

ROWS = ["best single", "equal average", "blend", "stack"]


def ladder_numbers(runs: list):
    got = {k: [] for k in ROWS}
    solo = {k: [] for k in NAMES}
    for d in runs:
        for k in NAMES:
            solo[k].append(rmse(d["y_te"], d["test"][k]))
        got["best single"].append(min(solo[k][-1] for k in NAMES))
        got["equal average"].append(
            rmse(d["y_te"], matrix(d, "test", NAMES).mean(axis=1))
        )
        w = fit_nnls(matrix(d, "blend_meta", NAMES), d["y_tr"][d["blend_idx"]])
        got["blend"].append(
            rmse(d["y_te"], apply_nnls(w, matrix(d, "blend_test", NAMES)))
        )
        w = fit_nnls(matrix(d, "oof", NAMES), d["y_tr"])
        got["stack"].append(
            rmse(d["y_te"], apply_nnls(w, matrix(d, "test", NAMES)))
        )
    mean = {k: float(np.mean(v)) for k, v in got.items()}
    sem = {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in got.items()}
    return mean, sem, {k: float(np.mean(v)) for k, v in solo.items()}


def ladder(exp: dict) -> None:
    stats = {}
    for dataset, runs in exp.items():
        mean, sem, solo = ladder_numbers(runs)
        gain = {k: 100 * (mean["best single"] - mean[k]) / mean["best single"]
                for k in ROWS}
        stats[dataset] = (mean, sem, gain)
        print(f"    {dataset}: "
              + ", ".join(f"{k} {v:.3f}" for k, v in solo.items()))
        for k in ROWS:
            print(f"      {k:<14} {mean[k]:.4f} +/- {sem[k]:.4f}  {gain[k]:+.2f}%")

    # The claims the lesson makes, each one a gate on this run.
    ca, co = stats["california"], stats["complementary"]
    assert abs(ca[2]["stack"]) < 1.0, (
        f"California stack gain {ca[2]['stack']:+.2f}% is no longer ~nothing")
    assert 1.0 < co[2]["stack"] < 8.0, (
        f"complementary stack gain {co[2]['stack']:+.2f}% outside the quoted band")
    for d in (ca, co):
        assert d[2]["equal average"] < 0, "the equal average did not lose"
        assert d[0]["stack"] <= d[0]["blend"], "blending beat stacking"

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    for ax, dataset in zip(axes, DATASETS):
        mean, sem, gain = stats[dataset]
        ys = np.arange(len(ROWS))[::-1]
        colours = [MUTED, WARM, BLUES[1], ACCENT]
        ax.barh(ys, [mean[k] for k in ROWS], height=0.62, color=colours,
                xerr=[sem[k] for k in ROWS],
                error_kw=dict(ecolor=INK, lw=1.1))
        top = max(mean.values())
        for y_, k in zip(ys, ROWS):
            txt = f"{mean[k]:.3f}" + ("" if k == "best single"
                                      else f"  ({gain[k]:+.1f}%)")
            ax.text(mean[k] + top * 0.025, y_, txt, va="center", fontsize=10.5)
        ax.set_yticks(ys, ROWS)
        ax.set_xlim(0, top * 1.45)
        ax.set_xlabel("test RMSE — lower is better")
        ax.set_title(TITLES[dataset], fontsize=11.5, loc="left", color=INK)
        tidy(ax)
        ax.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "stack-ladder.png")


# --------------------------------------------------------------------------- #
# (2) when averaging two models helps: an exact frontier, checked
# --------------------------------------------------------------------------- #


def frontier(rho):
    return np.sqrt(rho ** 2 + 3) - rho


def diversity(exp: dict) -> None:
    pts = []
    for dataset, runs in exp.items():
        for d in runs:
            e = {k: d["test"][k] - d["y_te"] for k in NAMES}
            for a, b in itertools.combinations(NAMES, 2):
                s = {k: float(np.sqrt(np.mean(e[k] ** 2))) for k in (a, b)}
                lo, hi = sorted((a, b), key=lambda k: s[k])
                rho = float(np.mean(e[a] * e[b]) / (s[a] * s[b]))
                ratio = s[hi] / s[lo]
                helped = rmse(d["y_te"],
                              0.5 * (d["test"][a] + d["test"][b])) < s[lo]
                pts.append((rho, ratio, helped, dataset))

    rho = np.array([p[0] for p in pts])
    ratio = np.array([p[1] for p in pts])
    helped = np.array([p[2] for p in pts])
    predicted = ratio < frontier(rho)
    wrong = int((predicted != helped).sum())
    print(f"    {len(pts)} pairs; averaging helped {helped.sum()}; "
          f"the frontier misclassifies {wrong}")
    print(f"    rho in [{rho.min():.2f}, {rho.max():.2f}], "
          f"ratio in [{ratio.min():.2f}, {ratio.max():.2f}]")
    assert wrong == 0, "the frontier is meant to be exact"
    assert 0 < helped.sum() < len(pts), "the figure needs both outcomes"

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    grid = np.linspace(0, 1, 300)
    ax.fill_between(grid, 1, frontier(grid), color=ACCENT_BG, zorder=0)
    ax.plot(grid, frontier(grid), color=ACCENT, lw=1.8,
            label=r"$\sigma_2/\sigma_1 = \sqrt{\rho^2+3}-\rho$")
    ax.scatter(rho[helped], ratio[helped], s=40, color=ACCENT, alpha=0.8,
               edgecolor="white", linewidth=0.7, zorder=3,
               label="the average beat the better model")
    ax.scatter(rho[~helped], ratio[~helped], s=40, color=WARM, alpha=0.8,
               marker="^", edgecolor="white", linewidth=0.7, zorder=3,
               label="it did not")
    ax.text(0.03, 1.12, "averaging wins here", color=ACCENT, fontsize=11)
    ax.set_xlabel(r"correlation $\rho$ of the two models' errors")
    ax.set_ylabel(r"how unequal they are,  $\sigma_2/\sigma_1$")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.98, min(ratio.max() * 1.08, 3.0))
    ax.legend(loc="upper right", fontsize=10.5)
    tidy(ax)
    fig.suptitle("A plain average pays only if the two are close and differ",
                 fontsize=13.5, x=0.02, ha="left")
    save(fig, "stack-diversity.png")


# --------------------------------------------------------------------------- #
# (3) the one diagram: how blending and stacking spend the training rows
# --------------------------------------------------------------------------- #


def split_diagram() -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(9.0, 5.0), gridspec_kw=dict(height_ratios=[1, 2.2])
    )
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    W = 0.74          # the training set, drawn full width
    k = 5
    cell = W / k

    def box(ax, x0, w, y0, h, colour, label="", fs=10.5, tc=INK):
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=colour,
                               edgecolor="white", linewidth=1.4))
        if label:
            ax.text(x0 + w / 2, y0 + h / 2, label, ha="center", va="center",
                    fontsize=fs, color=tc)

    # -- blending: one split of the training set, used once
    ax = axes[0]
    ax.text(0, 0.86, "Blending — one split", fontsize=13.5, color=INK)
    box(ax, 0, 4 * cell, 0.34, 0.34, ACCENT_BG, "base models fit here — 80%")
    box(ax, 4 * cell, cell, 0.34, 0.34, WARM, "meta\n20%", 10.5, "white")
    ax.text(W + 0.02, 0.51, "one fit per base model",
            fontsize=10.5, color=MUTED, va="center")
    ax.text(0, 0.14, "The meta-model is fitted on a fifth of the rows, and the "
                     "base models never see them.",
            fontsize=10.5, color=MUTED)

    # -- stacking: k folds, every row covered exactly once
    ax = axes[1]
    ax.text(0, 1.00, "Stacking — k-fold, every row covered", fontsize=13.5,
            color=INK)
    ax.text(0, 0.89, "fitted on the pale cells, predicts the dark one",
            fontsize=10.5, color=MUTED)
    h = 0.082
    for i in range(k):
        y0 = 0.80 - i * 0.097
        for j in range(k):
            box(ax, j * cell, cell, y0, h, WARM if i == j else ACCENT_BG)
        ax.text(W + 0.02, y0 + h / 2, f"fold {i + 1} held out",
                fontsize=9.5, color=MUTED, va="center")
    ax.add_patch(FancyArrowPatch((W / 2, 0.385), (W / 2, 0.315),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=INK, lw=1.3))
    box(ax, 0, W, 0.21, h, WARM,
        "the meta-model's training set — 100% of the rows", 10.5, "white")
    ax.text(0, 0.07, "Every prediction in that bar comes from a model that had "
                     "not seen the row. Cost: k fits per base model.",
            fontsize=10.5, color=MUTED)
    fig.subplots_adjust(hspace=0.12)
    save(fig, "stack-blend-vs-stack.png")


# --------------------------------------------------------------------------- #
# (4) the leak: in-sample meta-features against out-of-fold ones
# --------------------------------------------------------------------------- #


def leak(exp: dict) -> None:
    runs = exp["california"]
    w_in, w_oof, s_in, s_oof = [], [], [], []
    for d in runs:
        Zi = matrix(d, "insample", LEAK_NAMES)
        Zo = matrix(d, "oof", LEAK_NAMES)
        Zt = matrix(d, "test", LEAK_NAMES)
        a, b = fit_nnls(Zi, d["y_tr"]), fit_nnls(Zo, d["y_tr"])
        w_in.append(a[:-1])
        w_oof.append(b[:-1])
        s_in.append(rmse(d["y_te"], apply_nnls(a, Zt)))
        s_oof.append(rmse(d["y_te"], apply_nnls(b, Zt)))
    w_in, w_oof = np.mean(w_in, axis=0), np.mean(w_oof, axis=0)
    m_in, m_oof = float(np.mean(s_in)), float(np.mean(s_oof))
    r_in = float(np.mean([rmse(d["y_tr"], d["insample"]["1-NN"]) for d in runs]))
    print(f"    1-NN in-sample RMSE {r_in:.6f}; meta weight on it: "
          f"in-sample {w_in[-1]:.2f}, out-of-fold {w_oof[-1]:.2f}")
    print(f"    test RMSE: in-sample meta-features {m_in:.4f}, "
          f"out-of-fold {m_oof:.4f}  ({100 * (m_in - m_oof) / m_oof:+.1f}%)")

    assert r_in < 1e-9, "1-NN did not reproduce its training targets exactly"
    assert w_in[-1] > w_oof[-1], "the leak did not inflate the 1-NN weight"
    assert m_in > m_oof, "the leak did not cost test RMSE"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.0, 3.9), gridspec_kw=dict(width_ratios=[2.2, 1])
    )
    x = np.arange(len(LEAK_NAMES))
    ax1.bar(x - 0.19, w_in, width=0.36, color=WARM,
            label="meta-features taken in-sample")
    ax1.bar(x + 0.19, w_oof, width=0.36, color=ACCENT,
            label="meta-features taken out-of-fold")
    ax1.set_xticks(x, LEAK_NAMES, fontsize=10, rotation=18, ha="right")
    ax1.set_ylabel("meta-model weight")
    ax1.set_ylim(0, max(w_in.max(), w_oof.max()) * 1.62)
    ax1.legend(loc="upper left", fontsize=10.5)
    tidy(ax1)
    ax1.grid(axis="x", visible=False)
    ax1.annotate("zero training error,\nso the meta-model believes it",
                 xy=(x[-1] - 0.19, w_in[-1] * 1.02),
                 xytext=(x[-1] - 3.6, w_in[-1] * 1.16), fontsize=10.5,
                 color=WARM, va="bottom",
                 arrowprops=dict(arrowstyle="-|>", color=WARM, lw=1.2))

    ax2.bar([0, 1], [m_in, m_oof], width=0.55, color=[WARM, ACCENT])
    for i, v in enumerate([m_in, m_oof]):
        ax2.text(i, v + max(m_in, m_oof) * 0.03, f"{v:.3f}", ha="center",
                 fontsize=11.5)
    ax2.set_xticks([0, 1], ["in-sample", "out-of-fold"], fontsize=10.5)
    ax2.set_ylabel("test RMSE")
    ax2.set_ylim(0, max(m_in, m_oof) * 1.28)
    tidy(ax2)
    ax2.grid(axis="x", visible=False)
    fig.suptitle("One base model that memorises is enough to wreck the stack",
                 fontsize=13.5, x=0.02, ha="left")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "stack-leak.png")


# --------------------------------------------------------------------------- #
# (5) what the meta-model does with the weights
# --------------------------------------------------------------------------- #


def weights(exp: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.8))
    for col, dataset in enumerate(DATASETS):
        runs = exp[dataset]
        solo = np.mean([[rmse(d["y_te"], d["test"][k]) for k in NAMES]
                        for d in runs], axis=0)
        w = np.mean([fit_nnls(matrix(d, "oof", NAMES), d["y_tr"])[:-1]
                     for d in runs], axis=0)
        best = int(np.argmin(solo))
        print(f"    {dataset} solo:   "
              + ", ".join(f"{k} {v:.3f}" for k, v in zip(NAMES, solo)))
        print(f"    {dataset} weight: "
              + ", ".join(f"{k} {v:.2f}" for k, v in zip(NAMES, w)))
        assert w[best] > 0.4, "the best model did not lead the stack"

        x = np.arange(len(NAMES))
        colour = [ACCENT if i == best else BLUES[0] for i in range(len(NAMES))]
        for row, (vals, fmt, label) in enumerate((
            (solo, "{:.3f}", "alone: test RMSE"),
            (w, "{:.2f}", "in the stack: meta-model weight"),
        )):
            ax = axes[row][col]
            ax.bar(x, vals, width=0.62, color=colour)
            for i, v in enumerate(vals):
                ax.text(i, v + vals.max() * 0.04, fmt.format(v), ha="center",
                        fontsize=10)
            ax.set_xticks(x, NAMES, fontsize=9.5, rotation=20)
            ax.set_ylim(0, vals.max() * 1.3)
            ax.set_title(label if row else f"{TITLES[dataset]}\n{label}",
                         fontsize=11, loc="left", color=INK)
            tidy(ax)
            ax.grid(axis="x", visible=False)
    fig.subplots_adjust(wspace=0.2, hspace=0.62)
    save(fig, "stack-weights.png")


# --------------------------------------------------------------------------- #
# (6) combining distributions: linear pool against quantile averaging
# --------------------------------------------------------------------------- #

GRID = np.linspace(-8, 8, 4001)


def crps_from_cdf(F: np.ndarray, y: float) -> float:
    """CRPS by its definition, on the fixed grid: the integrated squared
    difference between the forecast CDF and the step at y."""
    return float(np.trapezoid((F - (GRID >= y).astype(float)) ** 2, GRID))


def pooling() -> None:
    mu, sigma = (-1.2, 1.2), 1.0
    F1 = norm.cdf(GRID, mu[0], sigma)
    F2 = norm.cdf(GRID, mu[1], sigma)
    pool = 0.5 * (F1 + F2)                            # average probabilities
    vinc = norm.cdf(GRID, float(np.mean(mu)), sigma)  # average quantiles

    rng = np.random.default_rng(0)
    n = 1500
    truths = {
        "disagreement is bias\n(truth in the middle)": rng.normal(0.0, sigma, n),
        "disagreement is real\n(truth is A or B)": np.where(
            rng.random(n) < 0.5,
            rng.normal(mu[0], sigma, n),
            rng.normal(mu[1], sigma, n),
        ),
    }
    scores = {}
    for name, ys in truths.items():
        scores[name] = {
            "linear pool": float(np.mean([crps_from_cdf(pool, y) for y in ys])),
            "quantile average": float(
                np.mean([crps_from_cdf(vinc, y) for y in ys])),
        }
        print(f"    {name.splitlines()[0]}: "
              + ", ".join(f"{k} {v:.4f}" for k, v in scores[name].items()))
    bias, real = list(scores)
    assert scores[bias]["quantile average"] < scores[bias]["linear pool"]
    assert scores[real]["linear pool"] < scores[real]["quantile average"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.2, 3.8), gridspec_kw=dict(width_ratios=[1.2, 1])
    )
    ax1.plot(GRID, np.gradient(F1, GRID), color=MUTED, lw=1.4,
             label="forecast A")
    ax1.plot(GRID, np.gradient(F2, GRID), color=MUTED, lw=1.4, ls=":",
             label="forecast B")
    ax1.plot(GRID, np.gradient(pool, GRID), color=WARM, lw=2.2,
             label="linear pool")
    ax1.plot(GRID, np.gradient(vinc, GRID), color=ACCENT, lw=2.2,
             label="quantile average")
    ax1.set_xlim(-5.5, 5.5)
    ax1.set_yticks([])
    ax1.set_xlabel("predicted value")
    ax1.legend(fontsize=10.5, loc="upper left")
    ax1.set_title("two ways to combine two forecasts", fontsize=12, loc="left")
    tidy(ax1)

    labels = list(scores)
    x = np.arange(len(labels))
    ax2.bar(x - 0.19, [scores[k]["linear pool"] for k in labels], width=0.36,
            color=WARM, label="linear pool")
    ax2.bar(x + 0.19, [scores[k]["quantile average"] for k in labels],
            width=0.36, color=ACCENT, label="quantile average")
    ax2.set_xticks(x, labels, fontsize=10)
    ax2.set_ylabel("CRPS — lower is better")
    ax2.legend(fontsize=10.5)
    ax2.set_title("which one wins depends on why", fontsize=12, loc="left")
    tidy(ax2)
    ax2.grid(axis="x", visible=False)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "stack-pooling.png")


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only",
                    default="ladder,diversity,split,leak,weights,pooling")
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--repeats", type=int, default=6)
    a = ap.parse_args()
    parts = set(a.only.split(","))
    print("generating stacking-and-blending figures ->", ASSETS)

    if "split" in parts:
        split_diagram()
    if "pooling" in parts:
        pooling()

    if parts & {"ladder", "diversity", "leak", "weights"}:
        exp = run_experiment(a.rows, a.repeats)
        if "ladder" in parts:
            ladder(exp)
        if "diversity" in parts:
            diversity(exp)
        if "leak" in parts:
            leak(exp)
        if "weights" in parts:
            weights(exp)
    print("done.")


if __name__ == "__main__":
    main()
