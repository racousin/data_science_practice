"""Figures for the `ms2a-machine-learning-practice` lesson
`s3-tabular-models/probabilistic-prediction.md`.

Every image this script writes is committed under
`content/ms2a-machine-learning-practice/assets/tabular/` with the prefix
`prob-`. Re-run with:

    uv run --no-project --with matplotlib --with numpy --with scipy \
        --with scikit-learn --with pandas \
        python courseware/tools/figures/ms2a_s3_probabilistic.py \
        --rain-data weather_europe_2020_2026.csv.gz

`--rain-data` is the challenge-177 ("European Rain Forecast") dataset file
`weather_europe_2020_2026.csv.gz`: hourly weather for 45 European cities,
2020-01-01 to 2026-03-03 UTC, one row per city-hour, with the columns
`timestamp`, `city_name`, `rain` (mm in that hour) among others. It draws one
figure, `prob-rain-europe.png`. Without the flag that figure is skipped and the
others, which use simulated data only, are still drawn. Every random draw is
seeded, so a re-run reproduces the committed PNGs.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "content"
    / "ms2a-machine-learning-practice"
    / "assets"
    / "tabular"
)

INK = "#1f2933"
MUTED = "#6b7684"
RULE = "#dfe3e8"
ACCENT = "#2f6f9f"
ACCENT_BG = "#e3eef7"
WARM = "#c1553b"
WARM_BG = "#f8e6e1"
# A single-hue ramp for ordered series (quantile levels): light to dark.
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


def crps_ensemble(members, y):
    """The lesson's estimator: CRPS of the members' empirical distribution
    (the standard, not the fair, one; |x - y| when M = 1)."""
    x = np.sort(np.asarray(members, dtype=float), axis=-1)
    y = np.asarray(y, dtype=float)[..., None]
    m = x.shape[-1]
    k = np.arange(1, m + 1)
    skill = np.abs(x - y).mean(axis=-1)
    spread = ((2 * k - m - 1) * x).sum(axis=-1) / m**2
    return skill - spread


# --------------------------------------------------------------------------- #
# Proper scoring rules — the optimal report under |p - y|^k
# --------------------------------------------------------------------------- #
def improper_scores() -> None:
    """For a belief q, the report p minimising q(1-p)^k + (1-q)p^k.
    k = 2 (Brier) gives p = q; k = 1 jumps to 0 or 1; k = 3 is pulled
    toward 0.5: p = r / (1 + r) with r = (q / (1 - q))^(1 / (k - 1))."""
    q = np.linspace(0.001, 0.999, 999)
    r3 = np.sqrt(q / (1 - q))
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot([0, 1], [0, 1], color=ACCENT, linewidth=2.6, zorder=3)
    ax.plot([0, 0.5, 0.5, 1], [0, 0, 1, 1], color=WARM, linewidth=2.2,
            linestyle="--", zorder=2)
    ax.plot(q, r3 / (1 + r3), color=WARM, linewidth=2.2, linestyle=":",
            zorder=2)
    ax.annotate("(p − y)², Brier:\nreport = belief", xy=(0.3, 0.3),
                xytext=(0.03, 0.62), fontsize=11.5, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(0.52, 0.93, "|p − y|: exaggerate\nto 0 or 1", color=INK,
            fontsize=11.5, ha="left", va="top")
    ax.text(0.80, 0.63, "|p − y|³:\nhedge to 0.5", color=INK,
            fontsize=11.5, ha="left", va="top")
    ax.set_xlabel("what you believe, q = P(rain)")
    ax.set_ylabel("report that minimises the expected score")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    tidy(ax)
    fig.tight_layout()
    save(fig, "prob-improper-scores.png")


# --------------------------------------------------------------------------- #
# Calibration — a reliability diagram, calibrated vs overconfident
# --------------------------------------------------------------------------- #
def reliability() -> None:
    """20 000 events whose true probability is known. One forecaster reports
    it; the other reports sigmoid(2.2 * logit), the shape of an overfit
    booster or naive Bayes. Bins are quantile bins of 2 000 forecasts."""
    from sklearn.calibration import calibration_curve

    rng = np.random.default_rng(7)
    n = 20_000
    true_p = rng.beta(1.3, 2.6, n)
    y = (rng.random(n) < true_p).astype(int)
    logit = np.log(true_p / (1 - true_p))
    forecasts = {
        "calibrated": (true_p, ACCENT, "-", "o"),
        "overconfident": (1 / (1 + np.exp(-2.2 * logit)), WARM, "--", "s"),
    }

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(7.2, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]})
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=":",
            zorder=1)
    ax.text(0.80, 0.86, "perfect calibration", color=MUTED, fontsize=10.5,
            rotation=45, ha="center", va="center", rotation_mode="anchor")
    bins = np.linspace(0, 1, 26)
    for name, (p, color, ls, marker) in forecasts.items():
        obs, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ece = np.mean(np.abs(obs - pred))  # equal-count bins: equal weights
        ax.plot(pred, obs, color=color, linewidth=2.2, linestyle=ls,
                marker=marker, markersize=7, markeredgecolor="white",
                label=f"{name}  (ECE {ece:.3f})", zorder=3)
        axh.hist(p, bins=bins, histtype="step", color=color, linewidth=1.8,
                 linestyle=ls)
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    tidy(ax)
    axh.set_xlabel("forecast probability")
    axh.set_ylabel("forecasts")
    axh.set_yticks([])
    tidy(axh)
    fig.tight_layout(h_pad=0.6)
    save(fig, "prob-reliability.png")


# --------------------------------------------------------------------------- #
# Quantile regression — the pinball loss
# --------------------------------------------------------------------------- #
def pinball() -> None:
    u = np.linspace(-2, 2, 401)            # y - q
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for tau, color in zip((0.1, 0.5, 0.9), (BLUES[0], BLUES[1], BLUES[3])):
        loss = np.maximum(tau * u, (tau - 1) * u)
        ax.plot(u, loss, color=color, linewidth=2.4)
        ax.text(2.04, tau * 2, f"τ = {tau}", color=INK, fontsize=11.5,
                va="center", ha="left")
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
    ax.text(-1.0, 1.72, "q above the outcome\ncost 1 − τ per mm", ha="center",
            va="top", fontsize=11, color=INK)
    ax.text(1.0, 1.72, "q below the outcome\ncost τ per mm", ha="center",
            va="top", fontsize=11, color=INK)
    ax.set_xlabel("y − q   (outcome minus predicted quantile)")
    ax.set_ylabel("pinball loss")
    ax.set_xlim(-2, 2.45)
    ax.set_ylim(0, 1.9)
    tidy(ax)
    fig.tight_layout()
    save(fig, "prob-pinball.png")


# --------------------------------------------------------------------------- #
# Quantile regression — a fan of five HistGradientBoosting models
# --------------------------------------------------------------------------- #
def quantile_fan() -> None:
    """y = mean(x) + skewed noise whose scale grows with x. Five independent
    quantile models; the printed coverage is on 20 000 fresh points."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(1)

    def draw(n):
        x = rng.uniform(0, 10, n)
        scale = 0.15 + 0.12 * x
        y = 2 + np.sin(x) + scale * (rng.gamma(2.0, 1.0, n) - 2.0)
        return x, y

    x_tr, y_tr = draw(3_000)
    x_te, y_te = draw(20_000)
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    grid = np.linspace(0, 10, 400)[:, None]
    q_grid, q_te = {}, {}
    for t in levels:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=t, max_iter=200, learning_rate=0.05,
            max_depth=2, min_samples_leaf=150, random_state=0,
        ).fit(x_tr[:, None], y_tr)
        q_grid[t] = m.predict(grid)
        q_te[t] = m.predict(x_te[:, None])
    cover = np.mean((y_te >= q_te[0.05]) & (y_te <= q_te[0.95]))

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.scatter(x_tr, y_tr, s=7, color=MUTED, alpha=0.35, linewidths=0,
               zorder=1)
    g = grid[:, 0]
    ax.fill_between(g, q_grid[0.05], q_grid[0.95], color=BLUES[0],
                    alpha=0.45, linewidth=0, zorder=2)
    ax.fill_between(g, q_grid[0.25], q_grid[0.75], color=BLUES[1],
                    alpha=0.55, linewidth=0, zorder=2)
    ax.plot(g, q_grid[0.5], color=BLUES[3], linewidth=2.2, zorder=3)
    ax.text(10.15, q_grid[0.95][-1], "95%", va="center", fontsize=11)
    ax.text(10.15, q_grid[0.75][-1], "75%", va="center", fontsize=11)
    ax.text(10.15, q_grid[0.5][-1], "median", va="center", fontsize=11)
    ax.text(10.15, q_grid[0.25][-1], "25%", va="center", fontsize=11)
    ax.text(10.15, q_grid[0.05][-1], "5%", va="center", fontsize=11)
    ax.text(0.2, 0.97,
            f"5–95% band covers {cover:.1%} of 20 000 new points",
            transform=ax.transAxes, fontsize=11, va="top", color=INK)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 11.3)
    ax.set_ylim(-1.5, 9)
    tidy(ax)
    fig.tight_layout()
    save(fig, "prob-quantile-fan.png")


# --------------------------------------------------------------------------- #
# CRPS — the forecast CDF, the observation step, and the squared gap
# --------------------------------------------------------------------------- #
def crps_area() -> None:
    """A rain forecast: P(wet) = 0.7, wet amount ~ Gamma(0.9, scale 0.8).
    Observed 1.2 mm. The CRPS is the area under (F - step)^2; the printed
    value is checked against the ensemble estimator on 20 000 quantile
    members of the same distribution."""
    p_wet, shape, scale, y = 0.7, 0.9, 0.8, 1.2
    x = np.linspace(-0.5, 5, 5501)
    F = np.where(x >= 0, 1 - p_wet + p_wet
                 * stats.gamma.cdf(np.maximum(x, 0), shape, scale=scale), 0.0)
    H = (x >= y).astype(float)
    gap2 = (F - H) ** 2
    crps = np.trapezoid(gap2, x)
    m = 20_000
    tau = (np.arange(1, m + 1) - 0.5) / m
    u = (tau - (1 - p_wet)) / p_wet
    members = np.where(u > 0, stats.gamma.ppf(np.clip(u, 1e-12, 1), shape,
                                              scale=scale), 0.0)
    check = float(crps_ensemble(members, y))
    assert abs(check - crps) < 2e-3, (check, crps)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.fill_between(x, F, H, color=ACCENT_BG, linewidth=0, zorder=1)
    a1.plot(x, F, color=ACCENT, linewidth=2.4, zorder=3)
    a1.plot(x, H, color=WARM, linewidth=2.2, linestyle="--", zorder=3)
    a1.text(2.6, 0.72, "forecast CDF F(x)", color=INK, fontsize=11.5)
    a1.text(1.32, 0.08, "observation: 1{x ≥ y}, y = 1.2 mm", color=INK,
            fontsize=11.5)
    a1.annotate("30% chance of\nexactly 0 mm", xy=(0.0, 0.3),
                xytext=(0.35, 0.47), fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    a1.set_title("F against the observation's step", fontsize=12.5,
                 color=INK, loc="left")
    a1.set_xlabel("rain x (mm)")
    a1.set_ylabel("P(rain ≤ x)")
    a1.set_ylim(-0.03, 1.05)

    a2.fill_between(x, gap2, color=ACCENT_BG, linewidth=0, zorder=1)
    a2.plot(x, gap2, color=ACCENT, linewidth=2.2, zorder=2)
    a2.axvline(y, color=WARM, linewidth=1.4, linestyle="--", zorder=1)
    a2.text(2.1, 0.55, f"shaded area = CRPS\n= {crps:.3f} mm", fontsize=12,
            color=INK)
    a2.set_title("(F − step)², integrated over x", fontsize=12.5, color=INK,
                 loc="left")
    a2.set_xlabel("rain x (mm)")
    a2.set_ylim(-0.02, 1.0)
    for a in (a1, a2):
        a.set_xlim(-0.5, 5)
        tidy(a)
    fig.tight_layout(w_pad=2.5)
    save(fig, "prob-crps-area.png")


# --------------------------------------------------------------------------- #
# CRPS — six forecasters on hours drawn from a known hurdle distribution
# --------------------------------------------------------------------------- #
def crps_forecasters() -> None:
    """Each hour has its own P(wet) ~ Beta(0.3, 1.3) and wet-amount scale;
    the amount given wet is Gamma(0.8, scale). CRPSS is the ratio of sums
    against climatology (the pooled quantiles, same for every hour)."""
    rng = np.random.default_rng(3)
    n, m, shape = 20_000, 20, 0.8
    p = rng.beta(0.3, 1.3, n)
    scale = np.exp(rng.normal(-0.3, 0.7, n))
    wet = rng.random(n) < p
    y = np.where(wet, rng.gamma(shape, scale), 0.0)
    tau = (np.arange(1, m + 1) - 0.5) / m
    u = (tau[None, :] - (1 - p[:, None])) / p[:, None]
    truth = np.where(u > 0, stats.gamma.ppf(np.clip(u, 1e-12, 1), shape,
                                            scale=scale[:, None]), 0.0)
    median = truth[:, [m // 2]]
    mean = (p * shape * scale)[:, None]
    forecasters = {
        "true distribution,\n20 quantile members": truth,
        "right centre, spread × 0.3": np.maximum(
            median + 0.3 * (truth - median), 0),
        "climatology (reference)": np.quantile(y, tau)[None, :]
        .repeat(n, 0),
        "always 0 mm, M = 1": np.zeros((n, 1)),
        "seed ensemble around\nthe mean, 20 members": np.maximum(
            mean + rng.normal(0, 0.02, (n, m)), 0),
        "the mean, M = 1": mean,
    }
    ref = crps_ensemble(forecasters["climatology (reference)"], y).sum()
    names = list(forecasters)
    skill = [1 - crps_ensemble(forecasters[k], y).sum() / ref for k in names]
    for k, s in zip(names, skill):
        print(f"    {k.replace(chr(10), ' '):45s} CRPSS {s:+.3f}")

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ypos = np.arange(len(names))[::-1]
    colors = [ACCENT if s >= 0 else WARM for s in skill]
    ax.barh(ypos, skill, height=0.55, color=colors, zorder=2)
    for yp, s in zip(ypos, skill):
        label = f"{s:+.2f}".replace("-", "\u2212")
        ax.text(s + (0.012 if s >= 0 else -0.012), yp, label,
                va="center", ha="left" if s >= 0 else "right", fontsize=11,
                color=INK)
    ax.axvline(0, color=INK, linewidth=1.2, zorder=3)
    ax.set_yticks(ypos, names, fontsize=11)
    ax.set_xlabel("CRPSS against climatology (higher is better)")
    ax.set_xlim(-0.52, 0.26)
    tidy(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "prob-crps-forecasters.png")


# --------------------------------------------------------------------------- #
# Checking a distribution — rank histograms, four diagnoses
# --------------------------------------------------------------------------- #
def rank_histograms() -> None:
    """Truth y ~ N(mu, 1); 19-member ensembles drawn from N(mu + b, s).
    Ranks of y among the members, 20 000 cases. Coverage is the share of y
    between the lowest and highest member: for 19 members exchangeable with
    y that range is a (19 - 1) / (19 + 1) = 90% interval."""
    rng = np.random.default_rng(11)
    n, m = 20_000, 19
    mu = rng.normal(0, 1, n)
    y = mu + rng.normal(0, 1, n)
    cases = [
        ("Flat: calibrated", 0.0, 1.0),
        ("U-shaped: under-dispersed", 0.0, 0.5),
        ("Dome: over-dispersed", 0.0, 2.0),
        ("Sloped: members too low", -0.7, 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), sharex=True,
                             sharey=True)
    for ax, (title, bias, sd) in zip(axes.ravel(), cases):
        ens = mu[:, None] + bias + sd * rng.normal(0, 1, (n, m))
        rank = (ens < y[:, None]).sum(axis=1)       # continuous: no ties
        freq = np.bincount(rank, minlength=m + 1) / n
        cover = np.mean((y >= ens.min(axis=1)) & (y <= ens.max(axis=1)))
        ax.bar(np.arange(m + 1), freq, width=0.8, color=ACCENT, zorder=2)
        ax.axhline(1 / (m + 1), color=INK, linewidth=1.2, linestyle="--",
                   zorder=3)
        ax.set_title(title, fontsize=12.5, loc="left", color=INK)
        ax.text(0.5, 0.95, f"member range covers {cover:.0%} (target 90%)",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, color=INK)
        tidy(ax)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("rank of the observation among 19 members")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of cases")
    axes[0, 0].set_ylim(0, 0.2)
    axes[0, 0].set_xticks([0, 5, 10, 15, 19])
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, "prob-rank-histograms.png")


# --------------------------------------------------------------------------- #
# Rain is zero-inflated — the challenge-177 data
# --------------------------------------------------------------------------- #
def rain_europe(path: pathlib.Path) -> None:
    """Left: Paris hourly rain, share of hours by amount. Right: share of wet
    hours (>= 0.1 mm) by calendar month, Paris against the other 44 cities."""
    import pandas as pd

    df = pd.read_csv(path, usecols=["timestamp", "city_name", "rain"],
                     parse_dates=["timestamp"])
    assert df["city_name"].nunique() == 45, df["city_name"].nunique()
    df["wet"] = df["rain"] >= 0.1
    df["month"] = df["timestamp"].dt.month
    wet_all = df["wet"].mean()
    amounts = df.loc[df["wet"], "rain"]
    print(f"    wet share {wet_all:.1%}; wet amounts median "
          f"{amounts.median():.1f}, p90 {amounts.quantile(0.9):.1f}, "
          f"p99 {amounts.quantile(0.99):.1f}, max {amounts.max():.1f} mm")
    by_city = df.groupby("city_name")["wet"].mean().sort_values()
    print(f"    driest {by_city.index[0]} {by_city.iloc[0]:.1%}, wettest "
          f"{by_city.index[-1]} {by_city.iloc[-1]:.1%}")

    paris = df.loc[df["city_name"] == "Paris", "rain"].to_numpy()
    edges = [0, 0.05, 0.15, 0.25, 0.55, 1.05, 2.05, 5.05, np.inf]
    labels = ["0", "0.1", "0.2", "0.3–0.5", "0.6–1", "1.1–2", "2.1–5", "> 5"]
    share = np.histogram(paris, bins=edges)[0] / len(paris)

    monthly = df.groupby(["city_name", "month"])["wet"].mean().unstack()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                                 gridspec_kw={"width_ratios": [1.05, 1]})
    xs = np.arange(len(labels))
    a1.bar(xs, share * 100, width=0.72,
           color=[MUTED] + [ACCENT] * (len(labels) - 1), zorder=2)
    for xpos, s in zip(xs, share):
        a1.text(xpos, s * 100 + 1.2, f"{s * 100:.1f}%", ha="center",
                fontsize=10.5, color=INK)
    a1.set_xticks(xs, labels, fontsize=10.5)
    a1.set_xlabel("rain in the hour (mm)")
    a1.set_ylabel("share of Paris hours (%)")
    a1.set_title("Paris, 2020–2026: a spike at zero and a long tail",
                 fontsize=12.5, loc="left", color=INK)
    a1.set_ylim(0, 92)
    tidy(a1)
    a1.grid(axis="x", visible=False)

    months = np.arange(1, 13)
    for city, row in monthly.iterrows():
        if city != "Paris":
            a2.plot(months, row.to_numpy() * 100, color=RULE, linewidth=1.1,
                    zorder=1)
    a2.plot(months, monthly.loc["Paris"].to_numpy() * 100, color=ACCENT,
            linewidth=2.6, marker="o", markersize=6,
            markeredgecolor="white", zorder=3)
    lo_city = monthly.mean(axis=1).idxmin()
    hi_city = monthly.mean(axis=1).idxmax()
    for city in (lo_city, hi_city):
        a2.plot(months, monthly.loc[city].to_numpy() * 100, color=MUTED,
                linewidth=1.6, zorder=2)
        a2.text(12.25, monthly.loc[city, 12] * 100, city, va="center",
                fontsize=10.5, color=MUTED)
    a2.text(12.25, monthly.loc["Paris", 12] * 100, "Paris", va="center",
            fontsize=11, color=INK, fontweight="bold")
    a2.set_xticks(months, list("JFMAMJJASOND"))
    a2.set_xlim(0.6, 14.2)
    a2.set_ylim(0, None)
    a2.set_ylabel("wet hours, ≥ 0.1 mm (%)")
    a2.set_title("Wet-hour share by month, 45 cities", fontsize=12.5,
                 loc="left", color=INK)
    tidy(a2)
    fig.tight_layout(w_pad=2.5)
    save(fig, "prob-rain-europe.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--rain-data", type=pathlib.Path,
        help="challenge-177 dataset file weather_europe_2020_2026.csv.gz; "
             "without it prob-rain-europe.png is skipped")
    args = ap.parse_args()

    print("generating probabilistic-prediction figures ->", ASSETS)
    improper_scores()
    reliability()
    pinball()
    quantile_fan()
    crps_area()
    crps_forecasters()
    rank_histograms()
    if args.rain_data is None:
        print("  skipped prob-rain-europe.png (no --rain-data)")
    else:
        if not args.rain_data.is_file():
            raise SystemExit(f"--rain-data: no such file {args.rain_data}")
        rain_europe(args.rain_data)
    print("done.")


if __name__ == "__main__":
    main()
