"""Figures for the `ms2a-machine-learning-practice` lesson
`s3-tabular-models/hyperparameter-optimization.md`.

Every image this script writes is committed under
`content/ms2a-machine-learning-practice/assets/tabular/` with the prefix
`optuna-`. Each one comes out of a real Optuna run, not a sketch. Re-run with:

    uv run --no-project --with optuna --with lightgbm --with scikit-learn \
        --with pandas --with matplotlib \
        python courseware/tools/figures/ms2a_s3_optuna.py

Pass `--only tpe,race,study` to redraw a subset. `study` fetches the Adult
census table from OpenML (id 1590, cached by scikit-learn after the first
download) and runs the lesson's LightGBM recipe for 60 trials: about four
minutes on a laptop. Every sampler, split and model is seeded, so a re-run
reproduces the committed PNGs.

Drawn against optuna 4.8.0, lightgbm 4.6.0, scikit-learn 1.8.0.
`tpe_anatomy` reads the two densities out of `TPESampler` itself by wrapping
its private `_compute_acquisition_func`; if a later Optuna renames that
method, the function fails loudly rather than drawing a guess.
"""

from __future__ import annotations

import argparse
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna

ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "content"
    / "ms2a-machine-learning-practice"
    / "assets"
    / "tabular"
)

# The same tokens as ms2a_s3_probabilistic.py, so Session 3's authored figures
# read as one set.
INK = "#1f2933"
MUTED = "#6b7684"
RULE = "#dfe3e8"
ACCENT = "#2f6f9f"      # "good" trials, l(x), TPE
ACCENT_BG = "#e3eef7"
WARM = "#c1553b"        # "bad" trials, g(x), random search
# A single-hue ramp for "trial number" (early = light, late = dark).
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

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
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "legend.frameon": False,
    }
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


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


# --------------------------------------------------------------------------- #
# (a) one TPE step on a 1-D objective, read out of the sampler itself
# --------------------------------------------------------------------------- #
def objective_1d(x):
    """Two basins; the global minimum is the right-hand one (x near 1.5)."""
    return np.sin(3 * x) + 0.12 * (x - 1.0) ** 2


def tpe_anatomy() -> None:
    """Run 100 seeded random trials, then ask TPESampler for trial 101 and
    record what it computes: the good/bad split, l(x), g(x), the 24
    candidates drawn from l(x) and the one it returns.

    The only non-default setting is `n_startup_trials=100`: a uniform history
    keeps the two densities apart on the page. With the default 10, TPE has
    already piled its trials into the best basin by trial 100, and l(x) and
    g(x) overlap there -- the same mechanics, harder to see. The kernels are
    wide because Optuna floors each bandwidth at (high - low) / (n + 2)."""
    lesson = "hyperparameter-optimization"
    low, high, n_done = -4.0, 4.0, 100

    sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=n_done)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda t: float(objective_1d(t.suggest_float("x", low, high))),
        n_trials=n_done,
    )

    captured = {}
    original = sampler._compute_acquisition_func  # private: see module doc

    def spy(samples, mpe_below, mpe_above):
        values = original(samples, mpe_below, mpe_above)
        captured.update(samples=samples["x"].copy(), acq=values.copy(),
                        below=mpe_below, above=mpe_above)
        return values

    sampler._compute_acquisition_func = spy
    trial = study.ask()
    x_next = trial.suggest_float("x", low, high)
    sampler._compute_acquisition_func = original

    cand, acq = captured["samples"], captured["acq"]
    assert np.isclose(cand[np.argmax(acq)], x_next), "capture mismatch"

    # The split TPESampler used: the best ceil(10% * n) values are "good".
    done = sorted(study.trials[:n_done], key=lambda t: t.value)
    n_good = optuna.samplers._tpe.sampler.default_gamma(n_done)
    good, bad = done[:n_good], done[n_good:]
    y_star = 0.5 * (good[-1].value + bad[0].value)

    xs = np.linspace(low, high, 800)
    log_l = captured["below"].log_pdf({"x": xs})
    log_g = captured["above"].log_pdf({"x": xs})
    ratio = np.exp(log_l - log_g)
    cand_ratio = np.exp(acq)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1, 1]})
    a1, a2, a3 = axes

    a1.plot(xs, objective_1d(xs), color=MUTED, linewidth=1.6,
            label="objective (unknown to the sampler)")
    a1.scatter([t.params["x"] for t in bad], [t.value for t in bad],
               s=46, facecolor="white", edgecolor=WARM, linewidth=1.6,
               zorder=3, label=f"bad trials ({len(bad)})")
    a1.scatter([t.params["x"] for t in good], [t.value for t in good],
               s=60, color=ACCENT, zorder=4,
               label=f"good trials ({n_good})")
    a1.axhline(y_star, color=INK, linestyle="--", linewidth=1.1)
    a1.text(high, y_star, "  y*", va="center", ha="left", fontsize=13)
    a1.set_ylabel("loss")
    a1.set_title(f"1. Split the {n_done} finished trials at y*: the best "
                 f"γ = 10% are \"good\"", loc="left")
    a1.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.02),
              fontsize=11.5, handletextpad=0.3, columnspacing=1.2)
    tidy(a1)

    a2.fill_between(xs, np.exp(log_l), color=ACCENT_BG)
    a2.plot(xs, np.exp(log_l), color=ACCENT, linewidth=2.2,
            label="l(x): density of the good trials")
    a2.plot(xs, np.exp(log_g), color=WARM, linewidth=2.2,
            label="g(x): density of the bad trials")
    a2.plot(cand, np.zeros_like(cand), "|", color=ACCENT, markersize=16,
            markeredgewidth=1.8,
            label=f"{len(cand)} candidates drawn from l(x)")
    a2.set_ylabel("density")
    top = max(np.exp(log_l).max(), np.exp(log_g).max())
    a2.set_ylim(-0.02, top * 1.55)        # headroom for the legend
    a2.set_title("2. Fit a Parzen estimator to each group, "
                 "sample candidates from l(x)", loc="left")
    a2.legend(loc="upper left", fontsize=11.5)
    tidy(a2)

    a3.plot(xs, ratio, color=INK, linewidth=2.0, label="l(x) / g(x)")
    a3.scatter(cand, cand_ratio, s=34, color=ACCENT, zorder=3,
               label="candidates")
    a3.scatter([x_next], [cand_ratio.max()], s=320, marker="*",
               color=WARM, edgecolor=INK, linewidth=0.8, zorder=4,
               label=f"next trial: x = {x_next:.2f}")
    a3.set_yscale("log")
    a3.set_ylabel("l(x) / g(x)")
    a3.set_xlabel("x (the hyperparameter)")
    a3.set_title("3. Evaluate the candidate that maximises l(x) / g(x)",
                 loc="left")
    a3.legend(loc="upper left", fontsize=11.5)
    tidy(a3)

    for ax in axes:
        ax.axvline(x_next, color=WARM, linestyle=":", linewidth=1.4,
                   zorder=1)
    a3.set_xlim(low, high)
    fig.tight_layout(h_pad=1.6)
    save(fig, "optuna-tpe-anatomy.png")
    print(f"    {lesson}: next x = {x_next:.4f}, "
          f"good = {n_good}, y* = {y_star:.3f}")


# --------------------------------------------------------------------------- #
# (b) grid vs random vs TPE, same budget, five seeds
# --------------------------------------------------------------------------- #
NAMES = ["a", "b", "c", "d", "e"]   # a, b matter; c, d, e barely do


def synthetic(t) -> float:
    v = [t.suggest_float(n, 0.0, 1.0) for n in NAMES]
    return ((v[0] - 0.71) ** 2 + (v[1] - 0.29) ** 2
            + 0.01 * sum((u - 0.4) ** 2 for u in v[2:]))


def search_race() -> None:
    """Best-so-far loss for the three samplers on a budget of 3**5 = 243
    trials: exactly the full grid of three values per parameter."""
    lesson = "hyperparameter-optimization"
    levels = [1 / 6, 1 / 2, 5 / 6]           # a fair grid: cell centres
    budget, seeds = len(levels) ** len(NAMES), range(5)

    def make(kind, seed):
        if kind == "grid":
            space = {n: levels for n in NAMES}
            return optuna.samplers.GridSampler(space, seed=seed)
        if kind == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        return optuna.samplers.TPESampler(seed=seed)

    curves = {}
    for kind in ("grid", "random", "tpe"):
        runs = []
        for seed in seeds:
            s = optuna.create_study(sampler=make(kind, seed))
            s.optimize(synthetic, n_trials=budget)
            runs.append(np.minimum.accumulate([t.value for t in s.trials]))
        curves[kind] = np.array(runs)

    style = {
        "grid": (MUTED, "--", "grid, 3 values per parameter"),
        "random": (WARM, "-", "random search"),
        "tpe": (ACCENT, "-", "TPE (Optuna default)"),
    }
    n = np.arange(1, budget + 1)
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for kind, runs in curves.items():
        colour, ls, label = style[kind]
        med = np.median(runs, axis=0)
        ax.fill_between(n, runs.min(axis=0), runs.max(axis=0),
                        color=colour, alpha=0.13, linewidth=0)
        ax.plot(n, med, color=colour, linestyle=ls, linewidth=2.4,
                label=label)
        ax.text(budget + 3, med[-1], f"{label}\n{med[-1]:.4f}",
                color=INK, va="center", fontsize=12)
    ax.set_yscale("log")
    ax.axvline(10, color=MUTED, linestyle=":", linewidth=1.3, zorder=0)
    ax.text(12, 0.97, "TPE's random startup ends (trial 10)",
            transform=ax.get_xaxis_transform(), va="top", fontsize=11.5,
            color=INK)
    ax.set_xlim(1, budget)
    ax.set_xlabel("trial number")
    ax.set_ylabel("best loss so far (log scale)")
    ax.set_title("Same budget, same objective (2 important + 3 unimportant "
                 "parameters): median and range over 5 seeds", loc="left",
                 fontsize=13)
    ax.legend(loc="lower left")
    tidy(ax)
    fig.tight_layout()
    save(fig, "optuna-grid-random-tpe.png")
    for kind, runs in curves.items():
        print(f"    {lesson}: {kind:6s} median final best = "
              f"{np.median(runs[:, -1]):.5f}")
    rand_final = np.median(curves["random"], axis=0)[-1]
    tpe_med = np.median(curves["tpe"], axis=0)
    print(f"    {lesson}: TPE median reaches random's final best at trial "
          f"{int(np.argmax(tpe_med <= rand_final)) + 1}")


# --------------------------------------------------------------------------- #
# (c) a real study: the lesson's LightGBM recipe on the Adult census table
# --------------------------------------------------------------------------- #
def run_lightgbm_study(n_trials: int = 60):
    """The lesson's recipe, verbatim apart from an in-memory study (no
    `storage=`), a progress print, and the three LightGBM flags
    (`deterministic`, `force_row_wise`, `n_jobs`) that make a re-run draw
    the same PNGs."""
    import lightgbm as lgb
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, train_test_split

    adult = fetch_openml(data_id=1590, as_frame=True)
    X_all, y_all = adult.data, (adult.target == ">50K").astype(int)
    X, _, y, _ = train_test_split(X_all, y_all, train_size=12_000,
                                  stratify=y_all, random_state=0)
    CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def sample_params(trial):
        return dict(
            learning_rate=trial.suggest_float(
                "learning_rate", 1e-2, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 256, log=True),
            min_child_samples=trial.suggest_int(
                "min_child_samples", 5, 200, log=True),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float(
                "reg_lambda", 1e-3, 100.0, log=True),
            n_estimators=5000,      # a ceiling: early stopping picks
            subsample=0.8, subsample_freq=1,
            random_state=0, verbose=-1,
            deterministic=True, force_row_wise=True, n_jobs=4,
        )

    def objective(trial):
        params, scores, trees = sample_params(trial), [], []
        for k, (tr, va) in enumerate(CV.split(X, y)):
            X_fit, X_es, y_fit, y_es = train_test_split(
                X.iloc[tr], y.iloc[tr], test_size=0.2,
                stratify=y.iloc[tr], random_state=k)
            model = lgb.LGBMClassifier(**params)
            model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)],
                      eval_metric="auc",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            p = model.predict_proba(X.iloc[va])[:, 1]
            scores.append(roc_auc_score(y.iloc[va], p))
            trees.append(model.best_iteration_)
            trial.report(float(np.mean(scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("n_trees", int(np.median(trees)))
        return float(np.mean(scores))

    def progress(study, t):
        shown = t.value if t.value is not None else float("nan")
        print(f"    trial {t.number:2d} {t.state.name:8s} {shown:.4f} "
              f"{t.duration.total_seconds():5.1f}s", flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=1),
    )
    study.enqueue_trial({"learning_rate": 0.1, "num_leaves": 31,
                         "min_child_samples": 20, "colsample_bytree": 1.0,
                         "reg_lambda": 1e-3}, skip_if_exists=True)
    study.optimize(objective, n_trials=n_trials, callbacks=[progress])
    return study


def study_history(study) -> None:
    lesson = "hyperparameter-optimization"
    from optuna.trial import TrialState

    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    best = np.maximum.accumulate([t.value for t in done])

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.axvspan(-0.5, 9.5, color=ACCENT_BG, zorder=0)
    ax.text(4.5, 0.97, "random startup\n(n_startup_trials = 10)",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=11.5, color=INK)
    ax.scatter([t.number for t in pruned],
               [t.intermediate_values[t.last_step] for t in pruned],
               marker="x", s=46, color=MUTED, linewidth=1.6, zorder=3,
               label=f"pruned ({len(pruned)}): mean AUC when stopped")
    ax.scatter([t.number for t in done], [t.value for t in done], s=40,
               color=ACCENT, zorder=4,
               label=f"completed ({len(done)}): 5-fold mean AUC")
    ax.step([t.number for t in done], best, where="post", color=WARM,
            linewidth=2.2, zorder=5, label="best so far")
    ax.annotate("trial 0 = the defaults of the five\nsearched parameters "
                "(enqueue_trial)",
                xy=(0, study.trials[0].value), xytext=(0.2, 0.22),
                textcoords=("data", "axes fraction"), fontsize=11.5,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlim(-1, len(study.trials))
    ax.set_xlabel("trial number")
    ax.set_ylabel("ROC AUC (cross-validated)")
    ax.set_title("Optimisation history: 60 trials of the LightGBM recipe "
                 "on 12,000 Adult census rows", loc="left", fontsize=13)
    ax.legend(loc="lower right")
    tidy(ax)
    fig.tight_layout()
    save(fig, "optuna-history.png")
    print(f"    {lesson}: best AUC = {study.best_value:.4f} at trial "
          f"{study.best_trial.number}; pruned = {len(pruned)}")


def study_importance(study) -> None:
    lesson = "hyperparameter-optimization"
    imp = optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
    names, vals = list(imp)[::-1], list(imp.values())[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.barh(names, vals, color=ACCENT, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=13)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("share of the objective's variance (fANOVA, sums to 1)")
    ax.set_title("Hyperparameter importance, same study", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=13, colors=INK)
    fig.tight_layout()
    save(fig, "optuna-importance.png")
    print(f"    {lesson}: " + ", ".join(f"{k}={v:.3f}"
                                         for k, v in imp.items()))


def study_slice(study) -> None:
    """Optuna's slice plot, redrawn: one panel per parameter, every
    completed trial, coloured by trial number."""
    lesson = "hyperparameter-optimization"
    from matplotlib.colors import LinearSegmentedColormap
    from optuna.trial import TrialState

    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    params = ["learning_rate", "num_leaves", "min_child_samples",
              "colsample_bytree"]
    log = {"learning_rate", "num_leaves", "min_child_samples"}
    cmap = LinearSegmentedColormap.from_list("trial", BLUES)

    fig, axes = plt.subplots(1, len(params), figsize=(13.5, 4.4),
                             sharey=True)
    for ax, p in zip(axes, params):
        sc = ax.scatter([t.params[p] for t in done], [t.value for t in done],
                        c=[t.number for t in done], cmap=cmap, s=36,
                        edgecolor=INK, linewidth=0.4,
                        vmin=0, vmax=len(study.trials))
        if p in log:
            ax.set_xscale("log")
        ax.set_xlabel(p)
        tidy(ax)
    axes[0].set_ylabel("ROC AUC")
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("trial number")
    cb.outline.set_visible(False)
    fig.suptitle("Slice plot: each completed trial against one parameter",
                 x=0.06, ha="left", fontsize=14)
    save(fig, "optuna-slice.png")
    print(f"    {lesson}: slice over {len(done)} completed trials")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="tpe,race,study")
    parts = set(ap.parse_args().only.split(","))
    print("generating hyperparameter-optimization figures ->", ASSETS)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    if "tpe" in parts:
        tpe_anatomy()
    if "race" in parts:
        search_race()
    if "study" in parts:
        study = run_lightgbm_study()
        study_history(study)
        study_importance(study)
        study_slice(study)
    print("done.")


if __name__ == "__main__":
    main()
