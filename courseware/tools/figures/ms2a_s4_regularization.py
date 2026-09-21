"""Figures for `s4-advanced-neural-networks/regularization.md` of
`ms2a-machine-learning-practice`. Re-run with:

    uv run --no-project --with torch --with torchvision --with matplotlib \
        --with numpy python courseware/tools/figures/ms2a_s4_regularization.py

(`--rerun` measures again instead of reading build/figures-cache/; a new
value in a sweep is measured on its own.) About 25 minutes on 10 cores.

One setting throughout, chosen so that a network can overfit: an MLP
784-512-512-10 (669,706 parameters) trained on 2,000 MNIST images, AdamW at
1e-3, batch 128, 3,000 steps (192 epochs), three seeds. Every strength is
chosen on the 10,000 validation images; the comparison is scored on the
10,000 test images, which nothing was chosen on. Training accuracy is always
measured in eval() mode on the un-augmented training images.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from ms2a_s4_common import *  # noqa: E402,F403

N, STEPS, SEEDS = 2_000, 3_000, (0, 1, 2)

WD = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
L1 = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
L1_PROX = (1e-3, 3e-3, 1e-2, 3e-2)
DROP = (0.2, 0.3, 0.5)
NOISE = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
SMOOTH = (0.1, 0.2, 0.3, 0.4, 0.5)
SIZES = (500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
STRENGTH = {"wd": "weight_decay", "l1": "l1", "l1_prox": "l1", "dropout": "dropout",
            "noise": "noise", "smooth": "label_smoothing"}


def run(task: dict) -> dict:
    set_threads()
    torch.manual_seed(task["seed"])
    X, y = subset(task.get("n", N))
    _, _, Xv, yv, Xt, yt = mnist()
    model = mlp(512, 2, p_drop=task.get("dropout", 0.0))
    kw = {k: task[k] for k in ("weight_decay", "l1", "l1_prox", "noise", "augment",
                                "label_smoothing") if k in task}
    hist, best = train(model, X, y, steps=task.get("steps", STEPS), seed=task["seed"],
                       eval_every=task.get("eval_every", 0), X_val=Xv, y_val=yv, **kw)
    out = dict(task, train_acc=accuracy(model, X, y), val_acc=accuracy(model, Xv, yv),
               test_acc=accuracy(model, Xt, yt), val_loss=mean_loss(model, Xv, yv),
               weight_norm=sum(p.pow(2).sum() for p in model.parameters()
                               if p.ndim > 1).sqrt().item())
    if task.get("eval_every"):
        out["history"] = hist
        model.load_state_dict(best[1])                  # early stopping's restore
        out.update(es_step=best[2], es_val_acc=accuracy(model, Xv, yv),
                   es_test_acc=accuracy(model, Xt, yt), es_train_acc=accuracy(model, X, y))
    if task.get("keep_first_layer"):
        w = model[0].weight.detach()                    # (512, 784)
        out["pixel_norm"] = w.norm(dim=0).tolist()
        a = w.abs().flatten()
        out["frac_zero"] = (a == 0).float().mean().item()
        out["frac_below_1e3"] = (a < 1e-3).float().mean().item()
    return out


def mean_by(rows, kind, metric="val_acc"):
    vals = {}
    for r in rows:
        if r["kind"] == kind:
            vals.setdefault(r[STRENGTH[kind]], []).append(r[metric])
    return {k: float(np.mean(v)) for k, v in sorted(vals.items())}


def best(rows, kind):
    m = mean_by(rows, kind)
    return max(m, key=m.get)


def measure():
    tasks = []
    for s in SEEDS:
        tasks.append(dict(kind="none", seed=s, eval_every=16))
        tasks += [dict(kind="wd", weight_decay=v, seed=s) for v in WD]
        tasks += [dict(kind="l1", l1=v, seed=s) for v in L1]
        tasks += [dict(kind="l1_prox", l1=v, l1_prox=True, seed=s) for v in L1_PROX]
        tasks += [dict(kind="dropout", dropout=v, seed=s) for v in DROP]
        tasks += [dict(kind="noise", noise=v, seed=s) for v in NOISE]
        tasks += [dict(kind="smooth", label_smoothing=v, seed=s) for v in SMOOTH]
        tasks.append(dict(kind="augment", augment=True, seed=s))
    rows = run_all("s4-reg-sweeps", run, tasks)

    wd, dr, ls = best(rows, "wd"), best(rows, "dropout"), best(rows, "smooth")
    stack = dict(weight_decay=wd, dropout=dr, augment=True)
    comb = run_all("s4-reg-combined", run, [
        dict(kind=k, seed=s, **extra) for s in SEEDS for k, extra in (
            ("combined", stack), ("all", dict(stack, label_smoothing=ls)))])

    # The picture uses L1 = 1e-5 rather than the validation-best strength, which
    # is weaker and harder to see; 1e-5 is within half a point of it.
    first = dict(seed=0, keep_first_layer=True)
    layers = run_all("s4-reg-first-layers", run, [
        dict(first, kind="none"), dict(first, kind="wd", weight_decay=wd),
        dict(first, kind="l1", l1=1e-5),
        dict(first, kind="l1_prox", l1=best(rows, "l1_prox"), l1_prox=True)])

    sizes = run_all("s4-reg-sizes", run, [
        dict(kind=k, n=n, seed=s, steps=6_000, **extra) for n in SIZES for s in SEEDS
        for k, extra in (("none", {}), ("combined", stack),
                         ("noise", dict(noise=best(rows, "noise"))))])
    return rows, comb, layers, sizes


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_weight_decay(rows):
    none = [r for r in rows if r["kind"] == "none"]
    x0 = 0.003                                        # where lambda = 0 is drawn
    lam = [x0] + list(WD)
    def series(metric):
        per = [[r[metric] for r in none]] + [
            [r[metric] for r in rows if r["kind"] == "wd" and r["weight_decay"] == v] for v in WD]
        return np.array([np.mean(p) for p in per]), np.array([np.min(p) for p in per]), \
            np.array([np.max(p) for p in per])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for metric, color, label in (("train_acc", NEUTRAL, "training accuracy (eval mode)"),
                                 ("val_acc", BLUE, "validation accuracy")):
        m, lo, hi = series(metric)
        a1.fill_between(lam, lo, hi, color=color, alpha=0.25, lw=0)
        a1.plot(lam, m, color=color, marker="o", ms=5, label=label)
    m, _, _ = series("val_acc")
    i = int(np.argmax(m))
    a1.annotate(f"best: λ = {lam[i]:g}\n{m[i]:.4f}", (lam[i], m[i]), (lam[i] * 0.06, 0.985),
                color=INK_2, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    a1.set_ylim(0.90, 1.005)
    a1.set_ylabel("accuracy")
    a1.legend(loc="lower left")
    m, lo, hi = series("val_loss")
    a2.fill_between(lam, lo, hi, color=BLUE, alpha=0.25, lw=0)
    a2.plot(lam, m, color=BLUE, marker="o", ms=5)
    a2.set_ylabel("validation cross-entropy")
    a2.set_ylim(0, None)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(lam, ["0"] + [f"{v:g}" for v in WD])
        ax.minorticks_off()
        ax.set_xlabel("weight_decay λ  (AdamW, lr 1e-3, 3,000 steps)")
        top = ax.secondary_xaxis("top", functions=(lambda x: x * 3, lambda x: x / 3))
        top.set_xticks([3 * v for v in WD], [f"{3 * v:g}" for v in WD])
        top.minorticks_off()
        top.set_xlabel("ηλT", color=INK_2)
        top.tick_params(colors=MUTED)
    fig.tight_layout()
    save(fig, "regularization-weight-decay-sweep.png")


def fig_first_layer(layers):
    titles = {"none": "no regularization", "wd": "weight decay (AdamW)",
              "l1": "L1 penalty, through Adam", "l1_prox": "L1, proximal step"}
    vmax = max(max(r["pixel_norm"]) for r in layers)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.7))
    for ax, r in zip(axes, layers):
        img = np.array(r["pixel_norm"]).reshape(28, 28)
        im = ax.imshow(img, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(SPINE)
        strength = r.get("weight_decay", r.get("l1"))
        ax.set_title(titles[r["kind"]] + (f"\nλ = {strength:g}" if strength else "\n"),
                     fontsize=10.5, color=INK)
        ax.text(0.5, -0.06, f"val acc {r['val_acc']:.4f}\n"
                f"|w| < 1e-3: {r['frac_below_1e3']:.1%}\nexactly 0: {r['frac_zero']:.1%}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10, color=INK_2)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cb.set_label("weight norm per pixel", color=INK_2)
    cb.outline.set_visible(False)
    save(fig, "regularization-l1-l2-first-layer.png")


def comparison_rows(rows, comb):
    def agg(sel, prefix=""):
        return (np.mean([r[prefix + "train_acc"] for r in sel]),
                np.mean([r[prefix + "test_acc"] for r in sel]),
                [r[prefix + "test_acc"] for r in sel])
    none = [r for r in rows if r["kind"] == "none"]
    out = [("none", *agg(none)), ("early stopping, best val accuracy restored", *agg(none, "es_"))]
    names = {"wd": "weight decay {:g}", "l1": "L1 {:g}", "dropout": "dropout {:g}",
             "noise": "input noise σ = {:g}", "smooth": "label smoothing {:g}"}
    single = []
    for kind, fmt in names.items():
        v = best(rows, kind)
        single.append((fmt.format(v), *agg([r for r in rows if r["kind"] == kind
                                             and r[STRENGTH[kind]] == v])))
    single.append(("augmentation", *agg([r for r in rows if r["kind"] == "augment"])))
    out += sorted(single, key=lambda row: row[2])
    out.append(("augmentation + dropout + weight decay",
                *agg([r for r in comb if r["kind"] == "combined"])))
    out.append(("the same + label smoothing", *agg([r for r in comb if r["kind"] == "all"])))
    return out


def fig_comparison(rows, comb):
    data = comparison_rows(rows, comb)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    base = data[0][2]
    for i, (name, tr, te, seeds) in enumerate(data[::-1]):
        ax.plot([te, tr], [i, i], color=GRID, lw=4, solid_capstyle="round", zorder=1)
        ax.scatter([tr], [i], s=90, color=NEUTRAL, zorder=2, edgecolor="white", lw=1.5)
        ax.scatter(seeds, [i] * len(seeds), s=14, color=BLUE, alpha=0.45, zorder=3, lw=0)
        ax.scatter([te], [i], s=110, color=BLUE, zorder=4, edgecolor="white", lw=1.5)
        ax.text(1.006, i, f"{te:.4f}   {100 * (te - base):+.1f}",
                ha="left", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(data)), [d[0] for d in data[::-1]], color=INK_2, fontsize=10.5)
    ax.axvline(base, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(0.90, 1.045)
    ax.set_xticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    ax.text(1.006, len(data) - 0.35, "test    points\n          vs none", ha="left",
            va="bottom", fontsize=9, color=INK_2)
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy — test (blue; small dots are the three seeds) and "
                  "training, eval mode (grey)")
    ax.scatter([], [], s=90, color=NEUTRAL, label="training accuracy (eval mode)")
    ax.scatter([], [], s=110, color=BLUE, label="test accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.38, 1.09), ncol=2)
    fig.tight_layout()
    save(fig, "regularization-comparison.png")


def fig_sizes(sizes):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    def curve(kind):
        return np.array([np.mean([r["test_acc"] for r in sizes if r["kind"] == kind
                                  and r["n"] == n]) for n in SIZES])
    none = curve("none")
    series = (("none", ORANGE, "no regularization"), ("noise", AQUA, "input noise"),
              ("combined", BLUE, "augmentation + dropout + weight decay"))
    for kind, color, label in series:
        c = curve(kind)
        a1.plot(SIZES, c, color=color, marker="o", ms=5, label=label)
        if kind != "none":
            a2.plot(SIZES, 100 * (c - none), color=color, marker="o", ms=5, label=label)
    a1.set_ylabel("test accuracy")
    a1.legend(loc="lower right")
    a2.axhline(0, color=INK_2, lw=1)
    a2.set_ylabel("gain over no regularization (points)")
    a2.legend(loc="upper right")
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(SIZES, [f"{n:,}" for n in SIZES])
        ax.minorticks_off()
        ax.set_xlabel("training images (6,000 steps each)")
    fig.tight_layout()
    save(fig, "regularization-vs-training-set-size.png")


def report(rows, comb, layers, sizes):
    for kind in STRENGTH:
        for metric in ("val_acc", "train_acc", "val_loss", "weight_norm"):
            print(kind, metric, {k: round(v, 4) for k, v in mean_by(rows, kind, metric).items()})
    for name, tr, te, seeds in comparison_rows(rows, comb):
        print(f"{name:45s} train {tr:.4f} test {te:.4f} seeds {[round(s, 4) for s in seeds]}")
    for r in layers:
        print("first layer", r["kind"], "val", round(r["val_acc"], 4), "zero",
              round(r["frac_zero"], 4), "below 1e-3", round(r["frac_below_1e3"], 4))
    for kind in ("none", "noise", "combined"):
        print(kind, [round(float(np.mean([r["test_acc"] for r in sizes if r["kind"] == kind
                                          and r["n"] == n])), 4) for n in SIZES])
    for r in rows:
        if r["kind"] == "none":
            h = r["history"]
            lo = min(h, key=lambda t: t[2])
            print(f"seed {r['seed']}: val loss min {lo[2]:.3f} at step {lo[0]} (acc {lo[1]:.4f});"
                  f" best acc {r['es_val_acc']:.4f} at step {r['es_step']};"
                  f" final loss {h[-1][2]:.3f} acc {h[-1][1]:.4f}")


def main() -> None:
    style()
    rows, comb, layers, sizes = measure()
    report(rows, comb, layers, sizes)
    fig_weight_decay(rows)
    fig_first_layer(layers)
    fig_comparison(rows, comb)
    fig_sizes(sizes)


if __name__ == "__main__":
    main()
