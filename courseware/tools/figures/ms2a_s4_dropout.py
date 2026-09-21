"""Figures for `s4-advanced-neural-networks/dropout.md` of
`ms2a-machine-learning-practice`. Re-run with:

    uv run --no-project --with torch --with torchvision --with matplotlib \
        --with numpy python courseware/tools/figures/ms2a_s4_dropout.py

(`--rerun` measures again instead of reading build/figures-cache/.)

The setting of the Regularization lesson: two hidden ReLU layers, 2,000 MNIST
training images, AdamW at 1e-3, batch 128, 3,000 steps, three seeds, strengths
chosen on the 10,000 validation images and scored on the 10,000 test images.
Dropout sits after each hidden ReLU. The BatchNorm experiment uses 10,000
training images instead, because that is where BatchNorm is used.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from ms2a_s4_common import *  # noqa: E402,F403

N, STEPS, SEEDS = 2_000, 3_000, (0, 1, 2)
WIDTHS = (64, 256, 1024)
RATES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)
TS = (1, 2, 3, 5, 10, 20, 50, 100)


def run_rate(task: dict) -> dict:
    set_threads()
    torch.manual_seed(task["seed"])
    X, y = subset(N)
    _, _, Xv, yv, Xt, yt = mnist()
    m = mlp(task["width"], 2, p_drop=task["p"])
    train(m, X, y, steps=STEPS, seed=task["seed"])
    return dict(task, train_acc=accuracy(m, X, y), val_acc=accuracy(m, Xv, yv),
                test_acc=accuracy(m, Xt, yt), val_loss=mean_loss(m, Xv, yv))


def mc_probs(m: nn.Module, X: torch.Tensor, T: int, gen_seed: int) -> torch.Tensor:
    """(T, N, 10) softmax outputs with every Dropout left on."""
    m.eval()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()
    torch.manual_seed(gen_seed)
    with torch.no_grad():
        out = torch.stack([m(X).softmax(1) for _ in range(T)])
    m.eval()
    return out


def fashion_test() -> torch.Tensor:
    import torchvision

    ds = torchvision.datasets.FashionMNIST(DATA, train=False, download=True)
    return ds.data.reshape(-1, 784).float() / 255.0


def run_mc(seed: int) -> dict:
    set_threads()
    torch.manual_seed(seed)
    X, y = subset(N)
    *_, Xt, yt = mnist()
    m = mlp(1024, 2, p_drop=0.5)
    train(m, X, y, steps=STEPS, seed=seed)
    with torch.no_grad():
        m.eval()
        p_eval = m(Xt).softmax(1)
    P = mc_probs(m, Xt, max(TS), gen_seed=100 + seed)
    single = (P.argmax(2) == yt).float().mean(1)            # one mask each
    by_T = {str(T): (P[:T].mean(0).argmax(1) == yt).float().mean().item()
            for T in TS}
    nll_eval = F.nll_loss(p_eval.log(), yt).item()
    nll_mc = F.nll_loss(P[:50].mean(0).clamp_min(1e-12).log(), yt).item()
    # Rejection: drop the least certain predictions first.
    pm = P[:50].mean(0)
    entropy = -(pm * pm.clamp_min(1e-12).log()).sum(1)
    maxprob = p_eval.max(1).values
    correct_eval = (p_eval.argmax(1) == yt).float()
    correct_mc = (pm.argmax(1) == yt).float()
    keep = np.linspace(1.0, 0.5, 11)
    def curve(score, correct):                              # higher score = keep
        order = score.argsort(descending=True)
        return [correct[order[: int(round(k * len(order)))]].mean().item() for k in keep]
    # Out of distribution: Fashion-MNIST through the digit model.
    Xf = fashion_test()
    with torch.no_grad():
        m.eval()
        pf_eval = m(Xf).softmax(1)
    Pf = mc_probs(m, Xf, 50, gen_seed=200 + seed).mean(0)
    ent_f = -(Pf * Pf.clamp_min(1e-12).log()).sum(1)
    def auroc(s_in, s_out):                                 # P(out scores higher)
        s = torch.cat([s_in, s_out]); lab = torch.cat([torch.zeros(len(s_in)), torch.ones(len(s_out))])
        r = s.argsort().argsort().float() + 1
        n1, n0 = len(s_out), len(s_in)
        return ((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()
    return dict(seed=seed, eval_acc=(p_eval.argmax(1) == yt).float().mean().item(),
                single_mean=single.mean().item(), single_min=single.min().item(),
                single_max=single.max().item(), by_T=by_T, nll_eval=nll_eval, nll_mc=nll_mc,
                keep=keep.tolist(), curve_maxprob=curve(maxprob, correct_eval),
                curve_mc=curve(-entropy, correct_mc),
                auroc_maxprob=auroc(-maxprob, -pf_eval.max(1).values),
                auroc_mc=auroc(entropy, ent_f),
                fashion_maxprob_mean=pf_eval.max(1).values.mean().item())


def run_bn(task: dict) -> dict:
    """Dropout then Linear then BatchNorm: how far is BN's running variance
    from the variance it meets once dropout is switched off?"""
    set_threads()
    torch.manual_seed(task["seed"])
    X, y = subset(10_000)
    *_, Xt, yt = mnist()
    p = task["p"]
    if task["order"] == "before":
        m = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p),
                          nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
                          nn.Linear(512, 10))
        bn = m[4]
    else:
        m = nn.Sequential(nn.Linear(784, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
                          nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
                          nn.Dropout(p), nn.Linear(512, 10))
        bn = m[4]
    train(m, X, y, steps=2_000, seed=task["seed"])
    acc = accuracy(m, Xt, yt)
    seen = {}
    h = bn.register_forward_hook(lambda mod, i, o: seen.__setitem__("x", i[0]))
    accuracy(m, X, y)                                       # eval-mode pass
    h.remove()
    ratio = (seen["x"].var(0) / bn.running_var).median().item()
    # The fix: re-estimate the running statistics with dropout switched off.
    bn.reset_running_stats(); bn.momentum = None            # cumulative average
    m.eval(); bn.train()
    with torch.no_grad():
        for i in range(0, len(X), 500):
            m(X[i:i + 500])
    acc_recal = accuracy(m, Xt, yt)
    return dict(task, test_acc=acc, var_ratio=ratio, test_acc_recal=acc_recal)


class FunctionalDropout(nn.Module):
    """The bug: F.dropout's `training` argument defaults to True."""

    def __init__(self, fixed: bool):
        super().__init__()
        self.fixed = fixed
        self.fc1, self.fc2, self.out = nn.Linear(784, 512), nn.Linear(512, 512), nn.Linear(512, 10)

    def forward(self, x):
        kw = {"training": self.training} if self.fixed else {}
        x = F.dropout(F.relu(self.fc1(x)), 0.5, **kw)
        x = F.dropout(F.relu(self.fc2(x)), 0.5, **kw)
        return self.out(x)


def run_functional(seed: int) -> dict:
    set_threads()
    out = {}
    X, y = subset(N)
    *_, Xt, yt = mnist()
    for fixed in (False, True):
        torch.manual_seed(seed)
        m = FunctionalDropout(fixed)
        train(m, X, y, steps=STEPS, seed=seed)
        m.eval()
        with torch.no_grad():
            a, b = m(Xt[:5]), m(Xt[:5])
        out["fixed" if fixed else "bug"] = dict(
            test_acc=accuracy(m, Xt, yt), same_twice=bool(torch.equal(a, b)))
    return out


# Width is ordered, so it gets one hue from light to dark, not three hues.
WIDTH_COLORS = {64: "#8fbbef", 256: "#2a78d6", 1024: "#12407a"}


def fig_rates(rates):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for w in WIDTHS:
        m = np.array([np.mean([r["val_acc"] for r in rates if r["width"] == w
                               and r["p"] == p]) for p in RATES])
        kw = dict(color=WIDTH_COLORS[w], marker="o", ms=5, label=f"width {w}")
        a1.plot(RATES, m, **kw)
        a2.plot(RATES, 100 * (m - m[0]), **kw)
    a1.set_ylabel("validation accuracy")
    a2.set_ylabel("change against p = 0 (points)")
    a2.axhline(0, color=INK_2, lw=1)
    for ax in (a1, a2):
        ax.set_xticks(RATES)
        ax.set_xlabel("dropout p, after each of 2 hidden layers (2,000 images)")
        ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "dropout-rate-by-width.png")


def fig_mc(mc):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    by_t = np.array([[r["by_T"][str(T)] for T in TS] for r in mc])
    a1.plot(TS, by_t.mean(0), color=BLUE, marker="o", ms=5,
            label="average of T random masks (train-mode dropout)")
    ev = np.mean([r["eval_acc"] for r in mc])
    a1.axhline(ev, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    a1.text(TS[-1], ev + 0.0006, f"eval(): weights scaled, one pass  {ev:.4f}",
            ha="right", va="bottom", fontsize=9.5, color=INK_2)
    a1.set_xscale("log")
    a1.set_xticks(TS, [str(t) for t in TS]); a1.minorticks_off()
    a1.set_xlabel("masks averaged, T")
    a1.set_ylabel("test accuracy")
    a1.legend(loc="lower right")
    keep = np.array(mc[0]["keep"])
    for key, color, label in (("curve_maxprob", ORANGE, "reject lowest max-probability, eval()"),
                              ("curve_mc", BLUE, "reject highest MC-dropout entropy, T = 50")):
        c = np.mean([r[key] for r in mc], 0)
        a2.plot(100 * (1 - keep), c, color=color, marker="o", ms=4, label=label)
    a2.set_xlabel("% of test predictions rejected, least certain first")
    a2.set_ylabel("accuracy on the predictions kept")
    a2.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "dropout-ensemble-and-uncertainty.png")


def main():
    style()
    rates = run_all("s4-drop-rates", run_rate, [
        dict(width=w, p=p, seed=s) for w in WIDTHS for p in RATES for s in SEEDS])
    for w in WIDTHS:
        for p in RATES:
            sel = [r for r in rates if r["width"] == w and r["p"] == p]
            print(f"width {w:4d} p {p:.1f} val {np.mean([r['val_acc'] for r in sel]):.4f}"
                  f" test {np.mean([r['test_acc'] for r in sel]):.4f}"
                  f" train {np.mean([r['train_acc'] for r in sel]):.4f}")
    fashion_test()                  # download once here, not in three workers at once
    mc = run_all("s4-drop-mc", run_mc, list(SEEDS))
    for r in mc:
        print({k: v for k, v in r.items() if not k.startswith("curve") and k != "keep"})
    bn = run_all("s4-drop-bn", run_bn, [
        dict(order=o, p=p, seed=s) for o in ("before", "after") for p in (0.2, 0.5)
        for s in SEEDS])
    for r in bn:
        print(r)
    print(run_all("s4-drop-functional", run_functional, list(SEEDS)))
    fig_rates(rates)
    fig_mc(mc)


if __name__ == "__main__":
    main()
