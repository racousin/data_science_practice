# Actor-Critic and PPO

Learn the baseline instead of guessing it, then stop the policy from moving so
far in one update that it destroys the data distribution it depends on. Those
two ideas produce PPO, which is the algorithm you should reach for by default.

<!-- notes: 35 minutes. GAE and the clipped objective are the two things they
must leave with. Draw the clip on the board as a piecewise line — the "when is
the gradient zero" question is what makes it click. End on the comparison table
and say the word "default" out loud. -->

---

## Add a critic

REINFORCE waits for the episode to end because it needs $G_t$. A learned value
function removes that constraint — estimate the return instead of observing it.
Two networks, or one trunk with two heads:

- the **actor** $\pi_\theta(a|s)$ — what to do
- the **critic** $V_\phi(s)$ — how good this state is

The critic supplies the baseline; the actor supplies the data the critic is
trained on. Neither optimises the other's loss, which is why this family needs
more care than a supervised model with one objective.

---

## A2C

Advantage Actor-Critic: collect a fixed number of steps from several parallel
environments, compute advantages, take one update on both networks.

```python
adv = returns - values.detach()
pg_loss = -(logp * adv).mean()
v_loss = F.mse_loss(values, returns)
entropy = dist.entropy().mean()
loss = pg_loss + 0.5 * v_loss - 0.01 * entropy
```

All three terms matter. The value coefficient stops the critic dominating the
shared trunk; the entropy bonus keeps the policy from collapsing to a
deterministic one before it has explored — remove it and CartPole frequently
converges to always-left.

---

## A3C, and why nobody runs it

A3C (2016) ran many workers asynchronously, each with its own environment copy,
each pushing gradients to a shared parameter server without locking. It was the
first result showing deep RL working on a CPU cluster.

Then people noticed the asynchrony was not the source of the gain — the
parallel environments were. A2C is A3C made synchronous: same throughput,
reproducible, and it uses the GPU properly because observations arrive
batched.

> If a design's benefit is "it is parallel", check whether a synchronous
> version with the same batch size does as well. It usually does, and you can
> debug it.

---

## Generalised advantage estimation

One-step advantage is biased by the critic's error; Monte-Carlo advantage is
unbiased and high-variance. GAE interpolates, from the TD residual

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

as the building block:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}
$$

An exponentially weighted average of n-step advantages, computed in a single
backward pass over the rollout.

---

## The $\lambda$ trade-off

| $\lambda$ | Estimator | Bias | Variance |
|---|---|---|---|
| 0 | one-step TD advantage | high | low |
| 0.95 | the usual choice | low | moderate |
| 1 | Monte-Carlo advantage | none | high |

$\gamma$ and $\lambda$ do different jobs and are often confused. $\gamma$ is
part of the problem definition — it says how much the future is worth.
$\lambda$ is part of the estimator — it says how much you trust your critic.

Tune $\lambda$ freely. Changing $\gamma$ changes the task.

---

## Why a large update is fatal

In supervised learning a bad step raises the loss and the next step lowers it
again. The dataset does not move.

In RL the policy generates the data. A step that makes the policy much worse
produces a batch of much worse trajectories, from which you learn a still worse
policy. There is no fixed dataset to recover from.

This is the most common failure in a hand-written policy gradient: it learns
for 200k steps, falls off a cliff, and stays there. The learning rate was never
the right knob, because the damage is measured in policy space, not parameter
space.

---

## TRPO — constrain the policy, not the weights

Trust Region Policy Optimization maximises the same surrogate objective subject
to a hard constraint on how far the policy distribution may move:

$$
\bar{D}_{KL}(\pi_{\theta_{old}} , \pi_\theta) \leq \delta
$$

KL divergence, not parameter distance — the quantity that predicts behaviour
change. With $\delta \approx 0.01$ TRPO gives monotonic improvement guarantees,
and it was the first policy gradient method to reliably train locomotion
controllers. The cost is a conjugate-gradient solve against the Fisher
information matrix plus a line search, every update. It works, and almost
nobody implements it.

---

## PPO — the same idea, made cheap

Define the probability ratio between the new and old policies:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

It is 1 before any update, and importance sampling makes reusing the old batch
valid while $r$ stays near 1. PPO enforces that by clipping, not by solving a
constraint:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t , \; clip(r_t(\theta), 1-\epsilon, 1+\epsilon) \, \hat{A}_t \right) \right]
$$

---

## What the clip actually does

Take $\epsilon = 0.2$, so the ratio is clipped to $[0.8, 1.2]$.

| Case | Behaviour |
|---|---|
| $\hat{A} > 0$, $r < 1.2$ | normal gradient, increase the probability |
| $\hat{A} > 0$, $r > 1.2$ | clipped, **gradient is zero** — no further reward |
| $\hat{A} < 0$, $r > 0.8$ | normal gradient, decrease the probability |
| $\hat{A} < 0$, $r < 0.8$ | clipped, gradient is zero |

The `min` makes the objective a pessimistic bound: an update is credited only
inside the trust region, and past it the optimiser gets nothing for pushing
further. Enforced by the loss surface, not by a projection step. The full loss
adds the critic and the entropy bonus:

$$
L = L^{CLIP} - c_1 L^{VF} + c_2 S[\pi_\theta]
$$

---

## The update

```python
for _ in range(n_epochs):
    for idx in minibatches(batch_size):
        ratio = (new_logp[idx] - old_logp[idx]).exp()
        clipped = ratio.clamp(1 - eps, 1 + eps)
        pg_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()
        (pg_loss + 0.5 * v_loss - 0.01 * ent).backward()
```

`old_logp` is stored at collection time and never recomputed. The exponent of a
log-probability difference is numerically far better behaved than a ratio of
probabilities — never compute the ratio directly. And normalise `adv` per
minibatch to zero mean and unit variance: not in the paper's equations, in
every implementation that works.

---

## PPO hyperparameters

| Parameter | Default | Note |
|---|---|---|
| Clip range $\epsilon$ | 0.2 | 0.1 for continuous control, 0.3 is loose |
| Epochs per batch | 4–10 | more epochs push $r$ out of the trust region |
| Rollout length | 128–2048 per env | times the number of envs = batch size |
| Minibatch size | 64–256 | |
| Learning rate | 3e-4, linearly annealed | annealing is worth more than tuning |
| $\gamma$ / $\lambda$ | 0.99 / 0.95 | |
| Entropy coefficient | 0.0–0.01 | raise it if the policy collapses early |
| Value coefficient | 0.5 | |
| Gradient clip | 0.5 | on the global norm |

When PPO stalls, check the fraction of samples being clipped before the
learning rate. Above ~30% the batch is being reused too aggressively: lower the
epoch count.

---

## Choosing

| Algorithm | Actions | Data | Sample efficiency | Stability | Use it when |
|---|---|---|---|---|---|
| DQN | discrete | off-policy | high | fragile | env steps are expensive, actions discrete |
| REINFORCE | any | on-policy | very low | poor | teaching, or a tiny problem |
| A2C | any | on-policy | low | moderate | a cheap fast simulator |
| PPO | any | on-policy | moderate | good | almost always |
| SAC | continuous | off-policy | high | good | robotics, expensive continuous control |

SAC adds a maximum-entropy objective — return *and* policy entropy — which
makes it explore well and stay off-policy-stable. On continuous control with a
slow simulator it beats PPO on sample efficiency by a wide margin.

---

## The default

> Start with PPO. Move to SAC only if your actions are continuous and
> environment steps are the bottleneck. Move to DQN only if your actions are
> discrete and you need off-policy replay.

PPO is the default because it is the least sensitive to getting the details
wrong. It is what trained OpenAI Five, and it is the algorithm behind RLHF in
Session 8. When a deep RL run fails, the useful question is almost never "would
a different algorithm fix this" — it is the subject of the next lesson.
