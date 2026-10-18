# From Tables to Networks

We stored one number per state-action pair. That works up to a few
thousand states and stops dead after. Replacing the table with a network is not
an optimisation — it changes what the algorithm converges to, and sometimes
whether it converges at all.

<!-- notes: 40 minutes. Do the state-count arithmetic on the board, out loud —
it is the moment the room understands why Session 9 was a toy. Spend the second
half on the two DQN fixes; the extensions can go fast, one slide each, they are
vocabulary not technique. -->

---

## The table stops working

| Environment | States | Table entries |
|---|---|---|
| Frozen Lake 4×4 | 16 | 64 |
| Tic-tac-toe | ~5,500 | ~50,000 |
| CartPole | continuous, 4-dim | infinite |
| Atari (4 stacked 84×84 grey frames) | $256^{28224}$ | infinite |
| Go | $\approx 10^{170}$ | infinite |

Two of these fit in a dict. The rest do not fit in the observable universe.

But size is the lesser problem. A table treats every state as unrelated to
every other: learning that a pole leaning 3.1° right needs a push right teaches
it nothing about 3.2°. Generalisation across similar states is what you are
buying; storage is incidental.

---
## Deep Q Learning Overview

Deep Q Learning extends Q Learning by using neural networks:

- Parametrize $Q$ function with $\theta$, $Q_\theta : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$.
- Objective: Find $\theta^*$ that approximates the optimal $Q$ function.
- Define Q target as: $y = R_{t+1} + \gamma \max_{a'} Q_\theta(S_{t+1}, a')$.
- Minimize loss (e.g., MSE): $L(\theta) = \mathbb{E}_{s, a \sim Q}\left[ \left( y - Q(s, a, \theta) \right)^2 \right]$.

---

## What naive DQN actually does

Train a network on transitions as they arrive, with the target computed from
the same network:

- Consecutive transitions are almost identical, so the network fits the last
  two seconds of experience and forgets the rest.
- Every gradient step moves the target. You chase a value that runs away.
- An overestimated action gets selected more, reinforcing the overestimate.

The symptom is a loss that looks fine while the mean Q value climbs past any
achievable return. Log the mean Q value — it is the earliest divergence signal
you have.

---

## Fix 1 — experience replay

Store transitions in a large circular buffer; train on random minibatches.

```python
buffer.append((obs, action, reward, next_obs, float(terminated)))
batch = random.sample(buffer, 64)
```

Sampling uniformly breaks the temporal correlation, so a minibatch looks
something like i.i.d. data — the assumption every optimiser in Session 4 was
built on. Each transition is also reused many times, which matters when the
environment step is the expensive part.

Typical buffer: $10^5$ to $10^6$ transitions, stored as numpy arrays. A million
tuples of tensors will exhaust your RAM.

---

## Fix 2 — the target network

Keep a frozen copy $\theta^-$ of the weights, compute targets with it, and sync
it to the live weights every few thousand steps.

$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]
$$

The target is fixed for the whole sync interval, which turns a moving-goalpost
problem into a sequence of ordinary supervised problems. Sync too often and you
are back to naive DQN; too rarely and you learn slowly against stale values.
1,000 to 10,000 steps is the usual range.

---

## The knobs that matter

| Knob | Typical | What goes wrong |
|---|---|---|
| Buffer size | $10^5$–$10^6$ | too small: forgets; too large: stale off-policy data |
| Batch size | 32–256 | small batches make the TD error estimate noisy |
| Target sync | 1k–10k steps | too fast: divergence; too slow: no progress |
| Learning rate | 1e-4 – 3e-4 | 1e-3 diverges on most control tasks |
| Learning starts | 1k–50k steps | training on 200 transitions overfits them |
| Train frequency | every 1–4 env steps | more updates per step is not free stability |
| $\epsilon$ schedule | 1.0 → 0.05 over 10% of the budget | decaying too fast locks in a bad policy |

Copy a published configuration for your environment family before you tune.
These interact, and a random search over seven of them is weeks of compute.

---

## Prioritized replay — sample what surprises you

Sample transitions with probability proportional to $|\delta|^\alpha$, the
magnitude of their last TD error, instead of uniformly.

Rare high-error transitions get replayed sooner. Biased sampling changes the
expectation of the gradient, corrected by importance-sampling weights with
exponent $\beta$ annealed to 1. It costs a sum-tree, two extra hyperparameters
and a real chance of a subtle bug in the weight correction. Take it from a
library or skip it.
