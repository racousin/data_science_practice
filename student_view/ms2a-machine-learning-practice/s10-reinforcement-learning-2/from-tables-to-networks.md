# From Tables to Networks

Session 9 stored one number per state-action pair. That works up to a few
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

## Function approximation

Replace the lookup with a parametric function: a network $Q_\theta$ that maps a
state to one value per action. The learning problem becomes finding $\theta$
that minimises the value error over the states you actually visit:

$$
\overline{VE}(\theta) = \sum_s \mu(s) \left[ v_\pi(s) - \hat{v}(s, \theta) \right]^2
$$

$\mu(s)$ is the on-policy state distribution — the fraction of time the agent
spends in $s$. You are not fitting the value function everywhere, only where
the agent goes, which is itself a function of the policy you are learning.

---

## The semi-gradient update

Take the tabular Q-learning update and swap the table entry for a network
output:

$$
\theta \leftarrow \theta + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q_\theta(S_{t+1}, a') - Q_\theta(S_t, A_t) \right] \nabla_\theta Q_\theta(S_t, A_t)
$$

The bracket is the TD error from Session 9; the gradient is new.

It is a **semi**-gradient because the target also depends on $\theta$ and we do
not differentiate through it. Not laziness: the full gradient performs worse in
practice and corresponds to no Bellman operator. In PyTorch the semi- is a
single `.detach()`.

---

## The deadly triad

Three ingredients, each individually fine. Together they can make the value
estimates diverge to infinity:

| Ingredient | Why you want it |
|---|---|
| Function approximation | the only way past a few thousand states |
| Bootstrapping | targets from your own estimates — TD, not Monte-Carlo |
| Off-policy training | learn from data another policy generated |

Tabular Q-learning has two of the three and converges; naive deep Q-learning
has all three and does not.

> Divergence in deep RL is usually the triad, not a bug in your code. The fix
> is structural, not a smaller learning rate.

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

## The training step

```python
q = net(obs).gather(1, action.unsqueeze(1)).squeeze(1)
with torch.no_grad():
    target = reward + gamma * (1 - done) * target_net(next_obs).max(1).values
loss = F.smooth_l1_loss(q, target)
opt.zero_grad(); loss.backward()
nn.utils.clip_grad_norm_(net.parameters(), 10.0)
opt.step()
```

`torch.no_grad()` is the semi-gradient. `(1 - done)` cuts the bootstrap at
terminal states — forget it and the agent learns that dying is worth
$\gamma V(s)$ of future reward. Huber loss bounds the gradient when one
transition produces a huge TD error.

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

## Double DQN — the overestimation bias

The `max` in the target both **selects** the action and **evaluates** it. With
noisy estimates, the action that gets selected is disproportionately the one
whose noise is positive, so the target is biased upward — systematically, not
on average zero.

Decouple them: select with the live network, evaluate with the target one.

$$
y = r + \gamma \, Q_{\theta^-} \left( s', \arg\max_{a'} Q_\theta(s', a') \right)
$$

Three lines of change, no extra parameters, better on nearly every benchmark.
There is no reason to run plain DQN.

---

## Dueling — separate value from advantage

Split the head into a scalar state value and a per-action advantage, then
recombine:

$$
Q_\theta(s,a) = V_\theta(s) + A_\theta(s,a) - \frac{1}{|A|} \sum_{a'} A_\theta(s, a')
$$

The subtraction makes the decomposition identifiable; without it $V$ and $A$
can drift by any constant. It helps where most actions are equivalent: the
state value is nearly all the signal, and the network stops re-deriving it once
per action.

---

## Prioritized replay — sample what surprises you

Sample transitions with probability proportional to $|\delta|^\alpha$, the
magnitude of their last TD error, instead of uniformly.

Rare high-error transitions get replayed sooner. Biased sampling changes the
expectation of the gradient, corrected by importance-sampling weights with
exponent $\beta$ annealed to 1. It costs a sum-tree, two extra hyperparameters
and a real chance of a subtle bug in the weight correction. Take it from a
library or skip it.

---

## n-step returns

Bootstrap after $n$ steps instead of one:

$$
y = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q_{\theta^-}(s_{t+n}, a')
$$

Less bias — more of the target is real reward — at the cost of more variance,
and technically incorrect off-policy because the intermediate actions came from
an older policy. In practice $n = 3$ is used everywhere, the objection does not
bite, and it is the cheapest speed-up here: a deque of length $n$ in the replay
writer.

---

## Rainbow

Rainbow (Hessel et al., 2017) is the combination, and the ablation study that
tells you which parts carry the result.

| Component | Contribution |
|---|---|
| Double Q-learning | moderate |
| Prioritized replay | large |
| Dueling | small |
| n-step returns | large |
| Distributional (C51) | large |
| Noisy nets | moderate |

n-step, prioritization and distributional targets do most of the work; dueling
is nearly free and nearly irrelevant. Read the ablation, not the headline
number — it is the standard by which "our method combines six tricks" papers
should be judged.

---

## What DQN cannot do

Every target contains a maximum over actions, `max_a' Q(s', a')`. Computing it
means enumerating the actions.

With a steering angle in $[-1, 1]$ there is nothing to enumerate: the max is
itself a continuous optimisation problem, solved at every training step and
every inference step. Discretising into bins works for one dimension and
explodes combinatorially for six.

DQN also cannot represent a stochastic optimal policy — its policy is `argmax`,
deterministic by construction, and there are games where any deterministic
policy is exploitable.

Both limitations have the same answer: stop learning values and learn the
policy itself.
