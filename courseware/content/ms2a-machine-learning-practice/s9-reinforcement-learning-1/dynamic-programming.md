# Dynamic Programming

When you know $P$ and $R$, solving an MDP is a computation, not a learning
problem. Nobody ships dynamic programming — but every model-free algorithm in
the next lesson is DP with the expectation replaced by a sample.

<!-- notes: 35 minutes. Run value iteration live on 4x4 Frozen Lake and print
the value grid every few sweeps; watching value propagate backwards from the
goal is worth more than the convergence proof. State the contraction result,
do not prove it. -->

---

## The setting

Dynamic programming assumes you have the **model**: the full transition
distribution $P(s'|s,a)$ and the reward $R(s,a)$, for every state and action.

That is a strong assumption and it is occasionally true — board games, queueing
systems, inventory control, any simulator whose internals you can read.
Gymnasium's tabular environments expose it directly:

```python
env = gym.make("FrozenLake-v1", is_slippery=True)
P = env.unwrapped.P          # P[state][action] -> [(prob, next_state, reward, done), ...]
print(P[0][1])
```

No interaction, no exploration, no variance. Only arithmetic.

---

## Policy evaluation

Given $\pi$, compute $V_{\pi}$. Initialise $V_0$ arbitrarily and apply the
Bellman expectation equation as an assignment:

$$
V_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left( R(s,a) + \gamma V_k(s') \right)
$$

$V_{\pi}$ is a fixed point of this map. The sequence $V_k$ converges to it from
any starting point, which is why the initialisation genuinely does not matter.

---


## Policy improvement

![The current policy](assets/rl/tikz_picture_16.png)

Start from a policy — any policy — and its value function $V_{\pi}$, computed
by the loop above. The arrows are what the agent currently does; the question is
whether any single-state change would do better.

---

## One step of lookahead

![The action values under that policy](assets/rl/tikz_picture_17.png)

$$
Q_{\pi}(s,a) = \sum_{s'} P(s'|s,a) \left( R(s,a) + \gamma V_{\pi}(s') \right)
$$

Deviate for one action, then fall back on $\pi$. Four numbers per cell, and the
model is what lets you compute them without ever taking the action.

---

## The greedy step

![The improved policy](assets/rl/tikz_picture_18.png)

$$
\pi'(s) = \arg\max_{a} Q_{\pi}(s, a)
$$

The **policy improvement theorem** guarantees $V_{\pi'}(s) \geq V_{\pi}(s)$ for
every $s$ — greedy action on a correct evaluation never makes things worse.

---

## Policy iteration

Alternate the two steps until the policy stops changing:

$$
\pi_0 \rightarrow V_{\pi_0} \rightarrow \pi_1 \rightarrow V_{\pi_1} \rightarrow \dots \rightarrow \pi^*
$$



There are finitely many deterministic policies and each round strictly improves,
so this terminates — usually in a handful of iterations, each of which contains
a full evaluation loop.

---


## Why it converges

It is an application of the Bellman operator $T$, and $T$ is a
$\gamma$-contraction in the sup norm:

$$
\max_{s} | (T V_1)(s) - (T V_2)(s) | \leq \gamma \max_{s} | V_1(s) - V_2(s) |
$$

Banach's fixed-point theorem then gives a unique fixed point and geometric
convergence to it at rate $\gamma$.
