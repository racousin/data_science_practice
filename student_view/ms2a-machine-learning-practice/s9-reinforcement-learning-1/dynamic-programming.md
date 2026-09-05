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

## Evaluation in code

```python
def sweep(P, pi, V, gamma):
    return np.array([
        sum(p * (r + gamma * V[s2]) for p, s2, r, _ in P[s][pi[s]])
        for s in range(len(V))])
```

The sum over actions has collapsed because `pi` is deterministic. Iterate
`sweep` until `np.abs(V_new - V).max() < theta`; call that loop `evaluate`.

Use the threshold as the only stopping rule. A fixed iteration count silently
returns an unconverged value function, and the bug surfaces three steps later as
a bad policy.

---

## Policy improvement

![The current policy](/api/academic_courses/assets/lessons/112/tikz_picture_16.png)

Start from a policy — any policy — and its value function $V_{\pi}$, computed
by the loop above. The arrows are what the agent currently does; the question is
whether any single-state change would do better.

---

## One step of lookahead

![The action values under that policy](/api/academic_courses/assets/lessons/112/tikz_picture_17.png)

$$
Q_{\pi}(s,a) = \sum_{s'} P(s'|s,a) \left( R(s,a) + \gamma V_{\pi}(s') \right)
$$

Deviate for one action, then fall back on $\pi$. Four numbers per cell, and the
model is what lets you compute them without ever taking the action.

---

## The greedy step

![The improved policy](/api/academic_courses/assets/lessons/112/tikz_picture_18.png)

$$
\pi'(s) = \arg\max_{a} Q_{\pi}(s, a)
$$

```python
def improve(P, V, gamma, nS, nA):
    Q = np.array([[sum(p * (r + gamma * V[s2]) for p, s2, r, _ in P[s][a])
                   for a in range(nA)] for s in range(nS)])
    return Q.argmax(axis=1)
```

The **policy improvement theorem** guarantees $V_{\pi'}(s) \geq V_{\pi}(s)$ for
every $s$ — greedy action on a correct evaluation never makes things worse.

---

## Policy iteration

Alternate the two steps until the policy stops changing:

$$
\pi_0 \rightarrow V_{\pi_0} \rightarrow \pi_1 \rightarrow V_{\pi_1} \rightarrow \dots \rightarrow \pi^*
$$

```python
while True:
    V = evaluate(P, pi, gamma)
    new_pi = improve(P, V, gamma, nS, nA)
    if np.array_equal(new_pi, pi):
        break
    pi = new_pi
```

There are finitely many deterministic policies and each round strictly improves,
so this terminates — usually in a handful of iterations, each of which contains
a full evaluation loop.

---

## Value iteration

Evaluation to convergence between improvements is wasted work. Truncate it to a
single sweep and fold the max in:

$$
V_{k+1}(s) = \max_{a} \sum_{s'} P(s'|s,a) \left( R(s,a) + \gamma V_k(s') \right)
$$

```python
while True:
    V_new = np.array([max(sum(p * (r + gamma * V[s2]) for p, s2, r, _ in P[s][a])
                          for a in range(nA)) for s in range(nS)])
    if np.abs(V_new - V).max() < theta:
        break
    V = V_new
pi = improve(P, V, gamma, nS, nA)
```

There is no explicit policy during the loop. One is extracted once, at the end.

---

## Which one

| | Policy iteration | Value iteration |
|---|---|---|
| Inner loop | evaluate to convergence | one sweep |
| Iterations to converge | few | more |
| Cost per iteration | high | low |
| Intermediate policy | yes | no |
| Default | — | **this one** |

Value iteration is simpler to write and simpler to stop. Reach for policy
iteration when evaluation is cheap relative to the action-space sweep, or when
you need a usable policy at every step.

---

## Why it converges

Both updates are applications of the Bellman operator $T$, and $T$ is a
$\gamma$-contraction in the sup norm:

$$
\max_{s} | (T V_1)(s) - (T V_2)(s) | \leq \gamma \max_{s} | V_1(s) - V_2(s) |
$$

Banach's fixed-point theorem then gives a unique fixed point and geometric
convergence to it at rate $\gamma$. Stated, not proved — the proof is four lines
and adds nothing operational.

What is operational: the error shrinks by a factor $\gamma$ per sweep, so
$\gamma = 0.999$ is not a free choice. It is roughly a thousand sweeps.

---

## The curse of dimensionality

One sweep touches every state, every action, and every successor: $O(|S|^2|A|)$
in the worst case.

| Problem | States |
|---|---|
| Frozen Lake 4x4 | 16 |
| Frozen Lake 8x8 | 64 |
| Tic-tac-toe | ~5,500 |
| Backgammon | $10^{20}$ |
| Go | $10^{170}$ |
| Anything with a continuous observation | uncountable |

$|S|$ grows exponentially in the number of state variables. Ten binary features
is 1024 states; thirty is a billion. DP dies at about $10^6$ states, and it dies
before that on memory.

---

## So why learn it

Two reasons, both practical.

It is the **reference implementation**. On a small MDP you can compute $V_*$
exactly, then check whether your Q-learning agent got close. That is the only
ground truth you will ever have in RL, and the lab uses it.

It is the **shape** of everything that follows. Monte-Carlo replaces the
expectation with an episode average; TD replaces it with one sampled step. The
sweep over all states becomes a walk through the states you actually visited.
The update is the same update.

---

## When the model is learned

Model-based RL fits $\hat{P}$ and $\hat{R}$ from experience, then plans with
them — MPC, Dyna, MCTS.

$$
\hat{s}_{t+1}, \hat{r}_{t+1} = f_{\theta}(s_t, a_t)
$$

The payoff is sample efficiency: one real transition trains the model, and the
model then generates thousands of imagined ones. The cost is **model bias** —
planning over a long horizon compounds prediction error, and the agent
enthusiastically exploits the places where its model is wrong.

Rule: keep the planning horizon short relative to how much you trust the model.
Model-free is the safer default; go model-based when real interaction is
expensive, slow, or dangerous.
