# Model-Free Prediction and Control

You almost never have $P$. What you have is an environment you can step. This
lesson replaces every expectation in the Bellman equations with a sample, and
that single substitution produces Q-learning.

<!-- notes: 45 minutes, the core of the session and the one they implement in
the lab. Budget 15 minutes for MC/TD prediction and 25 for control. The
on-policy/off-policy distinction is the thing they will get wrong on the exam
and in the lab — do the cliff-walk story on the board. -->

---

## No model

Dynamic programming computed

$$
\sum_{s'} P(s'|s,a) \left( R(s,a) + \gamma V(s') \right)
$$

exactly, because it knew $P$. Without $P$ that expectation is unavailable — but
it is estimable. Every transition the agent takes is a sample from exactly the
distribution you need to average over.

Two ways to use those samples: wait for the episode to end and average the
actual returns (**Monte-Carlo**), or update after every step using your own
current estimate (**temporal difference**).

---

## Monte-Carlo prediction

Run the policy to termination, then average the observed returns per state:

$$
V_{\pi}(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G^{(i)}
$$

**First-visit** MC counts only the first occurrence of $s$ in each episode;
**every-visit** counts them all. Both converge to $V_{\pi}$; first-visit gives
independent samples and the cleaner proof, every-visit is easier to write and
usually fine.

MC is unbiased. $G$ is the return, not an estimate of it.

---

## Monte-Carlo in code

```python
G, returns = 0.0, defaultdict(list)
for s, a, r in reversed(episode):
    G = r + gamma * G
    returns[(s, a)].append(G)
Q = {k: sum(v) / len(v) for k, v in returns.items()}
```

Walk the episode backwards and $G$ accumulates in one pass — computing each
return forwards is quadratic and a classic first implementation.

Two hard constraints: the task must be **episodic**, and nothing is learned
until an episode ends. On a task with 10,000-step episodes, that is 10,000 steps
of doing nothing with the data.

---

## The variance problem

$G$ is a sum of many random rewards through many random transitions. Its
variance grows with the episode length, so MC estimates are noisy and need many
episodes to settle.

| | Monte-Carlo |
|---|---|
| Bias | none |
| Variance | high, grows with horizon |
| Needs termination | yes |
| Bootstraps | no |
| Works on non-Markov states | yes |

The last row is a genuine advantage: MC does not assume the Markov property,
because it never uses a successor's value estimate.

---

## Temporal difference

Do not wait for the episode. After one step you already have a better estimate
of $V(S_t)$ than the one you are holding:

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

$$
V(S_t) \leftarrow V(S_t) + \alpha \delta_t
$$

$\delta_t$ is the **TD error** — the gap between what you predicted and what one
step of reality plus your own next prediction say. Move the estimate a fraction
$\alpha$ of the way towards the target and repeat.

---

## Bootstrapping

$R_{t+1} + \gamma V(S_{t+1})$ is the **TD target**. It contains $V(S_{t+1})$,
which is a guess. Updating a guess towards a guess is *bootstrapping*, and it is
the idea DP and TD share.

It introduces bias — the target is wrong while $V$ is wrong. It buys an enormous
variance reduction, because one reward and one transition are far less random
than a whole trajectory.

> Same substitution as MC, applied one step deep instead of all the way down.
> The bias vanishes as $V$ converges; the variance never comes back.

---

## MC versus TD

| | Monte-Carlo | TD(0) |
|---|---|---|
| Update after | an episode | a step |
| Bias | none | some, while $V$ is wrong |
| Variance | high | low |
| Online / incremental | no | yes |
| Continuing tasks | no | yes |
| Markov assumption | not needed | relied on |
| Converges faster in practice | rarely | usually |

Default to TD. Reach for MC when the state is not really Markov, when episodes
are short, or when you want an unbiased reference to check a TD implementation
against.

---

## Generalised policy iteration

Control is prediction plus a greedy step, exactly as in dynamic programming —
except neither half is ever run to completion.

1. Improve the estimate of $Q$ a little, from experience.
2. Make the policy a little greedier with respect to $Q$.
3. Repeat.

The policy must stay stochastic while learning, or states it currently dislikes
are never visited and their values are never corrected. $\epsilon$-greedy is
how that is enforced.

---

## SARSA

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right)
$$

```python
a = eps_greedy(Q, s, eps)
while not done:
    s2, r, term, trunc, _ = env.step(a)
    a2 = eps_greedy(Q, s2, eps)
    Q[s, a] += alpha * (r + gamma * Q[s2, a2] * (not term) - Q[s, a])
    s, a, done = s2, a2, (term or trunc)
```

The name is the tuple it consumes: $(S, A, R, S', A')$. $A'$ is the action the
agent will *actually take* — including when that action is a random exploratory
one. SARSA evaluates the policy it is following.

---

## Q-learning

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right)
$$

```python
while not done:
    a = eps_greedy(Q, s, eps)
    s2, r, term, trunc, _ = env.step(a)
    Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not term) - Q[s, a])
    s, done = s2, (term or trunc)
```

One symbol differs: $Q(S_{t+1}, A_{t+1})$ became $\max_a Q(S_{t+1}, a)$. The
target no longer refers to what the agent did next, so Q-learning learns the
*greedy* policy while behaving $\epsilon$-greedily. That is what off-policy
means.

---

## The bootstrap must stop at terminal states

`* (not term)` is not decoration. At a terminal state there is no successor, so
the target is $R_{t+1}$ alone.

Omit it and $V$ leaks value backwards from a state that does not exist. On
Frozen Lake the symptom is a Q-table where the holes have positive value and the
agent walks into them. It is the single most common bug in a first tabular
implementation.

Note also that `truncated` — a time limit — is **not** terminal. Bootstrap
through a truncation; do not bootstrap through a termination.

---

## On-policy versus off-policy, concretely

Put a cliff along the bottom row of a gridworld: falling in costs $-100$, and
the shortest path runs right along its edge.

| | Learns | Behaves on the cliff edge |
|---|---|---|
| SARSA | the $\epsilon$-greedy policy | takes the safe detour |
| Q-learning | the greedy policy | hugs the edge, falls sometimes |

Neither is wrong. SARSA accounts for the exploration it will actually do, so it
avoids states where a random action is catastrophic. Q-learning converges to the
optimal *greedy* policy, which is better once exploration is switched off.

Off-policy is the more useful property in general: it lets an agent learn from
replayed, logged, or another agent's data. Session 10 depends on that.

---

## Convergence conditions

Q-learning converges to $Q_*$ with probability 1 given three conditions:

1. Every state-action pair is visited infinitely often.
2. The learning rates satisfy the Robbins-Monro conditions.
3. Rewards are bounded and the MDP is finite.

$$
\sum_{t} \alpha_t(s,a) = \infty
$$

$$
\sum_{t} \alpha_t^2(s,a) < \infty
$$

The first sum diverging means the steps never shrink so fast that you stop
moving; the second converging means the noise eventually averages out.
$\alpha_t = 1/t$ satisfies both; a constant $\alpha$ satisfies only the first.

**GLIE** — Greedy in the Limit with Infinite Exploration — is the matching
condition on $\epsilon$: it must reach 0, but slowly enough that every pair is
still tried infinitely often. $\epsilon_t = 1/t$ works.

---

## What the theory does not say

None of the above tells you what to use on a run of 20,000 episodes. In
practice:

```python
eps = max(0.05, 1.0 * 0.9995 ** episode)
alpha = 0.1
```

- **Constant $\alpha$**, typically 0.05–0.2. It violates Robbins-Monro and it is
  what everyone uses, because a non-vanishing step size tracks a changing target
  and the noise floor is acceptable.
- **Exponentially decayed $\epsilon$** with a floor. Decay too fast and the
  agent commits to whatever it found first; decay too slowly and half the budget
  is spent on noise.
- **Optimistic initialisation** — start $Q$ at a value above any achievable
  return and even a greedy policy explores.

Report the schedule with every result. "Q-learning got 0.74" is not a claim
anyone can reproduce; the lab requires the ablation table for exactly this
reason.
