# Markov Decision Processes

The MDP is the formal object every RL algorithm assumes. Writing your problem
down as one takes ten minutes and tells you immediately whether the tools in
this session apply.

<!-- notes: 40 minutes, the densest lesson of the session. Work the gridworld on
the board in parallel with the slides — every symbol should land on a picture
before it lands on an equation. Do not skip partial observability; it is the
reason half of their real problems are not MDPs. -->

---

## The tuple

![An MDP: states, actions, transitions and rewards](assets/rl/mdp.jpg)

An MDP is $(S, A, P, R, \gamma)$:

- $S$ — the set of states
- $A$ — the set of actions
- $P$ — the transition model, $P(s'|s, a)$
- $R$ — the reward function, $R(s, a)$
- $\gamma$ — the discount factor, in $[0, 1]$

If you cannot name all five for your problem, you do not yet have an RL problem.

---

## The Markov property

$$
P(S_{t+1} | S_t, A_t) = P(S_{t+1} | S_0, A_0, \dots, S_t, A_t)
$$

The next state depends on the current state and action, and on nothing earlier.
The state is a **sufficient statistic for the future** — the history adds
nothing once you know it.

This is not a property of the world. It is a property of your *encoding* of the
world, and you control it.

---

## What breaks it

Partial observability. The agent sees an observation $o_t$ that is a lossy
function of the true state, and the process over observations is not Markov.

| Symptom | Fix |
|---|---|
| Velocity not in the observation | stack the last $k$ frames |
| Opponent's hidden cards | carry a belief state |
| Sensor noise | filter, then feed the estimate |
| Reward depends on an unseen phase | add the phase to the observation |

Frame stacking is the cheap fix and it covers most of what you will meet.
Anything richer is a POMDP, and a recurrent policy — Session 10 territory.

---

## A gridworld

![A 4x4 gridworld with an agent and a goal](assets/rl/tikz_picture_1.png)

Sixteen cells, one agent (A), one goal (G). $S$ has 16 elements; $A$ has four —
up, down, left, right — so $Q$ is a 16-by-4 array of numbers.

Small enough to fit in a table, which is exactly why every RL course starts
here. Everything in Session 10 is what you do when the table no longer fits.

---

## Transitions

![A stochastic transition spreads probability over several successors](assets/rl/tikz_picture_5.png)

Deterministic: `RIGHT` moves you right with probability 1, and $P$ is a lookup
table. Stochastic: `RIGHT` moves you right with probability 0.8 and sideways
with probability 0.1 each, and $P$ is a distribution.

Frozen Lake is the second kind. That single fact is why a policy which looks
obviously optimal scores 0.7 rather than 1.0 — and why you must never conclude
anything from one episode.

---

## Rewards

![A sparse goal reward](assets/rl/tikz_picture_6.png)

The sparse version: $+1$ on reaching the goal, 0 everywhere else. Honest, and
brutally hard to learn from — the agent must stumble into the goal by accident
before it has any signal at all.

The shaped alternative: a small penalty per step, a penalty for holes, a bonus
at the goal. Easier to learn, and now you are optimising something that is not
quite the task. Both choices cost you something; pick deliberately.

---

## Policies

![A policy assigns an action to each state](assets/rl/tikz_picture_8.png)

A **deterministic** policy is a map $\pi: S \rightarrow A$. A **stochastic**
policy is a distribution $\pi(a|s)$ over actions in each state.

Every MDP with a finite state space admits an optimal *deterministic* policy —
so stochasticity is never required for optimality here. It is required for
exploration during learning, and for partially observable or adversarial
settings where being predictable is exploitable.

---

## Trajectories and return

![A trajectory through the gridworld](assets/rl/tikz_picture_9.png)

A trajectory is $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots)$, and its return
from step $t$ is the discounted sum of what follows:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The agent maximises $G_t$ in expectation, not $R_{t+1}$ — the whole difference
between reinforcement learning and greedy control.

---

## Why gamma exists

![Discounted value along a trajectory, gamma = 0.95](assets/rl/tikz_picture_11.png)

1. **Convergence.** On a continuing task the undiscounted sum diverges; with
   $\gamma < 1$ and bounded rewards, $G_t$ is finite.
2. **Horizon.** $\gamma$ sets an effective horizon of about $1/(1-\gamma)$ steps
   — 100 at $\gamma = 0.99$.
3. **Uncertainty.** A distant reward is predicted by a model you trust less.

Set $\gamma$ from the horizon your task has. A $\gamma$ of 0.9 on a task needing
200-step lookahead will not learn, and the symptom looks like a bug in the
update rule.

---

## The value function

![State values in the gridworld](assets/rl/tikz_picture_13.png)

$$
V_{\pi}(s) = \mathbb{E}_{\pi} \left[ G_t | S_t = s \right]
$$

"How good is this state, if I keep behaving like $\pi$?" It is defined relative
to a policy — there is no such thing as the value of a state on its own.

---

## The action-value function

![Action values in the gridworld](assets/rl/tikz_picture_14.png)

$$
Q_{\pi}(s, a) = \mathbb{E}_{\pi} \left[ G_t | S_t = s, A_t = a \right]
$$

Take $a$ now, follow $\pi$ afterwards. $Q$ is the more useful of the two: given
$Q$ you can act without a model, by taking $\arg\max_a Q(s,a)$. Given only $V$
you need $P$ to look one step ahead.

That is why the algorithms you will ship learn $Q$.

---

## The Bellman expectation equations

The value of where you are is the reward you expect plus the discounted value of
where you land.

$$
V_{\pi}(s) = \sum_{a} \pi(a|s) \left( R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_{\pi}(s') \right)
$$

$$
Q_{\pi}(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q_{\pi}(s',a')
$$

Both are linear systems in the unknowns. For a finite MDP with a known model you
could solve them exactly; the next lesson iterates instead, because iteration
scales further and generalises to the model-free case.

---

## Optimality

An optimal policy $\pi^*$ satisfies $V_{\pi^*}(s) \geq V_{\pi}(s)$ for every
policy $\pi$ and every state $s$ — a single policy dominates everywhere at once.
That is a theorem, not a definition, and it is what makes the problem tractable.

$$
V_*(s) = \max_{a} \left( R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_*(s') \right)
$$

$$
Q_*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q_*(s',a')
$$

The sum over actions became a max. That is the only change, and it is what makes
these equations non-linear — no closed form, hence iteration.

---

## Reading the answer

![The optimal policy in the gridworld](assets/rl/tikz_picture_12.png)

Once you have $Q_*$, the optimal policy falls out with no further work:

$$
\pi^*(s) = \arg\max_{a} Q_*(s, a)
$$

Solving for $Q_*$ *is* solving the MDP. Everything that follows approximates
that one quantity.

---

## The same picture, a different problem

![State, action and reward in a video game](assets/rl/game.png)

State: the screen. Actions: the buttons. Reward: the score delta. The MDP does
not care that one problem is a 4x4 grid and the other is 60 frames per second of
RGB — only that you can name the five components.

> Write the tuple down before you write any code. If $S$ is not Markov, or $R$
> does not encode what you actually want, no algorithm will save you.
