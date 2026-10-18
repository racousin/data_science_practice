# Markov Decision Processes

The MDP is the formal object every RL algorithm assumes.

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

## A gridworld

![A 4x4 gridworld with an agent and a goal](assets/rl/tikz_picture_1.png)

Sixteen cells, one agent (A), one goal (G). $S$ has 16 elements; $A$ has four —
up, down, left, right.

---



## State/Observation Space $$S$$


16 discrete states.


![tikz_picture_2.png](assets/rl/tikz_picture_2.png)


---

## Action Space $$A$$

4 discrete actions (Up, Down, Left, Right).


![tikz_picture_3.png](assets/rl/tikz_picture_3.png)



---

## Transitions Model $$P$$ - Deterministic

$$\mathcal{P}_{ss'}^{a} = \mathbb{P}\left[ S_{t+1} = s' \mid S_t = s,\, A_t = a \right]$$
![tikz_picture_4.png](assets/rl/tikz_picture_4.png)


---

## Transitions Model $$P$$ - Stochastic

$$\mathcal{P}_{ss'}^{a} = \mathbb{P}\left[ S_{t+1} = s' \mid S_t = s,\, A_t = a \right]$$
![A stochastic transition spreads probability over several successors](assets/rl/tikz_picture_5.png)







---

## Rewards

$$r=R(s,a)=r(s')$$

![A sparse goal reward](assets/rl/tikz_picture_6.png)

The sparse version: $+1$ on reaching the goal, 0 everywhere else.

---

## Policies

![A policy assigns an action to each state](assets/rl/tikz_picture_8.png)

A **deterministic** policy is a map $\pi: S \rightarrow A$. A **stochastic**
policy is a distribution $\pi(a|s)$ over actions in each state.


---

## Trajectories and return

![A trajectory through the gridworld](assets/rl/tikz_picture_9.png)

A trajectory is $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots)$, and its return
from step $t$ is the discounted sum of what follows:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

---

## Why gamma exists

![Discounted value along a trajectory, gamma = 0.95](assets/rl/tikz_picture_11.png)

1. **Convergence.** On a continuing task the undiscounted sum diverges; with
   $\gamma < 1$ and bounded rewards, $G_t$ is finite.
2. **Horizon.** $\gamma$ sets an effective horizon of about $1/(1-\gamma)$ steps
   — 100 at $\gamma = 0.99$.
3. **Uncertainty.** A distant reward is predicted by a model you trust less.




---

## The objective

An episode produces a trajectory $\tau$ and a return $G(\tau)$. The agent looks
for the policy that maximises the expected return.

$$
\pi^* = \arg\max_{\pi} \; \mathbb{E}_{\tau \sim \pi}\left[ G(\tau) \right]
$$


![tikz_picture_12.png](assets/rl/tikz_picture_12.png)



---



## Epsilon-greedy


![explore_vs_exploit.jpeg](assets/rl/explore_vs_exploit.jpeg)


```python
def choose_action(self, state):
    if random.random() < self.epsilon:
        return random.randrange(self.n_actions)
    return int(np.argmax(self.Q[state]))
```

With probability $\epsilon$ act at random; otherwise act greedily. Three lines,
no tuning theory, and it is still the default for tabular control.

Fix $\epsilon$ and the agent never stops exploring, so it never converges to a
greedy policy. Decay it and it does. The schedule is a hyperparameter, not a
detail.


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
$Q$ you can act without a model, by taking $\arg\max_a Q(s,a)$.

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

Once you have $Q_*$, the optimal policy falls out with no further work:

$$
\pi^*(s) = \arg\max_{a} Q_*(s, a)
$$

Solving for $Q_*$ *is* solving the MDP. Everything that follows approximates
that one quantity.
