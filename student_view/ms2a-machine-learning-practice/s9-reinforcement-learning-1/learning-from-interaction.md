# Learning from Interaction

Every model so far was handed a dataset and asked to fit it. A reinforcement
learning agent is handed nothing: it acts, observes what happens, and the data
it will train on tomorrow is a consequence of what it did today.

<!-- notes: 30 minutes. Open with the bandit, not with the Bellman equation —
students who have only seen supervised learning need the "no labels, only a
score" shift before any formalism. Keep the AlphaGo slide short; it is a hook,
not a case study. -->

---

## Three paradigms

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| Data | pairs $(x, y)$ | inputs $x$ | trajectories |
| Signal | the correct answer | none | a scalar reward |
| Feedback | instructive | — | evaluative |
| Timing | immediate | — | delayed |
| Who generates the data | someone else | someone else | the agent |

The last row is the one that matters. In supervised learning the training
distribution is fixed before you start. In RL it moves every time the policy
changes.

---

## What actually changes

**No labels, only a score.** Nobody tells the agent that `LEFT` was the correct
move. It is told the episode ended with a return of 0.31. Whether `LEFT` helped
is something it has to infer.

**Feedback is delayed.** The move that lost the chess game was played forty
plies before the loss. Assigning blame across that gap is the *credit assignment
problem*, and it is the reason RL is hard.

**The data distribution depends on the policy.** A better policy visits
different states, so the training set is non-stationary by construction. Every
i.i.d. assumption from Sessions 2 and 3 is gone.

---

## The loop

![Agent-environment interaction loop](/api/academic_courses/assets/lessons/110/rl.png)

At each step $t$ the agent observes a state $s_t$, chooses an action $a_t$, and
the environment returns a reward $r_t$ and a next state $s_{t+1}$.

That is the entire interface. Everything else in this session is a way of
turning that stream into a decision rule.

---

## The loop in code

```python
obs, info = env.reset(seed=0)
done = False
while not done:
    action = agent.choose_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    agent.learn(obs, reward)
    done = terminated or truncated
```

Every RL algorithm in this course and the next is a different `learn`. The loop
itself never changes — which is why the environment API is worth taking
seriously (see the Gymnasium lesson).

---

## The objective

An episode produces a trajectory $\tau$ and a return $G(\tau)$. The agent looks
for the policy that maximises the expected return.

$$
\pi^* = \arg\max_{\pi} \; \mathbb{E}_{\tau \sim \pi}\left[ G(\tau) \right]
$$

Note where the policy appears: not only inside the expression being maximised,
but in the *distribution* the expectation is taken over. That coupling is what
makes the problem interesting and what makes gradients awkward — Session 10.

---

## What it costs

![AlphaGo against Ke Jie, 2017](/api/academic_courses/assets/lessons/110/alphago.jpg)

AlphaGo beat Ke Jie in 2017 on the back of millions of self-play games, TPU
clusters, and months of training. The result is real; the price tag is the point.

RL is the most sample-hungry family in this course by two or three orders of
magnitude. Budget for a simulator, not for a dataset.

---

## The smallest RL problem

![Reinforcement learning, multi-armed bandits and contextual bandits](/api/academic_courses/assets/lessons/110/bandit.png)

A **multi-armed bandit** has $k$ actions, no state, and an immediate reward
drawn from an unknown distribution per action. There is no sequencing, no
credit assignment, and no transition model.

What survives is the one dilemma RL never escapes: pull the arm that looks best,
or pull another one to find out whether it is.

---

## Three problems, increasing difficulty

![Bandit, contextual bandit and full RL](/api/academic_courses/assets/lessons/110/rl-bandit-diff.png)

| Setting | State | Action affects next state |
|---|---|---|
| Multi-armed bandit | none | no |
| Contextual bandit | sampled i.i.d. | no |
| Full RL | sampled i.i.d. | **yes** |

A/B testing and ad ranking are contextual bandits, and treating them as full RL
buys you nothing but variance. Check the middle column before reaching for
Q-learning.

---

## Epsilon-greedy

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

## UCB

$$
a_t = \arg\max_{a} \left( \hat{\mu}_a + c \sqrt{\frac{\ln t}{N_a}} \right)
$$

Upper Confidence Bound adds an optimism term that shrinks as an arm is pulled
more often. Untried arms look attractive because they are uncertain, not because
they are good.

UCB explores in a directed way and beats $\epsilon$-greedy on bandits with a
provable regret bound. On full RL problems with large state spaces its
advantage largely evaporates, which is why $\epsilon$-greedy survives.

---

## When RL is the wrong tool

Reach for RL only when all four hold:

1. The problem is **sequential** — actions change the situation you face next.
2. You have a **simulator**, or interaction is cheap and safe.
3. You can write down a **reward** that means what you want.
4. No supervised alternative exists — you cannot collect examples of the right
   answer.

> If you can label the correct action, label it and train a classifier. It will
> be faster, cheaper, and easier to debug.

Failure mode: a team writes a reward function as a proxy for the real objective,
and the agent maximises the proxy. Reward specification is where most applied RL
projects die, not algorithm choice.

---

## A map of the field

![Model-based and model-free reinforcement learning](/api/academic_courses/assets/lessons/110/model-based-free.jpg)

| Axis | Left | Right |
|---|---|---|
| Model | model-based: plan with $P$, $R$ | model-free: learn from samples |
| Learns | value-based: $Q$, act greedily | policy-based: $\pi_\theta$ directly |
| Data | on-policy: your own behaviour | off-policy: anyone's |

This session walks left to right and ends at Q-learning: model-free,
value-based, off-policy.
