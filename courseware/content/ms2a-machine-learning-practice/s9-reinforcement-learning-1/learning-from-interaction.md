# Learning from Interaction

Every model so far was handed a dataset and asked to fit it. A reinforcement
learning agent is handed nothing: it acts, observes what happens, and the data
it will train on tomorrow is a consequence of what it did today.

<!-- notes: 30 minutes. Open with the bandit, not with the Bellman equation —
students who have only seen supervised learning need the "no labels, only a
score" shift before any formalism. Keep the AlphaGo slide short; it is a hook,
not a case study. -->

---
In 2017, AlphaGo defeated Ke Jie, the world's top-ranked Go player.
![AlphaGo against Ke Jie, 2017](assets/rl/alphago.jpg)


---



## The loop

![Agent-environment interaction loop](assets/rl/rl.png)

At each step $t$ the agent observes a state $s_t$, chooses an action $a_t$, and
the environment returns a reward $r_t$ and a next state $s_{t+1}$.

That is the entire interface. Everything else in this session is a way of
turning that stream into a decision rule.

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

## Applications of RL


![game.png](assets/rl/game.png)


- Autonomous Systems : 
Robotics, self-driving vehicles, drone navigation

- Industrial Control : 
Process optimization, energy management

- Natural Language : 
Dialogue systems, chatbot policies

- Financial Markets : 
Trading strategies, risk management


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



---

## The smallest RL problem

![Reinforcement learning, multi-armed bandits and contextual bandits](assets/rl/bandit.png)

A **multi-armed bandit** has $k$ actions, no state, and an immediate reward
drawn from an unknown distribution per action. There is no sequencing, no
credit assignment, and no transition model.

What survives is the one dilemma RL never escapes: pull the arm that looks best,
or pull another one to find out whether it is.


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
