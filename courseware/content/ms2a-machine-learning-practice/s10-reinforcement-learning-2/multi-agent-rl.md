# Multi-Agent RL

Add a second learning agent and the environment stops being an environment. It
becomes another policy that is changing while you train against it, which
breaks the assumption every algorithm so far has rested on.

<!-- notes: 35 minutes. Non-stationarity first — it is the whole lesson in one
idea. The PettingZoo API and action masks are what they need for the lab, so
demo the AEC loop live. Finish on ELO because it is the number the platform
puts next to their name. -->

---

## The environment now contains a learner

An MDP assumes fixed transition dynamics. From one agent's point of view the
other agents are part of those dynamics — and they are updating their
weights.

The environment you learned against last epoch no longer exists. A policy that
was optimal against the opponent's old behaviour can be worthless against the
new one, and your replay buffer is full of transitions generated under dynamics
that will never recur.

> Non-stationarity is not a difficulty on top of RL. It removes the guarantee
> that made the algorithms converge in the first place.

---

## Two more things break

**Credit assignment across agents.** Five agents cooperate, the team scores one
point — which agent caused it? A shared reward gives all of them the same
gradient, including the one that did nothing: the *lazy agent* problem. Some
environments give per-agent rewards; most interesting ones do not.

**There is no optimum to converge to.** With one agent there is an optimal
policy. With several, the best policy depends on what the others do, so the
solution concept is an equilibrium: a set of policies where no agent gains by
changing unilaterally. Learning dynamics can cycle around it forever —
rock beats scissors beats paper beats rock — without anything being wrong.

---

## Three settings

| Setting | Rewards | Example | The hard part |
|---|---|---|---|
| Cooperative | shared | warehouse robots, traffic control | credit assignment |
| Competitive | zero-sum | Connect Four, Pong, Go | non-stationarity, exploitability |
| Mixed | partly aligned | trading, negotiation, autonomous driving | both, plus the equilibrium is not unique |

Zero-sum two-player games are the best-understood case and the one ML-Arena
mostly runs: there is a well-defined notion of a stronger policy, and self-play
converges usefully in practice.

---

## PettingZoo

PettingZoo is Gymnasium's multi-agent counterpart, from the same maintainers,
with two APIs for two kinds of environment.

| | AEC | Parallel |
|---|---|---|
| Turn structure | one agent acts at a time | all agents act simultaneously |
| Fits | board and card games | pursuit, particle worlds, most cooperative tasks |
| Loop | `agent_iter()` + `last()` | `step(dict_of_actions)` |
| Agent set | can change mid-episode | can change between steps |

Every AEC environment converts to Parallel and back, but converting Parallel to
AEC is only faithful when simultaneity does not matter.

---

## The AEC loop

```python
env = connect_four_v3.env()
env.reset(seed=0)
for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    action = None if (termination or truncation) else policy(agent, obs)
    env.step(action)
```

Three things catch people: `last()` returns the reward for the agent's
*previous* action, not the one it is about to take; a terminated agent must
still be stepped, with `None`; and `agent_iter()` cycles until every agent is
done, so the loop ends on its own.

---

## The Parallel loop

```python
observations, infos = env.reset(seed=0)
while env.agents:
    actions = {a: policy(a, observations[a]) for a in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

Dictionaries keyed by agent id, everywhere. `env.agents` shrinks as agents
finish, so iterate over it rather than over `possible_agents` — indexing a
dictionary with an agent that has already terminated is a `KeyError`, and that
is the correct behaviour.

---

## Action masks

In a board game most actions are illegal in most states. PettingZoo puts the
legal set in the observation:

```python
mask = obs["action_mask"]                       # 1 = legal
logits = net(obs["observation"])
logits[mask == 0] = -1e8
action = torch.distributions.Categorical(logits=logits).sample()
```

Mask the **logits**, not the sampled action; renormalising after the fact gives
gradients that push probability toward illegal moves.

An illegal action is not a small mistake — most environments end the episode and
award the loss, so an unmasked policy loses to a random one. It is the most
common reason a first submission scores below the baseline.

---

## Independent learners

The baseline that costs nothing: give every agent its own single-agent
algorithm, treat the others as part of the world, and train. Independent PPO —
IPPO — is exactly this.

It has no convergence guarantee at all and is a strong baseline anyway: on
cooperative benchmarks it matches or beats purpose-built multi-agent algorithms
often enough that a 2020 paper was titled to make the point. PPO's trust region
already limits how fast each policy moves, so the others' non-stationarity
stays slow relative to the learning rate. Start here; reach for a multi-agent
algorithm once you can show independent learners plateau below what you
need.

---

## Centralised training, decentralised execution

The critic may see everything. The policy may not.

During training, the value function takes the joint observation and joint
action of all agents — the training-time state is fully observable to you, even
if it is not to any single agent. At execution each agent runs its own policy on
its own local observation.

This resolves both problems at once: a centralised critic sees a stationary
environment because it conditions on what the others actually did, and it can
attribute the shared reward. MADDPG and MAPPO are the names to know. The
constraint is that it needs control of the training setup — which you have in a
simulator and do not in a live system.

---

## Self-play

Train against copies of yourself. The opponent improves exactly as fast as you
do, so difficulty stays calibrated: an automatic curriculum, no hand-designed
opponents.

![AlphaGo against Ke Jie, 2017](assets/rl/alphago.jpg)

AlphaGo Zero reached superhuman Go from random initialisation with self-play, a
policy-value network and tree search, and no human games. Naive self-play
forgets, though: training only against your current self produces a policy that
beats the current self and loses to the version from twenty iterations ago.

---

## The league

Keep a population. Sample opponents from a pool of past checkpoints rather than
only the latest, and the policy has to stay good against everything it used to
beat.

AlphaStar formalised this as a league: a main agent, past copies of itself, and
deliberately trained **exploiters** whose only job is to find and punish its
weaknesses. Closer to a tournament than to a training loop, and it produced a
StarCraft II policy that was hard to counter rather than merely strong on
average.

For a course lab, a pool of your last five checkpoints plus a random agent and
one scripted opponent is enough to see the effect.

---

## Evaluating a multi-agent policy

There is no fixed benchmark to report. "Mean reward" is meaningless without
naming the opponent, and a policy that beats opponent A can reliably lose to
opponent B that A beats — strength is not transitive.

What you report instead:

- Win / draw / loss rate against a **named** opponent pool.
- Both sides of an asymmetric game. Rotate the roles and report each.
- Results against opponents you did **not** train against. Beating only your
  own training pool is the multi-agent form of overfitting.
- The number of games. A 60% win rate over 20 games is not a result.

---

## ELO — what ML-Arena computes

For a two-player competition the platform ranks by ELO rather than by mean
reward. The expected score of A against B is

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$

and after a game with outcome $S_A$ (1 win, 0.5 draw, 0 loss):

$$
R_A \leftarrow R_A + K (S_A - E_A)
$$

A 400-point gap means a 10-to-1 expected score. Two consequences for the lab:
your rating depends on who else submitted, so it moves without you touching
your agent; and it is only meaningful after enough games — early positions on a
young leaderboard are mostly noise.
