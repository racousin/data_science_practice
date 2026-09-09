# Lab 2 — Ship an Agent to Connect Four

**This one is optional, and it is yours to run.** If you finished Lab 1 with
time to spare, this is where that time goes. There is no checkpoint to wait
for and no step to be walked through — the concepts are below, the notebook
opens in one click, and the leaderboard tells you whether you were right.

**The challenge:** [PettingZoo · Connect-Four](https://ml-arena.com/viewchallenge/65)
— two agents, alternating moves, ranked by **ELO**. You are not scored against
an answer key. You are scored against everybody else's agent, continuously, as
new ones arrive.

<!-- notes: for early finishers of Lab 1, and as homework for everyone else.
Nothing here is gated on the room. If a group asks "how good does it have to
be", point them at the measured table in "Plausible is not correct" — that
table is the lesson, not the agent. -->

---

## Start here — 15 minutes to a live agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml-arena/competition-baseline/blob/main/pettingzoo/agent_baseline.ipynb)

**[Open the starter notebook →](https://colab.research.google.com/github/ml-arena/competition-baseline/blob/main/pettingzoo/agent_baseline.ipynb)**

It runs in the browser — nothing to install, and it is the same Colab you met
in *Notebooks & Colab*. It contains a complete `Agent` that plays a random
legal move, plus the code to deploy it and read the leaderboard.

In the setup cell:

```python
COMPETITION_ID = 65
API_KEY = "mlk_user_..."        # Profile → API Keys, on ml-arena.com
```

Run it top to bottom. You now have an agent on the board — a bad one. That is
the point: get the pipeline working while it is still trivial, then make the
agent good. Everything after this section is about the second half.

Prefer to work locally in your Lab 1 repo? Skip the notebook and jump to
**Publish** at the bottom; the contract is identical.

---

## The contract

One file, `agent.py`, one class named `Agent`. The platform calls exactly four
methods:

```python
class Agent:
    def __init__(self):                                   # zero-arg. Required.
        ...
    def setup(self, observation_space, action_space):     # once, before episode 1
        return True
    def reset(self, env_player_name, episode_index):      # EVERY episode
        return True
    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        ...                                               # return a column, 0-6
```

`reset` is not optional. Seats rotate between episodes — you move first in some
games and second in others — and `reset` is how you are told which you are. An
agent that assumes it always moves first will score about half of what it
should.

| Argument | Shape | Meaning |
|---|---|---|
| `observation` | `(6, 7, 2)` array | plane `0` = **your** pieces, plane `1` = the opponent's |
| `action_mask` | length-7 array | `1` where the column is playable, `0` where it is full |
| `reward` | float | the reward from your previous move |
| `terminated` / `truncated` | bool | the game is over — **return `None`** |

Both planes are written from the point of view of whoever is to move, so the
same code works from either seat — you never need to ask which colour you are.

Row `0` is the **top** of the board and row `5` is the bottom, so a piece
dropped in column `c` lands in the largest `r` with both planes zero at
`(r, c)`. Getting that backwards produces an agent that blocks the wrong
square and still runs.

---

## Three ways to lose without losing a game

None of these raise on your laptop, and the platform does not forgive any of
them:

- **An illegal move.** A column with `action_mask[c] == 0` ends the game
  against you. Honour the mask on every single turn.
- **A crash.** Any exception out of `choose_action` ends the match the same way.
- **Slowness.** You get **250 ms per move**. The call is cut off at 250 ms and
  the run ends — there is no partial credit for a late move.

On the clock: the transport costs 2–4 ms on a typical call, but one call per
run spikes to 100–200 ms (cold start, or the machine scheduling something
else). Every agent pays it. So budget for about **100 ms of thinking on your
worst call**, not 250.

The rules-based agent below runs in well under a millisecond, so none of this
constrains it. It only matters if you go on to search — see **Where to go
next**.

---

## The strategy worth beating

Start here. It is three rules, it is about forty lines, and it is strong enough
to beat most first attempts:

> Consider only columns where `action_mask` is 1, and play the first rule that
> applies:
>
> 1. **Win now.** If dropping in a column gives you four in a row —
>    horizontal, vertical, or either diagonal — play it.
> 2. **Block.** If dropping in a column would give the *opponent* four in a row
>    on their next turn, play it.
> 3. **Centre.** Otherwise play the legal column closest to column 3, breaking
>    ties towards the lower index.
>
> Never return a column whose mask is 0, and never raise.

Rule 2 is the one that decides the lab.

---

## Plausible is not correct

Four hundred games against a uniform-random opponent, seats alternating.
Measured, not estimated — you can reproduce every row with the harness below:

| implementation | mean reward | wins |
|---|---|---|
| uniform random (the floor) | −0.052 | 0.472 |
| **rules 1 and 3 only — wins, never blocks** | **+0.485** | **0.743** |
| rule 3 only — always play the centre-most legal column | +0.780 | 0.890 |
| rules 1 and 2 — win and block, random otherwise | +0.940 | 0.970 |
| **all three rules** | **+0.975** | **0.988** |

Stare at rows two and three. **Dropping the blocking rule scores worse than
having no tactics at all** — an agent that hunts for its own win while ignoring
yours loses to an opponent playing at random. It reads like the smarter program
and it is 25 points worse.

That is the whole reason this lab exists. It is the one bug a coding agent
writes most readily, it survives code review, and no test you thought of
catches it. If your agent lands near +0.49, you have shipped exactly it.

---

## Measure before you believe

Passing tests you wrote is necessary and nowhere near sufficient: every board
in them is one *you* thought of, and the boards that beat you are the ones you
did not. Play a few hundred games instead.

```python
import random
from pettingzoo.classic import connect_four_v3
from connect4 import Agent                       # your module

def play(seed, my_seat):
    env = connect_four_v3.env(); env.reset(seed=seed)
    names, rng, me = list(env.agents), random.Random(seed), Agent()
    me.reset(names[my_seat], seed); reward = 0.0
    for name in env.agent_iter():
        obs, rew, term, trunc, _ = env.last()
        mine = name == names[my_seat]
        if mine:
            reward = rew
        if term or trunc:
            env.step(None); continue
        mask, board = obs["action_mask"], obs["observation"]
        legal = [i for i, ok in enumerate(mask) if ok]
        env.step(me.choose_action(board, rew, False, False, {}, mask)
                 if mine else rng.choice(legal))
    env.close()
    return reward

rs = [play(s, s % 2) for s in range(400)]        # seats alternate on s % 2
print(f"mean {sum(rs)/len(rs):+.3f}  wins {sum(r > 0 for r in rs)/len(rs):.3f}")
```

```bash
uv run --with "pettingzoo[classic]" python selfplay.py
```

`--with` installs pettingzoo for that one run without touching your
`pyproject.toml` — it is a test harness, not a dependency of your package.

`s % 2` alternates which seat you take. Measuring only as the first player
flatters you: moving first in Connect Four is a real advantage.

**Below +0.90, debug rather than submit.**

---

## If you are using a coding agent

Same discipline as the lecture, compressed — this lab is a good one to practise
it on, because the leaderboard grades you honestly a day later:

- **Pin the specification.** Paste the three rules verbatim into the prompt. If
  you let the agent invent the strategy, you have no standing to call it wrong.
- **Plan before code.** Ask for module layout, the win-detection helper, edge
  cases, and the tests it would write — with *no code*. Push back on the plan
  at least once, in writing, before anything is generated.
- **Tests before implementation**, and confirm they fail for the right reason
  first. One test per rule, each on a board you built by hand. For the blocking
  test, make sure the blocking column and the centre column differ, or it
  passes for the wrong reason.
- **Read the diff.** Reject and re-prompt on: a bare `except`, a default value
  for a required argument, a new dependency, an edited test, or a move chosen
  without consulting `action_mask`.

---

## Publish

Your `agent.py` must stand alone — it cannot import from your package, because
only the files you upload exist on the platform.

```bash
uv pip install mlarena-sdk        # imports as `mlarena`
```

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")     # Profile → API Keys
client.submit(competition_id=65, files=["agent.py"], agent_name="<you>-c4")
print(client.status())                               # queue / run / message
print(client.leaderboard(65))
```

Your class must define **every method the starter template declares, including
`__init__`** — upload validation compares your class against the template and
rejects the submission before anything runs.

Then watch it: **[the leaderboard](https://ml-arena.com/viewchallenge/65)**.
Your agent keeps playing after you submit — new matches are scheduled against
the live population every couple of hours, so your ELO moves while you sleep,
and it moves again when somebody else submits something better. You can replay
any match from the challenge page and watch your agent lose in a specific,
diagnosable way.

Submit as often as you like. That is the loop.

---

## Where to go next

Once the three rules are working and you are near +0.97, the interesting part
starts. In rough order of payoff:

- **Look two moves ahead.** Do not play a column that hands the opponent an
  immediate win on top of your piece. Cheap, and it fixes a whole class of
  losses.
- **Minimax with alpha-beta.** Mind the clock: on a mid-game board a plain
  list-of-lists board costs ~26 ms at depth 5, ~65 ms at depth 6, and ~214 ms
  at depth 7 — which is already too close to the 250 ms limit.
- **Iterative deepening.** Search depth 1, then 2, then 3, keeping the best
  move found so far, and stop when your own clock passes ~120 ms. An unlucky
  spike then costs you a shallower move instead of the match. A fixed depth
  tuned to just fit is a bet you lose eventually.
- **Bitboards.** Represent the position as two 49-bit integers and win
  detection becomes four shift-and-mask operations. Worth roughly four extra
  plies for the same time budget.

Connect Four is a solved game — the first player wins with perfect play by
taking the centre. You are not going to get there this week, and you do not
need to: you only need to be better than whoever is above you.
