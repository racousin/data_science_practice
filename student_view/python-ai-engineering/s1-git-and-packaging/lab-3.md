# Lab 2 — Ship an Agent to Connect Four

You will drive an agent through the loop from *Coding Agents* to build a
Connect-Four player, prove it works before you believe it, and put it on a
leaderboard where other people's agents get a vote.

**The challenge:** [PettingZoo · Connect-Four](https://ml-arena.com/viewchallenge/65),
competition `65`. Two agents, alternating moves, ranked by **ELO** — you are not
scored against a fixed answer key, you are scored against everybody else.

**Deliverable:** a merged pull request carrying a `connect4` module and a
`RETRO.md`, **and** your agent on the leaderboard of competition 65.

<!-- notes: 45 minutes in the room, the pull request as homework. The thing to
say out loud at the start: this challenge cannot be self-marked. Lab 1's
checks compared you to a specification; this one compares you to other
students, and a plausible-looking agent loses. That is the whole point of
putting the agentic lesson in front of it. -->

---

## Why this lab and not another feature

Three properties, and each one closes a hole the previous labs could not:

- **You cannot mark your own work.** There is no reference answer to diff
  against. The only honest evidence is games played.
- **Plausible and correct come apart visibly.** An agent that reads well and
  never blocks loses every game. You will see that in a number.
- **It is small.** The whole agent is about forty lines. The work is in
  specifying it, testing it, and refusing what the agent gets wrong.

---

## Part A — The contract

Your submission is one file, `agent.py`, exposing a class named `Agent`. The
platform's loop calls exactly these four methods:

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
        ...
```

`reset` is not optional. Seats rotate between episodes — you play first in some
games and second in others — and `reset` is how you are told which you are.

---

## Part A — What `choose_action` receives

| Argument | Shape | Meaning |
|---|---|---|
| `observation` | `(6, 7, 2)` array | plane `0` = **your** pieces, plane `1` = the opponent's |
| `action_mask` | length-7 array | `1` where the column is playable, `0` where it is full |
| `reward` | float | the reward from your previous move |
| `terminated` / `truncated` | bool | the game is over — **return `None`** |

Return an integer column, `0`–`6`.

Row `0` is the **top** of the board and row `5` is the bottom, so a piece
dropped in column `c` lands in the largest `r` with both planes zero at
`(r, c)`. Getting that backwards produces an agent that blocks the wrong
square and still runs.

---

## Part A — Three ways to lose without losing a game

The platform does not forgive these, and none of them raise on your laptop:

- **An illegal move.** A column with `action_mask[c] == 0` is a no-contest and
  the game is scored against you. Honour the mask on every single turn.
- **A crash.** Any exception out of `choose_action` ends the match the same way.
- **Slowness.** You get **0.5 s per move**. A heuristic takes microseconds; a
  search you did not bound does not.

---

## Part B — Branch, and give the agent its context

If you still have the throwaway `agent-sandbox` branch from the lecture, throw
it away first: `git restore . && git switch main && git branch -D agent-sandbox`.

1. `git switch main && git pull && git switch -c feature/connect4`
2. Confirm `git status` is clean — this is what makes `git diff` mean "what the
   agent did".
3. Create or extend `CLAUDE.md` (or `CONVENTIONS.md` for Aider) with:
   - the install / test / lint commands
   - your `src/` layout and test-mirroring convention
   - the fail-fast rule: no defaults for required arguments, no bare `except`
   - one explicit "do not": no new dependencies without asking

Write it yourself, or run `/init` and then **edit it** — an unedited `/init`
draft does not count.

---

## Part C — The pinned strategy

Hand your agent this, verbatim. It is short on purpose: a specification you
wrote is a specification you can hold the agent to, and "play well" is not one.

> On your turn, consider only columns where `action_mask` is 1, and pick the
> first rule that applies:
>
> 1. **Win now.** If dropping in a column gives you four in a row —
>    horizontal, vertical, or either diagonal — play it.
> 2. **Block.** If dropping in a column would give the *opponent* four in a
>    row on their next turn, play it.
> 3. **Centre.** Otherwise play the legal column closest to column 3, breaking
>    ties towards the lower index.
>
> Never return a column whose mask is 0, and never raise.

Rule 2 is the one that decides the lab. Part E puts a number on it.

---

## Part C — Ask for the plan, not the code

```text
> Read src/ and tests/. Here is the specification for a Connect-Four agent,
> which is fixed and not up for negotiation:
> <paste the three rules, verbatim>
> The platform calls Agent.choose_action(observation, reward, terminated,
> truncated, info, action_mask); observation is (6,7,2) with plane 0 = my
> pieces, action_mask is length 7. Propose how to build it: module layout,
> the win-detection helper, the edge cases, and the tests you would write.
> Do not write any code.
```

Pasting the rules is the whole trick. Without them the agent invents a
strategy, you have no standing to call it wrong, and you find out from the
leaderboard a day later.

**Save the plan** into `RETRO.md` under `## Plan`. Then push back on it at least
once — a real objection, in writing, before any code exists.

---

## Part D — Tests before implementation

```text
> Write the tests from the plan. Do not write the implementation.
```

Read every test. At minimum you must have one per rule, each built from a board
you constructed by hand:

- **Win now.** Three of your pieces in a row with an open fourth column → the
  agent plays that column.
- **Block.** Three *opponent* pieces in a row with an open fourth column → the
  agent plays that column. Set it up so the blocking column and the centre
  column differ, or the test passes for the wrong reason.
- **Win beats block.** A board where both are available → the agent takes the
  win.
- **Centre.** An empty board → column 3.
- **Mask.** A board with column 3 full → the agent never returns 3.

---

## Part D — Confirm they fail first

If the generated tests do not fail for the right reason, they are not tests.

```bash
uv run --all-extras pytest -v
```

If your Lab 1 `pyproject.toml` declares pytest under `[dependency-groups] dev`
— which is what Lab 1 asked for — a plain `uv run pytest` is enough and
`--all-extras` is a harmless no-op.

---

## Part E — Implement, then play real games

```text
> Now implement it so the tests pass. Do not modify the tests.
```

Passing tests is necessary and nowhere near sufficient — every board in them is
one *you* thought of, and the boards that beat you are the ones you did not.

So play four hundred games against a random opponent before you believe
anything. The harness on the next slide is fifteen lines and it is the only
evidence in this lab that does not come from your own imagination.

---

## Part E — The self-play harness

Save this as `selfplay.py`, next to your package:

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
```

---

## Part E — Run it

```python
rs = [play(s, s % 2) for s in range(400)]        # 400 games, seats alternating
print(f"mean {sum(rs)/len(rs):+.3f}  wins {sum(r > 0 for r in rs)/len(rs):.3f}")
```

```bash
uv run --with "pettingzoo[classic]" python selfplay.py
```

`--with` installs pettingzoo for that one run without touching your
`pyproject.toml`. It is a test harness, not a dependency of your package —
adding it to your dependencies is exactly the "do not" you wrote in Part B.

`s % 2` alternates which seat you take. Measuring only as the first player
flatters you: in Connect Four moving first is a real advantage.

---

## Part E — The numbers you are aiming at

Four hundred games against a uniform-random opponent, seats alternating.
Measured, not estimated — you can reproduce every row with the script above:

| implementation | mean reward | wins |
|---|---|---|
| uniform random (the floor) | −0.052 | 0.472 |
| **rules 1 and 3 only — wins, never blocks** | **+0.485** | **0.743** |
| rule 3 only — always play the centre-most legal column | +0.780 | 0.890 |
| rules 1 and 2 — win and block, random otherwise | +0.940 | 0.970 |
| **all three rules — the pinned specification** | **+0.975** | **0.988** |

Stare at rows two and three. **Dropping the blocking rule scores worse than
having no tactics at all** — an agent that hunts for its own win while ignoring
yours loses to an opponent playing at random. It reads like the smarter program
and it is 25 points worse.

That is what "plausible, not correct" costs, as a number. If your agent lands
near +0.49, you have almost certainly shipped exactly that bug.

**Below +0.90, do not submit — debug.** Then read the diff yourself:

```bash
git diff
```

**Reject and re-prompt** if you see any of: a bare `except`, a default value for
a required argument, a new dependency, an edited test, or a move chosen without
consulting `action_mask`.

---

## Part F — Submit

Copy your agent into a flat directory as `agent.py`. It must not import from
your package — only the files you upload exist on the platform. Then:

```bash
uv pip install mlarena-sdk
```

```python
import mlarena

client = mlarena.connect(api_key="mlk_user_...")   # from your Profile page
client.submit(competition_id=65, files=["agent.py"], agent_name="<you>-c4")
print(client.status())                             # queue_info / run_info / message
```

The package is `mlarena-sdk`; it imports as `mlarena`. Your class must define
**every method the starter template declares, including `__init__`** — upload
validation compares your class against the template and rejects the submission
before anything runs.
