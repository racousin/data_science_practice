# Reference Playgrounds — AquaControl and Bitcoin Intraday

Reference only — not covered in class. Two live ML-Arena environments are
attached to this module and no session lectures them. They exist so that
Sessions 9 and 10 have somewhere to go after the lab ends: a real board, a real
opponent population, and no marks riding on the result. This page says what each
one is, what the starter scores, and what a good score looks like.

<!-- notes: Self-study. Never lectured. Point at it from the end of Lab 9 and
Lab 10 — the students who finish early are exactly the ones who should be here.
Nothing on this page is graded; say so out loud or half of them will panic. -->

---

## The two, and where they belong

| | AquaControl (#168) | Bitcoin Intraday (#170) |
|---|---|---|
| Extends | Session 9 — tabular control | Session 10 — function approximation |
| Kernel | `flex_v1` (gymnasium loop) | `flex_v1` (gymnasium loop) |
| Action | `MultiBinary(6)` — six valves | discrete: buy / sell / hold |
| Observation | `Box(15,)` | minute bars |
| Episode | one simulated day, 1440 steps | continuous — the position carries |
| Graded | no | no |

Both use the same agent contract as Lab 9 and Lab 10, so a working lab agent is
already a valid submission with the policy swapped out.

---

## The contract both use

`flexkit`'s gym loop drives your agent: zero-argument constructor, one `setup`,
then one `choose_action` per step.

```python
class Agent:
    def setup(self, observation_space, action_space):
        from flexkit.spaces import decode_space
        self.action_space = decode_space(action_space)
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        return self.action_space.sample()
```

`setup` receives the spaces **dict-encoded**, not as Gymnasium objects —
`encode_space(gym.spaces.MultiBinary(6))` is the dict
`{'type': 'multi_binary', 'shape': [6]}`. Call `decode_space` before you touch
`.sample()`, `.n` or `.shape`, or you get `AttributeError: 'dict' object has no
attribute 'sample'` on the first step. This is the single most common way a
lab-shaped agent dies on the platform.

---

## AquaControl (#168) — the task

You operate a small drinking-water network: pump P1, six valves, two reservoirs,
two city zones with a morning and an evening demand peak. Each of the 1440 steps
is one simulated minute, and you set the open/closed state of all six valves.

Reward per step is $\frac{1}{2}(\text{supply}_N + \text{supply}_S) \cdot dt$,
minus a small pressure penalty, minus a **terminal `break_penalty` of 100** if
you burst a pipe. The five ways to burst one — tank overflow, deadhead,
overpressure, dry run — end the episode immediately.

That single 100 is the whole shape of the problem: a full clean day is worth
about 24 reward units, so one break costs four days of perfect operation.

---

## AquaControl (#168) — the numbers

**Metric** — mean episode reward. **Higher is better.**

| Agent on the board | Score | What it is |
|---|---|---|
| Ceiling | ~24 | a perfect operator, never breaks a pipe (from the competition page) |
| `Arwen` | **−49.98** | best on the board |
| `baseline-valve-heuristic` | **−98.33** | the shipped valve heuristic |
| `__benchmark__` | **−99.95** | the platform reference |
| `Jane Austen` | **−99.95** | — |

Read on the live board on 2 September 2026; reproduce with
`client.leaderboard(168)`.

A score near **−100** means you burst a pipe — the terminal penalty dominates
every other term, so the number tells you nothing about your control policy.
The first target is not efficiency, it is **surviving all 1440 steps**. Work out
the range: a clean day is worth at most about +24, and the pressure penalty
alone cannot cost more than about 2.4 over 1440 minutes, so a no-break episode
cannot score below roughly −2.4. Every agent on the board, the best of them at
−49.98, is therefore still bursting pipes in some episodes. Clear that and you
are top of the board. The cheapest way to fail is to open V1 with V2 and V4 both
closed: the pump deadheads and the pipe bursts on the first step.

---

## Bitcoin Intraday (#170) — the task

Stateful intraday BTCUSD trading on minute bars: buy, sell or hold, and **your
open position carries across runs**. The environment runs hourly, so the agent
you deploy tonight is still holding tomorrow morning's position. Reward is
mark-to-market fractional return.

Everything in the paragraph above comes from the competition's own description
field. Its overview page is still the platform's default placeholder, so there
is no written space specification. Until there is, discover the spaces rather
than guessing them:

```python
def setup(self, observation_space, action_space):
    from flexkit.spaces import decode_space
    print("obs:", decode_space(observation_space),
          "act:", decode_space(action_space))     # lands in your run log
    return True
```

---

## Bitcoin Intraday (#170) — the numbers

**Metric** — mean fractional return per run. **Higher is better.**

| Agent on the board | Score | Runs |
|---|---|---|
| `carry-test` | **+0.000468** | 2,112 |
| doing nothing | **0.000000** | — |
| `__benchmark__` | **−0.000024** | 2,174 |
| `Groupg7` | **−0.000040** | 1,733 |

Read on the live board on 2 September 2026; reproduce with
`client.leaderboard(170)`.

Two things follow. First, the platform reference is **negative** — it loses
money, so a flat "never trade" policy already beats it, and beating the
reference here is not evidence of anything. Second, the board is configured to
display two decimal places against a metric of order $10^{-4}$, so every row on
the web page currently renders as `0.00`. The numbers above are real; read them
with `client.leaderboard(170)` and ignore the rendered column until the
competition's display precision is fixed.

---

## Submitting to either

Same three lines as every other competition on the platform. The package is
`mlarena-sdk` and it imports as `mlarena`; `pip install mlarena` is an unrelated
project with no `connect`.

```python
import mlarena, os

client = mlarena.connect(api_key=os.environ["MLARENA_API_KEY"])
client.submit(competition_id=168, files=["agent.py"])   # or 170
print(client.status())
```

Neither board counts towards anything. That is the point: they are where you
find out that your Lab 10 agent had a `decode_space` bug, in a week when it
costs you nothing.

---

## Check yourself

1. Your AquaControl agent scores −99.9. Before you touch the policy, what does
   that number tell you happened, and what is the first target?

   **Answer.** You burst a pipe. A no-break day scores between about −2.4 and
   +24, so −99.9 is the terminal `break_penalty` of 100 and nothing else — the
   score is not reporting your control policy at all. The first target is
   surviving all 1440 steps, not efficiency.

2. Run this. You should get exactly the output shown.

   ```python
   encoded = {"type": "multi_binary", "shape": [6]}   # what setup() receives
   print(hasattr(encoded, "sample"))                  # -> False
   try:
       encoded.sample()
   except AttributeError as exc:
       print(exc)          # -> 'dict' object has no attribute 'sample'
   ```

   **Answer.** `setup` is handed dict-encoded space specs, not Gymnasium
   objects. `decode_space` turns that dict back into the object whose
   `.sample()` you were about to call.

3. On Bitcoin Intraday the platform reference `__benchmark__` scores −0.000024.
   Why is "I beat the benchmark" a weak claim on this board, and what is the
   honest comparison?

   **Answer.** The reference is negative, so it loses money and a policy that
   never trades already beats it. The honest comparison is against **0.000000**
   — doing nothing — and against `carry-test` at +0.000468.
