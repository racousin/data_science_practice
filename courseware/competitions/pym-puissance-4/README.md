# pym-puissance-4

Connect Four ("Puissance 4") challenge for the course *Python pour la médecine*,
on prod ML-Arena, created as a copy of challenge 65 (PettingZoo · Connect-Four,
flex_v1, `pettingzoo` env image, ELO-ranked). The challenge id is in `state.json`.

| File | Role |
| --- | --- |
| `create.py` | Idempotent create / verify / benchmark / start / publish script |
| `state.json` | The created challenge id (a re-run never creates a second one) |
| `source_snapshot.json` | Challenge 65 as read at creation (config, evaluation, env.py, overview) |
| `overview.md` | French overview page (`__CHALLENGE_ID__` is filled by `create.py`) |
| `agent_template.py` | French agent template handed to participants |
| `benchmark_agent.py` | Benchmark agent: win now, else block, else centre, else random legal |

Run with the released SDK 3.0.0 (prod does not run the 4.x API):

```bash
PYTHONPATH=<path to mlarena-sdk 3.0.0> python create.py
```

Keys come from `courseware/.env`: `MLARENA_CREATOR_API_KEY` (creator routes) and
`MLARENA_STUDENT_API_KEY` (read-only visibility check). Attaching the challenge to
the course is done separately with the teacher key.
