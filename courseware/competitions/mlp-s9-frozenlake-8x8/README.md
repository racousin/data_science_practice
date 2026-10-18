# mlp-s9-frozenlake-8x8

FrozenLake-v1 8x8 (slippery) challenge for *MS2A — Machine Learning Practice*,
Session 9, on prod ML-Arena: a copy of challenge 48 (Gymnasium · CartPole-v1,
flex_v1, `gymnasium` env image) with its own `env.py`. The challenge id is in
`state.json` (202).

| File | Role |
| --- | --- |
| `create.py` | Idempotent create / configure / benchmark / start / publish script |
| `state.json` | The created challenge id (a re-run never creates a second one) |
| `env.py` | `gym.make("FrozenLake-v1", map_name="8x8")`, driven by `flexkit.run_gymnasium_eval` |
| `overview.md` | Overview page (`__CHALLENGE_ID__` is filled by `create.py`) |
| `agent_template.py` | Template handed to participants: `agent.py` + `q_table.npy` |
| `benchmark_agent.py` | Benchmark: always move right |

Score: mean reward over 100 episodes = success rate. Measured with gymnasium
1.3.0 over 2,000 seeded episodes: random 0.00, always right 0.23 (platform
benchmark run: 0.25), tabular Q-learning (10,000 episodes, ε decay 0.9995, the notebook's recipe) 0.58–0.62 over 4 seeds, value
iteration (γ = 0.99) 0.63.

The Colab starter is
`website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s9-frozenlake-q-learning.ipynb`.

```bash
PYTHONPATH=<mlarena-sdk 4.6> python create.py
```

Keys come from `courseware/.env`: `MLARENA_CREATOR_API_KEY` (creator routes) and
`MLARENA_STUDENT_API_KEY` (read-only visibility check). The module attachment is
declared in `content/ms2a-machine-learning-practice/course.yaml` and made by
`make publish`.
