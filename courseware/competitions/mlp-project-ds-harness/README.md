# mlp-project-ds-harness — the MS2A-MLP project challenge

The project of MS2A — Machine Learning Practice: teams build an AI system around a small language
model that answers data-science objectives with one number each. Live as ML-Arena challenge
**194** ("MLP Project — DS-Harness"), the only challenge of the course module `mlp-project`.
Students get the public repository [racousin/ds-harness](https://github.com/racousin/ds-harness),
built from this folder. Spec: platform repo `docs/plan_challenge_194_ds_harness.md`.

```text
overview.md         the challenge page
config.py           settings, metrics, files (the platform side)
env.py              the platform env: private.json (scored run) / 16 dev tasks (test run) -> scoring.run_agent
hf_models.json      the five models mounted in the agent container
agent_template.py   the template uploads are validated against (class Agent, solve)
benchmark_agent.py  the start-gate benchmark: answers 0, reads every file, no model
public/             scoring.py (the source of truth), localtest.py, schema.md, the repository README
kit/                dsh.py, stage1_direct.py, stage2_tool_loop.py (loop body TODO), test_dsh.py
build_repo.py       assembles dist/ds-harness/, the public repository (refuses private markers)
build_starter.py    generates the Colab notebook (website/.../challenges/mlp-project-ds-harness.ipynb)
gen/                PRIVATE: the generators, the split builder, the independent checker
anchors2/           PRIVATE: stage2_solved.py (A1), agent_a2.py (A2), s1_designs.py, the measurements
data/               PRIVATE (gitignored): dev.json, private.json, *_specs.json, private_seed.txt, scoring.py
```

The repository is public: nothing that rebuilds the private split, names its private-only table
types or copies the anchors may be committed. `.gitignore` keeps `gen/`, `anchors2/`, `data/`.

## The data

`python3 -m gen.build` draws both splits: dev 180 (65 without files, 80 tables, 35 fits; gold
keys `answer`, `tol`, `type` shipped) and private 120 (45 / 50 with 20 of private-only table
types / 25), from `DEV_SEED` and `data/private_seed.txt`. Golds within tolerance of 0 are redrawn
(answering 0 scores nothing). `python3 -m gen.check data/dev.json data/dev_specs.json` (and the
private pair) re-solves every gold independently: the files are parsed back under the reading
rules the objective states, fits are refitted with scikit-learn. Both must report 0 mismatches.

## Commands

```bash
python3 -m gen.build && python3 -m gen.check data/dev.json data/dev_specs.json \
                     && python3 -m gen.check data/private.json data/private_specs.json
cp public/scoring.py data/scoring.py                 # the env's copy: byte-identical
python3 build_repo.py                                # dist/ds-harness/ (its own git)
python3 build_starter.py                             # the notebook
cd kit && cp ../public/scoring.py ../data/dev.json . && python3 -m pytest -q test_dsh.py
```

Measurements run on the GPU VM (GPU 0, the platform's torch agent image, a `--rm` container):
`anchors2/README.md`.

## Changing the live challenge

The platform refuses env edits on a started challenge: stop, upload `env.py`, `scoring.py`,
`dev.json`, `private.json`, `hf_models.json`, update the settings, the agent template and the
dataset, re-run the benchmark (expected 0.0), start. Old submissions are deleted when the data
change. `deploy.py` does all of it (`--dry-run` first; creator key from `courseware/.env`).

## The final split

After the freeze (2026-11-03 23:59): delete `data/private_seed.txt`, `gen.build`, `gen.check`,
upload `private.json` the same way, run each team's chosen submission once, and the anchors.
