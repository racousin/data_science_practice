# mlp-project-ds-harness — the MS2A-MLP project challenge

The project of MS2A — Machine Learning Practice: teams build a **harness** around a small LLM
(the models of the GPU VM's offline cache) that solves data-science tasks written in prose.
Live as ML-Arena challenge **194** ("MLP Project — DS-Harness"), the only challenge of the
course module `mlp-project`. Students get the public repository
[racousin/ds-harness](https://github.com/racousin/ds-harness), built from this folder.

```text
overview.md          the challenge page (ML-Arena "overview")
config.py            the package for tools/build_competitions.py
env.py               the platform env: loads private.json / dev.json, calls scoring.run_agent
benchmark_agent.py   the start-gate benchmark: a well-typed placeholder, no model
agent_template.py    the template uploads are validated against (class Agent, solve)
prepare_data.py      builds data/ (gitignored): private.json from the secret seed, dev.json, scoring.py
build_repo.py        assembles dist/ds-harness/, the public repository
build_starter.py     generates the starter notebook (website/.../challenges/mlp-project-ds-harness.ipynb)
kit/                 the starter kit: dsh.py, agent_naive.py (A0), agent_kit_baseline.py (A1), local_eval.py
draft/               the dataset: scoring.py (the source of truth), public/ (dev.json, schema.md,
                     localtest.py), and the PRIVATE generators, reference agent and tests (gitignored)
anchors/             PRIVATE: A2, the instructor harness (gitignored)
experiments/         PRIVATE: measured evidence, red team, fairness audit (gitignored)
thinking/            PRIVATE: design notes (keep out of git: they name task families)
```

The repository is public: nothing that rebuilds the private split or names the families
absent from dev.json may be committed. `build_repo.py` refuses files that mention private
material.

## One scorer, three copies

`draft/scoring.py` is the source. `draft/build_dataset.py dev` copies it to `draft/public/`,
`prepare_data.py` to `data/`, and `build_repo.py` to the repository. The platform runs the
copy uploaded as an env file. All of them must be byte-identical; check with
`shasum draft/scoring.py data/scoring.py dist/ds-harness/scoring.py` and the creator view.

## Commands

```bash
uv run --with pandas --with numpy python prepare_data.py         # data/ (private split, dev, scorer)
python3 build_repo.py                                            # dist/ds-harness/
python3 build_starter.py                                         # the starter notebook
cd draft && uv run --no-project --with pandas --with numpy --with pytest python -m pytest -q tests
cd kit && uv run --no-project --with numpy --with pandas --with pytest python -m pytest -q test_dsh.py
```

## Changing the live env files

The platform refuses env edits on a started challenge. With the creator key: `stop_challenge(194)`,
`upload_env_file` for `scoring.py` / `env.py` (each save syncs the folder to the GPU VM), then
`start_challenge(194)` (the start gate needs the benchmark submission active with a score; an
env edit does not reset it). Check the VM copy at
`/data/competitions/194/environment/187/` and redeploy one submission.

## Grading anchors (private set, platform rules)

| Anchor | Agent | Score |
|---|---|---|
| A0 | `kit/agent_naive.py`, Qwen2.5-1.5B | 0.6 |
| A1 | `kit/agent_kit_baseline.py`, Qwen2.5-1.5B | 13.9 (platform, the leaderboard row "A1 — kit baseline") |
| A2 | `anchors/agent_strong.py`, Qwen3-1.7B, numpy tools | 61.0 (platform, the leaderboard row "A2 — instructor harness", 2026-10-04) |

## The final split

After the freeze (2026-11-20), draw a new private split: delete `data/private_seed.txt`, run
`prepare_data.py`, re-run the red-team and leak checks (`draft/tools/`), upload `private.json`
the same way as the env files, and run each team's chosen submission twice.
