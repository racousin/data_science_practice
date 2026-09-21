# Lab 4 — Building Blocks, Measured

One network, one training function, one Weights & Biases project. The notebook of
this lab changes one building block at a time — activation, normalisation, dropout
and weight decay, optimiser and schedule, loss, an embedding — trains, and sends you
to the W&B workspace to read the difference. The data is MNIST with the pixel
positions and the label meanings permuted, which is what challenge 8, *1 Minute
Permuted MNIST*, hands your agent. The last three parts turn what you measured into
an `agent.py` that trains and predicts inside two 60-second deadlines, test it with
the challenge's own harness, and put it on the board.

**Time:** 60 minutes in the room — about 15 of compute on a Colab CPU, the rest
reading your runs. **Deliverable:** your copy of the notebook run end to end with
`QUICK = False`, a one-sentence answer under each of its nine questions, the link to
your W&B project, and your `agent.py` on the leaderboard of challenge 8.

<!-- notes: The notebook is a solution, not a skeleton: the work is the nine
sentences, each with a number from the workspace, and the agent. Students need two
Colab secrets before the session: WANDB_API_KEY (a free account; the old anonymous
mode is a no-op in current wandb, so without a key the prompt's third choice is
offline logging) and MLARENA_API_KEY. Have them run once with QUICK = True to see
the end before the 15-minute full run. Where they stall: Part 3's train()-mode
BatchNorm cell (they think it is a bug), and Part 9 when they rename a parameter and
the upload is rejected. A submission takes a few minutes to settle; start Part 10
before the debrief, not after. -->

---

## Setup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s4-building-blocks.ipynb)

- **Two secrets** in Colab's *Secrets* panel, both granted to the notebook:
  `WANDB_API_KEY` (W&B, free account → *User settings* → *API keys*) and
  `MLARENA_API_KEY` (ML-Arena, Profile → API Keys, starts with `mlk_user_`). Never
  in a cell.
- **No W&B key?** `wandb.login()` asks once; *don't visualise* writes the runs to
  `./wandb/` in offline mode, and `wandb sync wandb/offline-run-*` uploads them once
  you have an account. The comparisons below need the workspace, so get the account.
- **Locally:** download the `.ipynb` from the GitHub path of the badge and
  `uv add torch torchvision wandb mlarena-sdk`. The first run downloads MNIST (and
  Fashion-MNIST for the canary) into `./data/`.
- **`QUICK`.** The third cell holds one flag. Run once with `QUICK = True` (a few
  minutes) to see the whole notebook work, then set it back to `False` for every
  number you report.

The notebook trains on a Colab CPU on purpose: the challenge's agent runs on **3 CPU
cores**.
