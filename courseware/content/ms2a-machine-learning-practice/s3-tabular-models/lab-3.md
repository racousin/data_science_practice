# Lab 3 — Rain Over Europe

Six years of hourly weather for 45 European cities, and a live challenge that
asks for the rain 6 and 48 hours ahead, as a distribution. Read the past first,
fix a protocol you can defend, submit a minimal agent, then beat it with a
tuned model.

**Deliverable:** Agent
on the leaderboard of *European Rain Forecast* (#177) whose held-out score you
computed once, before submitting it.

**Challenge:** <https://ml-arena.com/viewchallenge/177>

<!-- notes: 90 minutes. Students need MLARENA_API_KEY in Colab Secrets before
the session; the two notebooks fetch the 28 MB European file by its label.
Nobody calls download_dataset(177): it also pulls 4.4 GB of global files. The
leaderboard cannot confirm anything in the room — a forecast is scored 48 h
after it is made — so the in-room check is the local score. Its scorer
reproduces the env's replay to four decimals on the reference, persistence and
always-0 rows; the climatology rows differ only by member count (the notebook's
100 quantiles against the env's 50). Where they stall: Part B's gap, and Part D
when one trial is too slow for the budget they declared. -->


## Part A — Read the past

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s3-rain-eda.ipynb)


## Part B — The minimal agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s3-rain-agent.ipynb)

- **The model:** for each city and month, 100 quantiles of hourly rain,
  saved to `rain_quantiles.json`. Splitting by hour of day loses about
  0.002 at both horizons: too little data per cell.
- **The agent:** `agent.py` loads the JSON from its own directory and
  looks the cities up by name.
- **The check:** requests built exactly as the env builds them; the shape,
  M ≤ 100, finite values.

---

## Part C — Evaluate and improve your weather forcast

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s3-rain-model.ipynb)

Train models build features and evaluate capacity to predict the weather risk.
