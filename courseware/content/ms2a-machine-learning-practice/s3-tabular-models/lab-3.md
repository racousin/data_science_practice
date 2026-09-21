# Lab 3 — Rain Over Europe

Six years of hourly weather for 45 European cities, and a live challenge that
asks for the rain 6 and 48 hours ahead, as a distribution. Read the past first,
fix a protocol you can defend, submit a minimal agent, then beat it with a
tuned model — scored the way the challenge scores it.

**Time:** 90 minutes. **Deliverable:** the analysis notebook run end to end
with your ten answers; your protocol, written before any model; and an agent
on the leaderboard of *European Rain Forecast* (#177) whose held-out score you
computed once, before submitting it.

<!-- notes: 90 minutes. Students need MLARENA_API_KEY in Colab Secrets before
the session; the two notebooks fetch the 28 MB European file by its label.
Nobody calls download_dataset(177): it also pulls 4.4 GB of global files. The
leaderboard cannot confirm anything in the room — a forecast is scored 48 h
after it is made — so the in-room check is the local score. Its scorer
reproduces the env's replay to four decimals on the reference, persistence and
always-0 rows; the climatology rows differ only by member count (the notebook's
100 quantiles against the env's 50). Where they stall: Part B's gap, and Part D
when one trial is too slow for the budget they declared. -->

---

## Setup (5 min)

**Challenge:** <https://ml-arena.com/viewchallenge/177>

- In Colab's *Secrets* panel, add `MLARENA_API_KEY` (ML-Arena, Profile → API
  Keys) and allow each notebook to read it. Never paste it into a cell.
- Both notebooks download `weather_europe_2020_2026.csv.gz` from the
  challenge: 2,434,320 rows, 45 cities × 54,096 hours, 2020-01-01 to
  2026-03-03 UTC.
- Pick that one file by its label from `client.datasets(177)`. Never call
  `client.download_dataset(177)`: the challenge also carries seven global
  yearly files, 4.4 GB.

Every hour of the file is known, so every number the notebooks print is
exact: a different number means different code.

---

## The task

Every 6 hours the challenge calls your agent once, with the whole panel:

- **In:** the last 48 hours of 8 features for the 45 cities, and the
  horizons `[6, 48]`.
- **Out:** `{"rain": (45, 2, M)}` — for each city and horizon, M plausible
  amounts in mm, 1 ≤ M ≤ 100.
- **Reference:** each city's own last 48 hours, used as 48 members.
- **Score:** per run and horizon, the reference's mean CRPS minus yours,
  divided by a fixed monthly constant, floored at −1; the mean of the two
  horizons, averaged over runs. The reference scores 0.

The constant is `REFERENCE_CRPS[h][month]`, the reference's own mean CRPS
over 2020–2024. It does not depend on the weather of the run, so the score
rewards exactly what the CRPS rewards (*Probabilistic Prediction*). A
perfect forecast scores about 1 — more in a wet run: 1 is not a ceiling.

**The course's bar is 0: beat the reference.**

---

## Part A — Read the past (25 min)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s3-rain-eda.ipynb)

Ten sections, each a question a forecaster asks before modelling: loading
and completeness, the panel, zero inflation, seasonality, the diurnal cycle,
persistence, spatial structure, the other features, and four baselines
scored with the challenge's CRPS skill.

Every section states its finding and leaves you one question. Answer each
in a text cell under it: one or two sentences, with the number that
supports it.

---

## Part A — What you should find

| Question | The notebook's number |
|---|---|
| How often is a city-hour wet? | 14.5%: Athens 7.9% to Glasgow 28.3% |
| How much, when it is? | median 0.2 mm, p99 4.5 mm |
| P(wet at +6 h / +48 h \| wet now) | 0.430 / 0.234, against 0.145 |
| At +48 h after a dry hour, × climatology | 0.92 |
| Rain in London shows up in Berlin | 22 h later |
| AUC of cloud cover, +6 h / +48 h | 0.718 / 0.576 |
| City × month climatology, CRPS skill | +0.057 |

Record them in your copy. The +48 h rows are the lesson: most of the time,
the best +48 h forecast is the city's climatology for the month.

---

## Part B — Freeze the protocol (10 min)

Before any model, write this cell and do not edit it again:

```python
TRAIN_END = pd.Timestamp("2024-01-01", tz="UTC")
VALID_END = pd.Timestamp("2025-01-01", tz="UTC")
GAP = pd.Timedelta(hours=48)            # the longest horizon
ISSUE_HOURS = (5, 11, 17, 23)           # UTC, as the challenge
SEED = 0

def block(issue):
    """The block a row issued at `issue` belongs to, or None."""
    if issue + GAP < TRAIN_END:
        return "train"                  # fit
    if TRAIN_END <= issue and issue + GAP < VALID_END:
        return "valid"                  # tune: the study reads this
    if VALID_END <= issue:
        return "test"                   # scored once, in Part E
    return None                         # inside a gap: dropped
```

---

## Part B — What it says

Under the cell, write down:

- **one row:** a city and an issue time, one target per horizon
- **the unit of generalisation:** the same 45 cities, at hours that have
  not happened yet — so the split is by time, never at random
- **the metric:** the challenge's CRPS skill per run, averaged over runs

**The gap.** A row issued at 23:00 on 31 December has its +48 h target on 2
January. Kept in training, that target sits inside the next block, and the
model is scored on hours it has seen. `block` drops it.

**Never at random.** An hour after a wet hour is wet 73% of the time, so a
random split gives every test row near-copies in training. Validation looks
good; the leaderboard does not.

---

## Part C — The minimal agent (15 min)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s3-rain-agent.ipynb)

- **The model:** for each city and month, 100 quantiles of hourly rain,
  saved to `rain_quantiles.json`. Splitting by hour of day loses about
  0.002 at both horizons: too little data per cell.
- **The agent:** `agent.py` loads the JSON from its own directory and
  looks the cities up by name.
- **The check:** requests built exactly as the env builds them; the shape,
  M ≤ 100, finite values.
- **The replay:** the 1,700 runs of Part B's test block, the env's scorer.
  You should get **+0.0574** — +0.026 at +6 h, +0.089 at +48 h.

---

## Part C — Submit it

```python
result = client.submit(177, files=["agent.py", "rain_quantiles.json"],
                       submission_name="city-month-climatology",
                       runtime_id=181)
status = client.status(result["submission_id"], 177)
print(status["status"], status["last_status_message"])
```

`runtime_id=181` is plain Python 3.12 with numpy and pandas; without it the
challenge's default runtime is used. 182 adds scikit-learn, LightGBM and
XGBoost.

**The first score appears about 48 hours after the first forecast.** A
forecast is scored once both of its hours have happened, so a new agent's
first eight runs report no score, not 0. Check the leaderboard next session.

---

## Part D — Beat it: a hurdle LightGBM (15 min)

One row per city and issue time, with the features Part A found:

| Feature | Part A section |
|---|---|
| rain over the last 1, 6 and 48 h | 6: persistence |
| cloud cover and humidity at the issue time | 8: AUC 0.66–0.72 at +6 h |
| rain in the city 300–500 km upwind | 7: a lag of +6 to +10 h |
| month, solar hour, city | 4 and 5 |

The model is the hurdle of *Probabilistic Prediction*: a LightGBM
classifier for P(wet), calibrated (`CalibratedClassifierCV`, or check its
reliability diagram — it sets how many members are exactly zero), LightGBM
quantile regressors on the wet rows in `log1p`, and `hurdle_members` to turn
both into M members. One model set per horizon, on runtime 182.

Score it on the `valid` runs with the agent notebook's `skill`. The bar is
the wet-now row of the ladder, +0.080.

---

## Part D — The Optuna study (10 min)

```python
def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
        min_child_samples=trial.suggest_int(
            "min_child_samples", 20, 500, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        n_estimators=300, random_state=SEED, verbose=-1)
    model = fit_hurdle(params, blocks=["train"])
    return skill_on(model, "valid")     # the challenge's scorer

study = optuna.create_study(
    direction="maximize", study_name="hurdle-v1",
    storage="sqlite:///rain.db", load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
```

`fit_hurdle` and `skill_on` are yours. The rules of *Hyperparameter
Optimisation*: time one trial, fix `N_TRIALS` to the time left and write it
down before the study starts — no `timeout`, which would make the trial count
depend on the machine; log scales; a seeded sampler; a persisted study;
`n_estimators` not searched.

---

## Part E — One test number, then submit (10 min)

Refit the best configuration on `train` and `valid` together, and score the
`test` block **once**:

```python
final = fit_hurdle(study.best_params, blocks=["train", "valid"])
print(skill_on(final, "test"))          # the one test number
```

Write it next to the best trial's validation score and the median of the
top five. A test score well below validation means the search overfitted
the 2024 runs — a finding, not a failure.

Then refit on every hour of the file, save the models next to `agent.py`
— the boosters with `booster_.save_model(...)`, reloaded with
`lgb.Booster(model_file=...)`; the calibrated classifier with joblib — load
them from the agent's own directory, pass the round-trip check of *Saving and
Loading Models and Pipelines*, and submit with `runtime_id=182`.

---

## The measured ladder

Replay of 1,700 scored runs, 2025-01-01 to 2026-03-01, through the
challenge's own `env.py`; fitted rows use 2020–2024 only.

| Forecast | Score | +6 h | +48 h |
|---|---|---|---|
| Persistence: the last hour, M = 1 | −0.497 | −0.443 | −0.550 |
| Reference: the last 48 h as members | 0 | 0 | 0 |
| Always 0 mm, M = 1 | +0.001 | −0.032 | +0.035 |
| City × month × hour climatology | +0.054 | +0.023 | +0.085 |
| City × month climatology (Part C) | +0.057 | +0.026 | +0.089 |
| The challenge's starter agent | +0.076 | +0.070 | +0.081 |
| Wet-now: +6 h split by rain in the last 3 h | +0.080 | +0.073 | +0.088 |

The Part C row is the agent notebook's replay. On the persistence, reference
and always-0 rows its scorer matches the env's to four decimals; the two
climatology rows differ only by member count — the env replays 50 quantiles,
the notebook 100, hence +0.0574 against the env's +0.0565 for city × month.
Consecutive runs share the weather, so standard errors come from weekly
blocks: about 0.005 per row (the starter: +0.076 ± 0.005). On the live board,
a gap under about 0.01 between two agents over a week or two is weather
noise, not skill.

---

## Grading and deductions

| Criterion | Weight |
|---|---|
| Part A: ten answers, each with its number | 25% |
| Protocol frozen first; time-ordered, with the gaps | 20% |
| Minimal agent submitted, local score reproduced | 15% |
| Hurdle model scored on `valid` with the challenge's scorer | 15% |
| Study: fixed budget, log scales, seeded, persisted | 15% |
| One test number, reported next to validation | 10% |

Automatic deductions: a random split; the `test` block read by the study
or scored twice; `n_estimators` searched; a best trial reported as the
model's score; a key pasted into a cell.

---

## Did you validate this session?

- [ ] The analysis notebook runs end to end and prints 14.5% wet
      city-hours; the ten answers are under their sections
- [ ] The protocol cell was written before any model and has not changed
- [ ] The minimal agent's replay prints **+0.0574** over 1,700 runs
- [ ] My key is in Colab Secrets, and in no cell
- [ ] My submission is `active` on European Rain Forecast (#177)
- [ ] The study ran a fixed `N_TRIALS`, declared first, with no `timeout`
- [ ] Exactly one `test` number, next to the validation score
- [ ] That number is above 0, the course's bar — and, the aim, above the
      minimal agent's +0.057

If the last three are not ticked you have not finished the lab, however
good the code is.
