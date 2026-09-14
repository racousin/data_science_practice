# Lab 2 — Preprocessing Notebook

Five years of a region's daily electricity demand: 1,909 training days with the
weather of the day (humidity, wind, ten temperature stations, a weather
condition), an oil-price indicator and the target `electricity_demand`. The
notebook is the complete, worked run of this session's checklist on that table:
seven functions, one per step, each measured with the same `LinearRegression` on
a 5-fold time-series split. Run it, explain every step, then change one step.

**Time:** 45 minutes. **Deliverable:** your copy of the notebook, run end to
end, with a one-sentence text cell under each of the seven steps and one extra
row in the final ladder that you added yourself.

<!-- notes: 45 minutes. Say at the start that the notebook is a solution, not a
skeleton: the work is the sentences and the extra ladder row, not the code.
Where they stall is section 3 — why some steps run before the split and others
inside the fold; send them to the docstring of `evaluate_pipeline`. Part D runs
over when they change the model: the model is fixed, one choice changes. -->

---

## Setup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s2-case-study.ipynb)

To run it locally, download the `.ipynb` from the same GitHub path and open it
in Jupyter (pandas, scikit-learn, matplotlib, requests). The first code cell
downloads the two CSV files.

---

## Part A — Explore the data (10 min)

Run sections 1 and 2 (*Data collection*, *Data analysis*). The train file is
1,909 rows by 16 columns, the test file 365 by 15: it has no target and the
notebook never uses it again; the time-series split on the training rows replaces it.

Before reading the notebook's own commentary, write down for each checklist
item what the printed output shows: the `km/h` and `m/s` mixed inside
`wind_speed`, the 90 duplicated dates, the 15% gaps in every temperature
station, the two categorical columns (one unordered, one ordered), the humidity
of 50,000 and the demand of −223,289. Then compare with the notebook's paragraph.

The rule: every number in this part is computed on the training rows. The test
file is opened once, to check that its wind units match, and closed.

---

## Part B — The evaluation protocol (8 min)

Read section 3 and the code of `evaluate_pipeline` until you can say, without
looking, which steps go in `steps_before` and which in `steps_in_fold`, and why.

- A step with **no learned parameter** (a unit conversion, a fixed category
  list, a rule on impossible values, a date feature) runs once, before the split.
- A step that **learns a statistic** (the medians that fill the gaps) runs inside
  each fold, fitted on that fold's training rows and applied to its validation
  rows.

The split is `TimeSeriesSplit(n_splits=5)` on the rows in date order: each fold
validates on days that come after the days it trained on. A random split would
let the model see the day after the one it predicts.

---

## Part C — The seven steps (20 min)

Run section 4, step 0 to step 7, and watch the ladder grow. Under each step add
a text cell of one sentence in your own words: what the data showed, what the
notebook chose, what the validation MSE did.

Steps 1 to 4 do not move the score. The single −223,289 demand sits in one
validation fold and its square is worth more than every other row together;
nothing is measurable until step 5 removes it (31,502,301.8 to 1,103.1). The
re-measured ladder just after step 5 shows what each step was really worth.

Step 7 barely changes the MSE (331.4 to 329.9) but changes the model: with ten
stations the temperature coefficient is split ten ways; with one column it is
readable. Selection is about a model you can explain, not the score.

---

## Part D — Change one step and measure (7 min)

Copy one of the seven functions, change **one** choice, and add the result to
the ladder with `record("8 my variant: ...", res)`. Some options, easiest first:

- step 3: fill a missing station with the column median instead of the same
  day's other stations
- step 6: move the degree-day thresholds from 18 / 24 °C to 15 / 22 °C
- step 3: compute the medians on the validation rows too, and watch the
  validation MSE *improve*: that is the leak of *The Preprocessing Contract*
  made visible

The model stays `LinearRegression()` and the split stays the five time-series
folds, or the row is not comparable. State the difference to 329.9 in one sentence.

---

## Did you validate this session?

- [ ] Section 1 prints `train: (1909, 16)   test: (365, 15)`
- [ ] Section 2 prints `duplicated rows: 90` and lists two humidity rows at 50,000
- [ ] Part B: for each of the seven functions I can say whether it runs before the split or inside the fold, and my answer matches the `steps_before` / `steps_in_fold` lists of step 7
- [ ] Part C: the ladder printed after step 7 has 8 rows; step 0 reads 31,502,301.8 and step 7 reads 329.9 with 20 features
- [ ] Part C: each of the seven steps has my one-sentence text cell under it
- [ ] Part D: re-running the section 5 cell prints a ninth row named after my change, with its own validation MSE, and one sentence compares it with 329.9

The same seven steps, fitted on the training rows only, are what the module's
challenge scores on real clinical data.

- [ ] My submission is on the leaderboard of Critical Care Survival (#192)
- [ ] My score beats the bar: **AUC ≥ 0.905**
