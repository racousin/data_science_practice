# Lab — Preprocessing Notebook

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
