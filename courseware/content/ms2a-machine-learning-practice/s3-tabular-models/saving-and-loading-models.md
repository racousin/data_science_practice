# Saving and Loading Models and Pipelines

Every model in this session has lived inside one notebook kernel, and would
have died with it. The agent you submit in the challenge runs in another process, on
another machine, with other library versions, and it never sees your training
data. What crosses that gap is a file. This lesson is about what goes into the
file, which format to write it in, and the ways it breaks once the notebook is
closed.

<!-- notes: 40 minutes, the last lesson before Lab 3. Two demos are worth doing
live. The __main__ one: define a function in a cell, dump a pipeline that uses
it, then load the file from a terminal and read the AttributeError out loud.
And the pickle that prints at load time: the room has never thought of loading
a file as running a program. Leave the version matrix up while they read it —
the point is that the warning cells and the failing cells look the same from
inside the notebook until you load. Every number here comes from
tools/figures/ms2a_s3_persistence.py, pinned to runtime 182's versions. -->

---

## Fit once, predict elsewhere

![The training process writes files; the serving process loads them](assets/tabular/persist-train-serve.png)

The serving side has the files and nothing else. Anything the model needs at
prediction time — a fitted scaler, a function you wrote, the order of the
columns — has to be inside one of them.

---

## What "fitted" means, in bytes

![The fitted state of a StandardScaler + 10-NN pipeline](assets/tabular/persist-fitted-state.png)

`fit` writes attributes whose names end in `_`. They are the model; the
constructor arguments are only settings. A ridge's fitted state is 8
coefficients and an intercept. A kNN's is the training set itself, stored
twice, which is why its file is 1,600 times larger.

---

## Save the pipeline, not the model

```python
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(10))
pipe.fit(X_tr, y_tr)
joblib.dump(pipe, "model.joblib")        # scaler and model, one file

model = joblib.load("model.joblib")      # later, anywhere
model.predict(X_new)                     # scaled with the TRAIN mean
```

The scaler's `mean_` and `scale_` are part of the model, for the reason
*The Preprocessing Contract* gave in Session 2: fitted on train, applied
unchanged to everything after. Save the estimator alone and the agent has
to rebuild that transformation — from which data?


---

## joblib: the default for scikit-learn

```python
import joblib

joblib.dump(forest, "forest.joblib")               # 24.0 MB
joblib.dump(forest, "forest.joblib", compress=3)   #  8.1 MB, same model
forest = joblib.load("forest.joblib")
```

joblib is Python's `pickle` with a faster path for large numpy arrays. It
writes **any** Python object, so a model made of several parts is one file:

```python
bundle = {"p_wet": calibrated_clf,          # a CalibratedClassifierCV
          "amounts": quantile_models,       # {0.1: model, 0.5: model, …}
          "features": FEATURES}
joblib.dump(bundle, "hurdle_6h.joblib", compress=3)
```

---

## The boosters' own formats

```python
import lightgbm as lgb
import xgboost as xgb

lgbm.booster_.save_model("lgbm.txt")           # LightGBM: plain text
booster = lgb.Booster(model_file="lgbm.txt")

model.save_model("xgb.ubj")                    # XGBoost: binary JSON
model = xgb.XGBRegressor()
model.load_model("xgb.ubj")
```

- The file holds the **booster only**: no scaler, no calibration, no
  column names. Keep those in a joblib file or `meta.json`.
- `lgb.Booster.predict` is not `LGBMClassifier.predict`: on a binary
  classifier it returns P(class 1), what `predict_proba(X)[:, 1]` gave.
- XGBoost's documentation recommends `save_model` over pickle for any file
  a later XGBoost version will have to read.

---

## Your own code has to travel too

A function defined in a notebook cell lives in the module `__main__`. The
pickle stores the name, not the code:

```python
def add_ratios(df):                  # a notebook cell
    return df.assign(RoomsPerOcc=df.AveRooms / df.AveOccup)

pipe = make_pipeline(FunctionTransformer(add_ratios), Ridge())
joblib.dump(pipe.fit(X, y), "model.joblib")        # works
```

Loaded anywhere else, `__main__` is someone else's script:

```text
AttributeError: Can't get attribute 'add_ratios'
on <module '__main__' from '…'>
```

A `lambda` fails earlier, at `dump`: `PicklingError: Can't pickle
<function <lambda> …>: it's not found as __main__.<lambda>`.

---

## Put it in a module, and ship the module

```text
submission/
├── agent.py
├── features.py        ← add_ratios lives here
└── model.joblib
```

```python
from features import add_ratios      # in the notebook AND in agent.py
```

The pickle now names `features.add_ratios`, and loading imports `features`.
The platform puts your submission's directory first on `sys.path`, so the
import finds your file. Forget to upload it and the load fails with
`ModuleNotFoundError: No module named 'features'` — loudly, which is the
good outcome.

---

## Load from next to `agent.py`, once

```python
import os
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))


class Agent:
    def __init__(self):                 # once, when the agent starts
        path = os.path.join(HERE, "model.joblib")
        self.model = joblib.load(path)

    def predict(self, request):         # every call
        X = make_table(request)         # yours: a DataFrame
        return make_response(self.model.predict(X))     # yours
```

- `joblib.load("model.joblib")` resolves against the **working
  directory**, which is yours in Colab and not yours on the platform.
- Loading inside `predict` repeats it on every call, inside the call's
  60-second budget.

---

## A metadata file next to the model

```python
import json
import sklearn, lightgbm

meta = {
    "features": list(X_tr.columns),           # the order predict expects
    "versions": {"sklearn": sklearn.__version__,
                 "lightgbm": lightgbm.__version__},
    "trained_on": ["2020-01-01", "2025-12-31"],
    "valid_score": 0.083,                     # yours, on `valid`
    "params": study.best_params,
}
with open("meta.json", "w") as f:
    json.dump(meta, f, indent=2)
```
