# Saving and Loading Models and Pipelines

Every model in this session has lived inside one notebook kernel, and would
have died with it. The agent you submit in Lab 3 runs in another process, on
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

## Where the file sits

![The data science life cycle: evaluation, then deployment](assets/tabular/data-science-life-cycle.png)

Evaluation ends with a model you trust. Deployment starts with a file that
someone else loads: Lab 3's agent this week, an API in a Docker image in
Session 11. Everything between those two steps is this lesson.

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

## The silent version of the mistake

![Test RMSE when the scaler is refitted on each request](assets/tabular/persist-serving-skew.png)

A `StandardScaler().fit_transform(X)` inside the agent is the usual way to
rebuild it. Nothing fails. On 45 neighbouring districts per request it scores
**1.04**, against **0.625** for the pipeline as fitted and 1.14 for predicting
the training mean: 80% of the model is gone.

---

## Why neighbouring rows are the bad case

Random rows are a small sample of the test set, so their mean drifts towards
the training mean as the batch grows: 45 random rows score 0.69. Rows that
arrive together are rarely random:

- 45 cities at the **same hour** of Lab 3 share that hour's weather;
- a morning's loan applications share a branch; a day's orders share a
  promotion.

Refitting on the batch subtracts exactly what the rows have in common, which
is the signal. With one row per request the scaler maps every input to
zeros, and the model gives the same answer to everyone: 1.47, worse than the
mean. Passing raw features to the bare estimator, with no scaling, scores
1.17.

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

## Formats, measured

![File size and load time for three models in each format](assets/tabular/persist-formats.png)

Every reloaded model predicted the 4,128 test rows exactly as the original
did. Size is the difference that matters: `compress=3` divides the forest by
3, and XGBoost's `.json` is 40% larger and slower to read than `.ubj`. Every
load took well under a second — load time only matters when you load in the
wrong place.

---

## Loading a pickle runs code

```python
import pickle

class Innocent:
    def __reduce__(self):
        return (print, ("this line ran inside pickle.loads",))

blob = pickle.dumps(Innocent())
pickle.loads(blob)
# this line ran inside pickle.loads
```

`__reduce__` tells pickle which function to call to rebuild the object, and
loading calls it. Put `os.system` where `print` is and loading the file runs
any command. `joblib.load` is `pickle.load`.

**Load only files you produced, or files from someone you would let run code
on your machine.** That includes a `.joblib` shared by a classmate.

---

## skops: a format that asks first

```python
import skops.io as sio

sio.dump(forest, "forest.skops")
print(sio.get_untrusted_types(file="forest.skops"))
# ['sklearn.tree._tree.Tree']
forest = sio.load("forest.skops", trusted=["sklearn.tree._tree.Tree"])
```

skops stores the object without arbitrary code, and `load` refuses every type
it does not know is safe until you name it — a LightGBM model lists three.
Use it for files that come from someone else.

For Lab 3 your agent loads files you wrote, and runtime 182 does not ship
skops: joblib and the native formats are the right tools there.

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

## Versions: what breaks, measured

![Which files load under which scikit-learn version](assets/tabular/persist-versions.png)

A pipeline and a forest load across all three versions, with an
`InconsistentVersionWarning`. The histogram booster does not.

---

## Versions: the two failures

Saved with one version, loaded with another, and nothing on the saving side
warned about either:

```text
AttributeError: Can't get attribute '__pyx_unpickle_CyHalfSquaredError'
on <module 'sklearn._loss._loss' from '…'>            # 1.4.2 → 1.8.0
ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known
BitGenerator module.                                  # 1.8.0 → 1.4.2
```

The second one is numpy 2 against numpy 1.26, not scikit-learn at all.

---

## Train with the versions you will be loaded with

Lab 3's runtime 182 is Python 3.12 with **scikit-learn 1.8.0, XGBoost
3.2.0, LightGBM 4.6.0**. In Colab, first cell, then restart the session:

```python
%pip install scikit-learn==1.8.0 lightgbm==4.6.0 xgboost==3.2.0
```

A warning is not a guarantee that the numbers are right, so turn it into an
error in the agent, before the load:

```python
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("error", InconsistentVersionWarning)
```

A submission that fails at start-up shows you the message. One that predicts
from a half-compatible model shows you a bad score two days later.

---

## Let the column names check the order

Fit on a DataFrame and every step stores `feature_names_in_`. Predict with
the columns in another order and scikit-learn refuses:

```text
ValueError: The feature names should match those that were passed
during fit. Feature names must be in the same order as they were in fit.
```

Predict with a numpy array and the check is off. A scaler + ridge fitted on
California housing, given the same five rows with the columns reversed:

```text
UserWarning: X does not have valid feature names, but StandardScaler
was fitted with feature names
[4.13 3.98 3.68 3.24 2.41]  ->  [96.48 1445.31 204.46 245.16 250.49]
```

In the agent, build the table as a DataFrame with the training column names.

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

An Optuna study saved in SQLite holds parameters and scores, not a fitted
model. Refit on `study.best_params`, then save — Lab 3's Part E.

---

## Test the round trip in a fresh process

```python
# check_agent.py — run it from another directory:
#   !cd /tmp && python /content/check_agent.py
import importlib.util, sys
import numpy as np
import pandas as pd

SUB = "/content/submission"
sys.path.insert(0, SUB)                          # as the platform does
spec = importlib.util.spec_from_file_location(
    "Agent", f"{SUB}/agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

X = pd.read_parquet("/content/check_X.parquet")  # saved in the notebook,
want = np.load("/content/check_pred.npy")        # with its predictions
got = mod.Agent().model.predict(X)
np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)
print("round trip OK")
```

A new process, another working directory: this one run catches `__main__`,
a relative path and a missing module at once.

---

## Why a tolerance, and not `==`

`atol=1e-9` is not there for the file. It is there for the model:

- A random forest on `n_jobs=-1` adds its trees' predictions in whatever
  order the threads finish. Predict the same rows twice with the same
  object and the results differ in the 15th significant digit.
- Every model in *Formats, measured* reloaded to identical numbers; a
  difference of 1e-15 is arithmetic, a difference of 1e-3 is a bug.

---

## Before you submit

1. The file holds the **whole pipeline**: every fitted step, not only the
   estimator.
2. Your own functions live in a **module** you upload with it.
3. Paths come from `__file__`; the model loads **once**, in `__init__`.
4. The versions are **pinned** to the runtime's and recorded in
   `meta.json`; a version warning is an error.
5. The table is built with the **training column names**.
6. The round trip passes in a **fresh process** from another directory.
7. You load only files you **trust**; skops for the rest.

---

## Check yourself

1. Run this. You should get exactly the output shown.

   ```python
   import numpy as np
   from sklearn.preprocessing import StandardScaler

   one_request = np.array([[8.3, 41.0, 6.98]])
   print(StandardScaler().fit_transform(one_request))
   # -> [[0. 0. 0.]]
   ```

   Which point of the serving figure is this, and why is it worse than
   predicting the mean?

   **Answer.** The batch of one, 1.47. A scaler fitted on one row maps it to
   zeros whatever it contains, so every request reaches the model as the same
   point, and gets the same prediction. That constant is the prediction for
   an average training row, which is not the training mean of the target.

2. Your agent runs in Colab, and on the platform its status says
   `AttributeError: Can't get attribute 'make_features' on <module
   '__main__' from …>`. What happened, and what is the fix?

   **Answer.** `make_features` was defined in a notebook cell, so the pickle
   refers to it as `__main__.make_features`, and on the platform `__main__`
   is the platform's script. Move the function to `features.py`, import it
   from there in the notebook, refit, dump again, and upload `features.py`
   with the model.

3. A classmate at the top of the leaderboard sends you their
   `hurdle.joblib`. What is the risk in loading it, and what do you ask for
   instead?

   **Answer.** `joblib.load` runs whatever the file tells pickle to call, with
   your permissions. Ask for the training code, and refit it yourself; or for
   a skops file, and read `get_untrusted_types` before you name anything as
   trusted.
