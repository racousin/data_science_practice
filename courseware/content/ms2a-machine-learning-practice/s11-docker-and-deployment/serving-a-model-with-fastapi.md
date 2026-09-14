# Serving a Model with FastAPI

Train a model, save it to a file, and put it behind two endpoints. FastAPI
turns a Python function into an HTTP endpoint and, for free, gives you a web
page to try it from the browser.

<!-- notes: 20 minutes, live-coded, students typing along. Iris on purpose: the
model is not the point, and the lab uses a different dataset with the same
steps. Every output shown was produced on 2026-09-14 with fastapi 0.141.1,
uvicorn 0.52.4, scikit-learn 1.9.1, Python 3.13. The moment that sells it is
/docs: let them press Execute themselves before moving on. Leave the server
running; the next lesson packs this exact folder. -->

---

## The project

```bash
uv init --python 3.13 iris-api
cd iris-api
uv add fastapi uvicorn scikit-learn joblib
```

| Package | Job |
|---|---|
| `fastapi` | turns Python functions into HTTP endpoints |
| `uvicorn` | the server: listens on a port, hands requests to FastAPI |
| `scikit-learn` | the model |
| `joblib` | saves the trained model to a file and loads it back |

`uv init` also wrote a hello-world `main.py`: you replace it in a minute.

---

## Step 1: train and save (`train.py`)

```python
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print("test accuracy:", model.score(X_test, y_test))

joblib.dump(model, "model.joblib")
```

`uv run python train.py` prints `test accuracy: 1.0` and writes
`model.joblib`, 991 bytes.

---

## What `model.joblib` is

- The fitted Python object, serialised: coefficients, classes, settings. Load
  it and it predicts, without the training data and without retraining.
- If you preprocess (Session 2), save the **whole pipeline**
  (`make_pipeline(StandardScaler(), model)`): the API must apply the
  transformation fitted on train, and a pipeline carries it with the model.
- It is a pickle. Loading a pickle runs code, so **never load a model file you
  did not produce or do not trust**.

---

## Step 2: the API (`main.py`)

```python
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

SPECIES = ["setosa", "versicolor", "virginica"]

model = joblib.load("model.joblib")     # once, when the server starts
app = FastAPI(title="Iris classifier")


class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
```

---

## Step 2, continued: the endpoints

```python
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(flower: Flower):
    x = [[flower.sepal_length, flower.sepal_width,
          flower.petal_length, flower.petal_width]]
    label = int(model.predict(x)[0])
    return {"species": SPECIES[label]}
```

- `@app.get("/health")` makes the function below answer `GET /health`.
- `flower: Flower` tells FastAPI to read the JSON body into a `Flower`, and to
  check it first.
- The returned dict becomes the JSON response.

---

## Three lines that matter

**`model = joblib.load(...)` at the top, not inside `predict`.** Loading runs
once when the server starts. Inside `predict` it would run on every request:
a 500-tree random forest takes 30 ms to load and 9 ms to predict, so every
answer would be four times slower. A deep network takes seconds.

**`class Flower(BaseModel)`** is the contract: four named floats. Anything
else is refused with `422` before your code runs.

**`int(...)`** turns numpy's `int64` into a Python `int`. Return the numpy
value in the dict instead and FastAPI cannot convert it to JSON: the client
gets `500 Internal Server Error`.

---

## Step 3: run it

```bash
uv run uvicorn main:app --reload
```

```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [68493] using StatReload
INFO:     Application startup complete.
```

- `main:app` means: in the file `main.py`, the object called `app`.
- `--reload` restarts the server whenever you save a file: for development
  only.
- The terminal is now busy serving. Open a second one for anything else;
  `Ctrl+C` stops the server.

---

## Step 4: use it from the browser

Open **http://localhost:8000/health**:

![GET /health in the browser](assets/deploy/browser-health.png)

The address bar sent `GET /health`; the page is the JSON your function
returned. That is the whole API seen from a browser.

---

## The documentation page

Open **http://localhost:8000/docs**: FastAPI generates this page from your
code, one row per endpoint.

![The interactive documentation page FastAPI generates](assets/deploy/docs-overview.png)

---

## Try it out

Click **POST /predict**, then **Try it out**, type a flower, press **Execute**:

![A request body typed into /docs](assets/deploy/docs-try-it.png)

---

## The answer

![The server's answer: 200 and the predicted species](assets/deploy/docs-response.png)

The page also shows the equivalent `curl` command. Change `6.7` to `"long"`
and execute again: `422`, with the field and the reason, `Input should be a
valid number`. You wrote no validation code.

---

## Check yourself

1. Why is the model loaded at the top of `main.py` rather than inside
   `predict`? **So it is read once at start-up, not on every request.**
2. You scaled the features during training. What do you save so the API
   scales them identically? **The whole pipeline, scaler and model together.**
3. `uv run uvicorn main:app` fails with `Error loading ASGI app. Could not
   import module "main"`. What is wrong? **You are not in the folder that
   contains `main.py`, or the file has another name.**
4. What does http://localhost:8000/docs give you that curl does not?
   **A form per endpoint, built from your code, with the expected JSON
   schema, to call the API from the browser.**
