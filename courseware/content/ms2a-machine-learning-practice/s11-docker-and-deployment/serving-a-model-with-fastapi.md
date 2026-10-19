# Step 2 — A Model API with uv and FastAPI

`model.joblib` becomes a service: a program on the laptop that loads it once
and answers prediction requests over HTTP.

<!-- notes: 30 minutes, live-coded, students typing along. Tested on
2026-10-19 with uv, fastapi 0.149.0, scikit-learn 1.6.1, Python 3.13. Where
students get stuck: (1) `uv init` run inside another project folder (a course
repo with a uv workspace): uv adds penguin-api as a member of the parent
workspace and the commands then behave oddly; create the folder outside every
other project. (2) model.joblib still in Downloads: `FileNotFoundError` at
start-up. (3) Windows PowerShell: `curl` is an alias of Invoke-WebRequest, and
even `curl.exe` mangles the JSON quotes of the POST; send them to Git Bash or
to /docs. (4) port 8000 taken by an old server: `--port 8001`. Let them press
Execute in /docs themselves. Keep the server running for the three birds. -->

---

## HTTP in one slide

![A client sends a request, the server runs the model and sends a response](assets/deploy/client-server.png)

- **Request**: a method, a URL, an optional JSON body. `GET` reads, `POST`
  sends data to process.
- **Response**: a status code and a JSON body. `200` OK, `404` no such path,
  `422` the body does not match the contract, `500` the server's code crashed.
- In `http://localhost:8000/predict`, `localhost` is the machine, `8000` the
  program (the port), `/predict` the endpoint (the path).

---

## The project, managed by uv

uv installs Python, creates the virtual environment and pins every package
([install](https://docs.astral.sh/uv/getting-started/installation/)):
`curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS / Linux,
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
on Windows. Then, **outside every other project folder**:

```bash
uv init --python 3.13 penguin-api && cd penguin-api
uv add "fastapi[standard]" pandas joblib "scikit-learn==1.6.1"
mv ~/Downloads/model.joblib .
```

`scikit-learn==1.6.1` is the version printed in Colab. `uv init` inside a
folder that holds a uv workspace adds the new project to that workspace.

**Checkpoint:** `uv run python -c "import sklearn; print(sklearn.__version__)"`
→ `1.6.1`

---

## The smallest API

`uv init` wrote a hello-world `main.py`. Replace it with:

```python
from fastapi import FastAPI
app = FastAPI()
@app.get("/health")
def health(): return {"status": "ok"}
```

`uv run fastapi dev main.py` starts the server and prints
`Server started at http://127.0.0.1:8000`. `dev` restarts it on every saved
file. The terminal is now busy serving: a second terminal sends the requests
([first steps](https://fastapi.tiangolo.com/tutorial/first-steps/)).

**Checkpoint:** `curl localhost:8000/health` → `{"status":"ok"}`

---

## `main.py`: the model and the contract

```python
model = joblib.load("model.joblib")   # once, at start-up
app = FastAPI(title="Penguin species")
```

```python
class Penguin(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
```

- The imports: `joblib`, `pandas as pd`, `FastAPI`, `pydantic.BaseModel`
  (complete file: [`penguin-api/main.py`](https://github.com/racousin/data_science_practice/tree/main/website/public/modules/ms2a-machine-learning-practice/deploy/penguin-api/main.py)).
- `joblib.load` runs once when the server starts, not on every request.
- `Penguin` is the contract: four named numbers. Any other body is refused
  with `422` before `predict` runs.

---

## `main.py`: the endpoint

```python
@app.post("/predict")
def predict(penguin: Penguin):
    X = pd.DataFrame([penguin.model_dump()])
    return {"species": model.predict(X)[0]}
```

- `@app.post("/predict")` binds the function to `POST /predict`.
- FastAPI reads the JSON body into a `Penguin`; the returned dict becomes the
  JSON response.
- The pipeline was fitted on a DataFrame with these four column names: it is
  called with the same.

---

## Testing it

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"bill_length_mm": 39.1, "bill_depth_mm": 18.7,
       "flipper_length_mm": 181, "body_mass_g": 3750}'
```

→ `{"species":"Adelie"}`. In the browser, `http://localhost:8000/docs` →
**POST /predict** → **Try it out** → **Execute**:

![The /docs page: a request body and the server's answer](assets/deploy/docs-penguin.png)

A body `{"bill_length_mm": "long"}` returns `422`, with `Field required` for
the three missing fields. Windows: PowerShell's `curl` is another command; use
Git Bash, or `/docs`.

---

## Three birds to check

The first penguin of each species in the file:

| `bill_length_mm` | `bill_depth_mm` | `flipper_length_mm` | `body_mass_g` | Expected |
|---|---|---|---|---|
| 39.1 | 18.7 | 181 | 3750 | `Adelie` |
| 46.5 | 17.9 | 192 | 3500 | `Chinstrap` |
| 46.1 | 13.2 | 211 | 4500 | `Gentoo` |

The same three answers as in the Colab notebook: the laptop loads the pickle
with the version that wrote it.

**Checkpoint:** the three `curl` (or `/docs`) requests return `Adelie`,
`Chinstrap`, `Gentoo`.

---

## Check yourself

1. Why is `joblib.load` at the top of `main.py` and not inside `predict`?
   **It runs once at start-up instead of on every request.**
2. `http://localhost:8000/predict` typed in the address bar answers
   `405 Method Not Allowed`. Why? **The address bar sends `GET`; the endpoint
   accepts only `POST`.**
3. A request body has `"body_mass_g": "heavy"`. Which status, and which code
   produces it? **`422`, produced by FastAPI from the `Penguin` contract, before
   `predict` runs.**
