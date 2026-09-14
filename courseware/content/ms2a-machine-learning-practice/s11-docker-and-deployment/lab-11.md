# Lab 11 — Ship a Penguin Classifier

Train a small model, put it behind an API, pack the API into a Docker image,
use it from your browser, and publish the image so that someone else runs it
with one command. Same steps as the Iris demo, on a dataset you have not seen
served yet, and without the answers typed out for you.

**Time:** 65 minutes. **Deliverable:** a GitHub repository with the files listed
at the end, and an image on Docker Hub that a classmate has run.

<!-- notes: 65 minutes. 5 setup, 10 train, 15 serve, 20 containerise, 10 ship,
5 checklist. Walk the room at Part C: the three failures you will see are the
missing --host 0.0.0.0 (empty reply), a forgotten uv server still holding
port 8000 (port is already allocated), and a model trained on a DataFrame but
called with a list (a UserWarning in the logs, and a wrong prediction the day
the columns get reordered). Pair students for Part D before the lab starts, so
nobody waits for a partner at the end. The reference solution scored 0.9884
test accuracy (85 of 86) with StandardScaler + LogisticRegression, and
predicts the three reference birds correctly from its linux/amd64 image. Part E
depends on the ML-Arena image submission, which is not live yet: say so, and
grade Parts A-D. -->

---

## Setup (5 min)

- Docker running: `docker run --rm hello-world` prints *Hello from Docker!*
- A free **Docker Hub** account: hub.docker.com, then `docker login` in a
  terminal. Your username is your image namespace.
- A new project, outside every other project folder:

```bash
uv init --python 3.13 penguin-api
cd penguin-api
uv add fastapi uvicorn scikit-learn joblib pandas
```

`uv init` also made the folder a git repository.

- Pair up with a neighbour now: in Part D you run each other's image.

---

## The data

The **Palmer penguins**: 344 penguins of three species measured on three
islands of Antarctica (public domain).

```python
URL = ("https://raw.githubusercontent.com/allisonhorst/"
       "palmerpenguins/main/inst/extdata/penguins.csv")
FEATURES = ["bill_length_mm", "bill_depth_mm",
            "flipper_length_mm", "body_mass_g"]
```

- **Features:** the four numeric measurements above. 2 rows have them missing:
  drop those, 342 remain.
- **Target:** `species`. After the drop: `Adelie` 151, `Gentoo` 123,
  `Chinstrap` 68.

---

## Part A — Train (10 min)

Write `train.py`:

1. Read the CSV, drop the rows with a missing feature.
2. Split with `train_test_split(X, y, test_size=0.25, stratify=y,
   random_state=0)`: 256 rows to train, 86 to test.
3. Print the **majority-class baseline** on the test part: always answer the
   most frequent training species.
4. Fit a **pipeline**: `make_pipeline(StandardScaler(),
   LogisticRegression(max_iter=1000))`, and print its test accuracy.
5. `joblib.dump` the pipeline to `model.joblib`.

| | Test accuracy |
|---|---|
| Always `Adelie` | 0.4419 |
| **Pass bar** | **0.95** |
| Scaler + logistic regression (reference) | 0.9884 |

Fit `X` as a **DataFrame** with the four named columns, not as a numpy array.
Part B depends on it.

---

## Part B — Serve (15 min)

Write `main.py`, following the Iris API of the lesson:

- the pipeline loaded **once**, at the top of the file;
- a `Penguin` model (`pydantic.BaseModel`) with the four features as `float`;
- `GET /health` answering `{"status": "ok"}`;
- `POST /predict` taking a `Penguin` and answering `{"species": "..."}`.

The pipeline was fitted on a DataFrame, so give it one, with the same column
names:

```python
X = pd.DataFrame([penguin.model_dump()])
```

A plain list works too, but prints `UserWarning: X does not have valid feature
names, but StandardScaler was fitted with feature names`, and would silently
mix up the columns if the order ever changed.

Run `uv run uvicorn main:app --reload`, open **http://localhost:8000/docs**,
and check the three birds below.

---

## Three birds to check

The first penguin of each species in the file:

| `bill_length_mm` | `bill_depth_mm` | `flipper_length_mm` | `body_mass_g` | Expected |
|---|---|---|---|---|
| 39.1 | 18.7 | 181 | 3750 | `Adelie` |
| 46.5 | 17.9 | 192 | 3500 | `Chinstrap` |
| 46.1 | 13.2 | 211 | 4500 | `Gentoo` |

Then break it on purpose: send `"body_mass_g": "heavy"`. You should get `422`,
not `500`. If you get `500`, read the traceback in the uvicorn terminal.

---

## Part C — Containerise (20 min)

Stop the uv server (`Ctrl+C`). Then, in the project folder:

1. `uv export --no-hashes -o requirements.txt`
2. A `.dockerignore` with `.venv`, `__pycache__` and `.git`.
3. A `Dockerfile`, from the lesson: `python:3.13-slim`, requirements first,
   then `main.py` and `model.joblib`, port 8000, `--host 0.0.0.0`.
4. Build it **for amd64**, the architecture of most servers:

```bash
docker build --platform linux/amd64 -t penguin-api:1.0 .
docker run -d -p 8000:8000 --name penguin penguin-api:1.0
docker logs penguin
```

5. Open http://localhost:8000/docs again and re-check the three birds, now
   answered by the container.

On an Apple silicon Mac, `docker run` prints a platform warning and the image
runs under emulation: that is expected.

---

## When Part C does not work

| Symptom | Cause | Fix |
|---|---|---|
| `port is already allocated` | the uv server, or an old container, holds 8000 | `Ctrl+C` the server; `docker rm -f penguin` |
| `curl: (52) Empty reply from server` / *localhost didn't send any data* | uvicorn listens on `127.0.0.1` | `--host 0.0.0.0` in `CMD` |
| container exits at once; `docker logs` shows `FileNotFoundError: ... 'model.joblib'` | the model was not copied in | `COPY main.py model.joblib ./`, and run `train.py` before building |
| `ModuleNotFoundError: No module named 'pandas'` | `requirements.txt` exported before `uv add pandas` | export again, rebuild |
| `InconsistentVersionWarning` in the logs | the model was trained with other package versions | export from the same project that trained it, retrain, rebuild |

After every fix: `docker rm -f penguin`, build, run again.

---

## Part D — Ship it (10 min)

Push the image to Docker Hub:

```bash
docker tag penguin-api:1.0 <your-user>/penguin-api:1.0
docker push <your-user>/penguin-api:1.0
```

Give your neighbour the name `<your-user>/penguin-api:1.0`. On **their**
machine, with nothing of yours installed:

```bash
docker run -d -p 8001:8000 --name theirs <your-user>/penguin-api:1.0
```

They open http://localhost:8001/docs and check your API on the three birds.
Then swap. Port 8001 on their side, so their own container keeps 8000.

This is the whole promise of the session: your model, answering on a machine
where you installed nothing.

---

## Part E — Submit the image to ML-Arena (opens later this term)

ML-Arena will run submitted images and score them by calling their API on a
held-out set of penguins. **Image submission is not open yet**; this part will
be updated with the exact command when it is. Build to this contract now: it
is the one the submission is being built against.

| | The contract |
|---|---|
| Platform | `linux/amd64` |
| Port | the server listens on `0.0.0.0:8000` |
| Health | `GET /health` answers `200` within 60 seconds of start |
| Request | `POST /predict` with the four features as JSON numbers |
| Response | `200` and `{"species": "Adelie"}`, `"Chinstrap"` or `"Gentoo"` |
| Network | none at run time: the model must be inside the image |
| Score | accuracy on the hidden penguins |

Everything in the contract is what you already built in Parts B and C.

---

## Hand in

A GitHub repository named `penguin-api` with at least:

```text
train.py   main.py   model.joblib   requirements.txt   Dockerfile
.dockerignore   pyproject.toml   uv.lock   README.md
```

`README.md` states, in five lines or fewer:

- the image name on Docker Hub, and the two commands that run it;
- the majority baseline and your test accuracy, both to four decimals;
- the name of the classmate who ran your image, and what their `/predict`
  answered for the Gentoo row.

---

## Grading

| Criterion | Weight |
|---|---|
| `train.py` runs and beats the pass bar: test accuracy ≥ 0.95 | 20% |
| `main.py`: model loaded once, `/health` and a validated `/predict` | 25% |
| Image builds for `linux/amd64` and answers the three birds correctly | 30% |
| Image on Docker Hub, run by a classmate | 15% |
| README complete | 10% |

---

## Automatic deductions

- `joblib.load` inside the `predict` function
- a `.venv` folder inside the image (check with `docker exec penguin ls -a`)
- `requirements.txt` without versions
- training data or an API key committed to the repository
- a bare `except` in `predict` that returns a species when the model failed

---

## Carry it forward

Every project that ends in a model ends here: the model has to leave the
notebook. What you packed today is the smallest honest version of it: a
pinned environment, a contract for the questions and the answers, and an
artefact any machine can run.

The next step, which this course stops short of, is to run that image
somewhere public: a cloud service that takes an image and gives it a URL.
The image does not change; only where `docker run` happens does.

---

## Did you validate this session?

- [ ] `uv run python train.py` prints a majority baseline of **0.4419** and a
      test accuracy
- [ ] My test accuracy is **≥ 0.95**
- [ ] With `uv run uvicorn main:app`, http://localhost:8000/health answers
      `{"status":"ok"}`
- [ ] In `/docs`, the three birds come back `Adelie`, `Chinstrap`, `Gentoo`,
      and `"body_mass_g": "heavy"` comes back `422`
- [ ] `docker image inspect penguin-api:1.0 --format '{{.Architecture}}'`
      prints `amd64`
- [ ] With the container running, the three birds come back correct from
      http://localhost:8000/docs, and `docker logs penguin` shows no warning
- [ ] `docker push` succeeded, and a classmate ran my image on their machine
      and got `Gentoo` for the third bird
- [ ] Once ML-Arena opens image submission: my image is accepted and scored
      on the penguins leaderboard
