# Workshop Deliverable — Your Model Online

The five steps of the workshop end with one model served three ways: from the
laptop, from an image anyone can pull, and from a public URL with a page.

<!-- notes: 10 minutes at the end of the session, the checks run live while
students post their links. The reference deployment of 2026-10-19
(racousin90/penguin-api:v2, https://penguin-api-y9gh.onrender.com) passes every
check below. A Free service asleep answers the first check after about a
minute: run the curl checks once, then again. -->

---

## Hand in

Three links, posted in the course channel:

1. **The image**: `USER/penguin-api:v2` on Docker Hub, public, `linux/amd64`.
2. **The service**: `https://<name>.onrender.com`, on a Free instance.
3. **The code**: a GitHub repository `penguin-api` with `main.py`,
   `index.html`, `Dockerfile`, `.dockerignore`, `pyproject.toml`, `uv.lock`
   and `model.joblib`.

The reference files are in
[deploy/penguin-api](https://github.com/racousin/data_science_practice/tree/main/website/public/modules/ms2a-machine-learning-practice/deploy/penguin-api).

---

## The checks

| Check | Expected |
|---|---|
| `curl <URL>/health` | `{"status":"ok"}` |
| the three birds on `POST <URL>/predict` | `Adelie`, `Chinstrap`, `Gentoo` |
| `{"body_mass_g": "heavy"}` on `POST <URL>/predict` | `422` |
| `<URL>/` in a browser, **Predict** with the default values | `Adelie` |
| `docker run --rm -p 8000:8000 USER/penguin-api:v2` on another machine | the same answers on `localhost:8000` |
| `pyproject.toml` | `scikit-learn==` the version Colab printed |

---

## Did you validate this session?

- [ ] The Colab notebook printed a test accuracy of **0.9884** and I have
      `model.joblib`
- [ ] `uv run fastapi dev main.py` answers the three birds on
      `localhost:8000`
- [ ] `docker run -p 8000:8000 penguin-api` answers the same
- [ ] `USER/penguin-api:v2` is on Docker Hub, `linux/amd64`
- [ ] The Render URL answers `/health`, `/predict` and shows the page
- [ ] (Compose) three replicas behind nginx: `X-Upstream` changes from one
      request to the next

---

## Going further

- **Batch and online**: a `POST /predict_batch` taking a list of penguins
  and returning a list of species, timed against 100 calls to `/predict`.
- **Latency**: p50 and p95 of 100 requests to the Render URL, with a
  10-line Python script, warm and after 15 idle minutes.
- **Probabilities**: `model.predict_proba` in the response, next to
  `species`.
- **Versioning**: a `model_version` field in the response, bumped with each
  new image tag.
- **Robustness**: a `HEALTHCHECK` instruction in the Dockerfile, and
  `docker run --restart on-failure`.
