# penguin-api

The reference project of the MS2A Session 10 workshop, *Model Deployment*: a
scikit-learn pipeline (`model.joblib`, trained in
`../../challenges/mlp-s10-penguins-train.ipynb`) served by FastAPI, with an HTML
page, packed in a Docker image.

| File | Step |
|---|---|
| `model.joblib` | 1 — trained in Colab (scikit-learn 1.6.1) |
| `pyproject.toml`, `uv.lock`, `main.py` | 2 — the API, `uv run fastapi dev main.py` |
| `Dockerfile`, `.dockerignore` | 3 — `docker build -t penguin-api .` |
| `index.html` | 5 — the page served at `/` |
| `compose/` | Docker Compose: 3 replicas behind nginx, `cd compose && docker compose up -d` |

```bash
docker build -t penguin-api . && docker run --rm -p 8000:8000 penguin-api
curl localhost:8000/health
```
