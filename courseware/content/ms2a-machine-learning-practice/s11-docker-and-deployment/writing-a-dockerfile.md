# Step 3 — The API in an Image, on Docker Hub

The `penguin-api` folder of Step 2 becomes an image: Python, the locked
packages, `main.py` and `model.joblib` in one package that runs on any
machine with Docker, then published on Docker Hub.

<!-- notes: 30 minutes, in the penguin-api folder. Stop the `fastapi dev`
server first (Ctrl+C): the container needs port 8000. Measured on 2026-10-19,
Apple M4, OrbStack, python:3.13-slim already pulled: cold build ~19 s, rebuild
after editing main.py < 1 s, image 687 MB (1.15 GB without --no-cache),
amd64 build 13.7 s, 156 MB compressed on Docker Hub. Where students get stuck:
(1) `port is already allocated`: the dev server or an old container holds 8000;
(2) `docker push` denied: not logged in, or the image name does not start with
their Docker Hub username; (3) an arm64 image pushed from a Mac, which Render
refuses later with `exec format error`; (4) Windows: Docker Desktop not
started. Do the edit-and-rebuild live. -->

---

## The Dockerfile

A file named `Dockerfile`, no extension, in `penguin-api/`:

```dockerfile
FROM python:3.13-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-cache
COPY main.py model.joblib ./
CMD ["uv", "run", "fastapi", "run", "main.py", "--port", "8000"]
```

And a `.dockerignore` with two lines, `.venv` and `__pycache__`: the laptop's
virtual environment is built for the laptop, not for the image's Linux.
Step 5 adds `index.html` to the `COPY main.py model.joblib ./` line.

---

## Line by line

| Instruction | Effect |
|---|---|
| `FROM python:3.13-slim` | Debian + Python 3.13, the version of Colab |
| `RUN pip install --no-cache-dir uv` | uv inside the image |
| `COPY pyproject.toml uv.lock ./` | the dependency list, with exact versions |
| `RUN uv sync --locked --no-cache` | installs exactly `uv.lock`; fails if it is out of date |
| `COPY main.py model.joblib ./` | the code and the model |
| `CMD [...]` | the command each container runs at start |

- `RUN` executes once, at build time. `CMD` executes at every container start.
- `fastapi run` listens on `0.0.0.0`, every interface of the container;
  `fastapi dev` listens on `127.0.0.1`, unreachable from outside the container.
- `--no-cache` keeps uv's download cache out of the image: 687 MB instead of
  1.15 GB. The JSON (exec) form of `CMD` avoids the `JSONArgsRecommended`
  build warning.

---

## Build and run

```bash
docker build -t penguin-api .
docker run --rm -p 8000:8000 penguin-api
```

- `-t penguin-api` names the image; `.` is the build context, this folder.
- `-p 8000:8000` maps the laptop's port 8000 to the container's port 8000.
- The container prints `Uvicorn running on http://0.0.0.0:8000`.

The `curl` requests and `/docs` of Step 2 now reach the container, which has
its own Python and packages.

**Checkpoint:** `curl localhost:8000/health` → `{"status":"ok"}`, and the three
birds → `Adelie`, `Chinstrap`, `Gentoo`.

---

## Layers and the cache

![Each instruction is a layer; editing main.py rebuilds only the layers above it](assets/deploy/dockerfile-layers.png)

Each instruction produces a layer. Docker reuses a layer when its inputs are
unchanged. `uv.lock` is copied and installed **before** the code: after an
edit to `main.py`, the 335 MB `uv sync` layer comes from the cache and the
rebuild takes under 1 s, against ~19 s for the first build.

---

## Publish: amd64, then Docker Hub

A Mac with an M chip builds `arm64` images; Render runs `linux/amd64` only
([multi-platform builds](https://docs.docker.com/build/building/multi-platform/)).

```bash
docker login
docker build --platform linux/amd64 -t USER/penguin-api:v1 .
docker push USER/penguin-api:v1
```

- `USER` is the Docker Hub username: the name decides where the push goes.
- The pushed image: 156 MB compressed, architecture amd64. An arm64 image
  fails on Render with `exec format error`.
- A new tag for each push (`v1`, `v2`, …). Render caches public images, so a
  re-pushed `latest` can be served stale.

Variants: [uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/),
[FastAPI Docker guide](https://fastapi.tiangolo.com/deployment/docker/).

---

## On Docker Hub

![The repository on Docker Hub after two pushes: tags v1 and v2, linux/amd64](assets/deploy/dockerhub-tags.png)

**Checkpoint:** on a neighbour's laptop, `docker pull USER/penguin-api:v1` →
`Downloaded newer image for USER/penguin-api:v1`.

---

## Check yourself

1. Only `main.py` changed. Why does the rebuild take under a second? **Every
   layer before `COPY main.py model.joblib ./`, including `uv sync`, is
   unchanged and comes from the cache.**
2. The `CMD` uses `fastapi dev` instead of `fastapi run`. The container starts
   but `curl` gets no reply. Why? **`dev` listens on `127.0.0.1`, the
   container's own loopback; `-p` cannot reach it.**
3. Why `--platform linux/amd64` on a Mac? **The Mac builds `arm64` by default
   and Render runs only `amd64` images.**
