# Writing a Dockerfile

Pack the Iris API of the previous lesson into an image: pin its packages,
write a ten-line Dockerfile, build, run, and share it.

<!-- notes: 15 minutes, in the iris-api folder from the previous lesson. Stop
the `uv run uvicorn` server first (Ctrl+C): the container wants port 8000
too. Build times were measured on an Apple M4 with python:3.13-slim already
pulled; a first pull of it is about 43 MB of download. Do the edit-and-rebuild
live, it is the slide that makes the layer order stick. The push slide needs a
Docker Hub account; show it, do not wait for every student to create one, the
lab does that. -->

---

## What goes into the image

```text
iris-api/
├── main.py            ← yes: the API
├── model.joblib       ← yes: the trained model
├── requirements.txt   ← yes: the pinned packages (next slide)
├── Dockerfile         ← the recipe
├── .dockerignore
├── train.py           ← no: training is already done
├── pyproject.toml, uv.lock
└── .venv/             ← never: built for your laptop, not for Linux
```

The image needs what **serving** needs: the code that answers, the model it
loads, the packages both import.

---

## Pin the versions you trained with

```bash
uv export --no-hashes -o requirements.txt
```

```text
fastapi==0.141.1
joblib==1.6.0
numpy==2.5.3
scikit-learn==1.9.1
uvicorn==0.52.4
...        (excerpt: 20 packages, each at its exact version)
```

`uv export` writes out the exact versions in `uv.lock`: the ones that trained
`model.joblib`. Load the model with another scikit-learn and you get:

```text
InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression
from version 1.7.2 when using version 1.9.1. This might lead to breaking
code or invalid results. Use at your own risk.
```

---

## The Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py model.joblib ./

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

A file named exactly `Dockerfile`, no extension, in the project folder.

---

## Line by line

| Instruction | What it does |
|---|---|
| `FROM python:3.13-slim` | start from an image with Debian and Python 3.13 (same as `.python-version`) |
| `WORKDIR /app` | create `/app` inside the image and work there |
| `COPY requirements.txt .` | copy a file from your folder into the image |
| `RUN pip install ...` | run a command **while building**; its result is saved in the image |
| `COPY main.py model.joblib ./` | copy the API and the model |
| `EXPOSE 8000` | document the port the server uses (it publishes nothing) |
| `CMD [...]` | the command a container runs **when it starts** |

`RUN` happens once, at build time. `CMD` happens every time a container starts.

---

## `--host 0.0.0.0`, the line everyone gets wrong once

- `uvicorn` alone listens on `127.0.0.1`: inside a container, that is the
  **container's** own loopback, unreachable from your laptop even with `-p`.
- `0.0.0.0` means *every network interface*, including the one Docker
  connects `-p` to.

Forget it and the container runs, the logs look healthy, and every request
fails: `curl: (52) Empty reply from server`, or in the browser *localhost
didn't send any data*.

---

## Keep the context small: `.dockerignore`

```text
.venv
__pycache__
.git
```

`docker build .` first sends the whole folder (the *build context*) to
Docker. `.dockerignore` works like `.gitignore`: this project's `.venv` alone
is 164 MB of packages compiled for your laptop, which no image should
contain.

---

## Build

```bash
docker build -t iris-api:1.0 .
```

- `-t iris-api:1.0` names the image, `name:tag`.
- `.` is the build context: this folder.
- Each instruction prints a step; the last line names the image.

```bash
docker images iris-api
```

About **600 MB** on disk, **131 MB** compressed to download: Debian, Python,
numpy, scipy and scikit-learn weigh far more than your 991-byte model.

---

## Layers and the cache

![Each instruction is a layer; editing main.py rebuilds only the layers above it](assets/deploy/dockerfile-layers.png)

Each instruction produces a **layer**, and Docker reuses a layer if nothing it
depends on changed. That is why `requirements.txt` is copied and installed
**before** the code: edit `main.py`, and the 315 MB `pip install` layer comes
from the cache. Copy everything first, and every code edit reinstalls every
package.

---

## Run

Stop the `uv run uvicorn` server first (`Ctrl+C`): port 8000 is needed.

```bash
docker run -d -p 8000:8000 --name iris iris-api:1.0
docker logs iris
```

```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Open **http://localhost:8000/docs**: the same page as before, now answered by
a container that owns its Python and its packages. `docker exec iris python -c
"import sklearn; print(sklearn.__version__)"` prints `1.9.1`, whatever your
laptop has.

---

## Share the image

A **Docker Hub** account (free, hub.docker.com) gives you a namespace:

```bash
docker login
docker tag iris-api:1.0 <your-user>/iris-api:1.0
docker push <your-user>/iris-api:1.0
```

On any other machine with Docker, nothing else installed:

```bash
docker run -d -p 8000:8000 <your-user>/iris-api:1.0
```

`docker tag` adds a second name to the same image; nothing is copied.

---

## Apple silicon laptops, amd64 servers

A Mac with an M chip builds `arm64` images by default. Most cloud servers run
`amd64`, and an `arm64`-only image does not start there.

```bash
docker build --platform linux/amd64 -t iris-api:1.0 .
```

Your Mac still runs it, through emulation, with a warning:

```text
WARNING: The requested image's platform (linux/amd64) does not match the
detected host platform (linux/arm64/v8)
```

Windows and Linux laptops are almost always `amd64` already.

---

## A variant: train inside the image

For a model that trains in seconds on public data, the Dockerfile can train it
itself:

```dockerfile
COPY train.py main.py ./
RUN python train.py
```

The model then comes from the exact packages of the image, by construction.
For anything that needs a GPU, hours, or private data, train outside and
`COPY` the file in, as above.

---

## Check yourself

1. You change one line of `main.py` and rebuild. Why does it take seconds, not
   half a minute? **Every layer before `COPY main.py` is unchanged and comes from
   the cache, including `pip install`.**
2. The container runs, `docker logs` shows `Uvicorn running on
   http://127.0.0.1:8000`, and the browser gets no data. Fix?
   **`--host 0.0.0.0` in `CMD`.**
3. What is the difference between `RUN` and `CMD`?
   **`RUN` executes at build time and is saved in the image; `CMD` executes
   each time a container starts.**
4. Why generate `requirements.txt` from `uv.lock` rather than write
   `scikit-learn` in it by hand? **So the image loads the model with the
   versions that trained it.**
