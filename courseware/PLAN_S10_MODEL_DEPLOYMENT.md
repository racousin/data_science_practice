# Plan — Session 10 rebuilt as a Model Deployment Workshop

**Status:** applied on 2026-10-19. Deviations found in the dry run (exec-form `CMD` on port 8000 with Render's port detection, Python 3.13 / scikit-learn 1.6.1 from Colab, Compose lesson added) are recorded in the lessons and in `course.yaml`.

Course 15 `ms2a-machine-learning-practice`, module **#35** `s11-docker-and-deployment`
(https://ml-arena.com/module-editor/35?course=15). Written 2026-10-19. Every
external fact below was checked on that date; the ones that could not be
checked are listed in §9 and are settled by the teacher dry run (§7, step 3).

## 1. Decisions

| | Decision |
|---|---|
| Scope | Every current lesson of module 35 is replaced. CVAT / Compose and Hugging Face Spaces go. |
| Thread | **One project from start to finish**: Palmer penguins → sklearn pipeline → FastAPI → Docker image → Docker Hub → Render → HTML page. |
| Model | `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))` on the 4 numeric features. The reference scores 0.9884 test accuracy, against 0.4419 for the majority class (existing `labs/.../penguin-api/train.py`). |
| Training | In **Colab** (new notebook). The student downloads `model.joblib`. |
| API | **FastAPI**, project managed by **uv**: `uv add "fastapi[standard]"`, then `uv run fastapi dev main.py`. |
| Frontend | `index.html` served by the same FastAPI app at `/`, calling `/predict` with `fetch`. Same image, same Render service, no CORS. |
| Hosting | **Render free web service**, source "Existing Image" from **Docker Hub**. |
| Deliverable | A checklist, no ML-Arena challenge: the Docker Hub image name and the Render URL, which the teacher calls with `curl`. |
| Module | Same module #35. Slugs cannot change. New title: **"Session 10 — Model Deployment Workshop"**. |
| Tone | Definitions, figures and worked examples. No "you should". |

## 2. Session timeline (3 h)

| Min | Lesson | Hands-on result |
|---|---|---|
| 0–20 | 1. Serving a model: the concepts | — |
| 20–30 | 2. Where models run | — |
| 30–45 | 3. Step 1: train and save (Colab) | `model.joblib` on the laptop |
| 45–75 | 4. Step 2: a model API with uv + FastAPI | `curl localhost:8000/predict` answers |
| 75–90 | 5. Docker: images, containers, registries | — |
| 90–100 | 6. Installing Docker, first containers | `docker run hello-world` |
| 100–130 | 7. Step 3: the API in an image, on Docker Hub | `docker push USER/penguin-api:v1` |
| 130–150 | 8. Step 4: serving on Render | `curl https://<name>.onrender.com/health` |
| 150–170 | 9. Step 5: an HTML page | the form works on the Render URL |
| 170–180 | 10. Deliverable and checklist | URL and image posted |

**Before the session** (goes in the module summary, and in a message to the
students the week before): install uv, install Docker (OrbStack on macOS, Docker
Desktop on Windows, Docker Engine on Linux), and create a Docker Hub account and
a Render account. Render can ask for a card check (§9).

## 3. Lesson map: what happens to every live lesson

| # | Slug (id) | Action | New title | Kind | Min |
|---|---|---|---|---|---|
| 1 | `why-deploy` (192) | rewrite | Serving a Model: The Concepts | lesson | 20 |
| 2 | `where-models-run` | **new** | Where Models Run: On-Premise, Cloud, Managed | lesson | 10 |
| 3 | `train-and-save-a-model` | **new** | Step 1 — Train and Save a Model in Colab | lesson | 15 |
| 4 | `serving-a-model-with-fastapi` (197) | rewrite | Step 2 — A Model API with uv and FastAPI | lesson | 30 |
| 5 | `docker-concepts` (193) | rewrite | Docker: Images, Containers, Registries | lesson | 15 |
| 6 | `docker-install-and-run` (194) | rewrite | Installing Docker and Running a Container | lesson | 10 |
| 7 | `writing-a-dockerfile` (198) | rewrite | Step 3 — The API in an Image, on Docker Hub | lesson | 30 |
| 8 | `deploy-on-render` | **new** | Step 4 — Serving the Image on Render | lesson | 20 |
| 9 | `html-frontend` | **new** | Step 5 — An HTML Page for the Model | lesson | 20 |
| 10 | `lab-11` (199) | rewrite | Workshop Deliverable — Your Model Online | exercise | 10 |
| — | `docker-compose-cvat` (195) | **delete** | | | |
| — | `apis-and-http` (196) | **delete**: the HTTP basics move into lesson 4 | | | |
| — | `deploy-on-hugging-face` (291) | **delete** | | | |

Each lesson has 5–7 `##` sections, one classic diagram, short code (3–5 lines
per block) and a closing "Check yourself" of 2–3 questions. Lessons 3–9 end with
a **Checkpoint** line: the one command whose output proves the step works.

## 4. Lesson contents

### 1. `why-deploy` — Serving a Model: The Concepts

Opening: a trained model is a file. **Serving** means that a program loads this
file once and answers prediction requests from other programs, at any time.

Sections, each one definition plus one concrete example:

1. **From a notebook to a service.** One cell run by hand, compared with a
   process that answers requests. Example: a hospital form calls a model every
   time it is saved.
2. **Reproducibility.** The same input gives the same output on the laptop, the
   server and a colleague's machine. It requires the same code, the same model
   file, the same Python and package versions, and the same OS libraries.
   Example: a `model.joblib` written with scikit-learn 1.9 and loaded with 1.7
   gives an `InconsistentVersionWarning` or an error. Step 1 shows it for real
   (Colab version vs local version).
3. **Robustness.** A process crashes, a machine reboots, and a supervisor
   restarts the process. Examples: Docker `--restart always | on-failure |
   unless-stopped` ([docs](https://docs.docker.com/engine/containers/start-containers-automatically/)),
   Kubernetes `restartPolicy` ([docs](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#restart-policy)),
   and a **health check**, an endpoint the supervisor calls (`GET /health`; [Render](https://render.com/docs/health-checks)
   stops traffic after 15 s of failures and restarts the instance after 60 s).
4. **Scalability.** One user, then 1,000, then a billion. **Vertical**
   scaling means a bigger machine. **Horizontal** scaling means more copies
   behind a load balancer ([Azure guide](https://learn.microsoft.com/en-us/azure/architecture/guide/design-principles/scale-out)).
   Image: load-balancer diagram (§6). Order of magnitude: Cloudflare serves 81
   million HTTP requests per second on average ([2025 Radar](https://blog.cloudflare.com/radar-2025-year-in-review/)).
   Horizontal copies require a stateless API, which is why the model is loaded
   read-only.
5. **Batch or online.** A table: input, trigger, output, typical latency,
   examples. Batch: the ML-Arena CSV submissions, a nightly scoring job. Online:
   one request, one answer. Definitions from [Azure ML endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints)
   and [SageMaker deployment options](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
   (real-time / serverless / asynchronous / batch).
6. **Latency.** The time between request and answer, described by
   **percentiles**: p50, p95, p99 ([Google SRE book](https://sre.google/sre-book/service-level-objectives/)).
   A **cold start** is the first request after an idle period. Measured figure:
   the latency of 200 requests to the workshop API, run locally and on Render,
   with Render's cold start (§6, figure `latency-local-vs-render.png`).
7. **Contract (interface).** The API's inputs and outputs are a promise: field
   names, types, units, and error codes. The model behind it can change, but the
   contract must stay the same. FastAPI writes that contract as an
   [OpenAPI](https://spec.openapis.org/oas/latest.html) document at `/openapi.json`.
   Example: the `Penguin` request with its 4 floats in mm/g, and the
   `{"species": "Adelie"}` answer.

### 2. `where-models-run` — On-Premise, Cloud, Managed

1. **On-premise.** The organisation owns the machines (hospital, bank, lab
   cluster). It controls the data, but it also buys and maintains the hardware.
2. **Cloud virtual machines (IaaS).** A rented machine, billed per second:
   [AWS EC2](https://aws.amazon.com/ec2/), [Google Compute Engine](https://cloud.google.com/products/compute),
   [Azure Virtual Machines](https://azure.microsoft.com/en-us/products/virtual-machines).
3. **Platforms (PaaS) and managed ML endpoints.** The platform runs a container
   or a model for you. Examples: Render, [SageMaker endpoints](https://aws.amazon.com/sagemaker/ai/deploy/),
   [Azure ML endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints),
   Google's Vertex AI. Vertex AI now redirects to "Gemini Enterprise Agent
   Platform", so check the name before the slide is published. Image:
   on-premise / IaaS / PaaS / SaaS responsibility diagram (§6).
4. **The operating system: Linux.** W3Techs (2026-10-19): Unix-like systems run
   **92.2 %** of websites whose OS is known
   ([source](https://w3techs.com/technologies/overview/operating_system)). A
   container image is a Linux userland, so the server, Render and the image are
   all Linux, even when the laptop runs macOS or Windows. This is the reason for
   `--platform linux/amd64` in Step 3.
5. **The workshop path.** Roadmap figure (§6, `workshop-path.png`): train
   (Colab) → API (laptop) → image (laptop) → Docker Hub (registry) → Render
   (cloud) → HTML page (browser). Each arrow is one lesson.

### 3. `train-and-save-a-model` — Step 1 (Colab)

The Colab badge for `mlp-s10-penguins-train.ipynb` (§5), and the penguins
artwork (credit "Artwork by @allison_horst").

1. The data: 344 penguins, 3 species, 4 measurements. Dataset licence CC0,
   [source](https://github.com/allisonhorst/palmerpenguins). The bill-measurement
   figure `culmen_depth.png`.
2. What the notebook does: look → split → baseline → pipeline → score → save.
   No code on the slide beyond the 3 key lines (`make_pipeline`, `fit`,
   `joblib.dump`).
3. Why a **pipeline**: the scaler is saved inside the model file, so the API
   sends raw millimetres.
4. The versions cell: the notebook prints `python`, `scikit-learn` and `joblib`
   versions. These numbers are the reproducibility contract for Step 2.
5. **Checkpoint**: `model.joblib` (~2 KB) is in the Downloads folder.

### 4. `serving-a-model-with-fastapi` — Step 2

1. **HTTP in one slide.** A client sends a method + URL + optional JSON body,
   and the server answers with a status code + JSON body. Reuse
   `client-server.png` and `url-anatomy.png`. `GET` reads, `POST` sends data.
   Status codes 200 / 404 / 422 / 500.
2. **uv.** Install ([docs](https://docs.astral.sh/uv/getting-started/installation/)):
   `curl -LsSf https://astral.sh/uv/install.sh | sh`. On Windows,
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
   Then:
   ```bash
   uv init --python 3.12 penguin-api && cd penguin-api
   uv add "fastapi[standard]" pandas joblib "scikit-learn==<Colab's version>"
   ```
   The scikit-learn pin is the reproducibility section made concrete. Set the
   Python minor to Colab's on the day (3.12 at the time of writing; recheck it
   in the dry run).
3. **The smallest API**, then `uv run fastapi dev main.py`
   ([FastAPI first steps](https://fastapi.tiangolo.com/tutorial/first-steps/)):
   ```python
   from fastapi import FastAPI
   app = FastAPI()
   @app.get("/health")
   def health(): return {"status": "ok"}
   ```
   Test it in the browser at `http://localhost:8000/health`, and with
   `curl localhost:8000/health`.
4. **The contract**: the `Penguin(BaseModel)` with 4 floats, `model =
   joblib.load("model.joblib")` once at import, and `POST /predict` returning
   `{"species": ...}` built from `pd.DataFrame([penguin.model_dump()])` (the
   reference `labs/.../penguin-api/main.py`, unchanged).
5. **Testing it.** With curl:
   ```bash
   curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"bill_length_mm":39.1,"bill_depth_mm":18.7,"flipper_length_mm":181,"body_mass_g":3750}'
   ```
   In the browser, `/docs` → "Try it out" (re-shoot `docs-*.png` on penguins).
   A wrong type returns **422**, which is the contract being enforced. Windows:
   PowerShell aliases `curl`, so use Git Bash or `curl.exe`, or `/docs`.
6. The three reference birds table (from the current lab) → `Adelie`,
   `Chinstrap`, `Gentoo`.
7. **Checkpoint**: the three curls return the three species.

### 5. `docker-concepts` — Images, Containers, Registries

Keep the current lesson's skeleton (the owner trimmed it on 2026-10-19, commit
fc8b272), with these changes:

1. "It works on my machine". The layer table, updated with the workshop's real
   numbers.
2. VM vs container: the existing `vm-vs-container.png`.
3. Docker architecture: client → daemon → registry. Image
   `docker-architecture.webp` from Docker Docs (§6).
4. **Image** (read-only template), **container** (running instance),
   **Dockerfile** (recipe), **registry** (image store, e.g. Docker Hub), **tag**
   (`USER/penguin-api:v1`).
5. The 6 commands, one line each: `build`, `run`, `ps`, `logs`, `stop`,
   `push`/`pull`. Reuse `build-push-pull-run.png`.
6. Ports: `-p 8000:8000` maps host to container. Reuse `port-mapping.png`.

### 6. `docker-install-and-run` — Installing Docker, First Containers

1. Install. macOS: **[OrbStack](https://orbstack.dev/download)** (free for
   personal use, education licences on request,
   [pricing](https://orbstack.dev/pricing)). Windows: [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
   with WSL 2. Linux: [Docker Engine](https://docs.docker.com/engine/install/).
2. Docker Hub: create an account at hub.docker.com, then `docker login`. Free
   accounts get unlimited public repositories and 200 pulls per 6 h when logged
   in ([limits](https://docs.docker.com/docker-hub/usage/)).
3. `docker run --rm hello-world`, with the explanation of what happened
   (pull → create → run → exit).
4. `docker run -d -p 8080:80 nginx`, then the browser on `localhost:8080`,
   then `docker ps`, `docker logs`, `docker stop`.
5. **Checkpoint**: `docker ps -a` shows the stopped nginx container.

### 7. `writing-a-dockerfile` — Step 3: the API in an Image, on Docker Hub

1. **A first Dockerfile**: a 3-line image that prints the Python version
   (`FROM python:3.12-slim` / `CMD ["python", "--version"]`), then
   build + run. It shows that the image's Python is not the laptop's.
2. **The workshop Dockerfile**, line by line (reuse `dockerfile-layers.png`):
   ```dockerfile
   FROM python:3.12-slim
   RUN pip install uv
   WORKDIR /app
   COPY pyproject.toml uv.lock ./
   RUN uv sync --locked
   COPY main.py model.joblib index.html ./
   CMD uv run fastapi run main.py --port ${PORT:-8000}
   ```
   `fastapi run` listens on `0.0.0.0`. `$PORT` is set by Render (default
   10000) and falls back to 8000 locally. `.dockerignore` contains `.venv`. The
   uv-native variant (`COPY --from=ghcr.io/astral-sh/uv`) is linked, not taught:
   [uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/),
   [FastAPI Docker guide](https://fastapi.tiangolo.com/deployment/docker/).
   `index.html` is added in Step 5. Until then, that line copies two files.
3. **Build and run**:
   `docker build -t penguin-api .` → `docker run --rm -p 8000:8000 penguin-api`,
   then the same curl and `/docs` as in Step 2. Same answers: reproducibility
   checked.
4. **Layer cache**: edit `main.py` and rebuild in ~3 s instead of ~30 s. The
   existing measured numbers apply.
5. **Push**. Render runs `linux/amd64`, and Apple Silicon builds `arm64` by
   default
   ([multi-platform docs](https://docs.docker.com/build/building/multi-platform/)):
   ```bash
   docker build --platform linux/amd64 -t USER/penguin-api:v1 .
   docker push USER/penguin-api:v1
   ```
   A new tag for every push (`v1`, `v2`, …). Render caches public images, so
   `latest` can come back stale.
6. **Checkpoint**: `docker pull USER/penguin-api:v1` works from a neighbour's
   laptop.

### 8. `deploy-on-render` — Step 4

1. What Render is: a PaaS. It runs a container image and gives it an HTTPS URL.
   Free instance: 512 MB RAM, 0.1 CPU, it **spins down after 15 min without
   traffic, and wakes in about 1 min**, with 750 instance-hours per month
   ([free tier](https://render.com/docs/free)). These are the latency and
   cold-start concepts of lesson 1, seen live.
2. Steps, one screenshot each (`render-*.png`, taken in the dry run):
   **New → Web Service → Existing Image** → `docker.io/USER/penguin-api:v1` →
   name, region (Frankfurt), instance type **Free** → **Deploy**
   ([docs](https://render.com/docs/deploying-an-image)).
3. Logs: the same uvicorn lines as on the laptop, with port 10000.
4. Test it with `curl https://<name>.onrender.com/health`, the `/predict` curl,
   and `/docs` in the browser. Time a cold request against a warm one with
   `curl -w "%{time_total}\n"`.
5. New version: push `:v2`, then **Settings → Image URL** → `:v2` → **Manual
   Deploy**. Mention the deploy hook (`?imgURL=`) as the automated form
   ([deploy hooks](https://render.com/docs/deploy-hooks)).
6. **Checkpoint**: `/health` answers on the public URL from a phone.

### 9. `html-frontend` — Step 5

1. A browser page is an HTTP client like curl. Same contract, different caller.
2. `index.html`: 4 `<input type="number">`, one button, one `<p id="out">`.
   The script, ~6 lines:
   ```js
   const r = await fetch("/predict", {method: "POST",
     headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
   out.textContent = (await r.json()).species;
   ```
3. Serve it from FastAPI:
   ```python
   from fastapi.responses import FileResponse
   @app.get("/")
   def home(): return FileResponse("index.html")
   ```
   The page and the API share one origin, so CORS does not apply. One sentence
   defines CORS, with a link to [MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS).
4. Local test, then rebuild → push `:v2` → redeploy on Render (Step 4's
   section 5).
5. **Checkpoint**: the three reference birds typed into the form on the Render
   URL.

### 10. `lab-11` — Workshop Deliverable (exercise, no slides)

- Post three things: the Docker Hub image `USER/penguin-api:v2`, the Render URL,
  and the GitHub repo containing `main.py`, `index.html`, `Dockerfile`,
  `.dockerignore`, `pyproject.toml`, `uv.lock` and `model.joblib`.
- Teacher check: `curl <URL>/health`, the three birds on `/predict`, and the form
  in a browser. `docker run -p 8000:8000 USER/penguin-api:v2` works on the
  teacher's machine.
- Going further: a `/predict_batch` endpoint taking a list (batch versus
  online), a Dockerfile `HEALTHCHECK`, `docker run --restart on-failure`, p50/p95
  over 100 requests with a 10-line script, returning class probabilities, and a
  `model_version` field in the response.
- Teacher notes: the common failures. A missing `--platform linux/amd64` gives
  `exec format error` on Render. Port 8000 already in use. A scikit-learn
  version mismatch between Colab and local. PowerShell curl quoting.

## 5. Code and notebook artefacts

| Artefact | Path | Notes |
|---|---|---|
| Notebook builder | `tools/notebooks/mlp_s10_penguins_train.py` | Same pattern as `mlp_s3_rain_eda.py`: imports `build_notebooks as nb` for the helpers only, never runs its `main`. Running it writes exactly one file. |
| Notebook | `website/public/modules/ms2a-machine-learning-practice/challenges/mlp-s10-penguins-train.ipynb` | Cells: versions → load CSV → `head`/`describe`/`value_counts` → one scatter (flipper vs bill length by species) → split (0.25, stratified, seed 0) → baseline → pipeline → accuracy + confusion matrix → predict the 3 birds → `joblib.dump` → `files.download("model.joblib")`. Executed once; the markdown quotes the real numbers. |
| Reference project | `labs/s11-docker-and-deployment/penguin-api/` | Updated to the final state: `main.py` (+ `/` route), `index.html`, the new `Dockerfile`, `.dockerignore`, `pyproject.toml`/`uv.lock` (fastapi[standard], scikit-learn pinned to Colab's), `model.joblib` from the notebook. `requirements.txt` removed. |
| Removed | `labs/s11-docker-and-deployment/iris-api/` | Not used any more. |
| Docs screenshots | `labs/s11-docker-and-deployment/shoot_docs.py` | Re-pointed at penguin-api to re-shoot `docs-*.png` and `browser-health.png`. |

Session 9 notebooks are named `mlp-s9-*`. This one is `mlp-s10-*`, after the
taught session number, not the module slug.

## 6. Images

**From the web** (fetched by a committed script
`tools/figures/ms2a_s10_web.py` into `assets/deploy/`, unmodified, each with a
credit line under it in the lesson):

| File | Source | Licence / credit | Lesson |
|---|---|---|---|
| `load-balancer.svg→png` | https://upload.wikimedia.org/wikipedia/commons/1/1f/Load_Balancing_Cluster_%28NAT%29.svg | CC BY-SA 4.0, HFWMan / Wikimedia Commons | 1 |
| `cloud-layers.svg→png` | https://upload.wikimedia.org/wikipedia/commons/7/7e/Cloud_computing_layers.svg | CC0, Bikeborg/Wylve | 2 |
| (alt.) `onprem-iaas-paas-saas.png` | https://upload.wikimedia.org/wikipedia/commons/f/ff/Comparison_of_on-premise%2C_IaaS%2C_PaaS%2C_and_SaaS.png | marked public domain (Rosati & Lynn). Check the file page before use. | 2 |
| `docker-architecture.webp→png` | https://docs.docker.com/get-started/images/docker-architecture.webp | Apache-2.0 (docker/docs), credit "Docker Docs" | 5 |
| `penguins.png` | https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png | "Artwork by @allison_horst" (teaching use invited, not CC0) | 3 |
| `culmen-depth.png` | https://allisonhorst.github.io/palmerpenguins/reference/figures/culmen_depth.png | same | 3 |

The docker.com container-vs-VM image is © Docker with no licence, so it is not
used. The repo's own `vm-vs-container.png` replaces it.

**Generated** (`tools/figures/s11_docker_and_deployment.py`, extended):
`workshop-path.png` (replaces `deploy-path.png`: Colab → laptop API → image →
Docker Hub → Render → browser), and `latency-local-vs-render.png` (measured:
200 sequential requests, local container vs warm Render, plus the cold start,
p50/p95/p99 marked).

**Screenshots** (dry run): `render-new-service.png`, `render-existing-image.png`,
`render-logs.png`, `html-form.png`, `docs-*.png` re-shot.

**Removed**: `cvat-*.png` (8 files), `deploy-path.png`, and their entries in the
figures script's `SCREENSHOTS`.

## 7. Execution order

1. **Sync** (website is the truth): `cd courseware && make pull-all`, then
   commit and push. Leave the unrelated dirty `mlp-project-ds-harness` files
   alone.
2. **Build the artefacts**: the notebook builder → execute the notebook in Colab
   or locally → note its versions → update `penguin-api/` (pin those versions)
   → the figures scripts.
3. **Teacher dry run, end to end, on an Apple Silicon laptop**: Colab →
   `uv run fastapi dev` → `docker build --platform linux/amd64` → push to a
   teacher Docker Hub → a **Render Free** service from the image. Record:
   whether the Free instance accepts an Existing Image, whether Render asks for a
   card, the cold-start time, and the latency measurements. Take the Render
   screenshots. Every number a lesson quotes comes from this run.
4. **Write the 10 lessons** (§4) and update `course.yaml`: module title,
   summary (with the before-session list), lesson list and order, minutes, and
   `kind: exercise` for `lab-11`. Delete the stale comment above the module
   ("Attach the penguins challenge…").
5. **Delete the old files**: `docker-compose-cvat.md`, `apis-and-http.md`,
   `deploy-on-hugging-face.md`, the iris-api, the cvat assets.
6. **Check**: `make check-slides COURSE=ms2a-machine-learning-practice
   MODULE=s11-docker-and-deployment`, `make slides-one …`, and
   `make publish-dry … DIFF=1`.
7. **Push the notebook to GitHub `main` first** (the Colab badge resolves
   against it), then `make publish COURSE=ms2a-machine-learning-practice
   MODULE=s11-docker-and-deployment`. The 4 new lessons get created and the 6
   rewrites update in place. The final reorder fails while 195/196/291 exist.
8. **The deletes run by the owner** (the auto-mode classifier blocks them): one
   line, `delete_lesson` for 195, 196 and 291. Then prune their entries
   from `.mlarena-state.json` (`lessons` and `published.lessons`), run
   `make publish … MODULE=s11-docker-and-deployment FORCE=1`, and
   `make check-sync`.
9. **Student check**: open the module as a student (`tools/student_walk.py`),
   follow every Colab badge, and check that every image returns 200.
10. Commit and push.

## 8. Acceptance

- A student with a fresh laptop reaches every checkpoint in 3 h with the lesson
  text alone (dry run 3 is the proof).
- The reference image `USER/penguin-api:v2` returns `Adelie`, `Chinstrap` and
  `Gentoo` for the three birds, locally and on Render.
- Every lesson has 5–7 sections, at least one image, and code blocks ≤ 5 lines,
  except the Dockerfile (7 lines).
- No lesson links to CVAT, Hugging Face or Iris. `make check-sync` is green.

## 9. Open points, settled by the dry run

1. **Render Free + Existing Image**: no Render page forbids it, and a 2023
   changelog opened public-image deploys to everyone, but no page states it for
   Free. Fallback: Render's "Git repository + Dockerfile" source on the same
   free tier, with the image on Docker Hub kept as the Step 3 deliverable.
2. **Card check**: Render sometimes asks for a card verification ($1 hold). If
   it does, students without a card pair with one who has one, and the module
   summary says so in advance.
3. **Colab versions**: Colab's Python and scikit-learn versions on the day fix
   the pins in Step 2 and the Dockerfile's `FROM`.
4. **OrbStack amd64 builds**: emulation speed on M-series. The cache-free
   `--platform linux/amd64` build time is measured in the dry run.
5. **Windows Home + Docker Desktop**: Docker's page is ambiguous. If a student
   cannot install it, a classmate builds the image, and Render still runs it.
6. **Render bandwidth** on Hobby workspaces (5 GB since 2026-08-01, per a
   third-party post) is irrelevant at workshop traffic.
