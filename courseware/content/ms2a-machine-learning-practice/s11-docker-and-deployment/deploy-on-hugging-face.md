# Deploying the Image on Hugging Face Spaces

The image runs on any machine with Docker. The last step gives it a public
URL: a service that takes the image, runs it, and answers the internet for it.

<!-- notes: 15 minutes, after "Writing a Dockerfile". Demo with the Iris image
already pushed to Docker Hub as linux/amd64. Creating a Docker Space needs a
Hugging Face PRO account ($9/month, checked on the Hugging Face docs on
2026-10-19): do the demo from the teacher account, and do not require it in
the lab. The first build of a Space takes a few minutes: start it before the
explanation, show the result at the end. -->

---

## Hugging Face Spaces

A **Space** is a git repository on huggingface.co that Hugging Face builds and
runs for you. With the **Docker** SDK, the repository holds a `Dockerfile`:
every push rebuilds the image and restarts the container.

| | |
|---|---|
| Public URL | `https://<user>-<space>.hf.space` |
| Default hardware | CPU Basic: 2 vCPU, 16 GB RAM, no hourly cost |
| Account | Docker Spaces need a **PRO** account ($9/month); Static Spaces are free |
| When unused | the Space goes to sleep; the next request wakes it, slowly |
| Disk | lost at every restart: the model must be inside the image |

---

## Two files in the Space

The Space repository needs only a `README.md` and a `Dockerfile`:

```text
iris-api/            (the Space repository)
├── README.md        ← metadata: sdk and port
└── Dockerfile       ← one line: the image you already pushed
```

The `Dockerfile` starts from your Docker Hub image, unchanged:

```dockerfile
FROM <your-user>/iris-api:1.0
```

The image must be **public** on Docker Hub and built for **`linux/amd64`**,
the architecture of the Space's servers.

---

## The README header

The YAML block at the top of `README.md` configures the Space:

```yaml
---
title: Iris API
sdk: docker
app_port: 8000
---
```

- `sdk: docker` builds the `Dockerfile`.
- `app_port: 8000` is the port the container listens on. The default is
  `7860`: without this line, the Space waits on a port where nothing answers.

---

## Create and push

1. huggingface.co → **New Space**: name `iris-api`, SDK **Docker**, template
   **Blank**, hardware **CPU Basic**, visibility **Public**.
2. Profile → **Settings → Access Tokens**: a token with **write** access.
3. Clone the empty Space, add the two files, push:

```bash
git clone https://huggingface.co/spaces/<hf-user>/iris-api
cd iris-api        # write README.md and Dockerfile here
git add README.md Dockerfile
git commit -m "Serve the Iris image"
git push           # password: the write token
```

---

## Watch it start

The Space page shows **Building**, then **Running**. The **Logs** tab shows
the same lines as `docker logs`:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Then, from any machine:

```bash
curl https://<hf-user>-iris-api.hf.space/health
curl -X POST https://<hf-user>-iris-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.7, "sepal_width": 3.0,
       "petal_length": 5.2, "petal_width": 2.3}'
```

`https://<hf-user>-iris-api.hf.space/docs` gives the same page as on your
laptop.

---

## When the Space does not start

| Symptom | Cause | Fix |
|---|---|---|
| stays on *Starting*, then fails | `app_port` is not the port of `CMD` | `app_port: 8000` in `README.md` |
| `exec format error` in the logs | the image was built for `arm64` | `docker build --platform linux/amd64`, push again |
| build fails at `FROM`: *pull access denied* | the Docker Hub image is private, or misspelt | make it public; check `docker pull <your-user>/iris-api:1.0` |
| `Permission denied` when the app writes a file | the container runs as user `1000`, not root | write nothing at run time, or create the folder in the `Dockerfile` |
| the first request is slow | the Space was asleep | expected on free hardware |

---

## Public means public

A public Space is open to everyone: its files, its logs, and its API. Anyone
can call `/predict` as often as they like.

- No API key, password or private data in the repository or the image: a
  **secret** set in the Space's Settings arrives as an environment variable.
- **Private** visibility hides the app as well as the code; the URL then
  answers `404` to everyone but you.

---

## Check yourself

1. The logs say `Uvicorn running on http://0.0.0.0:8000` and the Space never
   answers. What is missing? **`app_port: 8000` in the `README.md` header: the
   Space waits on 7860 by default.**
2. Why does the Space's `Dockerfile` need only one line? **The image already
   holds the code, the model and the packages; the Space only runs it.**
3. Built on an M-chip Mac without `--platform linux/amd64`, the Space fails.
   Why? **The image is `arm64`; the Space's servers are `amd64`.**
4. The Space restarts. Is the model still there? **Yes: it is in the image.
   A file the app wrote at run time is not.**
