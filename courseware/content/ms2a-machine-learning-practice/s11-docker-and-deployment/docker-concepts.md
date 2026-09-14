# Docker: Images, Containers, Registries

Docker runs a program together with everything it needs, isolated from the rest
of the machine. Four words carry the whole idea: **Dockerfile**, **image**,
**container**, **registry**.

<!-- notes: 12 minutes, no commands yet. The one thing that must land is image
versus container; everything later is a verb applied to one of them. The
class/object analogy works for this room, who all write Python. Keep the VM
comparison to one slide: they need to know why a container is light, not how a
hypervisor works. -->

---

## "It works on my machine"

What your model API needs, besides your code:

| Layer | On your laptop |
|---|---|
| Operating system | macOS 15, Windows 11, Ubuntu 24.04... |
| Python | 3.13.14 |
| Packages | fastapi 0.141.1, scikit-learn 1.9.1, numpy 2.5.3, and 17 more |
| Files | `main.py`, `model.joblib` |
| Start command | `uvicorn main:app --host 0.0.0.0 --port 8000` |

Every line can differ on the next machine. A README listing them is a hope;
an image containing them is a guarantee.

---

## Virtual machine or container

![A VM carries a whole guest OS; containers share the host's kernel](assets/deploy/vm-vs-container.png)

- A **virtual machine** emulates a whole computer, with its own operating
  system: gigabytes, minutes to boot.
- A **container** is an ordinary process on the host, given its own private
  filesystem, network and process list. It shares the host's kernel: megabytes,
  a second to start.

---

## Image and container

- An **image** is a read-only package: a filesystem (Debian, Python, your
  packages, your files) plus the command to run. It never changes.
- A **container** is a running instance of an image, with a thin writable layer
  on top. Stop it, delete it, start ten more from the same image.

| Python | Docker |
|---|---|
| a class | an image |
| an object, `Model()` | a container |
| `class Model: ...` in a `.py` file | a **Dockerfile** |

---

## Where images come from

![Build an image, push it to a registry, pull and run it anywhere](assets/deploy/build-push-pull-run.png)

- A **Dockerfile** is the recipe: start from this base image, copy these
  files, install these packages, run this command.
- `docker build` turns it into an **image**.
- A **registry** stores images by name, the way PyPI stores packages. The
  public one is **Docker Hub**: `python`, `nginx`, `postgres` all live there.
- `docker pull` downloads an image; `docker run` starts a container from it.

---

## Image names

```text
python:3.13-slim            official image, tag 3.13-slim
nginx                       no tag: means nginx:latest
cvat/server:v2.75.0         account cvat, image server, tag v2.75.0
racousin90/iris-api:1.0     an account on Docker Hub, your image
```

`name:tag`. The **tag** is a version label. `latest` is only a default name,
not a promise of anything: pin a real tag whenever the result matters.

---

## Where you meet containers

- **ML-Arena**: every submitted agent runs in a container built from a runtime
  image, with its code mounted in.
- **Cloud platforms**: Google Cloud Run, AWS ECS, Azure Container Apps run an
  image you give them; Kubernetes runs thousands.
- **Tools you install**: databases, annotation platforms, notebooks, MLflow;
  most ship as an image first, and some only as an image.
- **Reproducible research**: an image next to a paper is the environment,
  frozen.

---

## Check yourself

1. You run `docker run nginx` three times. How many images, how many
   containers? **One image, three containers.**
2. Which of Dockerfile, image, container is the running thing?
   **The container.** The Dockerfile is a recipe, the image a frozen package.
3. Why does a container start in about a second when a virtual machine takes
   minutes? **It does not boot an operating system: it is a process sharing the
   host's kernel.**
4. What does `nginx` with no tag pull? **`nginx:latest`, whatever that tag
   points at today.**
