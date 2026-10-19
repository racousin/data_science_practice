# Docker: Images, Containers, Registries

Docker runs a program together with everything it needs, isolated from the rest
of the machine. Four words carry the whole idea: **Dockerfile**, **image**,
**container**, **registry**.

<!-- notes: 15 minutes, no commands yet. The one thing that must land is image
versus container; everything later is a verb applied to one of them. The
class/object analogy works for this room, who all write Python. Keep the VM
comparison to one slide: they need to know why a container is light, not how a
hypervisor works. The versions in the first table are those of the workshop's
penguin API image, read inside it on 2026-10-19. -->

---

## "It works on my machine"

What the penguin API needs, besides its code:

| Layer | In the workshop image |
|---|---|
| Operating system | Debian 13 (your laptop: macOS, Windows, Ubuntu...) |
| Python | 3.13.16 |
| Packages | fastapi 0.149.0, scikit-learn 1.6.1, pandas 3.0.6, numpy 2.5.4, and 64 more |
| Files | `main.py`, `model.joblib`, `index.html` |
| Start command | `uv run fastapi run main.py --port 8000` |

Every line can differ on the next machine. A model saved with scikit-learn
1.6.1 is not guaranteed to load in another version. A README listing these
lines is a description; an image containing them is the environment itself.

---

## Virtual machine or container

![A VM carries a whole guest OS; containers share the host's kernel](assets/deploy/vm-vs-container.png)

- A **virtual machine** emulates a whole computer, with its own operating
  system: gigabytes, minutes to boot.
- A **container** is an ordinary process on the host, given its own private
  filesystem, network and process list. It shares the host's kernel: megabytes,
  a second to start.

---

## Client, daemon, registry

![The docker client sends commands to the daemon, which builds, stores and runs images and pulls them from a registry](assets/deploy/docker-architecture.png)

*Diagram: Docker Docs, Apache-2.0.*

- The **client** is the `docker` command you type.
- The **daemon** (Docker Engine) does the work: builds images, keeps them on
  disk, starts containers.
- A **registry** stores images by name, the way PyPI stores packages. The
  public one is **Docker Hub**: `python`, `nginx`, `postgres` all live there.

---

## The five words

- **Dockerfile**: the recipe. Start from this base image, copy these files,
  install these packages, run this command.
- **Image**: the read-only result of the recipe, a filesystem plus a start
  command. The penguin API image is 687 MB on disk.
- **Container**: a running instance of an image, with a thin writable layer on
  top. An image is to a container what a class is to an object.
- **Registry**: where images are pushed and pulled.
- **Tag**: the version part of an image name.

```text
python:3.13-slim          official image, tag 3.13-slim
nginx                     no tag: means nginx:latest
USER/penguin-api:v1       account USER on Docker Hub, image penguin-api, tag v1
```

---

## Six commands

![Build an image, push it to a registry, pull and run it anywhere](assets/deploy/build-push-pull-run.png)

| Command | Effect |
|---|---|
| `docker build -t USER/penguin-api:v1 .` | Dockerfile → image |
| `docker run USER/penguin-api:v1` | image → running container |
| `docker ps` | list the running containers |
| `docker logs <container>` | what the container printed |
| `docker stop <container>` | stop it |
| `docker push` / `docker pull` | image → registry / registry → image |

---

## Ports

![-p connects a port on your laptop to a port inside the container](assets/deploy/port-mapping.png)

A container has its own network. The API listens on port 8000 **inside** it;
nothing outside reaches it until `-p HOST:CONTAINER` connects a port of the
host to it.

```bash
docker run -p 8000:8000 USER/penguin-api:v1
```

`http://localhost:8000` on the laptop then reaches the API in the container.

---

## Check yourself

1. You run `docker run nginx` three times. How many images, how many
   containers? **One image, three containers.**
2. Which of Dockerfile, image, container is the running thing?
   **The container.** The Dockerfile is a recipe, the image a frozen package.
3. `docker run -p 9000:8000 USER/penguin-api:v1`: which URL reaches the API?
   **http://localhost:9000**, the host side of the mapping.
