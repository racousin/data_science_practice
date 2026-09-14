# Installing and Running Docker

Install Docker, check it works, then learn the dozen commands that cover
almost everything you will do with it: run, ports, logs, stop, remove, and
volumes.

<!-- notes: 15 minutes, everyone typing. Install should have been done before
the session; whoever arrives without it installs now and pairs with a
neighbour for the commands. Windows students without WSL 2 are the slowest
case (a reboot); send them to the WSL step first. All outputs on these slides
were produced on 2026-09-14 (docker 29.4.0). End with `docker rm -f web`: the
workshop that follows needs port 8080, and a forgotten nginx on it is the most
common failure there. -->

---

## Install

| System | What to install |
|---|---|
| macOS | **Docker Desktop** (pick Apple silicon or Intel), from docs.docker.com/desktop |
| Windows 10/11 | **WSL 2** first (`wsl --install` in an admin PowerShell, reboot), then **Docker Desktop** |
| Linux | **Docker Engine** for your distribution, from docs.docker.com/engine/install |

- Docker Desktop is free for students and personal use.
- On macOS and Windows, Docker Desktop runs a small Linux virtual machine
  in the background: containers are Linux processes, so they need a Linux
  kernel.
- On Windows, type every command of this session in the **Ubuntu (WSL)**
  terminal that `wsl --install` created: they then work exactly as written.
- On Linux, add yourself to the `docker` group, then log out and back in:
  `sudo usermod -aG docker $USER`.

---

## Check it works

```bash
docker version
docker run hello-world
```

```text
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
...
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

It did four things: looked for the image locally, pulled it from Docker Hub,
created a container, and ran it until it exited.

`Cannot connect to the Docker daemon`: Docker Desktop is not running. Start
it, wait for it to say *running*, and retry.

---

## A Python you did not install

```bash
docker run --rm python:3.13-slim python -c "import sys; print(sys.version)"
```

```text
3.13.14 (main, Jun 11 2026, 01:12:13) [GCC 14.2.0]
```

- `python:3.13-slim` is the image; everything after it is the command to run
  inside the container.
- `--rm` deletes the container when it exits.
- Your laptop's own Python, whatever version it is, played no part.

---

## A server in a container

```bash
docker run -d -p 8080:80 --name web nginx
```

Open **http://localhost:8080**: *Welcome to nginx!*

| Flag | Meaning |
|---|---|
| `-d` | detached: run in the background, give the terminal back |
| `-p 8080:80` | publish a port, `HOST:CONTAINER` |
| `--name web` | a name to use instead of the random id |

---

## Publishing a port

![-p connects a port on your laptop to a port inside the container](assets/deploy/port-mapping.png)

A container has its own network. nginx listens on port 80 **inside** it, and
nothing outside can reach that until `-p` connects a port of your laptop to it.

- No `-p`: the browser says *This site can't be reached*.
- Host port already taken: `Bind for :::8080 failed: port is already
  allocated`. Pick another host port, `-p 8081:80`, or stop what holds it.

---

## Look, stop, remove

```bash
docker ps              # running containers
docker logs web        # what the container printed
docker stop web        # stop it (it still exists)
docker ps -a           # all containers, stopped ones too
docker rm web          # delete it
```

```text
CONTAINER ID   IMAGE   STATUS         PORTS                   NAMES
b14952609bfe   nginx   Up 2 seconds   0.0.0.0:8080->80/tcp    web
```

`docker rm -f web` stops and deletes in one step.

---

## Inside a running container

```bash
docker exec web ls /usr/share/nginx/html
docker exec -it web bash
```

- `docker exec` runs a command in a container that is already running.
- `-it` gives you an interactive terminal: you get a shell inside it. `exit`
  leaves the shell; the container keeps running.

Useful for looking around. Anything you change in there is lost when the
container is removed.

---

## Containers are disposable, volumes are not

A container's files disappear with it. Data that must outlive the container
lives in a **volume**: a folder on the host mounted inside the container.

```bash
docker rm -f web
mkdir site && echo '<h1>Served from my laptop</h1>' > site/index.html
docker run -d -p 8080:80 --name web \
  -v "$PWD/site":/usr/share/nginx/html:ro nginx
```

Refresh the page: nginx serves your folder. Edit `index.html` and refresh
again: the container sees the change at once. `:ro` makes it read-only inside.

A host folder mounted this way is a *bind mount*. Docker can also keep a
**named volume** for you, a folder it manages and that survives the
container: that is where the database of the next workshop keeps its data.

---

## Housekeeping

```bash
docker images           # images on disk, with their size
docker rmi nginx        # delete an image (no container may use it)
docker system df        # disk used by images, containers, volumes
docker system prune     # delete stopped containers and unused data
```

Images add up: `nginx` is 270 MB, a Python image with scikit-learn about
600 MB. Check `docker system df` when your disk fills up.

Before the workshop, free port 8080: `docker rm -f web`.

---

## Check yourself

1. You run `docker run -p 9000:80 nginx`. Which address do you open in the
   browser? **http://localhost:9000**, the host side of the mapping.
2. `docker ps` shows nothing, but `docker ps -a` shows `web` as *Exited*. Can
   you see its logs? **Yes: `docker logs web` works on a stopped container.**
3. You wrote a file inside a container with `docker exec`, then ran
   `docker rm -f` on it. Where is the file? **Gone. Use a volume for anything
   you want to keep.**
4. What does `--rm` save you from? **A stopped container left behind after
   every run, to clean up by hand.**
