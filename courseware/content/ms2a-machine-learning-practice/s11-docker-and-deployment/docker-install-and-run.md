# Installing Docker and Running a Container

Install Docker, connect it to a Docker Hub account, then run two containers: one
that prints a message and exits, one web server that keeps running.

<!-- notes: 10 minutes, everyone typing. Install and the Docker Hub account
were asked for before the session; whoever arrives without them installs now
and pairs with a neighbour. Windows students without WSL 2 are the slowest case
(a reboot). Outputs on these slides were produced on 2026-10-19 on an Apple M4
laptop, OrbStack, docker 29.8.2. If port 8080 is taken on a laptop, any other
host port works: -p 8081:80. -->

---

## Install

| System | What to install |
|---|---|
| macOS | [OrbStack](https://orbstack.dev/download): free for personal use, education licences on request ([FAQ](https://docs.orbstack.dev/faq)) |
| Windows 10/11 | [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) with the WSL 2 backend |
| Linux | [Docker Engine](https://docs.docker.com/engine/install/) for your distribution |

- Containers are Linux processes. On macOS and Windows, OrbStack and Docker
  Desktop run a small Linux virtual machine in the background to host them.
- On Windows, the commands of this session are typed in the **Ubuntu (WSL)**
  terminal.
- On Linux, `sudo usermod -aG docker $USER`, then log out and back in, lets
  you run `docker` without `sudo`.

---

## A Docker Hub account

![The Repositories page of a Docker Hub account, with one public repository](assets/deploy/dockerhub-repositories.png)

Create an account at **hub.docker.com**. Your username is the namespace of
your images: `USER/penguin-api:v1`. Then connect the client to it:

```bash
docker login
```

Free accounts: unlimited public repositories, 1 private repository, 200 pulls
per 6 hours when logged in, 100 anonymously
([usage limits](https://docs.docker.com/docker-hub/usage/)).

---

## First container

```bash
docker run --rm hello-world
```

```text
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.
```

1. **Pull**: the image was not on disk; the daemon downloaded it from Docker Hub.
2. **Create**: a container was made from the image.
3. **Run**: its program printed the message.
4. **Exit**: the program ended, so the container stopped; `--rm` deleted it.

---

## A web server in a container

```bash
docker run -d -p 8080:80 --name web nginx
```

Open **http://localhost:8080**: the page *Welcome to nginx!*

| Flag | Meaning |
|---|---|
| `-d` | detached: runs in the background, the terminal is free |
| `-p 8080:80` | host port 8080 → container port 80 |
| `--name web` | a name instead of a random one |

---

## Look, read, stop

```bash
docker ps
docker logs web
docker stop web
```

```text
CONTAINER ID   IMAGE   STATUS         PORTS                  NAMES
f73d63c11c66   nginx   Up 2 seconds   0.0.0.0:8080->80/tcp   web
```

`docker logs web` ends with one line per request, here the page load:

```text
192.168.215.1 - - [19/Oct/2026:19:55:06 +0000] "GET / HTTP/1.1" 200 896
```

---

## Stopped is not deleted

```bash
docker ps -a
```

```text
CONTAINER ID   IMAGE   STATUS                              NAMES
f73d63c11c66   nginx   Exited (0) Less than a second ago   web
```

A stopped container still exists, with its logs, until `docker rm web`
deletes it. The image stays on disk too: `nginx` is 261 MB, listed by
`docker images`.

**Checkpoint**: `docker ps -a` shows the `web` container as *Exited*. Then
`docker rm web` frees the name for later.

---

## Check yourself

1. `docker run --rm hello-world` a second time prints no *Pulling* line. Why?
   **The image is already on disk; only the first run downloads it.**
2. `docker ps` shows nothing, but `docker ps -a` shows `web` as *Exited*. Can
   you read its logs? **Yes: `docker logs web` works on a stopped container.**
3. What happens to `localhost:8080` after `docker stop web`? **Nothing answers
   any more: the browser reports that the site cannot be reached.**
