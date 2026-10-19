# Docker Compose: Several Containers, One File

Docker Compose writes the options of `docker run` in a file, `compose.yaml`, and
starts every container it describes with one command. Three steps take the
penguin API from one container to three replicas behind a load balancer.

<!-- notes: 20 minutes, optional after the Dockerfile step. Everyone works in
the penguin-api folder, next to the Dockerfile. Every output here was produced
on 2026-10-19 on an Apple M4 laptop, OrbStack, docker 29.8.2, compose v5.5.0,
nginx 1.31.7. The tested compose.yaml and nginx.conf are in the public reference
project, deploy/penguin-api/compose/ on GitHub. Container IPs differ from
one laptop to the next; only their rotation matters. -->

---

## Why a file

A `docker run` command grows with every option: ports, restart policy,
environment variables, volumes. A second container doubles it, and the
command lives only in the shell history.

[Compose](https://docs.docker.com/compose/) keeps the same options in
`compose.yaml`, next to the Dockerfile. Each container to run is a
**service**; `docker compose up` creates them all, `docker compose down`
removes them all.

---

## Step 1: one service

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
```

Equivalent to `docker run -d -p 8000:8000` on the image built from the
Dockerfile; `image: USER/penguin-api:v1` instead of `build: .` pulls it from
Docker Hub.

```bash
docker compose up -d       # build if needed, create, start
docker compose ps          # the services and their state
docker compose logs -f api # follow the output of one service
docker compose down        # stop and delete containers and network
```

```text
NAME                 IMAGE              SERVICE   STATUS         PORTS
compose-test-api-1   compose-test-api   api       Up 4 seconds   0.0.0.0:8000->8000/tcp
```

---

## Step 2: restart and health check

Added under `api:`. The image has no `curl`, so the check uses Python:

```yaml
    restart: unless-stopped
    healthcheck:
      test: >-
        python -c "import urllib.request as r;
        r.urlopen('http://localhost:8000/health')"
      interval: 10s
```

```text
STATUS
Up Less than a second (health: starting)
Up 13 seconds (healthy)
```

- The **health check** calls `/health` every 10 s inside the container. Three
  failures in a row mark it *unhealthy*.
- **`restart: unless-stopped`**: when the server process was killed inside the
  container, Docker restarted it within 2 s (restart count 0 → 1). Only
  `docker stop` or `docker compose down` keeps it down.

---

## Step 3: three replicas behind nginx

![nginx receives every request on port 8080 and passes each one to the next of three API containers](assets/deploy/compose-scaling.png)

```yaml
  api:
    build: .
    deploy:
      replicas: 3
  nginx:
    image: nginx
    ports: ["8080:80"]
    volumes: ["./nginx.conf:/etc/nginx/nginx.conf:ro"]
    depends_on: [api]
```

`api` has no `ports:`: only nginx is reachable from the host. The tested
files: [`penguin-api/compose/`](https://github.com/racousin/data_science_practice/tree/main/website/public/modules/ms2a-machine-learning-practice/deploy/penguin-api/compose).
`depends_on` starts the replicas before nginx.

Services on the same Compose network reach each other by **service name**:
Docker's DNS answers `api` with the addresses of all three replicas.

---

## The load balancer

```nginx
worker_processes 1;
events {}
http {
    server {
        listen 80;
        location / {
            proxy_pass http://api:8000;
            add_header X-Upstream $upstream_addr always;
        }
    }
}
```

`curl -i localhost:8080/health`, six times; `X-Upstream` names the replica:

```text
192.168.156.3:8000   192.168.156.2:8000   192.168.156.4:8000
192.168.156.3:8000   192.168.156.2:8000   192.168.156.4:8000
```

- **Round robin**: each request goes to the next replica.
- `worker_processes 1`: each nginx worker keeps its own rotation. With the
  default, one worker per CPU, two runs of six requests sent 6 and 4 of them
  to the same replica.
- nginx resolves `api` once, at start: after `--scale api=4`, the fourth
  replica received traffic only after `docker compose restart nginx`.

---

## Robustness and scale

With one replica stopped (`docker stop compose-test-api-2`), all six requests
still returned 200. One of them shows nginx trying the stopped replica, then
the next:

```text
X-Upstream: 192.168.156.3:8000, 192.168.156.2:8000
```

- **Robustness**: restart policies and health checks bring a failed container
  back; a load balancer routes around it meanwhile.
- **Horizontal scaling**: more identical containers from one image, rather
  than a bigger machine.
- [Kubernetes](https://kubernetes.io/docs/concepts/overview/) applies the same
  ideas (replicas, health probes, restarts, service names) across a cluster
  of machines.

---

## Check yourself

1. Why has the `api` service no `ports:` in Step 3? **Only nginx is reached
   from the host; it reaches the replicas on the Compose network as
   `api:8000`. Three replicas could not all bind host port 8000 anyway.**
2. A replica's health check fails three times in a row. What does
   `docker compose ps` show for it? **`(unhealthy)`.**
3. What does `docker compose down` remove? **The containers and the network
   of the project. The images stay on disk.**
