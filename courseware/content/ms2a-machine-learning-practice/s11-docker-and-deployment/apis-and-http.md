# APIs and HTTP

In Session 1 you *called* APIs to collect data. To serve a model you write the
other side: a program that waits for requests and answers them. This lesson is
the vocabulary both sides share.

<!-- notes: 12 minutes. Draw the client/server picture on the board and keep
pointing at it. Do the three clients live against the class's own nginx or,
better, against the Iris API if it is already running from the demo; curl is
new to most of the room. The 422 slide sets up FastAPI's validation in the
next lesson. -->

---

## Client and server

![A client sends a request, the server runs the model and sends a response](assets/deploy/client-server.png)

- The **server** is a program that runs all the time and waits.
- A **client** sends it a **request**; the server computes and sends back a
  **response**.
- They talk **HTTP**, the protocol of the web: your browser is a client, and
  every page it shows came from a server.

---

## An address

![Each part of a URL chooses something](assets/deploy/url-anatomy.png)

- `localhost` is **this machine**. On a server it becomes a name like
  `api.example.com`.
- A **port** is a number that picks one program among all those listening on
  the machine: 8000 for our API, 8080 for CVAT, 5432 for PostgreSQL.
- The **path** picks the **endpoint**: one API offers several, `/health` and
  `/predict` here.

---

## Methods

| Method | Meaning | Our API |
|---|---|---|
| `GET` | read something, send no data | `GET /health`: are you up? |
| `POST` | send data for the server to process | `POST /predict`: here is a flower |

A browser's address bar only sends `GET`. Typing `localhost:8000/predict`
there answers `405 Method Not Allowed`: the endpoint exists, but only for
`POST`.

---

## JSON: the data in the request and the response

A `POST` request carries a **body**. Model APIs almost always use **JSON**,
which reads like a Python dict:

```json
{"sepal_length": 6.7, "sepal_width": 3.0,
 "petal_length": 5.2, "petal_width": 2.3}
```

The response is JSON too:

```json
{"species": "virginica"}
```

The request header `Content-Type: application/json` tells the server how to
read the body.

---

## Status codes: whose fault is it

| Code | Meaning | Who fixes it |
|---|---|---|
| `200 OK` | here is your answer | nobody |
| `404 Not Found` | no such path | the client: check the URL |
| `405 Method Not Allowed` | the path exists, not for this method | the client: `GET` or `POST`? |
| `422 Unprocessable Content` | the body does not match what the API expects | the client: check the fields |
| `500 Internal Server Error` | the server's code crashed | the server: read its logs |

4xx: the request is wrong. 5xx: the server is wrong.

---

## Three ways to call an API

**A browser**, for `GET`: open `http://localhost:8000/health`.

**curl**, in a terminal:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.7, "sepal_width": 3.0,
       "petal_length": 5.2, "petal_width": 2.3}'
```

```text
{"species":"virginica"}
```

**Python**, with `requests` (Session 1):

```python
import requests
flower = {"sepal_length": 5.1, "sepal_width": 3.5,
          "petal_length": 1.4, "petal_width": 0.2}
r = requests.post("http://localhost:8000/predict",
                  json=flower, timeout=5)
print(r.status_code, r.json())   # 200 {'species': 'setosa'}
```

---

## An endpoint is a function

| A Python function | An API endpoint |
|---|---|
| `predict(flower)` | `POST /predict` with a JSON body |
| arguments | the request body |
| return value | the response body |
| called with a wrong argument | answers `422` |
| raises an exception | answers `500` |

That is all a model API is: a `predict` function that other programs can call
over the network. The next lesson writes one in about twenty lines.

---

## Check yourself

1. In `http://localhost:8080/tasks`, which part picks the program, and which
   picks the page? **The port `8080` picks the program; the path `/tasks`, the
   page.**
2. You `POST` a body that forgets `petal_width`. Which status do you expect?
   **`422`: the body does not match what the endpoint expects.**
3. The server logs show a Python traceback and the client got a status. Which
   one? **`500`: the server's code crashed.**
4. Why does opening `localhost:8000/predict` in the browser fail when curl
   works? **The address bar sends `GET`; the endpoint only accepts `POST`.**
