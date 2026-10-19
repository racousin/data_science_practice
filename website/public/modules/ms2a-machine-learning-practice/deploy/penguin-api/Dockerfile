FROM python:3.13-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-cache
COPY main.py model.joblib index.html ./
CMD ["uv", "run", "fastapi", "run", "main.py", "--port", "8000"]
