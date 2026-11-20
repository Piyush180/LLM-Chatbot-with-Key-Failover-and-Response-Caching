# ---- build stage: install dependencies into a virtualenv
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- test stage (used by `make test`): runtime deps + pytest
FROM builder AS test
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
WORKDIR /app
COPY . .
CMD ["pytest", "-q"]

# ---- runtime stage: only the venv and the app code
FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////data/app.db \
    PORT=8000
RUN useradd --uid 10001 --create-home app \
 && mkdir -p /data && chown app:app /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY app ./app
USER app
VOLUME ["/data"]
EXPOSE 8000
# PORT is overridable (Render injects its own).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
