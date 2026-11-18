# Scoring image: the FastAPI service (default command) and the weekly batch job.
# The same image runs locally, in CI and on AWS Lambda. On Lambda, the Lambda Web Adapter
# extension passes requests to uvicorn, so the app needs no Lambda-specific code.
# Every base image is pinned by digest, so a rebuild cannot pick up a changed image unnoticed.
# Build after scripts/release_model.py has written the model bundle to build/model/.

FROM ghcr.io/astral-sh/uv:0.11.5@sha256:555ac94f9a22e656fc5f2ce5dfee13b04e94d099e46bb8dd3a73ec7263f2e484 AS uv
FROM public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0@sha256:17cfd08eff1dfea3f6a9a1e9c65fdac80aa4919b6085e746615530f43f57d2f1 AS lambda-adapter

# Stage 1: install the locked runtime dependencies into /app/.venv.
FROM python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Runtime dependencies only: no dev tools and no train group (MLflow, SHAP).
RUN uv sync --locked --no-default-groups --no-install-project

# Stage 2: the runtime image. No uv, no compiler, no package manager cache.
FROM python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
COPY --from=lambda-adapter /lambda-adapter /opt/extensions/lambda-adapter
# Debian security fixes released after the base image was built (the CI image scan fails on
# any fixable HIGH or CRITICAL finding).
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src/ src/
COPY sql/ sql/
COPY build/model/ model/
RUN /app/.venv/bin/python -m compileall -q src
# Files stay owned by root, so the app user can read them but not change them.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model \
    AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"]
CMD ["uvicorn", "early_warning.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
