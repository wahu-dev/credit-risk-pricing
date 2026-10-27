# Credit risk scoring API.
#
# Build (after `python -m src.build_serving`, which prints this exact command):
#   docker build --build-arg PYTHON_VERSION=3.13 -t credit-risk-api .
# Run:
#   docker run -p 8000:8000 credit-risk-api
# Then open http://localhost:8000/docs

# Match the Python version the models were trained with (printed by build_serving).
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# LightGBM needs the OpenMP runtime, which slim images do not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Exact library versions that trained the models (written by build_serving).
# Installed first, so code changes do not invalidate this slow layer.
COPY models/serving_requirements.txt .
RUN pip install --no-cache-dir -r serving_requirements.txt

COPY src/ src/
COPY config/ config/
ARG BUNDLE=models/serving_bundle.joblib
COPY ${BUNDLE} models/serving_bundle.joblib

ENV MODEL_PATH=models/serving_bundle.joblib \
    SCORE_LOG=/app/logs/scores.jsonl \
    PYTHONUNBUFFERED=1

# Run as a non-root user.
RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
