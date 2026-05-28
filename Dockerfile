FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; slim image + no build tools needed for wheels
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir -e .

# Train at build time so the image ships with a ready model artifact.
RUN python -m scripts.run train

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "predmaint.api:app", "--host", "0.0.0.0", "--port", "8000"]
