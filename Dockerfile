FROM python:3.11-slim

WORKDIR /app

# Issue #25: Non-root user for security
RUN groupadd -r ems && useradd -r -g ems ems

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R ems:ems /app
USER ems

# Issue #25: Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "scripts/run_pipeline.py"]
