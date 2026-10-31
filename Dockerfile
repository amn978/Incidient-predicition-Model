FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir ".[api]"
# model + CMDB are produced by `isp benchmark`; mount or bake them in at build time
COPY models ./models
COPY data/cmdb.csv ./data/cmdb.csv
ENV ISP_CONFIG=/app/config/config.yaml
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "incident_severity.api:app", "--host", "0.0.0.0", "--port", "8000"]
