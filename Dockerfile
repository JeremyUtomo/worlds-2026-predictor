FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORLDS_DATA=/data

WORKDIR /app
COPY requirements.lock .
RUN pip install -r requirements.lock

COPY . .
RUN useradd --uid 1000 --create-home worlds \
    && mkdir -p /data && chown -R worlds:worlds /data /app
USER worlds

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# the web app; the updater service overrides this command (see docker-compose.yml)
CMD ["sh", "-c", "python bootstrap.py && exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --server.fileWatcherType=none --server.runOnSave=false --browser.gatherUsageStats=false"]
