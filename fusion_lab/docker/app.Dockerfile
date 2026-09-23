# Fusion Lab app: Gradio + Rerun viewer + numpy. No torch, no CUDA, no dataset inside.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False FUSION_LAB_CACHE=/app/cache
WORKDIR /app
COPY requirements-app.txt .
# rerun_cli is the native desktop viewer (about 260 MB). The app only writes .rrd data
# and shows it in the web viewer bundled with gradio_rerun, so it is removed.
RUN pip install -r requirements-app.txt \
 && pip install --no-deps gradio_rerun==0.38.1 \
 && rm -rf /usr/local/lib/python3.11/site-packages/rerun_sdk/rerun_cli
COPY kflab ./kflab
COPY app ./app
EXPOSE 7860
CMD ["python", "-m", "app"]
