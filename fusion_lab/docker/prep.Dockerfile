# Fusion Lab data preparation. Runs once, writes ./cache on the host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/mpl
WORKDIR /app
COPY requirements-prep.txt .
RUN pip install -r requirements-prep.txt
COPY prep ./prep
COPY scenes.yaml .
ENTRYPOINT ["python", "-m", "prep"]
