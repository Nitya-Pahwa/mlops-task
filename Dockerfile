# ── base ──────────────────────────────────────────────────────────────────────
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# ── dependencies ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── source + data ──────────────────────────────────────────────────────────────
COPY run.py       .
COPY config.yaml  .
COPY data.csv     .

# ── run ────────────────────────────────────────────────────────────────────────
# Prints metrics JSON to stdout; exit 0 = success, non-zero = failure.
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
