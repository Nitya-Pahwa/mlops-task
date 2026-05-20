# MLOps Batch Job — Rolling-Mean Signal Pipeline

A minimal, reproducible MLOps pipeline that computes a binary trading signal
from OHLCV data using a configurable rolling mean.

---

## Project structure

```
.
├── run.py            # Main pipeline script
├── config.yaml       # Job configuration
├── data.csv          # Input OHLCV dataset (10 000 rows)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
├── metrics.json      # Sample output (successful run)
└── run.log           # Sample log (successful run)
```

---

## Local run

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Execute

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

No hard-coded paths — all file locations are passed via CLI flags.

---

## Docker build & run

```bash
# Build
docker build -t mlops-task .

# Run (prints metrics JSON to stdout; exit 0 = success)
docker run --rm mlops-task
```

The container bundles `data.csv` and `config.yaml`, runs the pipeline, writes
`metrics.json` and `run.log` inside the container, and prints the final JSON
to stdout.

---

## Configuration (`config.yaml`)

| Key       | Type    | Description                          |
|-----------|---------|--------------------------------------|
| `seed`    | int     | NumPy random seed for reproducibility |
| `window`  | int     | Rolling-mean window size             |
| `version` | string  | Pipeline version label               |

---

## Processing logic

1. **Config validation** — checks required fields (`seed`, `window`, `version`).
2. **Dataset validation** — missing file / empty file / missing `close` column.
3. **Rolling mean** — `pandas.Series.rolling(window, min_periods=window).mean()`
   on the `close` column. The first `window-1` rows produce NaN and are
   excluded from signal computation.
4. **Signal** — `1` if `close > rolling_mean`, else `0` (NaN rows excluded).
5. **Metrics** — `rows_processed` counts only rows with a valid signal;
   `signal_rate` is the mean of those signals.

---

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.499,
  "latency_ms": 134,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Available columns: [...]"
}
```

---

## Exit codes

| Code | Meaning          |
|------|------------------|
| `0`  | Pipeline success |
| `1`  | Pipeline failure |

Metrics JSON is **always** written — even on failure.
