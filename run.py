"""
MLOps Batch Job — rolling-mean signal pipeline
Usage:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")
    required = {"seed", "window", "version"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"Config 'seed' must be an integer, got: {type(cfg['seed'])}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got: {cfg['window']}")
    logger.info(f"Config validated — seed={cfg['seed']}, window={cfg['window']}, version={cfg['version']}")
    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _unwrap_quoted_csv(raw_text: str) -> str:
    """Strip outer quotes when every CSV row is wrapped in double-quotes."""
    lines = raw_text.splitlines()
    if not lines:
        return raw_text
    first = lines[0].strip()
    if first.startswith('"') and first.endswith('"') and ',' in first[1:-1]:
        unwrapped = []
        for line in lines:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            unwrapped.append(line)
        return "\n".join(unwrapped)
    return raw_text


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if path.stat().st_size == 0:
        raise ValueError("Input file is empty.")

    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = _unwrap_quoted_csv(raw)

    try:
        df = pd.read_csv(io.StringIO(raw), sep=",", engine="python", skipinitialspace=True)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Dataset contains no rows after parsing.")

    df.columns = [c.strip().strip('"').lower() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. "
            f"Available columns: {list(df.columns)}"
        )

    original_len = len(df)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    dropped = original_len - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with non-numeric 'close' values.")
    if df.empty:
        raise ValueError("No valid numeric 'close' values found in dataset.")

    logger.info(f"Dataset loaded — {len(df)} rows, columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Processing  (KEY FIX: signal uses 0/1 integers, not NaN for zeros)
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Computing rolling mean — window={window}")
    df = df.copy()

    # rolling mean — NaN for first (window-1) warm-up rows
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()

    valid_mask = df["rolling_mean"].notna()
    warmup_rows = int((~valid_mask).sum())
    logger.info(f"Warm-up rows excluded from signal: {warmup_rows}")
    logger.info(f"Rows used for signal computation: {int(valid_mask.sum())}")

    # Signal: 1 or 0 for valid rows; NaN only for warm-up rows
    df["signal"] = np.nan  # default NaN (warm-up rows stay NaN)
    df.loc[valid_mask, "signal"] = (
        df.loc[valid_mask, "close"] > df.loc[valid_mask, "rolling_mean"]
    ).astype(int)

    logger.info("Signal generation complete.")
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_metrics(df, version, seed, latency_ms):
    # Only rows with a computed signal (excludes warm-up NaNs)
    signal_series = df["signal"].dropna()
    rows_processed = int(len(signal_series))
    signal_rate = float(round(signal_series.mean(), 4))
    return {
        "version": version,
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": signal_rate,
        "latency_ms": int(latency_ms),
        "seed": seed,
        "status": "success",
    }

def write_metrics(metrics, output_path, logger):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics written to: {output_path}")

def write_error_metrics(version, error_msg, output_path, logger):
    with open(output_path, "w") as f:
        json.dump({"version": version, "status": "error",
                   "error_message": error_msg}, f, indent=2)
    logger.info(f"Metrics written to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps Batch Job")
    parser.add_argument("--input",    required=True)
    parser.add_argument("--config",   required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--log-file", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("MLOps Batch Job STARTED")
    logger.info(f"Input:   {args.input}")
    logger.info(f"Config:  {args.config}")
    logger.info(f"Output:  {args.output}")
    logger.info(f"Log:     {args.log_file}")
    logger.info("=" * 60)

    version = "unknown"
    try:
        logger.info(f"Loading config from: {args.config}")
        cfg = load_config(args.config, logger)
        version, seed, window = cfg["version"], cfg["seed"], cfg["window"]

        np.random.seed(seed)
        logger.info(f"Random seed set: {seed}")

        logger.info(f"Loading dataset from: {args.input}")
        df = load_dataset(args.input, logger)

        df = compute_signals(df, window, logger)

        latency_ms = (time.time() - start_time) * 1000
        metrics = build_metrics(df, version, seed, latency_ms)
        logger.info(
            f"Metrics summary — rows_processed={metrics['rows_processed']}, "
            f"signal_rate={metrics['value']}, latency_ms={metrics['latency_ms']}"
        )
        write_metrics(metrics, args.output, logger)
        logger.info("Job COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:
        logger.error(f"Pipeline error: {exc}")
        write_error_metrics(version, str(exc), args.output, logger)
        logger.info("Job FAILED")
        logger.info("=" * 60)
        print(json.dumps({"version": version, "status": "error",
                          "error_message": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())