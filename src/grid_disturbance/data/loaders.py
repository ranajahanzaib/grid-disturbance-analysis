"""File loaders and format handlers for synchrophasor and grid time series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_signal_from_file(
    file_path: str | Path,
    channel_col: str | None = None,
    time_col: str | None = None,
    sampling_rate: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    """Load signal array and time vector from CSV, Parquet, or NPY file.

    Returns:
        (time_array, signal_data, estimated_sampling_rate, channel_names)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".npy", ".npz"]:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            # If multiple arrays, take the first or named 'data'
            key = "data" if "data" in data else list(data.keys())[0]
            arr = data[key]
        else:
            arr = data

        arr = np.asarray(arr, dtype=np.float64)
        n = arr.shape[0] if arr.ndim == 1 else arr.shape[-1]
        t = np.arange(n) / sampling_rate
        channels = (
            ["channel_1"] if arr.ndim == 1 else [f"channel_{i + 1}" for i in range(arr.shape[0])]
        )
        return t, arr, sampling_rate, channels

    if suffix in [".csv", ".tsv", ".txt"]:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    elif suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Inspect columns
    cols = df.columns.tolist()

    # Detect time column
    inferred_time_col = time_col
    if inferred_time_col is None:
        for c in cols:
            if c.lower() in ["time", "timestamp", "t", "seconds", "sec"]:
                inferred_time_col = c
                break

    if inferred_time_col and inferred_time_col in df.columns:
        time_series = pd.to_numeric(df[inferred_time_col], errors="coerce").to_numpy()
        # Compute dt and sampling rate
        dt = np.diff(time_series)
        valid_dt = dt[dt > 0]
        if len(valid_dt) > 0:
            median_dt = float(np.median(valid_dt))
            if median_dt > 0:
                sampling_rate = round(1.0 / median_dt, 2)
        t = time_series
        numeric_df = df.drop(columns=[inferred_time_col]).select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])
        n = len(numeric_df)
        t = np.arange(n) / sampling_rate

    if numeric_df.empty:
        raise ValueError(f"No numeric channels found in {path}")

    if channel_col and channel_col in numeric_df.columns:
        arr = numeric_df[channel_col].to_numpy(dtype=np.float64)
        channels = [channel_col]
    elif len(numeric_df.columns) == 1:
        col_name = numeric_df.columns[0]
        arr = numeric_df[col_name].to_numpy(dtype=np.float64)
        channels = [col_name]
    else:
        # Multichannel array
        channels = numeric_df.columns.tolist()
        arr = numeric_df.to_numpy(dtype=np.float64).T  # (channels, samples)

    return t, arr, sampling_rate, channels
