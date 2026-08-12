"""OHLCV loading and sanity checks.

Source: fable-trading OKX kline cache CSV (columns: ts, open, high, low, close,
volume, open_time). Read-only; this project never writes into the parent repo.
The parent fetcher stores confirmed (closed) candles only; on top of that
assumption this module checks every bar sits on the exact bar_minutes grid.

Order of operations matters for holdout discipline: parse -> sort -> dedup ->
CUT AT data_end_boundary -> validity checks. Bars at/after the boundary are
dropped before any statistic is computed. Checks performed (doc section 6):
monotonic timestamps, duplicates, OHLC validity, missing values, explicit
timezone (UTC), grid alignment, gap census (gaps are reported, never filled).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DataCheckResult:
    n_rows_raw: int
    n_rows_final: int
    n_duplicates_dropped: int
    n_gaps: int
    gap_details: list = field(default_factory=list)
    start: str = ""
    end: str = ""
    passed: bool = False
    notes: list = field(default_factory=list)


def load_ohlcv(csv_path: str, bar_minutes: int, data_end_boundary: str) -> tuple[pd.DataFrame, DataCheckResult]:
    """Load kline CSV, run sanity checks, return (df, check_result).

    The returned frame is indexed 0..n-1 with a UTC ``timestamp`` column and
    float64 OHLCV, containing only bars strictly before ``data_end_boundary``.
    """
    raw = pd.read_csv(csv_path)
    result = DataCheckResult(n_rows_raw=len(raw), n_rows_final=0, n_duplicates_dropped=0, n_gaps=0)

    required = {"ts", "open", "high", "low", "close", "volume"}
    missing_cols = required - set(raw.columns)
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    df = raw[["ts", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    df = df.sort_values("ts", kind="mergesort")
    n_before = len(df)
    df = df.drop_duplicates(subset="ts", keep="first").reset_index(drop=True)
    result.n_duplicates_dropped = n_before - len(df)
    if result.n_duplicates_dropped:
        result.notes.append(f"dropped {result.n_duplicates_dropped} duplicate timestamps")

    boundary = pd.Timestamp(data_end_boundary)
    if boundary.tzinfo is None:
        boundary = boundary.tz_localize("UTC")
    n_before = len(df)
    df = df[df["timestamp"] < boundary].reset_index(drop=True)
    n_cut = n_before - len(df)
    if n_cut:
        result.notes.append(f"dropped {n_cut} bars at/after boundary {boundary} (holdout discipline)")
    if len(df) == 0:
        raise ValueError("no bars before data_end_boundary")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")

    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("NaN values in OHLCV")
    bad_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"]) | (df["high"] < df["close"])
        | (df["low"] > df["open"]) | (df["low"] > df["close"])
        | (df["low"] <= 0)
    )
    if bad_ohlc.any():
        raise ValueError(f"{int(bad_ohlc.sum())} bars violate OHLC bounds")
    if (df["volume"] < 0).any():
        raise ValueError("negative volume")

    step_ms = bar_minutes * 60_000
    if (df["ts"] % step_ms != df["ts"].iloc[0] % step_ms).any():
        raise ValueError(f"timestamps not aligned to the {bar_minutes}m grid (unfinished/partial bars?)")

    diffs = df["ts"].diff().dropna()
    gaps = diffs[diffs != step_ms]
    result.n_gaps = int(len(gaps))
    for idx in gaps.index[:20]:
        result.gap_details.append(
            {"after": str(df.loc[idx - 1, "timestamp"]), "before": str(df.loc[idx, "timestamp"]),
             "gap_bars": float(diffs.loc[idx] / step_ms)}
        )
    if result.n_gaps:
        result.notes.append(f"{result.n_gaps} non-uniform timestamp steps (gaps kept as-is, no future fill)")

    result.n_rows_final = len(df)
    result.start = str(df["timestamp"].iloc[0])
    result.end = str(df["timestamp"].iloc[-1])
    result.passed = True
    return df.reset_index(drop=True), result
