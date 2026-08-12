"""Causal indicators: SMA/EMA 20/60/120, ATR14, TR rolling means.

Every value at bar t depends only on bars <= t (pandas rolling/ewm are
trailing). No centered rolling, no future fill (doc section 6).

Columns used / windows:
- sma_N:  close, trailing window N (NaN during warmup)
- ema_N:  close, ewm(span=N, adjust=False); masked NaN for the first N-1 bars
          so warmup semantics match SMA
- tr:     high, low, prev close (shift(1) -- strictly past)
- atr_14: Wilder smoothing of TR, ewm(alpha=1/14, adjust=False), masked for
          the first 14 TR values
- tr_mean_5 / tr_mean_20: simple trailing means of TR (for atr_ratio_5_20)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MA_PERIODS = (20, 60, 120)
ATR_PERIOD = 14


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with sma_*/ema_*/tr/atr_14/tr_mean_* columns."""
    out = df.copy()
    close = out["close"]

    for n in MA_PERIODS:
        out[f"sma_{n}"] = close.rolling(n, min_periods=n).mean()
        ema = close.ewm(span=n, adjust=False).mean()
        ema.iloc[: n - 1] = np.nan
        out[f"ema_{n}"] = ema

    prev_close = close.shift(1)
    tr = pd.concat(
        [out["high"] - out["low"], (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = np.nan  # no previous close for the first bar
    out["tr"] = tr

    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False, ignore_na=True).mean()
    atr.iloc[: ATR_PERIOD] = np.nan  # warmup: first TR is bar 1, need ATR_PERIOD TRs
    out["atr_14"] = atr

    out["tr_mean_5"] = tr.rolling(5, min_periods=5).mean()
    out["tr_mean_20"] = tr.rolling(20, min_periods=20).mean()
    return out
