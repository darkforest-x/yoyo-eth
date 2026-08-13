"""Semantic features (doc section 10), all strictly causal.

Every feature at bar t uses only columns of bars <= t:
- rolling(...) is pandas trailing rolling (window ends at t)
- slopes are (x_t - x_{t-k}) differences, ATR- or price-normalised
- linear_slope_60 is an OLS slope over closes of bars [t-59, t]

Feature groups / columns / windows (see each helper's docstring):
  compression : ma_dispersion_atr (+mean5/min10/slope5), compression_duration
  ma slope    : ema20/sma20 slope5, ema60 slope10, sma120 slope20, gap
  price vs MA cluster : close_to_cluster_atr, above/below/inside ratios (10)
  reclaim/rejection   : penetration count, failed reclaims, max streak above (10)
  trend background    : return_20/60, linear_slope_60, distance_from_high_60_atr,
                        price_location_in_range_60
  bar & volatility    : body_ratio, range_atr, atr_ratio_5_20, volume_zscore_20,
                        distance_to_low_20_atr

Features are computed for the whole frame then sampled at decision positions,
which is equivalent to per-event computation (verified by the Future Mutation
Test in tests/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scanner import below_streak

# Longest finite look-back of any feature, in bars: sma120_slope_20 reads
# sma_120.shift(20) -> bars t-139..t. (EMA tails are recursive and technically
# unbounded, as is compression_duration's streak; both decay/reset in practice
# and are declared in the report instead of embargoed away.)
FEATURE_MAX_LOOKBACK = 139

FEATURE_COLUMNS = [
    # 10.1 compression
    "ma_dispersion_atr",
    "ma_dispersion_mean_5",
    "ma_dispersion_min_10",
    "ma_dispersion_slope_5",
    "compression_duration",
    # 10.2 MA direction
    "ema20_slope_5",
    "sma20_slope_5",
    "ema60_slope_10",
    "sma120_slope_20",
    "short_long_slope_gap",
    # 10.3 price vs cluster
    "close_to_cluster_atr",
    "close_above_cluster_ratio_10",
    "close_below_cluster_ratio_10",
    "close_inside_cluster_ratio_10",
    # 10.4 reclaim / rejection
    "cluster_penetration_count_10",
    "failed_reclaim_count_10",
    "max_consecutive_close_above_10",
    # 10.5 trend background
    "return_20",
    "return_60",
    "linear_slope_60",
    "distance_from_high_60_atr",
    "price_location_in_range_60",
    # 10.6 bar & volatility
    "body_ratio",
    "range_atr",
    "atr_ratio_5_20",
    "volume_zscore_20",
    "distance_to_low_20_atr",
]


def _rolling_ols_slope(y: pd.Series, window: int) -> pd.Series:
    """Trailing OLS slope of y against bar index over the last `window` bars."""
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def slope(arr: np.ndarray) -> float:
        return float(((x - x_mean) * (arr - arr.mean())).sum() / x_var)

    return y.rolling(window, min_periods=window).apply(slope, raw=True)


def add_features(
    df: pd.DataFrame,
    compression_threshold: float | None,
) -> pd.DataFrame:
    """Add all FEATURE_COLUMNS to a frame that already has indicators + dispersion.

    compression_threshold=None skips the bar-level compression_duration column;
    the walk-forward pipeline substitutes the event-level zone_length instead
    (same semantics -- "compression length confirmed at decision time" -- and
    keeps the rest of the feature matrix threshold-independent so it can be
    computed once per dataset instead of once per fold).
    """
    out = df.copy()
    atr = out["atr_14"]
    close = out["close"]
    disp = out["ma_dispersion_atr"]

    # --- 10.1 compression ---------------------------------------------------
    out["ma_dispersion_mean_5"] = disp.rolling(5, min_periods=5).mean()
    out["ma_dispersion_min_10"] = disp.rolling(10, min_periods=10).min()
    out["ma_dispersion_slope_5"] = disp - disp.shift(5)
    if compression_threshold is not None:
        out["compression_duration"] = below_streak(disp, compression_threshold).astype("float64")

    # --- 10.2 MA direction (ATR-normalised slopes) ---------------------------
    out["ema20_slope_5"] = (out["ema_20"] - out["ema_20"].shift(5)) / atr
    out["sma20_slope_5"] = (out["sma_20"] - out["sma_20"].shift(5)) / atr
    out["ema60_slope_10"] = (out["ema_60"] - out["ema_60"].shift(10)) / atr
    out["sma120_slope_20"] = (out["sma_120"] - out["sma_120"].shift(20)) / atr
    out["short_long_slope_gap"] = out["ema20_slope_5"] - out["sma120_slope_20"]

    # --- 10.3 price vs cluster ------------------------------------------------
    out["close_to_cluster_atr"] = (close - out["cluster_center"]) / atr
    above = (close > out["cluster_center"]).astype("float64")
    below = (close < out["cluster_center"]).astype("float64")
    inside = ((close >= out["ma_lower"]) & (close <= out["ma_upper"])).astype("float64")
    out["close_above_cluster_ratio_10"] = above.rolling(10, min_periods=10).mean()
    out["close_below_cluster_ratio_10"] = below.rolling(10, min_periods=10).mean()
    out["close_inside_cluster_ratio_10"] = inside.rolling(10, min_periods=10).mean()

    # --- 10.4 simple reclaim / rejection --------------------------------------
    # penetration: close crosses cluster_center relative to the previous bar
    side = np.sign(close - out["cluster_center"])
    crossed = (side * side.shift(1) < 0).astype("float64")
    out["cluster_penetration_count_10"] = crossed.rolling(10, min_periods=10).sum()
    # failed reclaim: the bar's high enters/exceeds the cluster but the close
    # ends back below cluster_center (simple single-bar version, doc 10.4)
    failed = ((out["high"] >= out["ma_lower"]) & (close < out["cluster_center"])).astype("float64")
    out["failed_reclaim_count_10"] = failed.rolling(10, min_periods=10).sum()
    # longest run of closes above cluster_center WITHIN the last 10 bars:
    # streak[j] counts back past the window start, so cap it at the number of
    # bars between window start and j before taking the max
    above_streak = below_streak(-close, -out["cluster_center"])  # streak of close > center

    def _max_in_window(arr: np.ndarray) -> float:
        return float(np.minimum(arr, np.arange(1, len(arr) + 1)).max())

    out["max_consecutive_close_above_10"] = above_streak.rolling(10, min_periods=10).apply(
        _max_in_window, raw=True
    )

    # --- 10.5 trend background --------------------------------------------------
    out["return_20"] = close / close.shift(20) - 1.0
    out["return_60"] = close / close.shift(60) - 1.0
    out["linear_slope_60"] = _rolling_ols_slope(close, 60) / atr
    high_60 = out["high"].rolling(60, min_periods=60).max()
    low_60 = out["low"].rolling(60, min_periods=60).min()
    out["distance_from_high_60_atr"] = (high_60 - close) / atr
    rng = (high_60 - low_60).replace(0.0, np.nan)
    out["price_location_in_range_60"] = (close - low_60) / rng

    # --- 10.6 bar & volatility ----------------------------------------------------
    bar_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    out["body_ratio"] = (close - out["open"]).abs() / bar_range
    out["range_atr"] = (out["high"] - out["low"]) / atr
    out["atr_ratio_5_20"] = out["tr_mean_5"] / out["tr_mean_20"]
    vol = out["volume"]
    vol_mean = vol.rolling(20, min_periods=20).mean()
    vol_std = vol.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    out["volume_zscore_20"] = (vol - vol_mean) / vol_std
    low_20 = out["low"].rolling(20, min_periods=20).min()
    out["distance_to_low_20_atr"] = (close - low_20) / atr

    return out
