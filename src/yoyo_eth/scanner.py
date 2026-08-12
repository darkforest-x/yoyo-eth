"""Loose MA-compression scanner (doc section 9).

Cluster geometry over the six MAs (sma/ema 20/60/120):
    ma_upper          = max of the six
    ma_lower          = min of the six
    cluster_center    = median of the six
    ma_dispersion_atr = (ma_upper - ma_lower) / atr_14

Threshold discipline: the compression threshold is a fixed quantile of
ma_dispersion_atr computed on TRAIN bars only, then frozen for the whole run.

Candidate bar: ma_dispersion_atr < threshold for >= min_duration consecutive
bars (the bar where the streak first reaches min_duration, and every later bar
of the streak, is a raw candidate). Deduplication: candidate bars whose gap is
<= cooldown_bars are merged into one event whose decision bar is the FIRST
qualifying bar of the merged group. Raw and deduplicated counts are both kept.

Deliberately absent (doc): trend filters, volume filters, future conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MA_COLS = ("sma_20", "sma_60", "sma_120", "ema_20", "ema_60", "ema_120")


def add_dispersion(df: pd.DataFrame) -> pd.DataFrame:
    """Add ma_upper/ma_lower/cluster_center/ma_dispersion_atr columns."""
    out = df.copy()
    mas = out[list(MA_COLS)]
    out["ma_upper"] = mas.max(axis=1)
    out["ma_lower"] = mas.min(axis=1)
    out["cluster_center"] = mas.median(axis=1)
    out["ma_dispersion_atr"] = (out["ma_upper"] - out["ma_lower"]) / out["atr_14"]
    return out


def freeze_threshold(df: pd.DataFrame, train_end_pos: int, quantile: float) -> dict:
    """Quantile of ma_dispersion_atr over TRAIN bars only (positions < train_end_pos)."""
    train_disp = df["ma_dispersion_atr"].iloc[:train_end_pos].dropna()
    if len(train_disp) == 0:
        raise ValueError("no valid dispersion values in train interval")
    return {
        "threshold": float(train_disp.quantile(quantile)),
        "quantile": quantile,
        "n_train_bars_used": int(len(train_disp)),
        "train_end_pos": int(train_end_pos),
        "train_end_ts": str(df["timestamp"].iloc[train_end_pos - 1]),
    }


def below_streak(dispersion: pd.Series, threshold: float) -> pd.Series:
    """Consecutive count of bars (ending at t) with dispersion < threshold. Causal."""
    below = (dispersion < threshold).fillna(False).to_numpy()
    streak = np.zeros(len(below), dtype=np.int64)
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        streak[i] = run
    return pd.Series(streak, index=dispersion.index)


def scan(df: pd.DataFrame, threshold: float, min_duration: int, cooldown_bars: int) -> tuple[pd.DataFrame, dict]:
    """Return (events, stats). Events carry the decision bar position + timestamp."""
    streak = below_streak(df["ma_dispersion_atr"], threshold)
    raw_positions = np.flatnonzero((streak >= min_duration).to_numpy())

    events = []
    prev_pos = None
    for pos in raw_positions:
        if prev_pos is not None and pos - prev_pos <= cooldown_bars:
            prev_pos = pos  # same event region, extend
            continue
        events.append(pos)
        prev_pos = pos

    ev = pd.DataFrame(
        {
            "decision_pos": np.asarray(events, dtype=np.int64),
            "decision_ts": df["timestamp"].iloc[events].to_numpy() if events else [],
        }
    )
    stats = {
        "raw_candidate_count": int(len(raw_positions)),
        "dedup_event_count": int(len(ev)),
        "threshold": float(threshold),
        "min_duration": int(min_duration),
        "cooldown_bars": int(cooldown_bars),
    }
    return ev, stats
