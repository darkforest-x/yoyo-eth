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


TRIGGERS = ("zone_start", "dispersion_exit", "price_breakout")


def _qualifying_positions(df: pd.DataFrame, threshold: float, min_duration: int, trigger: str):
    """Raw candidate bar positions + the confirmed zone length at each, per trigger.

    zone_start      : bar where the below-threshold streak first reaches
                      min_duration (and every later bar of the streak) --
                      fires at the START of a compression zone (MVP original).
    dispersion_exit : first bar back ABOVE the threshold after a zone of
                      >= min_duration bars -- fires when the zone ENDS
                      (MAs start fanning out).
    price_breakout  : bar whose close leaves the [ma_lower, ma_upper] band
                      while the previous bar sat in a confirmed zone -- fires
                      on the first price escape attempt.
    All three use only bars <= t. zone_length is the streak length already
    confirmed at decision time (for zone_start that is min_duration by
    construction; for the exit triggers, the full length of the ending zone).
    """
    streak = below_streak(df["ma_dispersion_atr"], threshold).to_numpy()
    close = df["close"].to_numpy()
    prev_streak = np.concatenate([[0], streak[:-1]])
    if trigger == "zone_start":
        mask = streak >= min_duration
        zone_len = streak
    elif trigger == "dispersion_exit":
        # explicit ">= threshold" so a NaN dispersion bar (warmup) neither
        # fires nor impersonates an upward exit (NaN comparisons are False)
        above = df["ma_dispersion_atr"].to_numpy() >= threshold
        mask = above & (prev_streak >= min_duration)
        zone_len = prev_streak
    elif trigger == "price_breakout":
        outside = (close > df["ma_upper"].to_numpy()) | (close < df["ma_lower"].to_numpy())
        mask = outside & (prev_streak >= min_duration)
        zone_len = prev_streak
    else:
        raise ValueError(f"unknown trigger {trigger!r}")
    pos = np.flatnonzero(mask)
    return pos, zone_len[pos]


def scan(
    df: pd.DataFrame,
    threshold: float,
    min_duration: int,
    cooldown_bars: int,
    trigger: str = "zone_start",
) -> tuple[pd.DataFrame, dict]:
    """Return (events, stats). Events carry decision position, timestamp and
    zone_length. Candidate bars closer than cooldown_bars merge into one event
    whose decision bar is the FIRST qualifying bar of the merged group."""
    raw_positions, raw_zone_len = _qualifying_positions(df, threshold, min_duration, trigger)

    events, zone_lens = [], []
    prev_pos = None
    for pos, zl in zip(raw_positions, raw_zone_len):
        if prev_pos is not None and pos - prev_pos <= cooldown_bars:
            prev_pos = pos  # same event region, extend
            continue
        events.append(int(pos))
        zone_lens.append(int(zl))
        prev_pos = pos

    ev = pd.DataFrame(
        {
            "decision_pos": np.asarray(events, dtype=np.int64),
            "decision_ts": df["timestamp"].iloc[events].to_numpy() if events else [],
            "zone_length": np.asarray(zone_lens, dtype=np.int64),
        }
    )
    stats = {
        "trigger": trigger,
        "raw_candidate_count": int(len(raw_positions)),
        "dedup_event_count": int(len(ev)),
        "threshold": float(threshold),
        "min_duration": int(min_duration),
        "cooldown_bars": int(cooldown_bars),
    }
    return ev, stats
