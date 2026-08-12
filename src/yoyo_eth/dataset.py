"""Chronological split and dataset assembly (doc section 12).

Random splits are forbidden. Bars [0, n) are cut at fixed fractions:
    train bars: [0, train_end)
    val bars:   [train_end, val_end)
    test bars:  [val_end, n)

Events are assigned by their DECISION bar position. Between consecutive splits
an embargo gap of max(context_window, horizon_bars) bars is carved out of the
START of val and test: an event whose decision bar falls inside a gap is
dropped. This guarantees (a) a train event's label horizon (which extends
horizon_bars past its decision bar) never overlaps any val/test decision bar,
and (b) a val/test event's context window is at least partially outside the
previous split's label horizon. One event never appears in two splits.

The scanner threshold is frozen from TRAIN BARS ONLY before any event exists;
this module only consumes it.
"""
from __future__ import annotations

import pandas as pd


def split_positions(n_bars: int, train_frac: float, val_frac: float) -> dict:
    train_end = int(n_bars * train_frac)
    val_end = int(n_bars * (train_frac + val_frac))
    if not (0 < train_end < val_end < n_bars):
        raise ValueError("degenerate split")
    return {"train_end": train_end, "val_end": val_end, "n_bars": n_bars}


def assign_split(
    events: pd.DataFrame,
    boundaries: dict,
    gap_bars: int,
) -> pd.DataFrame:
    """Add a `split` column: train / val / test / dropped_gap."""
    out = events.copy()
    te, ve = boundaries["train_end"], boundaries["val_end"]

    def which(pos: int) -> str:
        if pos < te:
            return "train"
        if pos < te + gap_bars:
            return "dropped_gap"
        if pos < ve:
            return "val"
        if pos < ve + gap_bars:
            return "dropped_gap"
        return "test"

    out["split"] = out["decision_pos"].map(which)
    return out


def build_dataset(
    events_labeled: pd.DataFrame,
    df_features: pd.DataFrame,
    feature_columns: list,
) -> pd.DataFrame:
    """Join per-bar features onto events at their decision positions.

    Events with any NaN feature (indicator warmup) are dropped and counted in
    ``attrs['n_dropped_nan_features']``.
    """
    feats = df_features.loc[events_labeled["decision_pos"], feature_columns].reset_index(drop=True)
    ds = pd.concat([events_labeled.reset_index(drop=True), feats], axis=1)
    n_before = len(ds)
    ds = ds.dropna(subset=feature_columns).reset_index(drop=True)
    ds.attrs["n_dropped_nan_features"] = n_before - len(ds)
    return ds
