"""Future outcome labels for SHORT direction (doc section 11).

Only labels may look forward. For a decision bar at position t:
    entry_price = open of bar t+1 (next bar open)
    ATR_at_entry = atr_14 of the DECISION bar t (known at decision time)
    horizon = bars t+1 .. t+horizon_bars inclusive (highs/lows of the entry
              bar itself count: price can run against a short intra-bar)

    MFE_short = (entry_price - min(low over horizon)) / ATR_at_entry
    MAE_short = (max(high over horizon) - entry_price) / ATR_at_entry
    short_utility = MFE_short - mae_penalty * MAE_short

    future_close_return_short = (entry_price - close[t+horizon_bars]) / entry_price
    gross_return = future_close_return_short           (exit at horizon close)
    net_return   = gross_return - round_trip_cost      (cost per costs.py route)

Events whose horizon extends past the end of data are dropped (no partial
horizons -- a truncated MFE/MAE would bias labels near the data edge).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COLUMNS = [
    "entry_price",
    "atr_at_entry",
    "mfe_short",
    "mae_short",
    "short_utility",
    "future_close_return_short",
    "gross_return",
    "net_return",
]


def horizon_bars_from_hours(horizon_hours: float, bar_minutes: int) -> int:
    hb = int(round(horizon_hours * 60 / bar_minutes))
    if hb <= 0:
        raise ValueError("horizon must be at least one bar")
    return hb


def add_labels(
    events: pd.DataFrame,
    df: pd.DataFrame,
    horizon_bars: int,
    mae_penalty: float,
    round_trip_cost: float,
) -> pd.DataFrame:
    """Return events + LABEL_COLUMNS, dropping rows with incomplete horizons.

    Every input row is labeled independently through a stable row-level key
    (event_row_id), merged back validate="one_to_one". A decision_pos that
    appears k times yields exactly k output rows -- never k*k. (iteration_v1
    Stage A repair: the old merge on decision_pos fanned out duplicated
    positions, silently inflating bootstrap-style control samples.)
    Invariant: len(output) <= len(input), equal when every row has a complete
    horizon and a valid ATR.
    """
    n = len(df)
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    atr = df["atr_14"].to_numpy()

    events = events.reset_index(drop=True).copy()
    events["event_row_id"] = np.arange(len(events))

    label_by_pos: dict = {}
    n_dropped_edge = 0
    n_dropped_bad_atr = 0
    rows = []
    for row_id, pos in zip(events["event_row_id"].to_numpy(), events["decision_pos"].to_numpy()):
        pos = int(pos)
        if pos not in label_by_pos:
            last = pos + horizon_bars
            if last >= n:
                label_by_pos[pos] = "incomplete_horizon"
            else:
                a = atr[pos]
                if not np.isfinite(a) or a <= 0:
                    label_by_pos[pos] = "bad_atr"
                else:
                    entry = opens[pos + 1]
                    window_low = lows[pos + 1 : last + 1].min()
                    window_high = highs[pos + 1 : last + 1].max()
                    mfe = (entry - window_low) / a
                    mae = (window_high - entry) / a
                    fut_ret = (entry - closes[last]) / entry
                    label_by_pos[pos] = {
                        "entry_price": float(entry),
                        "atr_at_entry": float(a),
                        "mfe_short": float(mfe),
                        "mae_short": float(mae),
                        "short_utility": float(mfe - mae_penalty * mae),
                        "future_close_return_short": float(fut_ret),
                        "gross_return": float(fut_ret),
                        "net_return": float(fut_ret - round_trip_cost),
                    }
        lab = label_by_pos[pos]
        if lab == "incomplete_horizon":
            n_dropped_edge += 1
        elif lab == "bad_atr":
            n_dropped_bad_atr += 1
        else:
            rows.append({"event_row_id": int(row_id), **lab})

    labeled = pd.DataFrame(rows, columns=["event_row_id"] + LABEL_COLUMNS)
    out = events.merge(labeled, on="event_row_id", how="inner", validate="one_to_one")
    assert len(out) <= len(events)
    assert len(out) == len(events) - n_dropped_edge - n_dropped_bad_atr
    out.attrs["n_dropped_incomplete_horizon"] = n_dropped_edge
    out.attrs["n_dropped_bad_atr"] = n_dropped_bad_atr
    return out
