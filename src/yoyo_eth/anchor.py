"""Compression episodes + causal anchored triggers (iteration_v1 Stage B).

Episode state machine (all causal, doc section 6.2):
  qualify : ma_dispersion_atr < threshold for min_duration consecutive bars.
            episode_qualified_pos = that bar; episode_start_pos = it minus
            (min_duration - 1).
  alive   : as long as compression persists. Single above-threshold bars do
            NOT end the episode.
  exit    : dispersion >= threshold for exit_confirm_bars consecutive bars;
            episode_end_pos = the confirming bar. Triggers may still be
            evaluated on that bar (doc: "在确认退出的当前 bar 上仍允许计算").
  NaN dispersion bars (indicator warmup only in real data) count as neither
  below nor above: they reset both counters but keep an alive episode alive.

Anchored triggers (doc 6.3), evaluated per bar from causal columns only:
  A failed_reclaim        : high >= ma_lower AND close < ma_lower AND
                            ema20_slope_5 < 0
  B compression_exit_down : disp[t-1] < threshold AND disp[t] >= threshold AND
                            close < ma_lower AND ema20_slope_5 < 0
  C local_low_break       : close < prior_local_low AND close < cluster_center
                            AND ema20_slope_5 < 0, where prior_local_low =
                            low.shift(1).rolling(local_low_window).min()
                            (shift(1): the current bar can never be its own
                            broken support).
  ema20_slope_5 < 0 is evaluated as ema_20[t] < ema_20[t-5] -- identical sign
  to the ATR-normalised feature since ATR > 0; NaN comparisons are False.

Canonical decision bar (doc 6.4): the FIRST bar in
[episode_qualified_pos, episode_end_pos] (window inclusive on both ends; the
qualified bar itself is eligible -- documented interpretation of "qualified 后")
where ANY trigger fires. One anchored event per episode at most. All trigger
flags on that bar are kept; trigger_type is the '+'-joined sorted flag names.
Episodes without any trigger are recorded but produce no event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRIGGER_NAMES = ("compression_exit_down", "failed_reclaim", "local_low_break")


def scan_compression_episodes(
    df: pd.DataFrame, threshold: float, min_duration: int, exit_confirm_bars: int
) -> pd.DataFrame:
    disp = df["ma_dispersion_atr"].to_numpy()
    below = disp < threshold  # NaN -> False
    above = disp >= threshold  # NaN -> False
    n = len(df)

    episodes = []
    state_alive = False
    streak = 0
    above_run = 0
    cur: dict = {}
    for i in range(n):
        streak = streak + 1 if below[i] else 0
        if not state_alive:
            if streak >= min_duration:
                state_alive = True
                above_run = 0
                cur = {
                    "episode_start_pos": i - min_duration + 1,
                    "episode_qualified_pos": i,
                }
        else:
            if above[i]:
                above_run += 1
                if above_run >= exit_confirm_bars:
                    cur["episode_end_pos"] = i
                    episodes.append(cur)
                    state_alive = False
                    cur = {}
            else:
                above_run = 0
    if state_alive:  # truncated by end of data
        cur["episode_end_pos"] = -1  # sentinel: unclosed
        episodes.append(cur)

    ep = pd.DataFrame(episodes, columns=["episode_start_pos", "episode_qualified_pos", "episode_end_pos"])
    ep["episode_id"] = np.arange(len(ep))
    ep["episode_closed"] = ep["episode_end_pos"] >= 0
    ep.loc[~ep["episode_closed"], "episode_end_pos"] = n - 1
    ts = df["timestamp"]
    ep["episode_start_ts"] = ts.iloc[ep["episode_start_pos"]].to_numpy() if len(ep) else []
    ep["episode_qualified_ts"] = ts.iloc[ep["episode_qualified_pos"]].to_numpy() if len(ep) else []
    ep["episode_end_ts"] = ts.iloc[ep["episode_end_pos"]].to_numpy() if len(ep) else []
    ep["episode_duration_bars"] = ep["episode_end_pos"] - ep["episode_start_pos"] + 1
    mins, means, quals = [], [], []
    for row in ep.itertuples():
        seg = disp[row.episode_start_pos : row.episode_end_pos + 1]
        mins.append(np.nanmin(seg) if len(seg) else np.nan)
        means.append(np.nanmean(seg) if len(seg) else np.nan)
        quals.append(disp[row.episode_qualified_pos])
    ep["min_dispersion"] = mins
    ep["mean_dispersion"] = means
    ep["qualified_dispersion"] = quals
    return ep


def compute_trigger_flags(df: pd.DataFrame, threshold: float, local_low_window: int) -> pd.DataFrame:
    disp = df["ma_dispersion_atr"]
    close = df["close"]
    slope_down = df["ema_20"] < df["ema_20"].shift(5)  # == (ema20_slope_5 < 0); NaN -> False

    failed_reclaim = (df["high"] >= df["ma_lower"]) & (close < df["ma_lower"]) & slope_down
    compression_exit_down = (
        (disp.shift(1) < threshold) & (disp >= threshold) & (close < df["ma_lower"]) & slope_down
    )
    prior_local_low = df["low"].shift(1).rolling(local_low_window, min_periods=local_low_window).min()
    local_low_break = (close < prior_local_low) & (close < df["cluster_center"]) & slope_down

    return pd.DataFrame(
        {
            "failed_reclaim": failed_reclaim.fillna(False),
            "compression_exit_down": compression_exit_down.fillna(False),
            "local_low_break": local_low_break.fillna(False),
            "prior_local_low": prior_local_low,
        },
        index=df.index,
    )


def build_anchored_events(
    episodes: pd.DataFrame, flags: pd.DataFrame, df: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """First-trigger canonical decision bar per episode (doc 6.4/6.5)."""
    disp = df["ma_dispersion_atr"].to_numpy()
    below = disp < threshold
    any_trigger = (
        flags["failed_reclaim"] | flags["compression_exit_down"] | flags["local_low_break"]
    ).to_numpy()

    rows = []
    for ep in episodes.itertuples():
        window = np.arange(ep.episode_qualified_pos, ep.episode_end_pos + 1)
        hits = window[any_trigger[window]]
        if len(hits) == 0:
            continue
        t = int(hits[0])
        seg_to_t = disp[ep.episode_start_pos : t + 1]
        min_so_far = float(np.nanmin(seg_to_t))
        rows.append(
            {
                "episode_id": int(ep.episode_id),
                "decision_pos": t,
                "decision_ts": df["timestamp"].iloc[t],
                "episode_start_pos": int(ep.episode_start_pos),
                "episode_qualified_pos": int(ep.episode_qualified_pos),
                "episode_age_at_decision": int(t - ep.episode_start_pos),
                "compression_duration_at_decision": int(below[ep.episode_start_pos : t + 1].sum()),
                "trigger_failed_reclaim": int(flags["failed_reclaim"].iloc[t]),
                "trigger_compression_exit_down": int(flags["compression_exit_down"].iloc[t]),
                "trigger_local_low_break": int(flags["local_low_break"].iloc[t]),
                "trigger_type": "+".join(
                    sorted(name for name in TRIGGER_NAMES if flags[name].iloc[t])
                ),
                "ma_dispersion_atr_at_decision": float(disp[t]) if np.isfinite(disp[t]) else np.nan,
                "episode_min_dispersion_so_far": min_so_far,
                "dispersion_change_from_min": (
                    float(disp[t]) - min_so_far if np.isfinite(disp[t]) else np.nan
                ),
            }
        )
    columns = [
        "episode_id", "decision_pos", "decision_ts", "episode_start_pos",
        "episode_qualified_pos", "episode_age_at_decision", "compression_duration_at_decision",
        "trigger_failed_reclaim", "trigger_compression_exit_down", "trigger_local_low_break",
        "trigger_type", "ma_dispersion_atr_at_decision", "episode_min_dispersion_so_far",
        "dispersion_change_from_min",
    ]
    ev = pd.DataFrame(rows, columns=columns)  # explicit columns: zero-trigger runs must not crash downstream
    ev["event_id"] = np.arange(len(ev))
    return ev
