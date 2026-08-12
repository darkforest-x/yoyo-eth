"""Human review charts (doc section 15).

Two strictly separated views per event:
  decision_view : context_window bars ENDING at the decision bar (no future
                  pixels at all), the last local_window bars shaded, the six
                  MAs overlaid, decision bar marked.
  outcome_view  : decision_view plus the future horizon bars, future region
                  shaded and entry price drawn.

The model never reads these images; they exist so a human can answer "are the
high-ranked samples visually the pattern I wanted?".
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MA_STYLE = {
    "sma_20": ("#1f77b4", "-"),
    "sma_60": ("#ff7f0e", "-"),
    "sma_120": ("#2ca02c", "-"),
    "ema_20": ("#1f77b4", "--"),
    "ema_60": ("#ff7f0e", "--"),
    "ema_120": ("#2ca02c", "--"),
}


def _draw_candles(ax, seg: pd.DataFrame, x0: int) -> None:
    x = np.arange(x0, x0 + len(seg))
    up = (seg["close"] >= seg["open"]).to_numpy()
    colors = np.where(up, "#26a69a", "#ef5350")
    ax.vlines(x, seg["low"], seg["high"], color=colors, linewidth=0.7)
    body_low = np.minimum(seg["open"], seg["close"])
    body_h = (seg["close"] - seg["open"]).abs().replace(0.0, 1e-9)
    ax.bar(x, body_h, bottom=body_low, width=0.7, color=colors, edgecolor=colors, linewidth=0.4)


def render_event(
    df: pd.DataFrame,
    decision_pos: int,
    context_window: int,
    local_window: int,
    horizon_bars: int,
    entry_price: float,
    title: str,
    out_decision: Path,
    out_outcome: Path,
) -> None:
    start = max(0, decision_pos - context_window + 1)

    for view, end in (("decision", decision_pos), ("outcome", min(len(df) - 1, decision_pos + horizon_bars))):
        seg = df.iloc[start : end + 1]
        fig, ax = plt.subplots(figsize=(13, 6), dpi=100)
        _draw_candles(ax, seg, start)
        for col, (color, ls) in MA_STYLE.items():
            ax.plot(np.arange(start, end + 1), seg[col], color=color, linestyle=ls, linewidth=0.9, label=col)
        ax.axvspan(decision_pos - local_window + 1, decision_pos + 0.5, color="#ffe082", alpha=0.18)
        ax.axvline(decision_pos, color="#333333", linewidth=1.0, linestyle=":")
        if view == "outcome":
            ax.axvspan(decision_pos + 0.5, end + 0.5, color="#90caf9", alpha=0.15)
            ax.axhline(entry_price, color="#d32f2f", linewidth=0.9, linestyle="-.", label="entry (next open)")
        ax.set_title(f"{title} [{view} view]", fontsize=10)
        ax.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.6)
        ax.set_xlim(start - 1, end + 1)
        fig.tight_layout()
        fig.savefig(out_decision if view == "decision" else out_outcome)
        plt.close(fig)


def render_review_sets(
    df: pd.DataFrame,
    ds_scored: pd.DataFrame,
    out_dir: Path,
    context_window: int,
    local_window: int,
    horizon_bars: int,
    n_top: int,
    n_random: int,
    n_bottom: int,
    seed: int,
) -> dict:
    """Render Top / middle-random / Bottom sets ranked by model prediction."""
    ranked = ds_scored.sort_values("prediction", ascending=False).reset_index(drop=True)
    n = len(ranked)
    n_top = min(n_top, n)
    n_bottom = min(n_bottom, max(0, n - n_top))
    middle = ranked.iloc[n_top : n - n_bottom]
    rng = np.random.default_rng(seed)
    take_mid = middle.sample(n=min(n_random, len(middle)), random_state=rng.integers(0, 2**31 - 1)) if len(middle) else middle

    sets = {
        "top50": ranked.head(n_top),
        "random50": take_mid.sort_values("prediction", ascending=False),
        "bottom50": ranked.tail(n_bottom),
    }
    counts = {}
    for name, subset in sets.items():
        set_dir = out_dir / name
        if set_dir.exists():
            shutil.rmtree(set_dir)  # stale charts from earlier runs must not mix in
        ddir = set_dir / "decision_view"
        odir = set_dir / "outcome_view"
        ddir.mkdir(parents=True, exist_ok=True)
        odir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(subset.itertuples()):
            ts = pd.Timestamp(row.decision_ts).strftime("%Y%m%d_%H%M")
            fname = f"{i:03d}_pos{row.decision_pos}_{ts}.png"
            title = (
                f"{name} #{i} pred={row.prediction:.3f} utility={row.short_utility:.3f} "
                f"mfe={row.mfe_short:.2f} mae={row.mae_short:.2f} {row.decision_ts} ({row.split})"
            )
            render_event(
                df, int(row.decision_pos), context_window, local_window, horizon_bars,
                float(row.entry_price), title, ddir / fname, odir / fname,
            )
        counts[name] = len(subset)
    return counts
