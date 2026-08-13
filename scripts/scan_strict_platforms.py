#!/usr/bin/env python3
"""Strict-platform candidate scan for owner semantic adjudication (P04).

Owner verdict 2026-08-13: the q0.30 pool is not the target pattern (30% of
all bars qualified; only ~30% of decisions sat on anything platform-like).
This round DEFINES ONLY -- no labels, no model, no ranking. Output is a set
of candidates + blind review charts; the owner's eyes are the acceptance test.

Strict definition (v2, owner-approved direction):
  threshold : q0.10 of ma_dispersion_atr over the SAME frozen train bars as
              all prior rounds (positions < 22610) -- ~10% of bars qualify
  core      : maximal run of dispersion < threshold, length 4..12 bars
              (owner's ETH reference: core ~4-7, we allow modest margin)
  flatness  : over the core, (max(high)-min(low)) / ATR(core end) < 3.0
              AND |sma20(end) - sma20(start)| / ATR(core end) < 0.5
  right edge: last bar of the core run (charts end here -- zero future pixels)

Charts mimic the owner's reference style: two vertical lines bracket the
platform core. Decision title is blind (no outcome info).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yoyo_eth import data as data_mod
from yoyo_eth import indicators as ind_mod
from yoyo_eth import scanner as scanner_mod
from yoyo_eth.render import MA_STYLE, _draw_candles

TRAIN_END = 22610  # same frozen boundary as MVP/P02/iteration_v1/P03
QUANTILE = 0.10
MIN_CORE, MAX_CORE = 4, 12
MAX_RANGE_ATR = 3.0
MAX_SMA20_DRIFT_ATR = 0.5
CONTEXT = 60
OUTCOME_BARS = 24
N_CHARTS = 50


def find_candidates(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    disp = df["ma_dispersion_atr"].to_numpy()
    below = disp < threshold
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    sma20, atr = df["sma_20"].to_numpy(), df["atr_14"].to_numpy()

    rows, reject = [], {"too_short": 0, "too_long": 0, "range": 0, "drift": 0}
    i, n = 0, len(df)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and below[j + 1]:
            j += 1
        core_len = j - i + 1
        if core_len < MIN_CORE:
            reject["too_short"] += 1
        elif core_len > MAX_CORE:
            reject["too_long"] += 1
        else:
            a = atr[j]
            rng = (high[i : j + 1].max() - low[i : j + 1].min()) / a
            drift = abs(sma20[j] - sma20[i]) / a
            if rng >= MAX_RANGE_ATR:
                reject["range"] += 1
            elif drift >= MAX_SMA20_DRIFT_ATR:
                reject["drift"] += 1
            else:
                rows.append(
                    {"core_start": i, "core_end": j, "core_len": core_len,
                     "range_atr": float(rng), "sma20_drift_atr": float(drift),
                     "min_dispersion": float(np.nanmin(disp[i : j + 1])),
                     "end_ts": df["timestamp"].iloc[j]}
                )
        i = j + 1
    cand = pd.DataFrame(rows)
    cand.attrs["reject"] = reject
    return cand


def render_candidate(df, row, idx, ddir, odir):
    start = max(0, row.core_start - CONTEXT)
    for view, end in (("decision", row.core_end), ("outcome", min(len(df) - 1, row.core_end + OUTCOME_BARS))):
        seg = df.iloc[start : end + 1]
        fig, ax = plt.subplots(figsize=(13, 6), dpi=100)
        _draw_candles(ax, seg, start)
        for col, (color, ls) in MA_STYLE.items():
            ax.plot(np.arange(start, end + 1), seg[col], color=color, linestyle=ls, linewidth=0.9, label=col)
        # owner-reference style: two vertical lines bracket the platform core
        ax.axvline(row.core_start - 0.5, color="#7b1fa2", linewidth=1.4)
        ax.axvline(row.core_end + 0.5, color="#7b1fa2", linewidth=1.4)
        ax.axvspan(row.core_start - 0.5, row.core_end + 0.5, color="#ce93d8", alpha=0.12)
        if view == "outcome":
            ax.axvspan(row.core_end + 0.5, end + 0.5, color="#90caf9", alpha=0.15)
        title = (f"platform #{idx:02d} {row.end_ts} core={row.core_len} bars "
                 f"range={row.range_atr:.1f}ATR drift={row.sma20_drift_atr:.2f}ATR [{view}]")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.6)
        ax.set_xlim(start - 1, end + 1)
        fig.tight_layout()
        fig.savefig((ddir if view == "decision" else odir) / f"{idx:02d}_pos{row.core_end}.png")
        plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "iteration_v1.yaml").read_text())
    out_base = ROOT / "reports" / "platform_v2"
    ddir, odir = out_base / "decision_view", out_base / "outcome_view"
    ddir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    art = ROOT / "artifacts" / "platform_v2"
    art.mkdir(parents=True, exist_ok=True)

    df, _ = data_mod.load_ohlcv(cfg["data"]["csv_path"], 15, cfg["data"]["data_end_boundary"])
    df = scanner_mod.add_dispersion(ind_mod.add_indicators(df))
    thr = float(df["ma_dispersion_atr"].iloc[:TRAIN_END].dropna().quantile(QUANTILE))

    cand = find_candidates(df, thr)
    stats = {
        "threshold_q0.10": thr,
        "pct_bars_below": float((df["ma_dispersion_atr"] < thr).mean()),
        "n_candidates": int(len(cand)),
        "rejects": cand.attrs["reject"],
        "core_len_dist": cand["core_len"].value_counts().sort_index().to_dict() if len(cand) else {},
        "definition": {
            "quantile": QUANTILE, "core_bars": [MIN_CORE, MAX_CORE],
            "max_range_atr": MAX_RANGE_ATR, "max_sma20_drift_atr": MAX_SMA20_DRIFT_ATR,
        },
    }
    cand.to_parquet(art / "candidates.parquet")
    (art / "scan_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[scan] thr={thr:.3f} ({stats['pct_bars_below']*100:.0f}% bars) candidates={len(cand)} rejects={stats['rejects']}")

    # evenly spaced over time so every regime is represented
    take = cand.iloc[np.linspace(0, len(cand) - 1, min(N_CHARTS, len(cand))).astype(int)] if len(cand) else cand
    for idx, row in enumerate(take.itertuples()):
        render_candidate(df, row, idx, ddir, odir)
    print(f"[charts] {len(take)} rendered")

    items = []
    for png in sorted(ddir.glob("*.png")):
        rel_d = f"decision_view/{png.name}"
        rel_o = f"outcome_view/{png.name}"
        items.append(
            f'<div class="item"><img loading="lazy" src="{rel_d}">'
            f'<details><summary>看 outcome(未来 6h)</summary><img loading="lazy" src="{rel_o}"></details></div>'
        )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>严平台定义 v2 盲审</title>
<style>body{{font-family:-apple-system,sans-serif;margin:20px;background:#fafafa;color:#222}}
img{{max-width:100%;border:1px solid #ddd;border-radius:4px;display:block}}
.item{{margin:14px 0;padding:10px;background:#fff;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
details{{margin-top:6px}}summary{{cursor:pointer;color:#1565c0;font-size:14px}}.tip{{color:#666;font-size:13px}}</style>
</head><body><h1>严平台定义 v2 — {len(take)} 个候选盲审</h1>
<p class="tip">定义:六均线带宽 &lt; {thr:.2f} ATR(train q0.10)连续 4–12 根,段内价格总幅 &lt; 3 ATR,
sma20 漂移 &lt; 0.5 ATR。紫色竖线 = 平台段两端(右端即图的最后一根,无未来)。
请逐张判断:这是不是你要的平台?整体命中率大概多少?</p>
{"".join(items)}</body></html>"""
    (out_base / "review_gallery.html").write_text(html)
    print(f"[done] {out_base / 'review_gallery.html'}")


if __name__ == "__main__":
    main()
