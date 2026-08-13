#!/usr/bin/env python3
"""P02 walk-forward grid runner.

    python3 scripts/run_walkforward.py --config configs/walkforward.yaml

Runs the (trigger x quantile) grid through the anchored walk-forward harness,
writes artifacts/p02/metrics_walkforward.json, renders OOS review charts for
the configured cells, and generates reports/P02_WALKFORWARD_REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yoyo_eth import data as data_mod
from yoyo_eth import indicators as ind_mod
from yoyo_eth import labels as labels_mod
from yoyo_eth import render as render_mod
from yoyo_eth import scanner as scanner_mod
from yoyo_eth import walkforward as wf
from yoyo_eth.features import FEATURE_MAX_LOOKBACK, add_features

NARRATIVE_MARKER = "<!-- Narrative sections are appended by the analyst. -->"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "walkforward.yaml"))
    ap.add_argument("--skip-charts", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    art = ROOT / cfg["paths"]["artifacts_dir"]
    rep = ROOT / cfg["paths"]["reports_dir"]
    art.mkdir(parents=True, exist_ok=True)

    df, check = data_mod.load_ohlcv(
        cfg["data"]["csv_path"], cfg["data"]["bar_minutes"], cfg["data"]["data_end_boundary"]
    )
    df = scanner_mod.add_dispersion(ind_mod.add_indicators(df))
    df_feat = add_features(df, compression_threshold=None)  # threshold-independent, computed once

    horizon_bars = labels_mod.horizon_bars_from_hours(cfg["label"]["horizon_hours"], cfg["data"]["bar_minutes"])
    gap_bars = max(cfg["windows"]["context_window"], horizon_bars, FEATURE_MAX_LOOKBACK + horizon_bars + 1)
    folds = wf.build_folds(len(df), cfg["walkforward"]["n_folds"], cfg["walkforward"]["initial_train_frac"])
    print(f"[setup] bars={len(df)} folds={folds} gap={gap_bars}")

    grid = {}
    oos_frames = {}
    for trigger in cfg["scanner"]["triggers"]:
        for q in cfg["scanner"]["quantiles"]:
            cell = f"{trigger}@{q}"
            fold_results = [
                wf.run_fold(df, df_feat, fold, trigger, q, cfg, horizon_bars, gap_bars) for fold in folds
            ]
            pooled = wf.pool_cell(fold_results)
            grid[cell] = pooled
            done = [f for f in fold_results if f and not f["skipped"]]
            if done:
                # chart pool: replace raw scores with within-fold percentile ranks --
                # fold models' score scales are not comparable across folds
                normed = [
                    f["test_events"].assign(prediction=f["test_events"]["prediction"].rank(pct=True))
                    for f in done
                ]
                oos_frames[cell] = pd.concat(normed, ignore_index=True)
            if "error" not in pooled:
                print(
                    f"[cell] {cell:>22}: oos={pooled['n_oos_events']:>4} "
                    f"rho/fold={pooled['per_fold_spearman']} wmean={pooled['weighted_mean_spearman']:+.3f} "
                    f"top10 gross={pooled['groups']['top_10pct']['mean_gross_return'] * 1e4:+.1f}bp "
                    f"all-vs-ctrl={pooled['edge_all_vs_control_bp']:+.1f}bp"
                )
            else:
                print(f"[cell] {cell}: {pooled}")

    metrics = {
        "data": {"start": check.start, "end": check.end, "n_bars": check.n_rows_final, "n_gaps": check.n_gaps},
        "harness": {
            "n_folds": cfg["walkforward"]["n_folds"],
            "initial_train_frac": cfg["walkforward"]["initial_train_frac"],
            "inner_val_frac": cfg["walkforward"]["inner_val_frac"],
            "gap_bars": gap_bars,
            "horizon_bars": horizon_bars,
            "folds": folds,
        },
        "owner_authorization": "package change 1+2+3 approved by owner in conversation, 2026-08-12",
        "grid": grid,
    }
    (art / "metrics_walkforward.json").write_text(json.dumps(metrics, indent=2, default=str))

    if not args.skip_charts:
        for cell in cfg["review_charts"]["cells"]:
            if cell not in oos_frames:
                continue
            counts = render_mod.render_review_sets(
                df, oos_frames[cell], rep / "review_charts_p02" / cell.replace("@", "_q"),
                cfg["windows"]["context_window"], cfg["windows"]["local_window"], horizon_bars,
                cfg["review_charts"]["n_top"], cfg["review_charts"]["n_random"], cfg["review_charts"]["n_bottom"],
                cfg["review_charts"]["random_seed"],
            )
            print(f"[charts] {cell}: {counts} (OOS events only)")

    write_report(cfg, metrics, rep / "P02_WALKFORWARD_REPORT.md")
    print(f"[done] {rep / 'P02_WALKFORWARD_REPORT.md'}")


def _cell_rows(grid: dict, cost_sweep: list) -> str:
    hdr = (
        "| cell | OOS n | rho per fold | rho wmean | top10 gross | "
        + " / ".join(f"top10 net@{c}" for c in cost_sweep)
        + " | all gross | control gross | all-ctrl | top10-ctrl |\n"
    )
    hdr += "|" + "---|" * 10 + "\n"
    rows = []
    for cell, p in grid.items():
        if "error" in p:
            rows.append(f"| {cell} | - | {p['error']} | | | | | | | |")
            continue
        g10, ga = p["groups"]["top_10pct"], p["groups"]["all"]
        nets = " / ".join(f"{(g10['mean_gross_return'] - c) * 1e4:+.0f}bp" for c in cost_sweep)
        rows.append(
            f"| {cell} | {p['n_oos_events']} | {p['per_fold_spearman']} | {p['weighted_mean_spearman']:+.3f} | "
            f"{g10['mean_gross_return'] * 1e4:+.1f}bp | {nets} | {ga['mean_gross_return'] * 1e4:+.1f}bp | "
            f"{p['control_mean_gross_return'] * 1e4:+.1f}bp | {p['edge_all_vs_control_bp']:+.1f}bp | "
            f"{p['edge_top10_vs_control_bp']:+.1f}bp |"
        )
    return hdr + "\n".join(rows)


def write_report(cfg: dict, m: dict, out_path: Path) -> None:
    h = m["harness"]
    body = f"""# P02 — Walk-forward 触发点 × 池宽 网格实验

Owner 2026-08-12 批准的打包改动(1 触发点语义 2 扩池 3 walk-forward)。归因方式:
同一 harness 下的 2x3 网格,任一行/列内只有一个变量在动。复现:

```bash
cd /Users/zhangzc/yoyo-eth
python3 scripts/run_walkforward.py --config configs/walkforward.yaml
python3 -m pytest tests/ -q
```

## Harness

| item | value |
|---|---|
| data | {m['data']['start']} .. {m['data']['end']}, {m['data']['n_bars']} bars, gaps={m['data']['n_gaps']} (holdout >= 2026-05-04 never read) |
| folds | {h['n_folds']} anchored, initial train {h['initial_train_frac']:.0%}, OOS = last {1 - h['initial_train_frac']:.0%} in equal slices |
| embargo gap | {h['gap_bars']} bars; horizon {h['horizon_bars']} bars |
| threshold freeze | per fold, inner-train bars only (inner val {h['inner_val_frac']:.0%} of train, early stopping only) |
| top-decile | selected PER FOLD, outcomes pooled (fold models' scores are not cross-comparable) |
| control | matched random per fold test segment (month x ATR tercile, 20 draws/event), pooled |
| costs | {cfg['label']['round_trip_cost']} round-trip (SWAP_TAKER, owner value); sweep {cfg['label']['cost_sweep']} |

## 网格结果(全部 out-of-sample)

{_cell_rows(m['grid'], cfg['label']['cost_sweep'])}

triggers: zone_start = 压缩带开头开火(MVP 原版,对照); dispersion_exit = 带向上穿出
(压缩结束); price_breakout = 收盘首次离开 [ma_lower, ma_upper](突破尝试)。
quantile 0.30 = MVP 原版(对照); 0.45 = 扩池。

{NARRATIVE_MARKER}
"""
    if out_path.exists() and NARRATIVE_MARKER in out_path.read_text():
        body = body.rstrip("\n").removesuffix(NARRATIVE_MARKER) + NARRATIVE_MARKER + out_path.read_text().split(NARRATIVE_MARKER, 1)[1]
    out_path.write_text(body)


if __name__ == "__main__":
    main()
