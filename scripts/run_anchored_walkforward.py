#!/usr/bin/env python3
"""P03: does iteration_v1's anchored val signal (rho +0.465, n=31) survive
walk-forward? Owner approved 2026-08-13 ("1 2 3", item 1).

    python3 scripts/run_anchored_walkforward.py

Two arms through the SAME 4-fold anchored harness as P02 (initial train 40%,
OOS = last 60% in equal slices, per-fold inner-train threshold freeze,
embargo 164, fixed matched controls):
  legacy   : zone_start scan (compression-start decision bar)
  anchored : compression episodes -> first causal short trigger (iteration_v1)
Model params = iteration_v1's (subsample_freq=1) for both arms -- the claim
under test was produced under that config. Everything else frozen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yoyo_eth import anchor as anchor_mod
from yoyo_eth import data as data_mod
from yoyo_eth import indicators as ind_mod
from yoyo_eth import scanner as scanner_mod
from yoyo_eth import walkforward as wf
from yoyo_eth.features import FEATURE_MAX_LOOKBACK, add_features

NARRATIVE_MARKER = "<!-- Analyst narrative appended below. -->"


def anchored_builder(cfg):
    def build(df, threshold):
        episodes = anchor_mod.scan_compression_episodes(
            df, threshold, cfg["scanner"]["min_duration"], cfg["anchor"]["exit_confirm_bars"]
        )
        flags = anchor_mod.compute_trigger_flags(df, threshold, cfg["anchor"]["local_low_window"])
        ev = anchor_mod.build_anchored_events(episodes, flags, df, threshold)
        # walkforward substitutes zone_length as the compression_duration feature;
        # for anchored events that is the episode's compressed-bar count at decision
        ev["zone_length"] = ev["compression_duration_at_decision"]
        return ev

    return build


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "iteration_v1.yaml").read_text())
    cfg["walkforward"] = {"n_folds": 4, "initial_train_frac": 0.40, "inner_val_frac": 0.15}
    art = ROOT / "artifacts" / "p03_anchored_wf"
    art.mkdir(parents=True, exist_ok=True)

    df, check = data_mod.load_ohlcv(
        cfg["data"]["csv_path"], cfg["data"]["bar_minutes"], cfg["data"]["data_end_boundary"]
    )
    df = scanner_mod.add_dispersion(ind_mod.add_indicators(df))
    df_feat = add_features(df, compression_threshold=None)
    horizon_bars = 24
    gap_bars = max(cfg["windows"]["context_window"], horizon_bars, FEATURE_MAX_LOOKBACK + horizon_bars + 1)
    folds = wf.build_folds(len(df), 4, 0.40)
    print(f"[setup] bars={len(df)} folds={folds} gap={gap_bars} model.subsample_freq={cfg['model']['subsample_freq']}")

    arms = {
        "legacy_zone_start": None,
        "anchored": anchored_builder(cfg),
    }
    q = cfg["scanner"]["compression_quantile"]
    results = {}
    for arm, builder in arms.items():
        fold_results = [
            wf.run_fold(df, df_feat, fold, "zone_start", q, cfg, horizon_bars, gap_bars, event_builder=builder)
            for fold in folds
        ]
        pooled = wf.pool_cell(fold_results)
        results[arm] = pooled
        if "error" not in pooled:
            print(
                f"[arm] {arm:>18}: oos={pooled['n_oos_events']:>4} folds_used={pooled['n_folds_used']} "
                f"rho/fold={pooled['per_fold_spearman']} wmean={pooled['weighted_mean_spearman']:+.3f} "
                f"top10 gross={pooled['groups']['top_10pct']['mean_gross_return'] * 1e4:+.1f}bp "
                f"all-vs-ctrl={pooled['edge_all_vs_control_bp']:+.1f}bp "
                f"counts={pooled['per_fold_counts']}"
            )
        else:
            print(f"[arm] {arm}: {pooled}")

    metrics = {
        "question": "does iteration_v1 anchored val rho=+0.465 (n=31) survive 4-fold walk-forward?",
        "owner_authorization": "items 1-3 approved in conversation 2026-08-13",
        "data": {"start": check.start, "end": check.end, "n_bars": check.n_rows_final},
        "harness": {"folds": folds, "gap_bars": gap_bars, "quantile": q,
                    "model": cfg["model"], "anchor": cfg["anchor"]},
        "arms": results,
    }
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    write_report(cfg, metrics, ROOT / "reports" / "P03_ANCHORED_WF_REPORT.md")
    print(f"[done] reports/P03_ANCHORED_WF_REPORT.md")


def write_report(cfg, m, out_path: Path) -> None:
    rows = []
    for arm, p in m["arms"].items():
        if "error" in p:
            rows.append(f"| {arm} | - | {p['error']} | | | | | |")
            continue
        g10, ga = p["groups"]["top_10pct"], p["groups"]["all"]
        rows.append(
            f"| {arm} | {p['n_oos_events']} | {p['per_fold_spearman']} | {p['weighted_mean_spearman']:+.3f} | "
            f"{g10['mean_gross_return'] * 1e4:+.1f}bp | {ga['mean_gross_return'] * 1e4:+.1f}bp | "
            f"{p['control_mean_gross_return'] * 1e4:+.1f}bp | {p['edge_all_vs_control_bp']:+.1f}bp |"
        )
    body = f"""# P03 — Anchored 触发的 walk-forward 复检

问题:iteration_v1 的 anchored val ρ=+0.465(p=0.01, n=31)是真信号还是单切分
regime 巧合?方法:与 P02 完全相同的 4 折 anchored harness,两臂(legacy
zone_start 对照 / anchored 因果触发)同参数(iteration_v1 模型配置,
subsample_freq=1)、同阈值冻结纪律、同修复后对照组。复现:

```bash
cd /Users/zhangzc/yoyo-eth
python3 scripts/run_anchored_walkforward.py
python3 -m pytest tests/ -q
```

| arm | OOS n | rho per fold | rho wmean | top10 gross | all gross | control gross | all-ctrl |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

folds: {m['harness']['folds']}(test 段依次 ≈ 牛市尾/暴跌前/暴跌(iteration_v1 的 val 期)/横盘(其 test 期))

{NARRATIVE_MARKER}
"""
    if out_path.exists() and NARRATIVE_MARKER in out_path.read_text():
        body = body.rstrip("\n").removesuffix(NARRATIVE_MARKER) + NARRATIVE_MARKER + out_path.read_text().split(NARRATIVE_MARKER, 1)[1]
    out_path.write_text(body)


if __name__ == "__main__":
    main()
