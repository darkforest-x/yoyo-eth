#!/usr/bin/env python3
"""One-command MVP pipeline (doc section 17).

    python scripts/run_mvp.py --config configs/mvp.yaml [--force] [--skip-charts]

Stages: data checks -> indicators -> split freeze -> scanner threshold (train
only) -> candidates -> labels -> features -> dataset -> LightGBM -> evaluation
-> review charts -> report. Intermediate artifacts are cached in artifacts/;
--force recomputes everything.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yoyo_eth import data as data_mod
from yoyo_eth import dataset as dataset_mod
from yoyo_eth import evaluate as eval_mod
from yoyo_eth import indicators as ind_mod
from yoyo_eth import labels as labels_mod
from yoyo_eth import render as render_mod
from yoyo_eth import scanner as scanner_mod
from yoyo_eth import train as train_mod
from yoyo_eth.features import FEATURE_COLUMNS, FEATURE_MAX_LOOKBACK, add_features


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cache_stamp(cfg: dict) -> str:
    """Cache key covering config + source data (size, mtime) + pipeline code.

    A config-only hash silently serves stale artifacts after a code fix or a
    data refresh (parent learning: artifacts-built-before-their-builder-landed).
    """
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode())
    st = Path(cfg["data"]["csv_path"]).stat()
    h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    for mod in sorted((ROOT / "src" / "yoyo_eth").glob("*.py")):
        h.update(mod.read_bytes())
    return h.hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mvp.yaml"))
    ap.add_argument("--force", action="store_true", help="ignore caches")
    ap.add_argument("--skip-charts", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    art = ROOT / cfg["paths"]["artifacts_dir"]
    rep = ROOT / cfg["paths"]["reports_dir"]
    art.mkdir(exist_ok=True)
    rep.mkdir(exist_ok=True)

    chash = config_hash(cfg)
    stamp = cache_stamp(cfg)
    stamp_file = art / "cache_stamp.txt"
    cache_ok = stamp_file.exists() and stamp_file.read_text().strip() == stamp and not args.force

    horizon_bars = labels_mod.horizon_bars_from_hours(cfg["label"]["horizon_hours"], cfg["data"]["bar_minutes"])
    # doc prescribes max(context, horizon) as the MINIMUM embargo; we widen it to
    # cover the longest finite feature look-back so a val/test event's features
    # never read bars that determined a train label
    gap_bars = max(cfg["windows"]["context_window"], horizon_bars, FEATURE_MAX_LOOKBACK + horizon_bars + 1)

    # ---- stage 1-2: data + indicators (cached as bars.parquet) ----------------
    bars_pq = art / "bars.parquet"
    if cache_ok and bars_pq.exists():
        df = pd.read_parquet(bars_pq)
        check = json.loads((art / "data_check.json").read_text())
        print(f"[cache] bars: {len(df)} rows")
    else:
        df, check_res = data_mod.load_ohlcv(
            cfg["data"]["csv_path"], cfg["data"]["bar_minutes"], cfg["data"]["data_end_boundary"]
        )
        check = dataclasses.asdict(check_res)
        (art / "data_check.json").write_text(json.dumps(check, indent=2))
        df = ind_mod.add_indicators(df)
        df = scanner_mod.add_dispersion(df)
        df.to_parquet(bars_pq)
        print(f"[data] {len(df)} bars {check['start']} .. {check['end']}, gaps={check['n_gaps']}")

    # ---- stage 3: split boundaries on bars -------------------------------------
    boundaries = dataset_mod.split_positions(len(df), cfg["split"]["train_frac"], cfg["split"]["val_frac"])
    print(f"[split] bars train_end={boundaries['train_end']} val_end={boundaries['val_end']} gap={gap_bars}")

    # ---- stage 4: freeze scanner threshold from TRAIN bars only -----------------
    thr_info = scanner_mod.freeze_threshold(df, boundaries["train_end"], cfg["scanner"]["compression_quantile"])
    (art / "scanner_threshold.json").write_text(json.dumps(thr_info, indent=2))
    threshold = thr_info["threshold"]
    print(f"[scanner] threshold={threshold:.4f} (q{cfg['scanner']['compression_quantile']} of train)")

    # ---- stage 5: scan candidates ------------------------------------------------
    events, scan_stats = scanner_mod.scan(
        df, threshold, cfg["scanner"]["min_duration"], cfg["scanner"]["cooldown_bars"]
    )
    print(f"[scan] raw={scan_stats['raw_candidate_count']} dedup={scan_stats['dedup_event_count']}")

    # ---- stage 6: labels ----------------------------------------------------------
    events_l = labels_mod.add_labels(
        events, df, horizon_bars, cfg["label"]["mae_penalty"], cfg["label"]["round_trip_cost"]
    )
    n_edge = events_l.attrs["n_dropped_incomplete_horizon"]
    n_bad_atr = events_l.attrs["n_dropped_bad_atr"]

    # ---- stage 7: features + dataset ----------------------------------------------
    df_feat = add_features(df, threshold)
    events_l = dataset_mod.assign_split(events_l, boundaries, gap_bars)
    n_gap_dropped = int((events_l["split"] == "dropped_gap").sum())  # count BEFORE NaN filtering
    ds = dataset_mod.build_dataset(events_l, df_feat, FEATURE_COLUMNS)
    n_nan = ds.attrs["n_dropped_nan_features"]
    ds.to_parquet(art / "candidates.parquet")
    ds_model = ds[ds["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    ds_model.to_parquet(art / "dataset.parquet")
    split_counts = ds_model["split"].value_counts().to_dict()
    print(f"[dataset] {split_counts} (edge-dropped={n_edge}, bad-atr={n_bad_atr}, nan-dropped={n_nan}, gap-dropped={n_gap_dropped})")

    train_ds = ds_model[ds_model["split"] == "train"].reset_index(drop=True)
    val_ds = ds_model[ds_model["split"] == "val"].reset_index(drop=True)
    test_ds = ds_model[ds_model["split"] == "test"].reset_index(drop=True)
    if len(train_ds) < 50 or len(val_ds) < 20 or len(test_ds) < 20:
        raise SystemExit(f"too few events per split: {split_counts} -- loosen scanner or fix data")

    # ---- stage 8: train --------------------------------------------------------------
    model, train_info = train_mod.train_model(train_ds, val_ds, FEATURE_COLUMNS, cfg["model"])
    model.booster_.save_model(str(art / "model.txt"))
    print(f"[train] best_iteration={train_info['best_iteration']}")

    # ---- stage 9: evaluate --------------------------------------------------------------
    prev_metrics_file = art / "metrics.json"
    prev_review_charts = (
        json.loads(prev_metrics_file.read_text()).get("review_charts") if prev_metrics_file.exists() else None
    )
    metrics = {
        "config_hash": chash,
        "symbol": cfg["data"]["symbol"],
        "timeframe": cfg["data"]["timeframe"],
        "data": {
            "start": check["start"], "end": check["end"], "n_bars": int(len(df)),
            "n_gaps": check["n_gaps"], "data_end_boundary": cfg["data"]["data_end_boundary"],
        },
        "scanner": {**scan_stats, "threshold_info": thr_info},
        "dataset": {
            "n_features": len(FEATURE_COLUMNS),
            "horizon_bars": horizon_bars,
            "gap_bars": gap_bars,
            "counts": {k: int(v) for k, v in split_counts.items()},
            "dropped_incomplete_horizon": int(n_edge),
            "dropped_bad_atr": int(n_bad_atr),
            "dropped_nan_features": int(n_nan),
            "dropped_gap": n_gap_dropped,
            "positive_utility_rate_train": float((train_ds["short_utility"] > 0).mean()),
        },
        "label_assumptions": {
            "horizon_hours": cfg["label"]["horizon_hours"],
            "mae_penalty": cfg["label"]["mae_penalty"],
            "round_trip_cost": cfg["label"]["round_trip_cost"],
            "cost_source": "fable-trading yoyo/contracts/costs.py SWAP_TAKER (owner value)",
        },
        "model": {"params": cfg["model"], **train_info},
        "results": {},
    }

    split_ranges = {
        "validation": (boundaries["train_end"] + gap_bars, boundaries["val_end"]),
        "test": (boundaries["val_end"] + gap_bars, boundaries["n_bars"]),
    }
    for split_name, split_ds in (("validation", val_ds), ("test", test_ds)):
        pred = train_mod.predict(model, split_ds, FEATURE_COLUMNS)
        split_ds = split_ds.assign(prediction=pred)
        lo, hi = split_ranges[split_name]
        metrics["results"][split_name] = {
            "model": eval_mod.evaluate_split(split_ds, pred),
            "baseline_dispersion_asc": eval_mod.evaluate_split(split_ds, train_mod.baseline_score(split_ds)),
            "matched_random_control": eval_mod.matched_random_control(
                df, split_ds, lo, hi, horizon_bars,
                cfg["label"]["mae_penalty"], cfg["label"]["round_trip_cost"],
            ),
        }
        # cost sensitivity on gross returns, symmetric across model/baseline/control
        for cost in cfg["label"]["cost_sweep"]:
            key = f"mean_net_return_at_{cost}"
            for score_key in ("model", "baseline_dispersion_asc"):
                for gstats in metrics["results"][split_name][score_key]["groups"].values():
                    gstats[key] = gstats["mean_gross_return"] - cost
            ctrl = metrics["results"][split_name]["matched_random_control"]
            if "mean_gross_return" in ctrl:
                ctrl[key] = ctrl["mean_gross_return"] - cost

    metrics["permutation_importance_val"] = eval_mod.permutation_importance(model, val_ds, FEATURE_COLUMNS)
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[eval] val spearman={metrics['results']['validation']['model']['spearman_r']:.4f} "
          f"test spearman={metrics['results']['test']['model']['spearman_r']:.4f}")

    # ---- stage 10: review charts -----------------------------------------------------------
    chart_counts = {}
    if args.skip_charts:
        # keep the last real chart inventory in metrics/report instead of blanking it
        if prev_review_charts:
            metrics["review_charts"] = prev_review_charts
            (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
    else:
        review_pool = test_ds.assign(prediction=train_mod.predict(model, test_ds, FEATURE_COLUMNS))
        pool_name = "test"
        need = cfg["review_charts"]["n_top"] + cfg["review_charts"]["n_random"] + cfg["review_charts"]["n_bottom"]
        if len(review_pool) < need:
            review_pool = pd.concat(
                [
                    val_ds.assign(prediction=train_mod.predict(model, val_ds, FEATURE_COLUMNS)),
                    review_pool,
                ],
                ignore_index=True,
            )
            pool_name = "val+test"
        if len(review_pool) < need:
            # still short of 50+50+50: include train events too, split-tagged in
            # every chart title so the reviewer can discount in-sample ranks
            review_pool = pd.concat(
                [
                    train_ds.assign(prediction=train_mod.predict(model, train_ds, FEATURE_COLUMNS)),
                    review_pool,
                ],
                ignore_index=True,
            )
            pool_name = "train+val+test"
        chart_counts = render_mod.render_review_sets(
            df, review_pool, rep / "review_charts",
            cfg["windows"]["context_window"], cfg["windows"]["local_window"], horizon_bars,
            cfg["review_charts"]["n_top"], cfg["review_charts"]["n_random"], cfg["review_charts"]["n_bottom"],
            cfg["review_charts"]["random_seed"],
        )
        metrics["review_charts"] = {"pool": pool_name, **chart_counts}
        (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[charts] {chart_counts} from {pool_name}")

    stamp_file.write_text(stamp)
    write_report(cfg, metrics, rep / "MVP_REPORT.md")
    print(f"[done] report: {rep / 'MVP_REPORT.md'}")


NARRATIVE_MARKER = "<!-- Narrative sections (interpretation, risks, next steps) are appended by the analyst. -->"


def _fmt_group_row(score_name: str, gname: str, g: dict, cost_sweep: list, primary_cost: float) -> str:
    nets = " | ".join(
        f"{g.get(f'mean_net_return_at_{c}', g['mean_gross_return'] - c) * 1e4:+.1f}bp" for c in cost_sweep
    )
    return (
        f"| {score_name} | {gname} | {g['sample_count']} | {g['mean_short_utility']:.3f} | "
        f"{g['median_short_utility']:.3f} | {g['mean_mfe']:.3f} | {g['mean_mae']:.3f} | "
        f"{g['mean_gross_return'] * 1e4:+.1f}bp | {nets} | {g['positive_net_return_ratio']:.2f} |"
    )


def _fmt_group_table(results: dict, cost_sweep: list, primary_cost: float) -> str:
    header = (
        "| score | group | n | mean utility | median utility | mean MFE | mean MAE | mean gross | "
        + " | ".join(f"net@{c}" for c in cost_sweep)
        + f" | pos-net ratio@{primary_cost} |\n"
    )
    header += "|" + "---|" * (9 + len(cost_sweep)) + "\n"
    rows = []
    for score_name, res in (("model", results["model"]), ("baseline (dispersion asc)", results["baseline_dispersion_asc"])):
        for gname, g in res["groups"].items():
            rows.append(_fmt_group_row(score_name, gname, g, cost_sweep, primary_cost))
    ctrl = results.get("matched_random_control", {})
    if "mean_gross_return" in ctrl:
        rows.append(_fmt_group_row("matched random control", "all", ctrl, cost_sweep, primary_cost))
    return header + "\n".join(rows)


def write_report(cfg: dict, m: dict, out_path: Path) -> None:
    r = m["results"]
    cs = cfg["label"]["cost_sweep"]
    gain_top = list(m["model"]["gain_importance"].items())[:10]
    perm_top = list(m["permutation_importance_val"]["mean_drop"].items())[:10]
    lines = f"""# yoyo-eth Semantic MVP Report

Generated by `scripts/run_mvp.py` (config hash `{m['config_hash']}`). Reproduce:

```bash
cd /Users/zhangzc/yoyo-eth
python3 scripts/run_mvp.py --config configs/mvp.yaml --force
python3 -m pytest tests/ -q
```

## Setup

| item | value |
|---|---|
| symbol | {m['symbol']} |
| timeframe | {m['timeframe']} |
| data range | {m['data']['start']} .. {m['data']['end']} |
| bars | {m['data']['n_bars']} (gaps: {m['data']['n_gaps']}) |
| data end boundary | {m['data']['data_end_boundary']} (fable-trading frozen holdout start; bars at/after it never read) |
| scanner threshold | {m['scanner']['threshold']:.4f} = q{m['scanner']['threshold_info']['quantile']} of ma_dispersion_atr on train bars only |
| raw candidates | {m['scanner']['raw_candidate_count']} |
| dedup events | {m['scanner']['dedup_event_count']} |
| features | {m['dataset']['n_features']} |
| horizon | {cfg['label']['horizon_hours']}h = {m['dataset']['horizon_bars']} bars; mae_penalty {cfg['label']['mae_penalty']} |
| split gap | {m['dataset']['gap_bars']} bars |
| train / val / test events | {m['dataset']['counts'].get('train', 0)} / {m['dataset']['counts'].get('val', 0)} / {m['dataset']['counts'].get('test', 0)} (gap-dropped {m['dataset']['dropped_gap']}, edge-dropped {m['dataset']['dropped_incomplete_horizon']}, nan-dropped {m['dataset']['dropped_nan_features']}) |
| cost | {cfg['label']['round_trip_cost']} round-trip ({m['label_assumptions']['cost_source']}); sensitivity: {cs} |
| model | LightGBM regressor, fixed params, best_iteration={m['model']['best_iteration']} |

## Correlation (prediction vs short_utility)

| split | score | pearson r | pearson p | spearman r | spearman p |
|---|---|---|---|---|---|
| validation | model | {r['validation']['model']['pearson_r']:.4f} | {r['validation']['model']['pearson_p']:.3g} | {r['validation']['model']['spearman_r']:.4f} | {r['validation']['model']['spearman_p']:.3g} |
| validation | baseline | {r['validation']['baseline_dispersion_asc']['pearson_r']:.4f} | {r['validation']['baseline_dispersion_asc']['pearson_p']:.3g} | {r['validation']['baseline_dispersion_asc']['spearman_r']:.4f} | {r['validation']['baseline_dispersion_asc']['spearman_p']:.3g} |
| test | model | {r['test']['model']['pearson_r']:.4f} | {r['test']['model']['pearson_p']:.3g} | {r['test']['model']['spearman_r']:.4f} | {r['test']['model']['spearman_p']:.3g} |
| test | baseline | {r['test']['baseline_dispersion_asc']['pearson_r']:.4f} | {r['test']['baseline_dispersion_asc']['pearson_p']:.3g} | {r['test']['baseline_dispersion_asc']['spearman_r']:.4f} | {r['test']['baseline_dispersion_asc']['spearman_p']:.3g} |

## Validation groups

{_fmt_group_table(r['validation'], cs, cfg['label']['round_trip_cost'])}

## Test groups

{_fmt_group_table(r['test'], cs, cfg['label']['round_trip_cost'])}

## Feature importance

Top 10 by LightGBM gain: {', '.join(f'{k} ({v:.0f})' for k, v in gain_top)}

Top 10 by permutation (val spearman drop): {', '.join(f'{k} ({v:+.4f})' for k, v in perm_top)}

## Review charts

{json.dumps(m.get('review_charts', {}), indent=2)}

See `reports/review_charts/{{top50,random50,bottom50}}/{{decision_view,outcome_view}}/`.

{NARRATIVE_MARKER}
"""
    # keep analyst-written narrative (everything after the marker) across reruns
    if out_path.exists() and NARRATIVE_MARKER in out_path.read_text():
        narrative = out_path.read_text().split(NARRATIVE_MARKER, 1)[1]
        lines = lines.rstrip("\n").removesuffix(NARRATIVE_MARKER) + NARRATIVE_MARKER + narrative
    out_path.write_text(lines)


if __name__ == "__main__":
    main()
