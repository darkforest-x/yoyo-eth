#!/usr/bin/env python3
"""iteration_v1 one-command pipeline: Correctness Repair + Causal Anchor.

    python3 scripts/run_iteration_v1.py --config configs/iteration_v1.yaml --force

Two arms through identical controls (data, boundary, threshold, 27 features,
labels, split, embargo, model params, seed, costs):
  Legacy   : MVP zone_start scan (compression-start decision bar)
  Anchored : compression episodes -> first causal short trigger

Outputs land in artifacts/iteration_v1/ and reports/iteration_v1/ only; MVP
and P02 artifacts are untouched baseline evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import lightgbm
import pyarrow

from yoyo_eth import anchor as anchor_mod
from yoyo_eth import data as data_mod
from yoyo_eth import dataset as dataset_mod
from yoyo_eth import evaluate as eval_mod
from yoyo_eth import indicators as ind_mod
from yoyo_eth import labels as labels_mod
from yoyo_eth import render as render_mod
from yoyo_eth import scanner as scanner_mod
from yoyo_eth import train as train_mod
from yoyo_eth.features import FEATURE_COLUMNS, FEATURE_MAX_LOOKBACK, add_features

SEMANTIC_COLS = [
    "compression_duration_at_decision",
    "episode_age_at_decision",
    "failed_reclaim_count_10",
    "cluster_penetration_count_10",
    "ema20_slope_5",
    "close_to_cluster_atr",
    "ma_dispersion_slope_5",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    )
    return commit, dirty


def _quantiles(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"mean": None, "median": None, "p25": None, "p75": None}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
    }


def build_arm_dataset(events_labeled, df_feat, boundaries, gap_bars):
    ev = dataset_mod.assign_split(events_labeled, boundaries, gap_bars)
    ds = dataset_mod.build_dataset(ev, df_feat, FEATURE_COLUMNS)
    return ds[ds["split"].isin(["train", "val", "test"])].reset_index(drop=True)


def eval_arm(name, ds_model, df, cfg, boundaries, gap_bars, horizon_bars):
    """Train + evaluate one arm; returns (results dict, scored val/test frames)."""
    tr = ds_model[ds_model["split"] == "train"].reset_index(drop=True)
    va = ds_model[ds_model["split"] == "val"].reset_index(drop=True)
    te = ds_model[ds_model["split"] == "test"].reset_index(drop=True)
    model, train_info = train_mod.train_model(tr, va, FEATURE_COLUMNS, cfg["model"])
    (ROOT / cfg["paths"]["artifacts_dir"] / f"{name}_model.txt").parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(ROOT / cfg["paths"]["artifacts_dir"] / f"{name}_model.txt"))

    ranges = {
        "validation": (boundaries["train_end"] + gap_bars, boundaries["val_end"]),
        "test": (boundaries["val_end"] + gap_bars, boundaries["n_bars"]),
    }
    out = {"counts": {"train": len(tr), "val": len(va), "test": len(te)}, "train_info": train_info}
    scored = {}
    for split_name, split_ds in (("validation", va), ("test", te)):
        pred = train_mod.predict(model, split_ds, FEATURE_COLUMNS)
        split_ds = split_ds.assign(prediction=pred)
        scored[split_name] = split_ds
        lo, hi = ranges[split_name]
        res = eval_mod.evaluate_split(split_ds, split_ds["prediction"])
        if "error" in res:
            raise RuntimeError(f"{name}/{split_name}: evaluate_split failed: {res['error']}")
        ctrl = eval_mod.matched_random_control(
            df, split_ds, lo, hi, horizon_bars,
            cfg["label"]["mae_penalty"], cfg["label"]["round_trip_cost"],
            seed=cfg["review_charts"]["random_seed"],
        )
        if "error" in ctrl:
            raise RuntimeError(f"{name}/{split_name}: matched_random_control failed: {ctrl['error']}")
        # LOW_SAMPLE marking (doc 9.3)
        for gname, g in res.get("groups", {}).items():
            g["low_sample"] = bool(g["sample_count"] < 10)
        res["matched_random_control"] = ctrl
        res["candidate_minus_control_gross_bp"] = (
            (res["groups"]["all"]["mean_gross_return"] - ctrl["mean_gross_return"]) * 1e4
        )
        res["candidate_minus_control_net_bp"] = (
            (res["groups"]["all"]["mean_net_return"] - ctrl["mean_net_return"]) * 1e4
        )
        out[split_name] = res
    return out, scored, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "iteration_v1.yaml"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-charts", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    art = ROOT / cfg["paths"]["artifacts_dir"]
    rep = ROOT / cfg["paths"]["reports_dir"]
    art.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)

    # ---- data + indicators -----------------------------------------------------
    df, check = data_mod.load_ohlcv(
        cfg["data"]["csv_path"], cfg["data"]["bar_minutes"], cfg["data"]["data_end_boundary"]
    )
    df = scanner_mod.add_dispersion(ind_mod.add_indicators(df))
    bars_pq = art / "bars.parquet"
    df.to_parquet(bars_pq)

    horizon_bars = labels_mod.horizon_bars_from_hours(cfg["label"]["horizon_hours"], cfg["data"]["bar_minutes"])
    gap_bars = max(cfg["windows"]["context_window"], horizon_bars, FEATURE_MAX_LOOKBACK + horizon_bars + 1)
    boundaries = dataset_mod.split_positions(len(df), cfg["split"]["train_frac"], cfg["split"]["val_frac"])

    thr_info = scanner_mod.freeze_threshold(df, boundaries["train_end"], cfg["scanner"]["compression_quantile"])
    threshold = thr_info["threshold"]
    df_feat = add_features(df, threshold)
    print(f"[setup] bars={len(df)} threshold={threshold:.4f} split={boundaries} gap={gap_bars}")

    # ---- Legacy arm ---------------------------------------------------------------
    legacy_events, legacy_stats = scanner_mod.scan(
        df, threshold, cfg["scanner"]["min_duration"], cfg["scanner"]["cooldown_bars"], "zone_start"
    )
    legacy_labeled = labels_mod.add_labels(
        legacy_events, df, horizon_bars, cfg["label"]["mae_penalty"], cfg["label"]["round_trip_cost"]
    )
    legacy_labeled.to_parquet(art / "legacy_events.parquet")
    legacy_ds = build_arm_dataset(legacy_labeled, df_feat, boundaries, gap_bars)
    legacy_ds.to_parquet(art / "legacy_dataset.parquet")

    # ---- Anchored arm ----------------------------------------------------------------
    episodes = anchor_mod.scan_compression_episodes(
        df, threshold, cfg["scanner"]["min_duration"], cfg["anchor"]["exit_confirm_bars"]
    )
    flags = anchor_mod.compute_trigger_flags(df, threshold, cfg["anchor"]["local_low_window"])
    anchored_events = anchor_mod.build_anchored_events(episodes, flags, df, threshold)
    episodes["has_trigger"] = episodes["episode_id"].isin(
        anchored_events["episode_id"] if len(anchored_events) else []
    )
    episodes.to_parquet(art / "episodes.parquet")
    anchored_labeled = labels_mod.add_labels(
        anchored_events, df, horizon_bars, cfg["label"]["mae_penalty"], cfg["label"]["round_trip_cost"]
    )
    anchored_labeled.to_parquet(art / "anchored_events.parquet")
    anchored_ds = build_arm_dataset(anchored_labeled, df_feat, boundaries, gap_bars)
    anchored_ds.to_parquet(art / "anchored_dataset.parquet")

    counts_a = anchored_ds["split"].value_counts().to_dict()
    print(f"[anchored] episodes={len(episodes)} triggered={int(episodes['has_trigger'].sum())} "
          f"events={len(anchored_events)} splits={counts_a}")

    # ---- sample gate (doc section 8) ----------------------------------------------------
    gate = cfg["sample_gate"]
    blocked = (
        counts_a.get("train", 0) < gate["anchored_train_min"]
        or counts_a.get("val", 0) < gate["anchored_val_min"]
        or counts_a.get("test", 0) < gate["anchored_test_min"]
    )

    metrics = {
        "status": "blocked_by_sample_size" if blocked else "completed",
        "data": {"start": check.start, "end": check.end, "n_bars": check.n_rows_final, "n_gaps": check.n_gaps},
        "threshold_info": thr_info,
        "split": {**boundaries, "gap_bars": gap_bars,
                  "train_end_ts": str(df["timestamp"].iloc[boundaries["train_end"] - 1]),
                  "val_end_ts": str(df["timestamp"].iloc[boundaries["val_end"] - 1])},
        "legacy_scan": legacy_stats,
        "episode_stats": {
            "episode_count": int(len(episodes)),
            "episode_with_trigger": int(episodes["has_trigger"].sum()),
            "episode_without_trigger": int((~episodes["has_trigger"]).sum()),
            "anchored_event_count": int(len(anchored_events)),
            "episode_duration": _quantiles(episodes["episode_duration_bars"]),
        },
        "legacy_dataset_counts": legacy_ds["split"].value_counts().to_dict(),
        "anchored_dataset_counts": counts_a,
        "trigger_distribution": (
            anchored_ds.groupby("trigger_type")
            .agg(n=("trigger_type", "size"), mean_utility=("short_utility", "mean"),
                 mean_net=("net_return", "mean"))
            .reset_index().to_dict("records")
            if len(anchored_ds) else []
        ),
    }

    # ---- semantic alignment (doc 9.2) -----------------------------------------------------
    legacy_sem = legacy_ds.copy()
    legacy_sem["compression_duration_at_decision"] = legacy_sem["compression_duration"]
    legacy_sem["episode_age_at_decision"] = legacy_sem["compression_duration"] - 1
    sem = {}
    for col in SEMANTIC_COLS:
        sem[col] = {
            "legacy": _quantiles(legacy_sem[col]) if col in legacy_sem else None,
            "anchored": _quantiles(anchored_ds[col]) if col in anchored_ds else None,
        }
    metrics["semantic_alignment"] = sem
    if len(anchored_ds):
        metrics["anchored_minus_legacy_decision_bars"] = float(
            anchored_ds["episode_age_at_decision"].mean() - legacy_sem["episode_age_at_decision"].mean()
        )

    if not blocked:
        legacy_res, legacy_scored, _ = eval_arm("legacy", legacy_ds, df, cfg, boundaries, gap_bars, horizon_bars)
        anchored_res, anchored_scored, _ = eval_arm("anchored", anchored_ds, df, cfg, boundaries, gap_bars, horizon_bars)
        metrics["results"] = {"legacy": legacy_res, "anchored": anchored_res}
        for arm, res in metrics["results"].items():
            for sp in ("validation", "test"):
                r = res[sp]
                print(f"[{arm:>8}/{sp:<10}] n={r['n']} rho={r['spearman_r']:+.3f} "
                      f"all-ctrl={r['candidate_minus_control_gross_bp']:+.1f}bp "
                      f"ctrl req/act/uniq={r['matched_random_control']['requested_sample_count']}/"
                      f"{r['matched_random_control']['actual_sample_count']}/"
                      f"{r['matched_random_control']['unique_decision_pos_count']}")

        # ---- review charts (doc section 10): val/test only, N=min(20, avail) -----------
        if not args.skip_charts:
            for arm, scored in (("legacy", legacy_scored), ("anchored", anchored_scored)):
                for sp, frame in scored.items():
                    sp_dir = "validation" if sp == "validation" else "test"
                    n_set = min(cfg["review_charts"]["n_per_set"], len(frame))
                    if n_set == 0:
                        continue
                    render_mod.render_review_sets(
                        df, frame, rep / "review_charts" / arm / sp_dir,
                        cfg["windows"]["context_window"], cfg["windows"]["local_window"], horizon_bars,
                        n_set, n_set, n_set, cfg["review_charts"]["random_seed"],
                        dynamic_set_names=True,
                    )
            print("[charts] legacy/anchored x validation/test done")

    (art / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # ---- run manifest (doc 5.5) --------------------------------------------------------------
    commit, dirty = git_info()
    manifest = {
        "git_commit": commit,
        "git_dirty": dirty,
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
        "source_csv_path": cfg["data"]["csv_path"],
        "source_csv_sha256": sha256_file(Path(cfg["data"]["csv_path"])),
        "data_end_boundary": cfg["data"]["data_end_boundary"],
        "bars_sha256": sha256_file(bars_pq),
        "legacy_dataset_sha256": sha256_file(art / "legacy_dataset.parquet"),
        "anchored_dataset_sha256": sha256_file(art / "anchored_dataset.parquet"),
        "scanner_threshold": threshold,
        "train_end_ts": metrics["split"]["train_end_ts"],
        "val_end_ts": metrics["split"]["val_end_ts"],
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "lightgbm_version": lightgbm.__version__,
        "pyarrow_version": pyarrow.__version__,
        "random_seed": cfg["model"]["random_state"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (art / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] status={metrics['status']} manifest+metrics written; report skeleton next")
    write_report_skeleton(cfg, metrics, manifest, rep / "ITERATION_V1_REPORT.md")


NARRATIVE_MARKER = "<!-- Analyst narrative (15 questions, FACT/OBSERVATION/HYPOTHESIS/DECISION) appended below. -->"


def _fmt_arm_table(res: dict, cost_sweep: list) -> str:
    net_hdr = " | ".join(f"net@{c}" for c in cost_sweep)
    hdr = ("| split | n | pearson | spearman | group | n | mean util | median util | mean MFE | mean MAE | "
           f"gross | {net_hdr} | pos-net@{cost_sweep[0]} | flag |\n")
    hdr += "|" + "---|" * (13 + len(cost_sweep)) + "\n"

    def nets(g):
        return " | ".join(f"{(g['mean_gross_return'] - c) * 1e4:+.1f}bp" for c in cost_sweep)

    rows = []
    for sp in ("validation", "test"):
        r = res[sp]
        for gname, g in r["groups"].items():
            rows.append(
                f"| {sp} | {r['n']} | {r['pearson_r']:+.3f} | {r['spearman_r']:+.3f} | {gname} | "
                f"{g['sample_count']} | {g['mean_short_utility']:+.3f} | {g['median_short_utility']:+.3f} | "
                f"{g['mean_mfe']:.2f} | {g['mean_mae']:.2f} | {g['mean_gross_return'] * 1e4:+.1f}bp | "
                f"{nets(g)} | {g['positive_net_return_ratio']:.2f} | {'LOW_SAMPLE' if g['low_sample'] else ''} |"
            )
        c = r["matched_random_control"]
        rows.append(
            f"| {sp} | {r['n']} | | | matched control | {c['actual_sample_count']} | "
            f"{c['mean_short_utility']:+.3f} | {c['median_short_utility']:+.3f} | {c['mean_mfe']:.2f} | "
            f"{c['mean_mae']:.2f} | {c['mean_gross_return'] * 1e4:+.1f}bp | "
            f"{nets(c)} | {c['positive_net_return_ratio']:.2f} | req={c['requested_sample_count']} uniq={c['unique_decision_pos_count']} |"
        )
    return hdr + "\n".join(rows)


def write_report_skeleton(cfg, m, manifest, out_path: Path) -> None:
    sem_rows = []
    for col, v in m["semantic_alignment"].items():
        for arm in ("legacy", "anchored"):
            q = v[arm]
            if q and q["mean"] is not None:
                sem_rows.append(
                    f"| {col} | {arm} | {q['mean']:+.3f} | {q['median']:+.3f} | {q['p25']:+.3f} | {q['p75']:+.3f} |"
                )
    trig_rows = [
        f"| {t['trigger_type']} | {t['n']} | {t['mean_utility']:+.3f} | {t['mean_net'] * 1e4:+.1f}bp |"
        for t in m["trigger_distribution"]
    ]
    results_block = ""
    if "results" in m:
        results_block = f"""## Legacy 模型结果

{_fmt_arm_table(m['results']['legacy'], cfg['label']['cost_sweep'])}

## Anchored 模型结果

{_fmt_arm_table(m['results']['anchored'], cfg['label']['cost_sweep'])}
"""
    body = f"""# ITERATION V1 — Correctness Repair + Causal Anchor

状态:**{m['status']}**。复现:

```bash
cd /Users/zhangzc/yoyo-eth
python3 scripts/run_iteration_v1.py --config configs/iteration_v1.yaml --force
python3 -m pytest tests/ -q
```

## Setup

| item | value |
|---|---|
| data | {m['data']['start']} .. {m['data']['end']}, {m['data']['n_bars']} bars, gaps={m['data']['n_gaps']} |
| boundary | {cfg['data']['data_end_boundary']}(边界后行在 CSV 解析后、任何指标/扫描/特征/标签/训练之前剔除)|
| threshold | {m['threshold_info']['threshold']:.4f} = q{cfg['scanner']['compression_quantile']} on train bars only |
| split | train_end={m['split']['train_end']} ({m['split']['train_end_ts']}), val_end={m['split']['val_end']} ({m['split']['val_end_ts']}), gap={m['split']['gap_bars']} |
| model | LightGBM fixed params, subsample={cfg['model']['subsample']} subsample_freq={cfg['model']['subsample_freq']} seed={cfg['model']['random_state']} |
| manifest | git={manifest['git_commit'][:8]} dirty={manifest['git_dirty']} csv_sha={manifest['source_csv_sha256'][:12]}.. |

## 正确性指标(doc 9.1)

| item | value |
|---|---|
| Legacy raw candidates | {m['legacy_scan']['raw_candidate_count']} |
| Legacy dedup events | {m['legacy_scan']['dedup_event_count']} |
| Compression episodes | {m['episode_stats']['episode_count']} |
| Episodes with trigger | {m['episode_stats']['episode_with_trigger']} |
| Episodes without trigger | {m['episode_stats']['episode_without_trigger']} |
| Anchored events | {m['episode_stats']['anchored_event_count']} |
| Legacy splits | {m['legacy_dataset_counts']} |
| Anchored splits | {m['anchored_dataset_counts']} |

## 语义对齐(doc 9.2)

| quantity | arm | mean | median | p25 | p75 |
|---|---|---|---|---|---|
{chr(10).join(sem_rows)}

按 trigger_type(anchored,val+test+train 全体):

| trigger_type | n | mean utility | mean net |
|---|---|---|---|
{chr(10).join(trig_rows) if trig_rows else '| (none) | | | |'}

{results_block}
{NARRATIVE_MARKER}
"""
    if out_path.exists() and NARRATIVE_MARKER in out_path.read_text():
        body = body.rstrip("\n").removesuffix(NARRATIVE_MARKER) + NARRATIVE_MARKER + out_path.read_text().split(NARRATIVE_MARKER, 1)[1]
    out_path.write_text(body)


if __name__ == "__main__":
    main()
