"""Anchored walk-forward evaluation harness (P02).

Owner-approved package change of 2026-08-12 ("1 2 3"): trigger variants +
pool expansion + walk-forward. Attribution is preserved by running a grid of
(trigger x quantile) cells through THIS identical harness; within any row or
column exactly one variable moves.

Fold layout over the pre-holdout bars (holdout >= 2026-05-04 is cut upstream
in data.py and never reaches this module):

    fold k (k = 1..n_folds):
        train bars: [0, test_lo)              anchored, expanding
        test bars:  [test_lo, test_hi)        consecutive equal slices of the
                                              tail (1 - initial_train_frac)
    inner split for early stopping: the last inner_val_frac of the fold's
    train bars, with the same embargo gap as train/test.

Per fold, in order: freeze scanner threshold on INNER-TRAIN bars only ->
scan events over the whole frame -> label -> assign inner-train / inner-val /
test with embargo gaps -> train LightGBM (early stop on inner-val) -> predict
the fold's test events. Out-of-sample results are pooled across folds; the
top-decile is selected PER FOLD (scores from different fold models are not
comparable) and outcomes pooled. A matched random control is drawn per fold
test segment and pooled the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

from . import evaluate as eval_mod
from . import labels as labels_mod
from . import scanner as scanner_mod
from . import train as train_mod
from .features import FEATURE_COLUMNS


def build_folds(n_bars: int, n_folds: int, initial_train_frac: float) -> list:
    """Anchored expanding folds: [(train_end, test_lo, test_hi), ...]."""
    tail = 1.0 - initial_train_frac
    folds = []
    for k in range(n_folds):
        lo = initial_train_frac + tail * k / n_folds
        hi = initial_train_frac + tail * (k + 1) / n_folds
        folds.append((int(n_bars * lo), int(n_bars * lo), int(n_bars * hi)))
    return folds


def run_fold(
    df: pd.DataFrame,
    df_feat: pd.DataFrame,
    fold: tuple,
    trigger: str,
    quantile: float,
    cfg: dict,
    horizon_bars: int,
    gap_bars: int,
    event_builder=None,
) -> dict | None:
    """One fold. event_builder(df, threshold) -> events frame overrides the
    default scanner.scan(trigger) source; it must provide decision_pos,
    decision_ts and a zone_length column (used as the compression_duration
    feature -- 'compression length confirmed at decision time')."""
    train_end, test_lo, test_hi = fold
    inner_val_start = int(train_end * (1.0 - cfg["walkforward"]["inner_val_frac"]))

    thr_info = scanner_mod.freeze_threshold(df, inner_val_start, quantile)
    threshold = thr_info["threshold"]
    if event_builder is not None:
        events = event_builder(df, threshold)
        scan_stats = {"event_source": "custom", "n_events": int(len(events)), "threshold": threshold}
    else:
        events, scan_stats = scanner_mod.scan(
            df, threshold, cfg["scanner"]["min_duration"], cfg["scanner"]["cooldown_bars"], trigger
        )
    events = labels_mod.add_labels(
        events, df, horizon_bars, cfg["label"]["mae_penalty"], cfg["label"]["round_trip_cost"]
    )

    pos = events["decision_pos"]
    events = events.assign(
        split=np.select(
            [
                pos < inner_val_start,
                # right edge shrunk by horizon so no val label window reaches
                # into the fold's test bars (strict purged-CV)
                (pos >= inner_val_start + gap_bars) & (pos < train_end - horizon_bars),
                (pos >= test_lo + gap_bars) & (pos < test_hi),
            ],
            ["train", "val", "test"],
            default="dropped",
        )
    )

    feats = df_feat.loc[events["decision_pos"], [c for c in FEATURE_COLUMNS if c != "compression_duration"]]
    feats = feats.reset_index(drop=True)
    ds = pd.concat([events.reset_index(drop=True), feats], axis=1)
    # event-level compression length stands in for the bar-level streak feature
    ds["compression_duration"] = ds["zone_length"].astype("float64")
    ds = ds.dropna(subset=FEATURE_COLUMNS)

    tr = ds[ds["split"] == "train"].reset_index(drop=True)
    va = ds[ds["split"] == "val"].reset_index(drop=True)
    te = ds[ds["split"] == "test"].reset_index(drop=True)
    if len(tr) < 40 or len(va) < 10 or len(te) < 10:
        return {"skipped": True, "counts": {"train": len(tr), "val": len(va), "test": len(te)}}

    model, train_info = train_mod.train_model(tr, va, FEATURE_COLUMNS, cfg["model"])
    te = te.assign(prediction=train_mod.predict(model, te, FEATURE_COLUMNS))

    control = eval_mod.matched_random_control(
        df, te, test_lo + gap_bars, test_hi, horizon_bars,
        cfg["label"]["mae_penalty"], cfg["label"]["round_trip_cost"],
        seed=cfg["review_charts"]["random_seed"],
    )
    rho = sstats.spearmanr(te["prediction"], te["short_utility"])
    return {
        "skipped": False,
        "threshold": threshold,
        "counts": {"train": len(tr), "val": len(va), "test": len(te)},
        "best_iteration": train_info["best_iteration"],
        "spearman": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "test_events": te,
        "control_stats": control,
        "scan_stats": scan_stats,
    }


def pool_cell(fold_results: list) -> dict:
    """Pool OOS outcomes across folds; top-decile selected per fold."""
    done = [f for f in fold_results if f and not f["skipped"]]
    if not done:
        return {"error": "all folds skipped"}
    te_all = pd.concat([f["test_events"] for f in done], ignore_index=True)
    top10 = pd.concat(
        [f["test_events"].nlargest(max(1, int(np.ceil(len(f["test_events"]) * 0.10))), "prediction") for f in done],
        ignore_index=True,
    )
    top20 = pd.concat(
        [f["test_events"].nlargest(max(1, int(np.ceil(len(f["test_events"]) * 0.20))), "prediction") for f in done],
        ignore_index=True,
    )
    ns = np.array([len(f["test_events"]) for f in done], dtype=float)
    rhos = np.array([f["spearman"] for f in done])
    pooled_rho = sstats.spearmanr(te_all["prediction"], te_all["short_utility"])  # cross-model ranks, report as secondary
    ctrl_gross = np.array([f["control_stats"].get("mean_gross_return", np.nan) for f in done])
    ctrl_n = np.array([f["control_stats"].get("sample_count", 0) for f in done], dtype=float)
    ctrl_ok = np.isfinite(ctrl_gross) & (ctrl_n > 0)  # a failed fold control must not poison the pool
    ctrl_gross, ctrl_n = ctrl_gross[ctrl_ok], ctrl_n[ctrl_ok]
    if len(ctrl_n) == 0:
        ctrl_gross, ctrl_n = np.array([np.nan]), np.array([1.0])
    out = {
        "n_folds_used": len(done),
        "n_folds_with_control": int(ctrl_ok.sum()),
        "n_oos_events": int(len(te_all)),
        "per_fold_spearman": [round(float(r), 3) for r in rhos],
        "weighted_mean_spearman": float((rhos * ns).sum() / ns.sum()),
        "pooled_spearman_secondary": float(pooled_rho.statistic),
        "groups": {
            "top_10pct": eval_mod._group_stats(top10),
            "top_20pct": eval_mod._group_stats(top20),
            "all": eval_mod._group_stats(te_all),
        },
        "control_mean_gross_return": float((ctrl_gross * ctrl_n).sum() / ctrl_n.sum()),
        "per_fold_counts": [f["counts"] for f in done],
        "per_fold_threshold": [round(float(f["threshold"]), 3) for f in done],
        "per_fold_best_iteration": [f["best_iteration"] for f in done],
    }
    out["edge_all_vs_control_bp"] = (out["groups"]["all"]["mean_gross_return"] - out["control_mean_gross_return"]) * 1e4
    out["edge_top10_vs_control_bp"] = (out["groups"]["top_10pct"]["mean_gross_return"] - out["control_mean_gross_return"]) * 1e4
    return out
