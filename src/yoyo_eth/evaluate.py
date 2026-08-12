"""Evaluation (doc section 14): correlations + Top-decile group tables.

For each of validation and test, and for both model prediction and the
ma_dispersion_atr baseline: Pearson/Spearman correlation with short_utility,
then group stats for Top 10% / Top 20% / all events ranked by the score.

MVP acceptance is "pipeline runs correctly", not "model is profitable" -- the
numbers are reported as-is either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats


def _group_stats(g: pd.DataFrame) -> dict:
    return {
        "sample_count": int(len(g)),
        "mean_short_utility": float(g["short_utility"].mean()),
        "median_short_utility": float(g["short_utility"].median()),
        "mean_mfe": float(g["mfe_short"].mean()),
        "mean_mae": float(g["mae_short"].mean()),
        "mean_gross_return": float(g["gross_return"].mean()),
        "mean_net_return": float(g["net_return"].mean()),
        "positive_net_return_ratio": float((g["net_return"] > 0).mean()),
    }


def evaluate_split(ds: pd.DataFrame, score: pd.Series, top_fracs=(0.10, 0.20)) -> dict:
    """Correlation of score vs short_utility + grouped outcome tables."""
    if len(ds) < 3:
        return {"error": f"too few samples ({len(ds)})"}
    y = ds["short_utility"]
    pearson = sstats.pearsonr(score, y)
    spearman = sstats.spearmanr(score, y)
    out = {
        "n": int(len(ds)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "groups": {},
    }
    ranked = ds.assign(_score=score.to_numpy()).sort_values("_score", ascending=False)
    for frac in top_fracs:
        k = max(1, int(np.ceil(len(ranked) * frac)))
        out["groups"][f"top_{int(frac * 100)}pct"] = _group_stats(ranked.head(k))
    out["groups"]["all"] = _group_stats(ranked)
    return out


def matched_random_control(
    df: pd.DataFrame,
    split_ds: pd.DataFrame,
    pos_lo: int,
    pos_hi: int,
    horizon_bars: int,
    mae_penalty: float,
    round_trip_cost: float,
    n_per_event: int = 20,
    seed: int = 42,
) -> dict:
    """Matched random-entry control (parent CLAUDE.md requirement).

    For each real event in the split, sample ``n_per_event`` random decision
    bars from the SAME month x same ATR-tercile bucket within the same split's
    bar range, then apply identical entry / horizon / barriers / costs. This is
    what any directional result table must be compared against: it prices the
    short beta of the period, which permutation of ranks cannot see.
    ATR terciles are computed from the split's own bars (evaluation-only
    construct; nothing feeds back into training).
    """
    from .labels import add_labels  # local import to avoid cycle at module load

    rng = np.random.default_rng(seed)
    hi = min(pos_hi, len(df) - horizon_bars - 1)
    pool = df.iloc[pos_lo:hi].copy()
    pool = pool[pool["atr_14"].notna() & (pool["atr_14"] > 0)]
    if len(pool) < 100:
        return {"error": "control pool too small"}
    terc = pool["atr_14"].quantile([1 / 3, 2 / 3]).to_numpy()
    pool["_bucket"] = list(
        zip(pool["timestamp"].dt.strftime("%Y-%m"), np.digitize(pool["atr_14"], terc))
    )
    by_bucket = {k: g.index.to_numpy() for k, g in pool.groupby("_bucket")}

    ev_atr = df["atr_14"].iloc[split_ds["decision_pos"]].to_numpy()
    ev_month = df["timestamp"].iloc[split_ds["decision_pos"]].dt.strftime("%Y-%m").to_numpy()
    ev_bucket = list(zip(ev_month, np.digitize(ev_atr, terc)))

    picks, n_fallback = [], 0
    all_pool = pool.index.to_numpy()
    for b in ev_bucket:
        cand = by_bucket.get(b)
        if cand is None or len(cand) == 0:
            cand = all_pool
            n_fallback += 1
        picks.extend(rng.choice(cand, size=n_per_event, replace=True).tolist())

    ctrl_events = pd.DataFrame(
        {"decision_pos": np.asarray(picks, dtype=np.int64),
         "decision_ts": df["timestamp"].iloc[picks].to_numpy()}
    )
    ctrl = add_labels(ctrl_events, df, horizon_bars, mae_penalty, round_trip_cost)
    stats = _group_stats(ctrl)
    stats["n_fallback_events"] = int(n_fallback)
    stats["n_per_event"] = int(n_per_event)
    return stats


def permutation_importance(model, ds: pd.DataFrame, feature_columns: list, n_repeats: int = 5, seed: int = 42) -> dict:
    """Spearman-based permutation importance on the given (validation) split."""
    rng = np.random.default_rng(seed)
    y = ds["short_utility"].to_numpy()
    base_pred = model.predict(ds[feature_columns])
    base = sstats.spearmanr(base_pred, y).statistic
    drops = {}
    X = ds[feature_columns].reset_index(drop=True)
    for col in feature_columns:
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            pred = model.predict(Xp)
            vals.append(base - sstats.spearmanr(pred, y).statistic)
        drops[col] = float(np.mean(vals))
    return {
        "baseline_spearman": float(base),
        "mean_drop": dict(sorted(drops.items(), key=lambda kv: kv[1], reverse=True)),
    }
