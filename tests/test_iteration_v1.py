"""iteration_v1 mandated tests (task book section 11)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from test_mvp import make_bars  # noqa: E402  (shared synthetic bars)

from yoyo_eth import anchor as anchor_mod
from yoyo_eth import evaluate as eval_mod
from yoyo_eth import indicators as ind_mod
from yoyo_eth import labels as labels_mod
from yoyo_eth import scanner as scanner_mod
from yoyo_eth import train as train_mod
from yoyo_eth.render import FORBIDDEN_IN_DECISION_TITLE, build_review_titles


def _bars_with_indicators(n=300, seed=5):
    df = make_bars(n, seed=seed)
    df = ind_mod.add_indicators(df)
    return scanner_mod.add_dispersion(df)


def _episode_frame():
    """dispersion: >=thr except bars 10..24 below (15-bar zone); exit_confirm=2."""
    n = 60
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    disp = np.full(n, 2.0)
    disp[10:25] = 0.5
    close = np.full(n, 100.0)
    df = pd.DataFrame(
        {"ts": ts.view("int64") // 10**6, "timestamp": ts, "open": close,
         "high": close + 1, "low": close - 1, "close": close, "volume": np.ones(n),
         "ma_dispersion_atr": disp, "ma_upper": 101.0, "ma_lower": 99.0,
         "cluster_center": 100.0, "ema_20": np.linspace(100, 90, n)}  # ema falling
    )
    return df


# ---- 11.1 duplicate decision_pos labels -------------------------------------------

def test_duplicate_decision_pos_labels():
    df = _bars_with_indicators()
    ev = pd.DataFrame(
        {"decision_pos": [50, 50, 51],
         "decision_ts": df["timestamp"].iloc[[50, 50, 51]].to_numpy()}
    )
    lab = labels_mod.add_labels(ev, df, 24, 0.7, 0.001)
    assert len(lab) == 3
    # the two duplicated rows carry identical labels, once each
    dup = lab[lab["decision_pos"] == 50]
    assert len(dup) == 2
    assert dup["short_utility"].nunique() == 1


# ---- 11.2 control exact count ------------------------------------------------------

def test_control_exact_count():
    df = _bars_with_indicators(n=600, seed=9)
    events = pd.DataFrame(
        {"decision_pos": [200, 210, 220, 230, 240, 250, 260]}
    )
    events["decision_ts"] = df["timestamp"].iloc[events["decision_pos"]].to_numpy()
    split_ds = labels_mod.add_labels(events, df, 24, 0.7, 0.001)
    assert len(split_ds) == 7
    ctrl = eval_mod.matched_random_control(df, split_ds, 150, 560, 24, 0.7, 0.001, n_per_event=20)
    assert ctrl["requested_sample_count"] == 140
    assert ctrl["actual_sample_count"] == 140
    assert ctrl["unique_decision_pos_count"] + ctrl["duplicate_draw_count"] == 140
    assert ctrl["sample_count"] == 140


# ---- 11.3 episode start / qualified / end ------------------------------------------

def test_episode_start_qualified_end():
    df = _episode_frame()
    ep = anchor_mod.scan_compression_episodes(df, 1.0, 3, 2)
    assert len(ep) == 1
    row = ep.iloc[0]
    assert row["episode_start_pos"] == 10
    assert row["episode_qualified_pos"] == 12  # streak reaches 3
    assert row["episode_end_pos"] == 26  # 2nd consecutive above-threshold bar (25, 26)
    assert row["episode_duration_bars"] == 17
    assert row["episode_closed"]
    assert np.isclose(row["min_dispersion"], 0.5)
    assert np.isclose(row["qualified_dispersion"], 0.5)


def test_episode_single_above_bar_does_not_end():
    df = _episode_frame()
    df.loc[18, "ma_dispersion_atr"] = 2.0  # one above bar inside the zone
    ep = anchor_mod.scan_compression_episodes(df, 1.0, 3, 2)
    assert len(ep) == 1
    assert ep.iloc[0]["episode_end_pos"] == 26  # still ends at the confirmed exit


# ---- 11.4 first trigger wins ----------------------------------------------------------

def test_first_trigger_selected():
    df = _episode_frame()
    # local_low_break ONLY at bar 15 (high pushed below ma_lower so
    # failed_reclaim cannot co-fire), failed_reclaim at bar 20 (later, "better")
    df.loc[15, "close"] = 90.0  # below prior lows (99) and cluster_center
    df.loc[15, "high"] = 95.0  # below ma_lower: failed_reclaim condition off
    df.loc[20, "close"] = 98.0  # below ma_lower with high above it
    df.loc[20, "high"] = 100.0
    ep = anchor_mod.scan_compression_episodes(df, 1.0, 3, 2)
    flags = anchor_mod.compute_trigger_flags(df, 1.0, 10)
    ev = anchor_mod.build_anchored_events(ep, flags, df, 1.0)
    assert len(ev) == 1
    assert ev.iloc[0]["decision_pos"] == 15  # earliest trigger, not the later one
    assert ev.iloc[0]["trigger_type"] == "local_low_break"


# ---- 11.5 no trigger -> episode recorded, no event ---------------------------------------

def test_no_trigger_no_event():
    df = _episode_frame()  # flat closes at 100: no trigger conditions ever true
    ep = anchor_mod.scan_compression_episodes(df, 1.0, 3, 2)
    flags = anchor_mod.compute_trigger_flags(df, 1.0, 10)
    ev = anchor_mod.build_anchored_events(ep, flags, df, 1.0)
    assert len(ep) == 1
    assert len(ev) == 0


# ---- 11.6 local low causality ---------------------------------------------------------------

def test_local_low_causality():
    df = _episode_frame()
    flags1 = anchor_mod.compute_trigger_flags(df, 1.0, 10)
    df2 = df.copy()
    df2.loc[30, "low"] = 1.0  # crash the CURRENT bar's low
    flags2 = anchor_mod.compute_trigger_flags(df2, 1.0, 10)
    # prior_local_low at bar 30 must not see bar 30's own low
    assert flags1["prior_local_low"].iloc[30] == flags2["prior_local_low"].iloc[30]
    # it must see it from bar 31 onward
    assert flags2["prior_local_low"].iloc[31] == 1.0


# ---- 11.7 anchored future mutation ------------------------------------------------------------

def test_anchored_future_mutation():
    df = _episode_frame()
    df.loc[15, "close"] = 90.0  # trigger at bar 15
    ep1 = anchor_mod.scan_compression_episodes(df, 1.0, 3, 2)
    f1 = anchor_mod.compute_trigger_flags(df, 1.0, 10)
    ev1 = anchor_mod.build_anchored_events(ep1, f1, df, 1.0)
    decision = int(ev1.iloc[0]["decision_pos"])

    df2 = df.copy()
    rng = np.random.default_rng(3)
    idx = df2.index[decision + 1 :]
    for col in ("open", "high", "low", "close", "ma_dispersion_atr", "ema_20"):
        df2.loc[idx, col] = df2.loc[idx, col] * rng.uniform(0.5, 2.0, len(idx))
    ep2 = anchor_mod.scan_compression_episodes(df2, 1.0, 3, 2)
    f2 = anchor_mod.compute_trigger_flags(df2, 1.0, 10)
    ev2 = anchor_mod.build_anchored_events(ep2, f2, df2, 1.0)

    assert len(ev2) >= 1
    r1, r2 = ev1.iloc[0], ev2.iloc[0]
    for col in ("decision_pos", "episode_start_pos", "episode_qualified_pos",
                "episode_age_at_decision", "compression_duration_at_decision",
                "trigger_type", "episode_min_dispersion_so_far"):
        assert r1[col] == r2[col], col


# ---- 11.8 blind decision title -----------------------------------------------------------------

def test_blind_decision_title():
    meta = {
        "set_name": "top20", "rank": 0, "candidate_id": 7, "prediction": 1.234,
        "decision_ts": "2026-01-01", "split": "test", "trigger_type": "failed_reclaim",
        "episode_age": 5, "short_utility": 2.5, "mfe_short": 3.0, "mae_short": 0.7,
        "gross_return": 0.004, "net_return": 0.003,
    }
    d, o = build_review_titles(meta)
    for word in FORBIDDEN_IN_DECISION_TITLE:
        assert word not in d.lower(), word
    assert "utility" in o and "trig=failed_reclaim" in d


# ---- 11.9 model parameter test -------------------------------------------------------------------

def test_model_subsample_effective():
    cfg = yaml.safe_load((ROOT / "configs" / "iteration_v1.yaml").read_text())
    if cfg["model"]["subsample"] < 1.0:
        assert cfg["model"].get("subsample_freq", 0) > 0
    # and the param actually reaches LightGBM
    df = _bars_with_indicators(n=800, seed=13)
    from yoyo_eth.features import FEATURE_COLUMNS, add_features
    feat = add_features(df, compression_threshold=1.5)
    ev = pd.DataFrame({"decision_pos": np.arange(200, 700, 5)})
    ev["decision_ts"] = df["timestamp"].iloc[ev["decision_pos"]].to_numpy()
    lab = labels_mod.add_labels(ev, df, 24, 0.7, 0.001)
    ds = pd.concat(
        [lab.reset_index(drop=True), feat.loc[lab["decision_pos"], FEATURE_COLUMNS].reset_index(drop=True)],
        axis=1,
    ).dropna(subset=FEATURE_COLUMNS)
    tr, va = ds.iloc[:70], ds.iloc[70:]
    model, _ = train_mod.train_model(tr, va, FEATURE_COLUMNS, cfg["model"])
    assert model.get_params()["subsample_freq"] == cfg["model"]["subsample_freq"]
