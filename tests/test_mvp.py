"""Unit tests (doc section 16): indicators, scanner, ordering, duplicates,
labels, split, and the Future Mutation Test (no-lookahead guarantee)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yoyo_eth import data as data_mod
from yoyo_eth import dataset as dataset_mod
from yoyo_eth import indicators as ind_mod
from yoyo_eth import labels as labels_mod
from yoyo_eth import scanner as scanner_mod
from yoyo_eth.features import FEATURE_COLUMNS, add_features


def make_bars(n=400, seed=7, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = np.roll(close, 1) + rng.normal(0, 0.1, n)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.5, n)
    vol = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"ts": (ts.view("int64") // 10**6), "timestamp": ts, "open": open_,
         "high": high, "low": low, "close": close, "volume": vol}
    )


@pytest.fixture(scope="module")
def bars():
    df = make_bars()
    df = ind_mod.add_indicators(df)
    return scanner_mod.add_dispersion(df)


# ---- indicator tests ---------------------------------------------------------

def test_sma(bars):
    # SMA20 at position 25 equals mean of closes 6..25
    expected = bars["close"].iloc[6:26].mean()
    assert np.isclose(bars["sma_20"].iloc[25], expected)
    assert bars["sma_20"].iloc[:19].isna().all()


def test_ema(bars):
    # recursive definition: ema_t = alpha*close_t + (1-alpha)*ema_{t-1}
    alpha = 2 / 21
    e = bars["close"].iloc[0]
    for i in range(1, 30):
        e = alpha * bars["close"].iloc[i] + (1 - alpha) * e
    assert np.isclose(bars["ema_20"].iloc[29], e)
    assert bars["ema_20"].iloc[:19].isna().all()


def test_atr(bars):
    tr1 = max(
        bars["high"].iloc[1] - bars["low"].iloc[1],
        abs(bars["high"].iloc[1] - bars["close"].iloc[0]),
        abs(bars["low"].iloc[1] - bars["close"].iloc[0]),
    )
    assert np.isclose(bars["tr"].iloc[1], tr1)
    assert bars["atr_14"].iloc[:14].isna().all()
    assert (bars["atr_14"].dropna() > 0).all()


def test_dispersion(bars):
    row = bars.iloc[200]
    mas = [row[c] for c in scanner_mod.MA_COLS]
    assert np.isclose(row["ma_dispersion_atr"], (max(mas) - min(mas)) / row["atr_14"])
    assert (bars["ma_dispersion_atr"].dropna() >= 0).all()


# ---- data check tests ---------------------------------------------------------

def test_time_ordering_and_duplicates(tmp_path):
    df = make_bars(200)
    shuffled = df.sample(frac=1.0, random_state=0)
    dup = pd.concat([shuffled, shuffled.iloc[:5]])
    p = tmp_path / "k.csv"
    dup[["ts", "open", "high", "low", "close", "volume"]].to_csv(p, index=False)
    loaded, check = data_mod.load_ohlcv(str(p), 15, "2099-01-01T00:00:00+00:00")
    assert loaded["ts"].is_monotonic_increasing
    assert loaded["ts"].duplicated().sum() == 0
    assert check.n_duplicates_dropped == 5
    assert len(loaded) == 200


def test_boundary_cut(tmp_path):
    df = make_bars(200, start="2026-05-01")
    p = tmp_path / "k.csv"
    df[["ts", "open", "high", "low", "close", "volume"]].to_csv(p, index=False)
    loaded, _ = data_mod.load_ohlcv(str(p), 15, "2026-05-02T00:00:00+00:00")
    assert (loaded["timestamp"] < pd.Timestamp("2026-05-02", tz="UTC")).all()


# ---- scanner tests -------------------------------------------------------------

def test_below_streak():
    s = pd.Series([1.0, 0.1, 0.1, 0.1, 1.0, 0.1, np.nan, 0.1])
    streak = scanner_mod.below_streak(s, 0.5)
    assert streak.tolist() == [0, 1, 2, 3, 0, 1, 0, 1]


def test_scan_dedup(bars):
    thr = scanner_mod.freeze_threshold(bars, 280, 0.3)["threshold"]
    ev, stats = scanner_mod.scan(bars, thr, 3, 5)
    assert stats["raw_candidate_count"] >= stats["dedup_event_count"]
    if len(ev) > 1:
        assert (np.diff(ev["decision_pos"]) > 5).all()


def test_threshold_train_only(bars):
    # threshold must not move when post-train data changes
    thr1 = scanner_mod.freeze_threshold(bars, 280, 0.3)["threshold"]
    mutated = bars.copy()
    mutated.loc[300:, "ma_dispersion_atr"] = 0.0001
    thr2 = scanner_mod.freeze_threshold(mutated, 280, 0.3)["threshold"]
    assert thr1 == thr2


# ---- label tests ------------------------------------------------------------------

def test_labels():
    df = make_bars(100)
    df = ind_mod.add_indicators(df)
    df = scanner_mod.add_dispersion(df)
    ev = pd.DataFrame({"decision_pos": [50], "decision_ts": [df["timestamp"].iloc[50]]})
    lab = labels_mod.add_labels(ev, df, 24, 0.7, 0.001)
    row = lab.iloc[0]
    entry = df["open"].iloc[51]
    assert row["entry_price"] == entry
    assert np.isclose(row["mfe_short"], (entry - df["low"].iloc[51:75].min()) / df["atr_14"].iloc[50])
    assert np.isclose(row["mae_short"], (df["high"].iloc[51:75].max() - entry) / df["atr_14"].iloc[50])
    assert np.isclose(row["short_utility"], row["mfe_short"] - 0.7 * row["mae_short"])
    assert np.isclose(row["net_return"], row["gross_return"] - 0.001)


def test_labels_drop_incomplete_horizon():
    df = make_bars(100)
    df = ind_mod.add_indicators(df)
    ev = pd.DataFrame({"decision_pos": [90], "decision_ts": [df["timestamp"].iloc[90]]})
    lab = labels_mod.add_labels(ev, df, 24, 0.7, 0.001)
    assert len(lab) == 0
    assert lab.attrs["n_dropped_incomplete_horizon"] == 1


# ---- split tests --------------------------------------------------------------------

def test_split_chronological_with_gap():
    b = dataset_mod.split_positions(1000, 0.7, 0.15)
    ev = pd.DataFrame({"decision_pos": [0, 699, 700, 723, 724, 849, 850, 873, 874, 999]})
    ev = dataset_mod.assign_split(ev, b, gap_bars=24)
    got = dict(zip(ev["decision_pos"], ev["split"]))
    assert got[0] == "train" and got[699] == "train"
    assert got[700] == "dropped_gap" and got[723] == "dropped_gap"
    assert got[724] == "val" and got[849] == "val"
    assert got[850] == "dropped_gap" and got[873] == "dropped_gap"
    assert got[874] == "test" and got[999] == "test"
    # no event in two splits, ordering preserved
    for name, lo, hi in (("train", 0, 699), ("val", 724, 849), ("test", 874, 999)):
        sub = ev[ev["split"] == name]["decision_pos"]
        assert sub.between(lo, hi).all()


def test_max_consecutive_close_above_window_capped():
    """Streaks that began before the 10-bar window must be capped at the number
    of bars actually inside the window (adversarial-review finding M2)."""
    n = 60
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    # closes above center for bars 10..29 (20-bar run), below elsewhere
    center = np.zeros(n)
    close = np.where((np.arange(n) >= 10) & (np.arange(n) < 30), 1.0, -1.0)
    df = pd.DataFrame(
        {"ts": ts.view("int64") // 10**6, "timestamp": ts, "open": close,
         "high": close + 0.1, "low": close - 0.1, "close": close, "volume": np.ones(n)}
    )
    df = ind_mod.add_indicators(df)
    df = scanner_mod.add_dispersion(df)
    # override cluster geometry with the synthetic center to isolate the feature
    df["cluster_center"] = center
    df["ma_upper"] = center + 0.5
    df["ma_lower"] = center - 0.5
    df["atr_14"] = 1.0
    df["tr_mean_5"] = 1.0
    df["tr_mean_20"] = 1.0
    out = add_features(df, compression_threshold=1.5)
    col = out["max_consecutive_close_above_10"]
    assert col.iloc[29] == 10  # 20-bar run, but only 10 bars fit in the window
    assert col.iloc[33] == 6  # run ended at bar 29; window 24..33 holds 6 run bars
    assert col.iloc[45] == 0  # run fully outside the window


# ---- Future Mutation Test (doc 16, mandatory) -----------------------------------------

def test_future_mutation():
    """Mutating OHLCV strictly after the decision bar must not change any
    feature at the decision bar. Failure = MVP not complete."""
    n = 400
    decision = 250
    df1 = make_bars(n, seed=11)
    df2 = df1.copy()
    rng = np.random.default_rng(99)
    future = slice(decision + 1, n)
    for col in ("open", "high", "low", "close", "volume"):
        df2.loc[df2.index[future], col] = df2.loc[df2.index[future], col] * rng.uniform(0.5, 2.0, n - decision - 1)
    df2["high"] = df2[["open", "high", "low", "close"]].max(axis=1)
    df2["low"] = df2[["open", "low", "close"]].min(axis=1)

    feats = []
    for d in (df1, df2):
        d = ind_mod.add_indicators(d)
        d = scanner_mod.add_dispersion(d)
        d = add_features(d, compression_threshold=1.5)
        feats.append(d.loc[decision, FEATURE_COLUMNS].astype(float))
    pd.testing.assert_series_equal(feats[0], feats[1], check_exact=False, rtol=1e-12)


def test_future_mutation_scanner_events():
    """Events at/before the decision bar must be identical after future mutation."""
    n = 400
    decision = 250
    df1 = make_bars(n, seed=11)
    df2 = df1.copy()
    df2.loc[df2.index[decision + 1 :], "close"] = 999.0
    df2["high"] = df2[["open", "high", "low", "close"]].max(axis=1)
    df2["low"] = df2[["open", "low", "close"]].min(axis=1)

    evs = []
    for d in (df1, df2):
        d = ind_mod.add_indicators(d)
        d = scanner_mod.add_dispersion(d)
        ev, _ = scanner_mod.scan(d, 1.5, 3, 5)
        evs.append(ev[ev["decision_pos"] <= decision]["decision_pos"].tolist())
    assert evs[0] == evs[1]
