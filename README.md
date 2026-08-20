# ARCHIVED

This repository has been consolidated into `darkforest-x/fable-trading`.

- Canonical repository: `darkforest-x/fable-trading`
- Source frozen commit: `6147810afb46be1c664128e9a5359e8e7d0a3923`
- Migration PR: https://github.com/darkforest-x/fable-trading/pull/1
- Migration merge commit: `31d6b2aa39cc8cc454144228980c8eb427f868c4` (PR #1, merged into main)
- Canonical module: `yoyo/layers/l1_detection/numeric_baseline/, yoyo/evaluation/, experiments/historical/yoyo_eth/`
- Status: read-only historical research — closed_negative — the compression pool lost to matched random entry

No further development occurs in this repository. Its conclusions, including the
negative ones, are registered in `experiments/registry.yaml` and summarised in
`experiments/historical/` in the canonical repository. Nothing here was deleted.

---

# yoyo-eth — Semantic MVP

Standalone mini-experiment spun out of `fable-trading`. One question:

> Without YOLO — using only OHLCV, SMA/EMA 20/60/120, ATR and a small set of
> semantic features — can we rank MA-compression candidates so that the top of
> the ranking is more likely to precede a profitable SHORT?

Acceptance for this phase is **the pipeline runs end-to-end correctly**, not
that the model is profitable. Honest negative results are valid results.

## One command

```bash
python3 scripts/run_mvp.py --config configs/mvp.yaml          # cached
python3 scripts/run_mvp.py --config configs/mvp.yaml --force  # from scratch
python3 -m pytest tests/ -q                                   # incl. Future Mutation Test
```

## Scope (deliberately minimal)

- One symbol (ETH-USDT-SWAP), one timeframe (15m), one scanner, one LightGBM
  regressor on `short_utility = MFE_short − 0.7·MAE_short` (6h horizon).
- Chronological split 70/15/15 with `max(context_window, horizon_bars)` embargo
  gaps; scanner threshold frozen from TRAIN bars only.
- Data is read from the fable-trading kline cache (read-only). Rows at or
  after the frozen holdout boundary (`2026-05-04`) are excluded before
  indicator calculation, scanner statistics, feature construction, label
  construction and model training. (The raw CSV is fully read once; the cut
  happens immediately after parsing — this is a code-level exclusion, not
  physical isolation.)
- ~27 causal features (compression, MA slopes, price-vs-cluster, simple
  reclaim/rejection, trend background, bar/volatility). Only labels see the
  future; `tests/test_mvp.py::test_future_mutation` enforces this.

## Layout

```
configs/mvp.yaml      all knobs; cost values are owner decisions from fable-trading
src/yoyo_eth/         data / indicators / scanner / features / labels / dataset /
                      train / evaluate / render
scripts/run_mvp.py    one-command pipeline
artifacts/            candidates.parquet dataset.parquet model.txt
                      scanner_threshold.json metrics.json
reports/MVP_REPORT.md + review_charts/{top50,random50,bottom50}/{decision_view,outcome_view}/
```

## Forbidden in this phase

YOLO / images / manual labels / second symbol / second timeframe / multi-model
stacks / hyperparameter search / touching the parent repo. After the report and
review charts are generated, STOP — whether to continue is a human decision.
