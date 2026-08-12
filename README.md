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
- Data is read from the fable-trading kline cache (read-only) and cut at
  `2026-05-04` — the parent project's frozen holdout start. Holdout bars are
  never loaded.
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
