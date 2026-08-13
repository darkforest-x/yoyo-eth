"""LightGBM regressor on short_utility (doc section 13).

One model, fixed conservative params from configs/mvp.yaml, early stopping on
validation. Test is never used for any training decision. Baseline: rank by
ma_dispersion_atr ascending (tighter compression = better), no learning.
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd

TARGET = "short_utility"


def train_model(
    train_ds: pd.DataFrame,
    val_ds: pd.DataFrame,
    feature_columns: list,
    params: dict,
) -> tuple[lgb.LGBMRegressor, dict]:
    model = lgb.LGBMRegressor(
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        # LightGBM default subsample_freq=0 silently disables row bagging;
        # iteration_v1 sets 1 explicitly, older configs keep 0 (their recorded
        # behaviour: subsample was inert). Report the effective params.
        subsample_freq=params.get("subsample_freq", 0),
        colsample_bytree=params["colsample_bytree"],
        random_state=params["random_state"],
        verbose=-1,
    )
    model.fit(
        train_ds[feature_columns],
        train_ds[TARGET],
        eval_set=[(val_ds[feature_columns], val_ds[TARGET])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False)],
    )
    info = {
        "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
        "n_features": len(feature_columns),
        "gain_importance": dict(
            sorted(
                zip(feature_columns, model.booster_.feature_importance(importance_type="gain").tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }
    return model, info


def predict(model: lgb.LGBMRegressor, ds: pd.DataFrame, feature_columns: list) -> pd.Series:
    return pd.Series(model.predict(ds[feature_columns]), index=ds.index, name="prediction")


def baseline_score(ds: pd.DataFrame) -> pd.Series:
    """Baseline ranking score: tighter compression ranks higher (negated dispersion)."""
    return -ds["ma_dispersion_atr"]
