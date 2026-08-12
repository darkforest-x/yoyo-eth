"""yoyo-eth: semantic-feature MVP for MA-compression short candidates.

Standalone mini-project spun out of fable-trading. No YOLO, no images:
OHLCV -> SMA/EMA/ATR -> loose compression scanner -> semantic features ->
future MFE/MAE/short_utility labels -> chronological split -> LightGBM
regressor -> ranking metrics + human review charts.

The parent repository is read-only from here (kline cache CSV only).
"""

__version__ = "0.1.0"
