from __future__ import annotations

import pandas as pd
from .base import Strategy


class RSIStrategy(Strategy):
    name = 'RSI Mean Reversion'

    def __init__(self, window: int = 14, oversold: float = 30, overbought: float = 70):
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    def _rsi(self, prices: pd.DataFrame) -> pd.DataFrame:
        from .indicators import rsi
        return rsi(prices, self.window)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        rsi = self._rsi(prices)
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[rsi < self.oversold] = 1.0
        signals[rsi > self.overbought] = 0.0
        signals = signals.ffill().shift(1).fillna(0.0)
        return signals.div(len(prices.columns))
