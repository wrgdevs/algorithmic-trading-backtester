from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class RegimeSwitchingStrategy(Strategy):
    """Risk-on/risk-off strategy using benchmark trend as a simple market regime filter."""

    name = 'Regime Switching Momentum'

    def __init__(self, trend_window: int = 200, momentum_lookback: int = 63, top_n: int = 2):
        self.trend_window = int(trend_window)
        self.momentum_lookback = int(momentum_lookback)
        self.top_n = int(top_n)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        benchmark = prices.iloc[:, 0]
        risk_on = benchmark > benchmark.rolling(self.trend_window).mean()
        momentum = prices.pct_change(self.momentum_lookback)
        ranks = momentum.rank(axis=1, ascending=False, method='first')
        selected = (ranks <= self.top_n) & (momentum > 0)
        weights = selected.astype(float)
        counts = weights.sum(axis=1).replace(0, np.nan)
        weights = weights.div(counts, axis=0).fillna(0.0)
        weights = weights.mul(risk_on.astype(float), axis=0)
        return weights.shift(1).fillna(0.0)
