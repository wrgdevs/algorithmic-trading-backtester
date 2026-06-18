from __future__ import annotations

import pandas as pd
from .base import Strategy


class InverseVolatilityPortfolio(Strategy):
    name = 'Inverse Volatility Portfolio'

    def __init__(self, window: int = 63):
        self.window = window

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        vol = prices.pct_change().rolling(self.window).std()
        inv = 1 / vol.replace(0, pd.NA)
        weights = inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
        return weights
