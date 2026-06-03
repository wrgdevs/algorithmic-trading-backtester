from __future__ import annotations

import numpy as np
import pandas as pd
from .base import Strategy


class CrossSectionalMomentum(Strategy):
    name = 'Cross-Sectional Momentum'

    def __init__(self, lookback: int = 63, top_n: int = 2):
        self.lookback = lookback
        self.top_n = top_n

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        momentum = prices.pct_change(self.lookback)
        ranks = momentum.rank(axis=1, ascending=False, method='first')
        selected = (ranks <= min(self.top_n, len(prices.columns))).astype(float)
        weights = selected.div(selected.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        return weights.shift(1).fillna(0.0)
