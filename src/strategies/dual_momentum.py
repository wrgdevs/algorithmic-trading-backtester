from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class DualMomentum(Strategy):
    """Ranks assets by momentum, then holds only those above a cash/zero-return filter."""

    name = 'Dual Momentum'

    def __init__(self, lookback: int = 126, top_n: int = 2, min_return: float = 0.0):
        self.lookback = int(lookback)
        self.top_n = int(top_n)
        self.min_return = float(min_return)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        momentum = prices.pct_change(self.lookback)
        ranks = momentum.rank(axis=1, ascending=False, method='first')
        selected = (ranks <= self.top_n) & (momentum > self.min_return)
        weights = selected.astype(float)
        counts = weights.sum(axis=1).replace(0, np.nan)
        return weights.div(counts, axis=0).fillna(0.0)
