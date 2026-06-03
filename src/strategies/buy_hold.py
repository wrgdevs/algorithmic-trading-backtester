from __future__ import annotations

import pandas as pd
from .base import Strategy


class BuyAndHold(Strategy):
    name = 'Buy and Hold'

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        weights = pd.DataFrame(self.weight / len(prices.columns), index=prices.index, columns=prices.columns)
        return weights
