from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import zscore


class PairsTradingStrategy(Strategy):
    """Simple two-asset spread mean reversion using price-ratio z-score."""

    name = 'Pairs Trading Spread Reversion'

    def __init__(self, asset_a: str | None = None, asset_b: str | None = None, window: int = 30, entry_z: float = 1.5, exit_z: float = 0.25):
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.window = int(window)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        if len(prices.columns) < 2:
            raise ValueError('PairsTradingStrategy requires at least two assets.')
        a = self.asset_a or prices.columns[0]
        b = self.asset_b or prices.columns[1]
        if a not in prices.columns or b not in prices.columns:
            raise ValueError(f'Pair assets must exist in prices. Got {a}, {b}.')
        if a == b:
            raise ValueError('PairsTradingStrategy requires two different assets.')

        ratio = (prices[a] / prices[b]).to_frame('spread')
        spread_z = zscore(ratio, self.window)['spread']
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        position = 0.0
        for i in range(1, len(prices)):
            z = spread_z.iloc[i]
            if pd.isna(z):
                pass
            elif z > self.entry_z:
                position = -1.0  # short A, long B
            elif z < -self.entry_z:
                position = 1.0   # long A, short B
            elif abs(z) < self.exit_z:
                position = 0.0
            weights.iloc[i, weights.columns.get_loc(a)] = 0.5 * position
            weights.iloc[i, weights.columns.get_loc(b)] = -0.5 * position
        return weights.shift(1).fillna(0.0)
