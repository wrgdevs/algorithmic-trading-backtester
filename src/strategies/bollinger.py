from __future__ import annotations

import pandas as pd
from .base import Strategy
from .indicators import bollinger_bands


class BollingerMeanReversion(Strategy):
    name = 'Bollinger Mean Reversion'

    def __init__(self, window: int = 20, num_std: float = 2.0, exit_on_mean: bool = True):
        self.window = window
        self.num_std = num_std
        self.exit_on_mean = exit_on_mean

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        lower, mid, upper = bollinger_bands(prices, self.window, self.num_std)
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        in_position = pd.DataFrame(False, index=prices.index, columns=prices.columns)
        buy = prices < lower
        sell = prices > mid if self.exit_on_mean else prices > upper
        for i in range(1, len(prices)):
            in_position.iloc[i] = in_position.iloc[i - 1]
            in_position.iloc[i] = in_position.iloc[i].mask(buy.iloc[i], True)
            in_position.iloc[i] = in_position.iloc[i].mask(sell.iloc[i], False)
        weights[in_position] = 1.0
        weights = weights.shift(1).fillna(0.0)
        return weights.div(len(prices.columns))
