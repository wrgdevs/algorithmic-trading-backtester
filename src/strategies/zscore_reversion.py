from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import zscore


class ZScoreMeanReversion(Strategy):
    """Long oversold assets and optionally short overbought assets using rolling z-scores."""

    name = 'Z-Score Mean Reversion'

    def __init__(self, window: int = 20, entry_z: float = 1.5, exit_z: float = 0.25, long_only: bool = True):
        self.window = int(window)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.long_only = bool(long_only)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        z = zscore(prices, self.window)
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        long_signal = z < -self.entry_z
        exit_long = z > -self.exit_z
        short_signal = z > self.entry_z
        exit_short = z < self.exit_z

        state = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for i in range(1, len(prices)):
            prev = state.iloc[i - 1].copy()
            current = prev.copy()
            current[long_signal.iloc[i]] = 1.0
            current[exit_long.iloc[i] & (prev > 0)] = 0.0
            if not self.long_only:
                current[short_signal.iloc[i]] = -1.0
                current[exit_short.iloc[i] & (prev < 0)] = 0.0
            state.iloc[i] = current

        gross = state.abs().sum(axis=1).replace(0, np.nan)
        weights = state.div(gross, axis=0).fillna(0.0)
        return weights
