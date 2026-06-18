from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class EnsembleStrategy(Strategy):
    """Combines multiple strategies by averaging their target weights."""

    name = 'Weighted Strategy Ensemble'

    def __init__(self, strategies: list[Strategy], weights: list[float] | None = None):
        if not strategies:
            raise ValueError('EnsembleStrategy needs at least one strategy.')
        if weights is not None and len(weights) != len(strategies):
            raise ValueError('weights must have one value per strategy.')
        self.strategies = strategies
        self.ensemble_weights = np.array(weights if weights is not None else [1.0] * len(strategies), dtype=float)
        if not np.isfinite(self.ensemble_weights).all() or (self.ensemble_weights < 0).any():
            raise ValueError('Ensemble weights must be finite and non-negative.')
        if self.ensemble_weights.sum() <= 0:
            raise ValueError('Ensemble weights must have a positive sum.')
        self.ensemble_weights = self.ensemble_weights / self.ensemble_weights.sum()

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for weight, strategy in zip(self.ensemble_weights, self.strategies):
            combined = combined.add(strategy.generate_signals(prices).reindex_like(combined).fillna(0.0) * weight, fill_value=0.0)
        gross = combined.abs().sum(axis=1).replace(0, np.nan)
        scale = (1.0 / gross).clip(upper=1.0).fillna(1.0)
        return combined.mul(scale, axis=0)
