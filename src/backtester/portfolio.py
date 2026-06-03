from __future__ import annotations

import numpy as np
import pandas as pd


class Portfolio:
    """Vectorized portfolio simulator using target weights.

    Features:
    - commission and slippage costs
    - optional long-only clipping
    - max per-asset weight and max gross exposure controls
    - volatility targeting
    - configurable rebalancing frequency
    - cash exposure tracking
    """

    def __init__(
        self,
        initial_cash: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_weight: float = 1.0,
        max_gross_exposure: float = 1.0,
        rebalance_frequency: str = 'D',
        long_only: bool = True,
        volatility_target: float | None = None,
        volatility_window: int = 20,
    ):
        self.initial_cash = float(initial_cash)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.max_weight = float(max_weight)
        self.max_gross_exposure = float(max_gross_exposure)
        self.rebalance_frequency = rebalance_frequency
        self.long_only = bool(long_only)
        self.volatility_target = volatility_target
        self.volatility_window = int(volatility_window)

    def _apply_risk_controls(self, weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
        weights = weights.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        lower = 0.0 if self.long_only else -self.max_weight
        weights = weights.clip(lower=lower, upper=self.max_weight)

        gross = weights.abs().sum(axis=1).replace(0, np.nan)
        scale = (self.max_gross_exposure / gross).clip(upper=1.0).fillna(1.0)
        weights = weights.mul(scale, axis=0)

        if self.volatility_target:
            raw_returns = (weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1)
            realized_vol = raw_returns.rolling(self.volatility_window).std() * np.sqrt(252)
            vol_scale = (self.volatility_target / realized_vol).clip(upper=1.5).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            weights = weights.mul(vol_scale, axis=0)
            gross = weights.abs().sum(axis=1).replace(0, np.nan)
            scale = (self.max_gross_exposure / gross).clip(upper=1.0).fillna(1.0)
            weights = weights.mul(scale, axis=0)
        return weights

    def _apply_rebalance_frequency(self, weights: pd.DataFrame) -> pd.DataFrame:
        freq = self.rebalance_frequency.upper()
        if freq in {'D', 'DAILY'}:
            return weights
        if freq in {'W', 'WEEKLY'}:
            mask = weights.index.to_series().dt.to_period('W').ne(weights.index.to_series().dt.to_period('W').shift(1))
        elif freq in {'M', 'MONTHLY'}:
            mask = weights.index.to_series().dt.to_period('M').ne(weights.index.to_series().dt.to_period('M').shift(1))
        else:
            raise ValueError('rebalance_frequency must be D, W, or M.')
        rebalanced = weights.where(mask).ffill().fillna(0.0)
        return rebalanced

    def simulate(self, prices: pd.DataFrame, target_weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        prices = prices.copy().ffill().dropna()
        asset_returns = prices.pct_change().fillna(0.0)
        weights = target_weights.reindex(prices.index).fillna(0.0)
        weights = self._apply_risk_controls(weights, asset_returns)
        weights = self._apply_rebalance_frequency(weights)

        turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
        trading_costs = turnover * (self.commission + self.slippage)
        portfolio_returns_before_costs = (weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1)
        portfolio_returns = portfolio_returns_before_costs - trading_costs
        equity = self.initial_cash * (1 + portfolio_returns).cumprod()

        positions_value = weights.mul(equity, axis=0)
        shares = positions_value / prices
        trades = shares.diff().fillna(shares)
        trade_log = trades.stack().reset_index()
        trade_log.columns = ['Date', 'Ticker', 'Shares']
        price_frame = prices.copy()
        price_frame.index.name = 'Date'
        price_frame.columns.name = 'Ticker'
        price_lookup = price_frame.stack().rename('Price')
        trade_log = trade_log.merge(price_lookup.reset_index(), on=['Date', 'Ticker'], how='left')
        trade_log['Side'] = np.where(trade_log['Shares'] > 0, 'BUY', 'SELL')
        trade_log['Notional'] = trade_log['Shares'] * trade_log['Price']
        trade_log['Estimated Cost'] = trade_log['Notional'].abs() * (self.commission + self.slippage)
        trade_log = trade_log[trade_log['Shares'].abs() > 1e-8]

        history = pd.DataFrame({
            'Equity': equity,
            'Portfolio Return': portfolio_returns,
            'Return Before Costs': portfolio_returns_before_costs,
            'Turnover': turnover,
            'Trading Cost': trading_costs,
            'Cash Weight': 1 - weights.sum(axis=1),
            'Gross Exposure': weights.abs().sum(axis=1),
        })
        for col in weights.columns:
            history[f'Weight {col}'] = weights[col]
        return history, trade_log
