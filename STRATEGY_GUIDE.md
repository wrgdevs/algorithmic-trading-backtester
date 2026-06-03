# Strategy Guide

This project uses a simple target-weight interface: every strategy returns a `pandas.DataFrame` with dates as rows, tickers as columns, and desired portfolio weights as values.

## Built-in strategies

- **Buy and Hold**: equal-weight long exposure.
- **Moving Average Crossover**: trend-following strategy based on short and long moving averages.
- **RSI Mean Reversion**: buys oversold assets and exits/reduces exposure when RSI becomes overbought.
- **MACD Trend Following**: uses MACD line versus signal line to identify trend direction.
- **Bollinger Mean Reversion**: buys assets below the lower band and exits near the mean.
- **Cross-Sectional Momentum**: ranks assets by lookback return and buys the strongest names.
- **Inverse Volatility Portfolio**: allocates more weight to lower-volatility assets.
- **Z-Score Mean Reversion**: buys statistically oversold assets and optionally shorts overbought assets.
- **Donchian Channel Breakout**: enters when price breaks above a prior high and exits on channel weakness.
- **Dual Momentum**: combines relative strength with an absolute return filter.
- **Pairs Trading Spread Reversion**: long/short market-neutral spread trading between two assets.
- **Regime Switching Momentum**: only takes risk-on momentum trades when the benchmark trend filter is positive.
- **Weighted Strategy Ensemble**: combines several strategies into one averaged target-weight model.
- **Custom Rule Strategy**: lets users define their own rule expressions in the dashboard.

## Custom rule examples

Variables available in custom rules:

```text
close, returns, sma_fast, sma_slow, ema_fast, ema_slow, rsi,
macd_line, macd_signal, macd_hist, bb_lower, bb_mid, bb_upper, zscore
```

Examples:

```text
(sma_fast > sma_slow) & (rsi < 70)
```

```text
(close < bb_lower) | (zscore < -1.5)
```

```text
(macd_hist > 0) & (close > ema_slow)
```

For long/short mode, you can also provide a short rule such as:

```text
(sma_fast < sma_slow) & (rsi > 30)
```

## Creating your own Python strategy

Create a new file in `src/strategies/`:

```python
from __future__ import annotations

import pandas as pd
from .base import Strategy


class MyStrategy(Strategy):
    name = "My Strategy"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        signal = prices > prices.rolling(50).mean()
        weights = signal.astype(float)
        counts = weights.sum(axis=1).replace(0, pd.NA)
        return weights.div(counts, axis=0).fillna(0.0)
```

Then import it in `app.py` or `run_backtest.py`.


## Improved Custom Rule Strategy

The custom strategy can be used in three ways from the dashboard: preset templates, a visual rule builder, or manual expressions. This lets you describe strategies without writing a new Python class every time.

Example rules:

```python
(sma_fast > sma_slow) & (rsi < 70)
(close < bb_lower) & (zscore < -1.5)
(momentum_63 > 0) & (rolling_vol < rolling_vol_slow)
```

The Signal Diagnostics tab helps explain what the custom rule is doing by showing a heatmap of generated target weights and the percent of days with active signals.