# Custom Rule Builder

The Streamlit dashboard now includes a no-code custom strategy designer.

## Modes

### Preset
Choose a ready-made template, then edit the generated rules.

Included presets:
- Trend + RSI filter
- Buy the dip in uptrend
- MACD confirmation
- Bollinger snapback
- Low volatility breakout
- Defensive momentum

### Visual Builder
Create rules by stacking conditions such as:

```text
sma_fast > sma_slow
rsi < 70
momentum_63 > 0
```

You can combine conditions with `AND` or `OR`. The app shows the generated rule before the backtest runs.

### Manual
Write pandas-style boolean rules directly:

```python
(close > sma_slow) & (rsi < 70) & (momentum_63 > 0)
```

Use `&`, `|`, `~`, and parentheses for complex logic.

## Available variables

- `close`: adjusted close price
- `returns`: daily percentage return
- `sma_fast`, `sma_slow`: simple moving averages
- `ema_fast`, `ema_slow`: exponential moving averages
- `rsi`: Relative Strength Index
- `macd_line`, `macd_signal`, `macd_hist`: MACD values
- `bb_lower`, `bb_mid`, `bb_upper`: Bollinger Bands
- `zscore`: rolling price z-score
- `momentum_21`, `momentum_63`, `momentum_126`: lookback returns
- `rolling_vol`, `rolling_vol_slow`: annualized volatility
- `rolling_high_20`, `rolling_low_20`: previous 20-day channel levels
- `drawdown`: current asset drawdown from its running high

## Safety and validation

Rules are checked before execution. Unsafe tokens such as imports, builtins, `eval`, `exec`, and dunder access are blocked. Unknown variables are also rejected with a readable error.
