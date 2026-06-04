from __future__ import annotations

import re
import numpy as np
import pandas as pd

from .base import Strategy
from .indicators import bollinger_bands, macd, rsi, zscore


CUSTOM_PRESETS: dict[str, dict[str, str]] = {
    'Trend + RSI filter': {
        'long_rule': '(sma_fast > sma_slow) & (rsi < 70)',
        'short_rule': '(sma_fast < sma_slow) & (rsi > 30)',
        'description': 'Classic trend-following with an RSI filter to avoid buying into extreme overbought moves.',
    },
    'Buy the dip in uptrend': {
        'long_rule': '(close > sma_slow) & (rsi < 35)',
        'short_rule': '(close < sma_slow) & (rsi > 65)',
        'description': 'Mean reversion entries only when the asset is still above its longer-term trend.',
    },
    'MACD confirmation': {
        'long_rule': '(macd_hist > 0) & (ema_fast > ema_slow)',
        'short_rule': '(macd_hist < 0) & (ema_fast < ema_slow)',
        'description': 'Momentum confirmation using both MACD histogram and EMA trend direction.',
    },
    'Bollinger snapback': {
        'long_rule': '(close < bb_lower) & (zscore < -1.5)',
        'short_rule': '(close > bb_upper) & (zscore > 1.5)',
        'description': 'Mean-reversion setup looking for stretched prices outside Bollinger Bands.',
    },
    'Low volatility breakout': {
        'long_rule': '(close > rolling_high_20) & (rolling_vol < rolling_vol_slow)',
        'short_rule': '(close < rolling_low_20) & (rolling_vol < rolling_vol_slow)',
        'description': 'Breakout after quiet volatility regimes, similar to volatility compression logic.',
    },
    'Defensive momentum': {
        'long_rule': '(momentum_63 > 0) & (close > sma_slow) & (drawdown > -0.15)',
        'short_rule': '(momentum_63 < 0) & (close < sma_slow)',
        'description': 'Momentum strategy that avoids deeply damaged trends.',
    },
}


CUSTOM_VARIABLES = {
    'close': 'Adjusted close price.',
    'returns': 'Daily percentage return.',
    'sma_fast': 'Simple moving average using fast_window.',
    'sma_slow': 'Simple moving average using slow_window.',
    'ema_fast': 'Exponential moving average using fast_window.',
    'ema_slow': 'Exponential moving average using slow_window.',
    'rsi': 'Relative Strength Index.',
    'macd_line': 'MACD line.',
    'macd_signal': 'MACD signal line.',
    'macd_hist': 'MACD histogram.',
    'bb_lower': 'Lower Bollinger Band.',
    'bb_mid': 'Middle Bollinger Band.',
    'bb_upper': 'Upper Bollinger Band.',
    'zscore': 'Rolling z-score of price.',
    'momentum_21': '21-trading-day return.',
    'momentum_63': '63-trading-day return.',
    'momentum_126': '126-trading-day return.',
    'rolling_vol': 'Annualized volatility over z_window.',
    'rolling_vol_slow': 'Annualized volatility over slow_window.',
    'rolling_high_20': 'Yesterday\'s 20-day rolling high.',
    'rolling_low_20': 'Yesterday\'s 20-day rolling low.',
    'drawdown': 'Current drawdown from rolling all-time high.',
}


class CustomRuleStrategy(Strategy):
    """User-defined rules evaluated against a safe indicator namespace.

    Use pandas-style boolean expressions such as:
    ``(sma_fast > sma_slow) & (rsi < 70)``.
    """

    name = 'Custom Rule Strategy'

    def __init__(
        self,
        long_rule: str = 'sma_fast > sma_slow',
        short_rule: str = '',
        fast_window: int = 20,
        slow_window: int = 50,
        rsi_window: int = 14,
        z_window: int = 20,
        long_only: bool = True,
    ):
        self.long_rule = long_rule
        self.short_rule = short_rule
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self.rsi_window = int(rsi_window)
        self.z_window = int(z_window)
        self.long_only = bool(long_only)

    @staticmethod
    def _normalize_rule(rule: str) -> str:
        normalized = rule.strip()
        normalized = re.sub(r'\bAND\b', '&', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bOR\b', '|', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bNOT\b', '~', normalized, flags=re.IGNORECASE)
        return normalized

    @staticmethod
    def validate_rule(rule: str) -> None:
        if not rule.strip():
            return
        banned = ['__', 'import', 'open(', 'exec(', 'eval(', 'compile(', 'globals', 'locals', 'lambda', 'class ', 'def ']
        lowered = rule.lower()
        for token in banned:
            if token in lowered:
                raise ValueError(f'Blocked unsafe token in custom rule: {token}')
        allowed_words = set(CUSTOM_VARIABLES) | {'and', 'or', 'not', 'AND', 'OR', 'NOT'}
        words = set(re.findall(r'[A-Za-z_]\w*', rule))
        unknown = sorted(w for w in words if w not in allowed_words)
        if unknown:
            raise ValueError(f'Unknown variable(s): {", ".join(unknown)}')

    @staticmethod
    def build_rule(conditions: list[tuple[str, str, str]], joiner: str = 'AND') -> str:
        """Build a boolean expression from UI-style condition tuples.

        Each condition is ``(left_variable, operator, right_value_or_variable)``.
        """
        pieces = []
        for left, op, right in conditions:
            if not left or not op or right in {'', None}:
                continue
            pieces.append(f'({left} {op} {right})')
        connector = ' & ' if joiner.upper() == 'AND' else ' | '
        return connector.join(pieces)

    def _context(self, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
        returns = prices.pct_change()
        bb_lower, bb_mid, bb_upper = bollinger_bands(prices, self.z_window, 2.0)
        macd_line, macd_signal, macd_hist = macd(prices)
        running_peak = prices.cummax().replace(0, np.nan)
        return {
            'close': prices,
            'returns': returns,
            'sma_fast': prices.rolling(self.fast_window).mean(),
            'sma_slow': prices.rolling(self.slow_window).mean(),
            'ema_fast': prices.ewm(span=self.fast_window, adjust=False).mean(),
            'ema_slow': prices.ewm(span=self.slow_window, adjust=False).mean(),
            'rsi': rsi(prices, self.rsi_window),
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'bb_lower': bb_lower,
            'bb_mid': bb_mid,
            'bb_upper': bb_upper,
            'zscore': zscore(prices, self.z_window),
            'momentum_21': prices.pct_change(21),
            'momentum_63': prices.pct_change(63),
            'momentum_126': prices.pct_change(126),
            'rolling_vol': returns.rolling(self.z_window).std() * np.sqrt(252),
            'rolling_vol_slow': returns.rolling(self.slow_window).std() * np.sqrt(252),
            'rolling_high_20': prices.rolling(20).max().shift(1),
            'rolling_low_20': prices.rolling(20).min().shift(1),
            'drawdown': prices / running_peak - 1,
        }

    def _eval_rule(self, rule: str, prices: pd.DataFrame) -> pd.DataFrame:
        if not rule.strip():
            return pd.DataFrame(False, index=prices.index, columns=prices.columns)
        self.validate_rule(rule)
        context = self._context(prices)
        safe_rule = self._normalize_rule(rule)
        try:
            result = eval(safe_rule, {'__builtins__': {}}, context)  # noqa: S307 - restricted namespace, validated names
        except Exception as exc:
            raise ValueError(f'Invalid custom rule: {rule}. Error: {exc}') from exc
        if isinstance(result, pd.Series):
            result = pd.concat([result] * len(prices.columns), axis=1)
            result.columns = prices.columns
        if not isinstance(result, pd.DataFrame):
            raise ValueError('Custom rule must evaluate to a pandas DataFrame or Series.')
        missing = set(result.columns) - set(prices.columns)
        if missing:
            raise ValueError(f'Custom rule returned unknown columns: {missing}')
        return result.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        long_signal = self._eval_rule(self.long_rule, prices)
        short_signal = self._eval_rule(self.short_rule, prices) if not self.long_only else pd.DataFrame(False, index=prices.index, columns=prices.columns)
        raw = long_signal.astype(float) - short_signal.astype(float)
        gross = raw.abs().sum(axis=1).replace(0, np.nan)
        weights = raw.div(gross, axis=0).fillna(0.0)
        return weights.shift(1).fillna(0.0)
