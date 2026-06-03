from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

import numpy as np
import pandas as pd

from strategies.custom import CustomRuleStrategy
from strategies.donchian import DonchianBreakout
from strategies.dual_momentum import DualMomentum
from strategies.ensemble import EnsembleStrategy
from strategies.moving_average import MovingAverageCrossover
from strategies.pairs import PairsTradingStrategy
from strategies.zscore_reversion import ZScoreMeanReversion


def sample_prices() -> pd.DataFrame:
    idx = pd.date_range('2021-01-01', periods=180, freq='B')
    base = np.linspace(100, 150, len(idx))
    return pd.DataFrame({
        'AAA': base + np.sin(np.arange(len(idx))) * 2,
        'BBB': 100 + np.cos(np.arange(len(idx)) / 3) * 3 + np.linspace(0, 20, len(idx)),
        'CCC': 80 + np.sin(np.arange(len(idx)) / 5) * 4,
    }, index=idx)


def assert_valid_weights(weights: pd.DataFrame, prices: pd.DataFrame):
    assert list(weights.columns) == list(prices.columns)
    assert weights.index.equals(prices.index)
    assert np.isfinite(weights.to_numpy()).all()
    assert (weights.abs().sum(axis=1) <= 1.000001).all()


def test_extra_strategies_generate_valid_weights():
    prices = sample_prices()
    strategies = [
        ZScoreMeanReversion(window=20),
        DonchianBreakout(entry_window=30, exit_window=10),
        DualMomentum(lookback=30, top_n=2),
        CustomRuleStrategy('(sma_fast > sma_slow) & (rsi < 80)', fast_window=10, slow_window=30),
        EnsembleStrategy([MovingAverageCrossover(10, 30), DualMomentum(30, 1)]),
    ]
    for strategy in strategies:
        assert_valid_weights(strategy.generate_signals(prices), prices)


def test_pairs_strategy_is_market_neutral():
    prices = sample_prices()[['AAA', 'BBB']]
    weights = PairsTradingStrategy(window=20, entry_z=0.5).generate_signals(prices)
    assert_valid_weights(weights, prices)
    assert np.allclose(weights.sum(axis=1), 0.0)


def test_custom_rule_builder_and_new_variables():
    from strategies.custom import CustomRuleStrategy
    prices = sample_prices()

    rule = CustomRuleStrategy.build_rule([
        ('momentum_21', '>', '0'),
        ('rolling_vol', '<', 'rolling_vol_slow'),
    ], 'AND')
    strategy = CustomRuleStrategy(long_rule=rule, fast_window=10, slow_window=30, z_window=20)
    signals = strategy.generate_signals(prices)
    assert list(signals.columns) == list(prices.columns)
    assert signals.abs().sum(axis=1).max() <= 1.0


def test_custom_rule_validation_blocks_unsafe_code():
    from strategies.custom import CustomRuleStrategy

    try:
        CustomRuleStrategy.validate_rule('__import__("os").system("echo bad")')
    except ValueError as exc:
        assert 'Blocked unsafe token' in str(exc)
    else:
        raise AssertionError('unsafe custom rule should have been rejected')
