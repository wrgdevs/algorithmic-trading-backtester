import numpy as np
import pandas as pd

from src.backtester.metrics import calculate_metrics, drawdown_series, value_at_risk, conditional_value_at_risk
from src.strategies.indicators import rsi


def test_drawdown_known_case():
    equity = pd.Series([100, 120, 90, 150, 75], dtype=float)
    dd = drawdown_series(equity)
    assert dd.iloc[0] == 0
    assert dd.iloc[2] == -0.25
    assert dd.iloc[4] == -0.5


def test_total_return_and_cagr_known_case():
    equity = pd.Series([100.0, 101.0, 102.01])
    metrics = calculate_metrics(equity)
    assert np.isclose(metrics['Total Return'], 0.0201)
    expected_cagr = (1.0201) ** (252 / 2) - 1
    assert np.isclose(metrics['Annual Return'], expected_cagr)


def test_var_and_cvar_known_case():
    returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])
    assert np.isclose(value_at_risk(returns, 0.2), -0.06)
    assert np.isclose(conditional_value_at_risk(returns, 0.2), -0.10)


def test_rsi_edge_cases():
    dates = pd.date_range('2024-01-01', periods=20)
    up = pd.DataFrame({'A': np.arange(1, 21, dtype=float)}, index=dates)
    down = pd.DataFrame({'A': np.arange(20, 0, -1, dtype=float)}, index=dates)
    flat = pd.DataFrame({'A': np.ones(20) * 10}, index=dates)
    assert rsi(up, 14).iloc[-1, 0] == 100
    assert rsi(down, 14).iloc[-1, 0] == 0
    assert rsi(flat, 14).iloc[-1, 0] == 50
