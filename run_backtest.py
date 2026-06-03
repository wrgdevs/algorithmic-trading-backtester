from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / 'src'))

from backtester.engine import BacktestEngine
from backtester.optimizer import grid_search
from backtester.reporting import export_report
from data.loader import load_prices
from strategies.moving_average import MovingAverageCrossover


def main() -> None:
    prices = load_prices(['AAPL', 'MSFT', 'NVDA', 'SPY'], start='2019-01-01')
    benchmark = load_prices(['SPY'], start='2019-01-01')
    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    engine = BacktestEngine(
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
        max_weight=0.5,
        max_gross_exposure=1.0,
        rebalance_frequency='W',
        volatility_target=0.15,
    )
    result = engine.run(prices, strategy, benchmark=benchmark)

    print('\nPerformance Metrics')
    print(pd.Series(result['metrics']).to_string())

    paths = export_report(result, ROOT / 'reports')
    print('\nSaved report files:')
    for label, path in paths.items():
        print(f'- {label}: {path}')

    print('\nSmall MA parameter sweep')
    opt = grid_search(
        prices,
        MovingAverageCrossover,
        {'short_window': [10, 20, 30], 'long_window': [50, 100, 150]},
        engine_kwargs={'initial_cash': 100_000, 'commission': 0.001, 'slippage': 0.0005, 'max_weight': 0.5},
        benchmark=benchmark,
    )
    print(opt.head().to_string(index=False))
    opt.to_csv(ROOT / 'reports' / 'optimizer_results.csv', index=False)


if __name__ == '__main__':
    main()
