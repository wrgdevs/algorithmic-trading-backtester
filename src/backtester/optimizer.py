from __future__ import annotations

from itertools import product
from typing import Any, Callable

import pandas as pd

from .engine import BacktestEngine
from .metrics import calculate_metrics

try:
    from strategies.base import Strategy
except ModuleNotFoundError:
    from src.strategies.base import Strategy


def grid_search(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Strategy],
    param_grid: dict[str, list[Any]],
    engine_kwargs: dict[str, Any] | None = None,
    benchmark: pd.DataFrame | None = None,
    objective: str = 'Sharpe Ratio',
) -> pd.DataFrame:
    """Run a simple parameter sweep and rank strategies by one metric."""
    engine_kwargs = engine_kwargs or {}
    rows: list[dict[str, Any]] = []
    keys = list(param_grid)
    engine = BacktestEngine(**engine_kwargs)
    for values in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))
        try:
            strategy = strategy_factory(**params)
            result = engine.run(prices, strategy, benchmark=benchmark)
            row = {**params, **result['metrics']}
            rows.append(row)
        except Exception as exc:
            rows.append({**params, 'error': str(exc)})
    out = pd.DataFrame(rows)
    if objective in out.columns:
        out = out.sort_values(objective, ascending=False)
    return out.reset_index(drop=True)


def walk_forward_search(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Strategy],
    param_grid: dict[str, list[Any]],
    train_size: int,
    test_size: int,
    engine_kwargs: dict[str, Any] | None = None,
    benchmark: pd.DataFrame | None = None,
    objective: str = 'Sharpe Ratio',
    step_size: int | None = None,
) -> pd.DataFrame:
    """Select parameters on rolling training windows and score unseen windows.

    The strategy receives the training window as warm-up context when generating
    each test result. Only observations in the following test window contribute
    to the reported out-of-sample metrics.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError('prices must be a non-empty pandas DataFrame.')
    if train_size < 2 or test_size < 2:
        raise ValueError('train_size and test_size must both be at least 2.')
    step = test_size if step_size is None else int(step_size)
    if step < 1:
        raise ValueError('step_size must be positive.')
    if train_size + test_size > len(prices):
        raise ValueError('Not enough price rows for one train/test fold.')

    engine_kwargs = engine_kwargs or {}
    engine = BacktestEngine(**engine_kwargs)
    rows: list[dict[str, Any]] = []
    fold = 1
    for start in range(0, len(prices) - train_size - test_size + 1, step):
        train_end = start + train_size
        test_end = train_end + test_size
        train_prices = prices.iloc[start:train_end]
        train_benchmark = benchmark.reindex(train_prices.index) if benchmark is not None else None
        ranked = grid_search(
            train_prices,
            strategy_factory,
            param_grid,
            engine_kwargs=engine_kwargs,
            benchmark=train_benchmark,
            objective=objective,
        )
        if objective not in ranked or ranked[objective].dropna().empty:
            rows.append({'Fold': fold, 'error': f'No valid training result for {objective}.'})
            fold += 1
            continue

        best = ranked.loc[ranked[objective].first_valid_index()]
        # A mixed result DataFrame may upcast integer parameters to floats.
        # Recover the original grid objects so rolling-window strategies still
        # receive integers (for example, 20 rather than 20.0).
        params = {
            key: next(candidate for candidate in param_grid[key] if candidate == best[key])
            for key in param_grid
        }
        context_prices = prices.iloc[start:test_end]
        context_benchmark = benchmark.reindex(context_prices.index) if benchmark is not None else None
        result = engine.run(context_prices, strategy_factory(**params), benchmark=context_benchmark)

        # Include the last training observation as the test curve's baseline so
        # the first out-of-sample return is measured rather than discarded.
        test_history = result['history'].iloc[train_size - 1:]
        test_benchmark_curve = test_history.get('Benchmark')
        test_metrics = calculate_metrics(
            test_history['Equity'],
            test_benchmark_curve,
            risk_free_rate=engine.risk_free_rate,
        )
        rows.append({
            'Fold': fold,
            'Train Start': train_prices.index[0],
            'Train End': train_prices.index[-1],
            'Test Start': prices.index[train_end],
            'Test End': prices.index[test_end - 1],
            **params,
            f'Train {objective}': float(best[objective]),
            **{f'Test {key}': value for key, value in test_metrics.items()},
        })
        fold += 1
    return pd.DataFrame(rows)
