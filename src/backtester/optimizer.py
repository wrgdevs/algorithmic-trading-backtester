from __future__ import annotations

from itertools import product
from typing import Any, Callable

import pandas as pd

from .engine import BacktestEngine
from strategies.base import Strategy


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
    for values in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))
        try:
            strategy = strategy_factory(**params)
            result = BacktestEngine(**engine_kwargs).run(prices, strategy, benchmark=benchmark)
            row = {**params, **result['metrics']}
            rows.append(row)
        except Exception as exc:
            rows.append({**params, 'error': str(exc)})
    out = pd.DataFrame(rows)
    if objective in out.columns:
        out = out.sort_values(objective, ascending=False)
    return out.reset_index(drop=True)
