from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def _normalize_tickers(tickers: str | Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        return [t.strip().upper() for t in tickers.split(',') if t.strip()]
    return [str(t).strip().upper() for t in tickers if str(t).strip()]


def load_prices(
    tickers: str | Iterable[str],
    start: str = '2018-01-01',
    end: str | None = None,
    source: str = 'yfinance',
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load adjusted close prices for one or more tickers.

    Returns a DataFrame indexed by date with one column per ticker.
    CSV mode expects either a Date column plus ticker columns, or OHLCV rows with
    Date, Ticker and Adj Close/Close columns.
    """
    tickers_list = _normalize_tickers(tickers)
    if not tickers_list:
        raise ValueError('At least one ticker is required.')

    if source == 'csv':
        if csv_path is None:
            raise ValueError('csv_path is required when source="csv".')
        raw = pd.read_csv(csv_path, parse_dates=['Date'])
        if 'Ticker' in raw.columns:
            price_col = 'Adj Close' if 'Adj Close' in raw.columns else 'Close'
            prices = raw.pivot(index='Date', columns='Ticker', values=price_col)
        else:
            prices = raw.set_index('Date')
        prices = prices[[c for c in prices.columns if c.upper() in tickers_list]]
    else:
        raw = yf.download(tickers_list, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError('No data returned. Check tickers and date range.')
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw['Close']
        else:
            prices = raw[['Close']].rename(columns={'Close': tickers_list[0]})

    prices = prices.sort_index().ffill().dropna(how='all')
    prices.columns = [str(c).upper() for c in prices.columns]
    missing = [t for t in tickers_list if t not in prices.columns]
    if missing:
        raise ValueError(f'Missing price data for: {missing}')
    return prices[tickers_list].dropna()
