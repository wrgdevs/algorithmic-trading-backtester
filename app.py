from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / 'src'))

from backtester.engine import BacktestEngine
from backtester.optimizer import grid_search
from backtester.reporting import export_report
from data.loader import load_prices
from strategies.buy_hold import BuyAndHold
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi import RSIStrategy
from strategies.macd import MACDTrendStrategy
from strategies.bollinger import BollingerMeanReversion
from strategies.momentum import CrossSectionalMomentum
from strategies.volatility import InverseVolatilityPortfolio
from strategies.zscore_reversion import ZScoreMeanReversion
from strategies.donchian import DonchianBreakout
from strategies.dual_momentum import DualMomentum
from strategies.pairs import PairsTradingStrategy
from strategies.custom import CUSTOM_PRESETS, CUSTOM_VARIABLES, CustomRuleStrategy
from strategies.regime import RegimeSwitchingStrategy
from strategies.ensemble import EnsembleStrategy

st.set_page_config(page_title='Algorithmic Trading Backtester', layout='wide')
st.title('Algorithmic Trading Backtester')
st.caption('Python · Pandas · NumPy · Plotly · Streamlit · Portfolio Analytics')


def parse_tickers(raw: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in raw.replace('\n', ',').split(','):
        ticker = token.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def safe_top_n_slider(label: str, asset_count: int, default: int = 2) -> int:
    max_assets = max(1, int(asset_count))
    return st.slider(label, 1, max_assets, min(default, max_assets))


with st.sidebar:
    st.header('Backtest Setup')
    tickers = st.text_input('Tickers', value='AAPL,MSFT,NVDA,SPY', help='Comma-separated symbols, for example: AAPL,MSFT,NVDA,SPY')
    ticker_list = parse_tickers(tickers)
    benchmark_ticker = st.text_input('Benchmark', value='SPY')
    start = st.date_input('Start date', value=pd.to_datetime('2019-01-01'))
    end = st.date_input('End date', value=pd.Timestamp.today())
    data_source = st.radio('Price data source', ['Yahoo Finance', 'CSV upload'], horizontal=True)
    csv_file = st.file_uploader('Upload price CSV', type=['csv'], help='Wide CSV: Date,AAPL,MSFT... or long CSV: Date,Ticker,Close/Adj Close.') if data_source == 'CSV upload' else None

    input_errors: list[str] = []
    if not ticker_list:
        input_errors.append('Enter at least one ticker.')
    if pd.to_datetime(end) <= pd.to_datetime(start):
        input_errors.append('End date must be after start date.')
    if data_source == 'CSV upload' and csv_file is None:
        input_errors.append('Upload a CSV file or switch back to Yahoo Finance.')

    st.header('Execution Model')
    initial_cash = st.number_input('Initial cash', min_value=1_000, value=100_000, step=10_000)
    commission = st.number_input('Commission', min_value=0.0, value=0.001, step=0.0005, format='%.4f')
    slippage = st.number_input('Slippage', min_value=0.0, value=0.0005, step=0.0005, format='%.4f')
    rebalance_frequency = st.selectbox('Rebalance frequency', ['D', 'W', 'M'], format_func=lambda x: {'D':'Daily','W':'Weekly','M':'Monthly'}[x])

    st.header('Risk Controls')
    long_only = st.checkbox('Long only', value=True)
    max_weight = st.slider('Max weight per asset', 0.05, 1.0, 0.5, 0.05)
    max_gross = st.slider('Max gross exposure', 0.1, 2.0, 1.0, 0.1)
    use_vol_target = st.checkbox('Use volatility targeting')
    vol_target = st.slider('Annual volatility target', 0.05, 0.50, 0.15, 0.01) if use_vol_target else None
    risk_free_rate = st.number_input('Risk-free rate', min_value=0.0, value=0.0, step=0.005, format='%.3f')

    st.header('Strategy')
    strategy_name = st.selectbox('Strategy', [
        'Moving Average Crossover',
        'RSI Mean Reversion',
        'MACD Trend Following',
        'Bollinger Mean Reversion',
        'Cross-Sectional Momentum',
        'Inverse Volatility Portfolio',
        'Z-Score Mean Reversion',
        'Donchian Channel Breakout',
        'Dual Momentum',
        'Pairs Trading Spread Reversion',
        'Regime Switching Momentum',
        'Custom Rule Strategy',
        'Ensemble: Trend + Mean Reversion',
        'Buy and Hold',
    ])

    if strategy_name == 'Moving Average Crossover':
        short = st.slider('Short MA window', 5, 100, 20)
        long = st.slider('Long MA window', short + 1, 250, 50)
        strategy = MovingAverageCrossover(short_window=short, long_window=long, long_only=long_only)
    elif strategy_name == 'RSI Mean Reversion':
        window = st.slider('RSI window', 5, 40, 14)
        oversold = st.slider('Oversold threshold', 10, 45, 30)
        overbought = st.slider('Overbought threshold', 55, 90, 70)
        strategy = RSIStrategy(window=window, oversold=oversold, overbought=overbought)
    elif strategy_name == 'MACD Trend Following':
        fast = st.slider('Fast EMA', 5, 20, 12)
        slow = st.slider('Slow EMA', fast + 1, 60, 26)
        signal = st.slider('Signal EMA', 3, 20, 9)
        strategy = MACDTrendStrategy(fast=fast, slow=slow, signal=signal, long_only=long_only)
    elif strategy_name == 'Bollinger Mean Reversion':
        window = st.slider('Bollinger window', 10, 80, 20)
        num_std = st.slider('Standard deviations', 1.0, 3.5, 2.0, 0.25)
        strategy = BollingerMeanReversion(window=window, num_std=num_std)
    elif strategy_name == 'Cross-Sectional Momentum':
        lookback = st.slider('Momentum lookback', 20, 252, 63)
        top_n = safe_top_n_slider('Top N assets', len(ticker_list), 2)
        strategy = CrossSectionalMomentum(lookback=lookback, top_n=top_n)
    elif strategy_name == 'Inverse Volatility Portfolio':
        window = st.slider('Volatility window', 20, 252, 63)
        strategy = InverseVolatilityPortfolio(window=window)
    elif strategy_name == 'Z-Score Mean Reversion':
        window = st.slider('Z-score window', 10, 120, 20)
        entry_z = st.slider('Entry z-score', 0.5, 3.5, 1.5, 0.25)
        exit_z = st.slider('Exit z-score', 0.0, 1.5, 0.25, 0.05)
        strategy = ZScoreMeanReversion(window=window, entry_z=entry_z, exit_z=exit_z, long_only=long_only)
    elif strategy_name == 'Donchian Channel Breakout':
        entry_window = st.slider('Entry breakout window', 20, 252, 55)
        exit_window = st.slider('Exit channel window', 5, min(entry_window, 100), 20)
        strategy = DonchianBreakout(entry_window=entry_window, exit_window=exit_window, long_only=long_only)
    elif strategy_name == 'Dual Momentum':
        lookback = st.slider('Momentum lookback', 20, 252, 126)
        top_n = safe_top_n_slider('Top N assets', len(ticker_list), 2)
        min_return = st.slider('Minimum lookback return', -0.20, 0.30, 0.0, 0.01)
        strategy = DualMomentum(lookback=lookback, top_n=top_n, min_return=min_return)
    elif strategy_name == 'Pairs Trading Spread Reversion':
        if len(ticker_list) < 2:
            st.warning('Pairs trading needs at least two tickers.')
            input_errors.append('Pairs trading needs at least two tickers.')
            strategy = BuyAndHold()
        else:
            first = st.selectbox('First pair asset', ticker_list, index=0)
            second_options = [ticker for ticker in ticker_list if ticker != first]
            second = st.selectbox('Second pair asset', second_options, index=0)
            window = st.slider('Spread z-score window', 10, 120, 30)
            entry_z = st.slider('Pair entry z-score', 0.5, 3.5, 1.5, 0.25)
            exit_z = st.slider('Pair exit z-score', 0.0, 1.5, 0.25, 0.05)
            strategy = PairsTradingStrategy(first, second, window=window, entry_z=entry_z, exit_z=exit_z)
    elif strategy_name == 'Regime Switching Momentum':
        trend_window = st.slider('Market regime trend window', 50, 300, 200)
        lookback = st.slider('Risk-on momentum lookback', 20, 252, 63)
        top_n = safe_top_n_slider('Risk-on top N assets', len(ticker_list), 2)
        strategy = RegimeSwitchingStrategy(trend_window=trend_window, momentum_lookback=lookback, top_n=top_n)
    elif strategy_name == 'Custom Rule Strategy':
        st.caption('No-code rule designer with presets, validation, and advanced manual mode.')
        custom_mode = st.radio('Custom strategy mode', ['Preset', 'Visual Builder', 'Manual'], horizontal=True)
        fast_window = st.slider('Custom fast window', 5, 100, 20)
        slow_window = st.slider('Custom slow window', fast_window + 1, 250, 50)
        rsi_window = st.slider('Custom RSI window', 5, 40, 14)
        z_window = st.slider('Custom z/vol window', 10, 120, 20)

        variable_options = list(CUSTOM_VARIABLES.keys())
        operator_options = ['>', '<', '>=', '<=', '==', '!=']

        def condition_builder(prefix: str, defaults: list[tuple[str, str, str]]) -> str:
            joiner = st.selectbox(f'{prefix} condition joiner', ['AND', 'OR'], key=f'{prefix}_joiner')
            condition_count = st.slider(f'{prefix} condition count', 1, 5, min(2, len(defaults) or 2), key=f'{prefix}_count')
            conditions = []
            for i in range(condition_count):
                d = defaults[i] if i < len(defaults) else ('close', '>', 'sma_slow')
                c1, c2, c3 = st.columns([1.1, 0.7, 1.1])
                left = c1.selectbox(f'{prefix} left {i+1}', variable_options, index=variable_options.index(d[0]) if d[0] in variable_options else 0, key=f'{prefix}_left_{i}')
                op = c2.selectbox(f'{prefix} operator {i+1}', operator_options, index=operator_options.index(d[1]) if d[1] in operator_options else 0, key=f'{prefix}_op_{i}')
                right = c3.text_input(f'{prefix} right {i+1}', value=d[2], key=f'{prefix}_right_{i}', help='Use a number like 70 or a variable like sma_slow.')
                conditions.append((left, op, right))
            return CustomRuleStrategy.build_rule(conditions, joiner)

        if custom_mode == 'Preset':
            preset_name = st.selectbox('Preset strategy template', list(CUSTOM_PRESETS.keys()))
            preset = CUSTOM_PRESETS[preset_name]
            st.info(preset['description'])
            long_rule = st.text_area('Long rule', value=preset['long_rule'], height=70)
            short_rule = st.text_area('Short rule', value=preset['short_rule'], height=70, disabled=long_only)
        elif custom_mode == 'Visual Builder':
            st.caption('Build rules by stacking conditions. Right side accepts numbers or indicator names.')
            long_rule = condition_builder('Long', [('sma_fast', '>', 'sma_slow'), ('rsi', '<', '70')])
            short_rule = '' if long_only else condition_builder('Short', [('sma_fast', '<', 'sma_slow'), ('rsi', '>', '30')])
            st.code(f'Long rule: {long_rule or "None"}')
            if not long_only:
                st.code(f'Short rule: {short_rule or "None"}')
        else:
            st.caption('Manual rules use pandas boolean logic: &, |, ~ and parentheses. Example: (close > sma_slow) & (rsi < 70)')
            long_rule = st.text_area('Long rule', value='(sma_fast > sma_slow) & (rsi < 70)', height=80)
            short_rule = st.text_area('Short rule', value='(sma_fast < sma_slow) & (rsi > 30)', height=80, disabled=long_only)
            with st.expander('Available variables'):
                st.dataframe(pd.DataFrame({'Variable': list(CUSTOM_VARIABLES.keys()), 'Meaning': list(CUSTOM_VARIABLES.values())}), use_container_width=True, hide_index=True)

        try:
            CustomRuleStrategy.validate_rule(long_rule)
            CustomRuleStrategy.validate_rule(short_rule if not long_only else '')
            st.success('Custom rules passed validation.')
        except Exception as rule_error:
            st.warning(f'Rule validation issue: {rule_error}')
        strategy = CustomRuleStrategy(long_rule=long_rule, short_rule=short_rule, fast_window=fast_window, slow_window=slow_window, rsi_window=rsi_window, z_window=z_window, long_only=long_only)
    elif strategy_name == 'Ensemble: Trend + Mean Reversion':
        trend_weight = st.slider('Trend model weight', 0.0, 1.0, 0.6, 0.05)
        strategy = EnsembleStrategy(
            [MovingAverageCrossover(20, 80, long_only=long_only), RSIStrategy(14, 30, 70), MACDTrendStrategy(long_only=long_only)],
            weights=[trend_weight, (1 - trend_weight) / 2, (1 - trend_weight) / 2],
        )
    else:
        strategy = BuyAndHold()

    if input_errors:
        for error in input_errors:
            st.warning(error)
    run = st.button('Run Backtest', type='primary', disabled=bool(input_errors))
    run_optimizer = st.checkbox('Run MA parameter optimizer')

if run:
    try:
        source = 'csv' if data_source == 'CSV upload' else 'yfinance'
        prices = load_prices(ticker_list, start=str(start), end=str(end), source=source, csv_path=csv_file)
        benchmark_symbols = parse_tickers(benchmark_ticker)
        benchmark = None
        if benchmark_symbols:
            try:
                benchmark = load_prices(benchmark_symbols[:1], start=str(start), end=str(end), source=source, csv_path=csv_file)
            except Exception as bench_exc:
                st.warning(f'Benchmark could not be loaded and will be skipped: {bench_exc}')
        engine_kwargs = dict(
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            max_weight=max_weight,
            max_gross_exposure=max_gross,
            rebalance_frequency=rebalance_frequency,
            long_only=long_only,
            volatility_target=vol_target,
            risk_free_rate=risk_free_rate,
        )
        engine = BacktestEngine(**engine_kwargs)
        result = engine.run(prices, strategy, benchmark=benchmark)

        hist = result['history']
        metrics = result['metrics']
        trades = result['trades']

        st.subheader('Performance Summary')
        cols = st.columns(5)
        display_metrics = ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Daily VaR 95%']
        for col, key in zip(cols, display_metrics):
            value = metrics.get(key)
            if value is None or pd.isna(value):
                col.metric(key, 'N/A')
            elif 'Return' in key or 'Drawdown' in key or 'VaR' in key:
                col.metric(key, f'{value:.2%}')
            else:
                col.metric(key, f'{value:.2f}')

        tabs = st.tabs(['Charts', 'Metrics', 'Trades', 'Weights', 'Signal Diagnostics', 'Optimizer', 'Downloads'])
        with tabs[0]:
            equity_df = hist[['Equity']].copy()
            if 'Benchmark' in hist:
                equity_df['Benchmark'] = hist['Benchmark']
            st.plotly_chart(px.line(equity_df, title='Portfolio vs Benchmark'), use_container_width=True)
            left, right = st.columns(2)
            with left:
                st.plotly_chart(px.area(hist, y='Drawdown', title='Portfolio drawdown'), use_container_width=True)
            with right:
                st.plotly_chart(px.histogram(hist, x='Portfolio Return', nbins=60, title='Daily return distribution'), use_container_width=True)
            st.plotly_chart(px.line(hist, y=['Gross Exposure', 'Cash Weight'], title='Exposure and cash weight'), use_container_width=True)
            monthly_returns = hist['Equity'].resample('ME').last().pct_change().dropna().rename('Monthly Return').reset_index()
            if not monthly_returns.empty:
                st.plotly_chart(px.bar(monthly_returns, x='Date', y='Monthly Return', title='Monthly returns'), use_container_width=True)

        with tabs[1]:
            metric_df = pd.DataFrame({'Metric': list(metrics.keys()), 'Value': list(metrics.values())})
            st.dataframe(metric_df, use_container_width=True)

        with tabs[2]:
            st.dataframe(trades.tail(300), use_container_width=True)

        with tabs[3]:
            weight_cols = [c for c in hist.columns if c.startswith('Weight ')]
            if weight_cols:
                st.plotly_chart(px.area(hist[weight_cols], title='Portfolio allocation over time'), use_container_width=True)
                st.dataframe(hist[weight_cols].tail(100), use_container_width=True)
            else:
                st.info('No weights available.')

        with tabs[4]:
            signal_sample = result['signals'].tail(252)
            if not signal_sample.empty:
                active_days = (result['signals'].abs().sum(axis=1) > 0).mean()
                avg_names = result['signals'].abs().mean().sort_values(ascending=False).reset_index()
                avg_names.columns = ['Ticker', 'Average absolute weight']
                d1, d2, d3 = st.columns(3)
                d1.metric('Active signal days', f'{active_days:.1%}')
                d2.metric('Average turnover', f'{hist["Turnover"].mean():.2%}')
                d3.metric('Total estimated cost drag', f'{hist["Trading Cost"].sum():.2%}')
                st.plotly_chart(px.imshow(signal_sample.T, aspect='auto', title='Signal heatmap, last 252 trading days'), use_container_width=True)
                st.dataframe(avg_names, use_container_width=True, hide_index=True)
                if strategy_name == 'Custom Rule Strategy':
                    st.markdown('**Custom rule used**')
                    st.code(f'Long: {strategy.long_rule}\nShort: {strategy.short_rule if not strategy.long_only else "disabled / long-only"}')
            else:
                st.info('No signal diagnostics available.')

        with tabs[5]:
            if run_optimizer:
                opt = grid_search(
                    prices,
                    MovingAverageCrossover,
                    {'short_window': [10, 20, 30], 'long_window': [50, 100, 150]},
                    engine_kwargs=engine_kwargs,
                    benchmark=benchmark,
                    objective='Sharpe Ratio',
                )
                st.dataframe(opt, use_container_width=True)
                if not opt.empty and 'Sharpe Ratio' in opt:
                    st.plotly_chart(px.scatter(opt, x='Annual Volatility', y='Annual Return', size='Sharpe Ratio', hover_data=['short_window', 'long_window'], title='Optimizer: return vs risk'), use_container_width=True)
            else:
                st.info('Enable optimizer in the sidebar to run a small MA parameter sweep.')

        with tabs[6]:
            report_paths = export_report(result, ROOT / 'reports')
            st.download_button('Download trade log CSV', trades.to_csv(index=False), file_name='trade_log.csv')
            st.download_button('Download equity curve CSV', hist.to_csv(index=True), file_name='equity_curve.csv')
            st.download_button('Download metrics CSV', pd.Series(metrics).to_csv(), file_name='metrics.csv')
            st.caption(f'HTML report saved to {report_paths["html"]}')
    except Exception as exc:
        st.error(f'Backtest failed: {exc}')
else:
    st.info('Choose tickers and a strategy, then run the backtest.')
