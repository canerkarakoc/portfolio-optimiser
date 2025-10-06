import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from data_fetch import fetch_prices
from plots import (
    plot_allocation_pie,
    plot_cumulative_returns,
    plot_efficient_frontier,
)
from portfolio import (
    compute_covariance,
    compute_returns,
    optimize_max_sharpe,
    optimize_min_volatility,
    random_portfolios,
)
from utils import export_dataframe, parse_tickers


st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")


def _show_kpis(expected_return: float, volatility: float, sharpe: float) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Annual Return", f"{expected_return:.2%}")
    col2.metric("Volatility (Annualized)", f"{volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")


def _compute_cumulative_returns(returns: pd.DataFrame) -> pd.Series:
    cumulative = (1 + returns.mean(axis=1)).cumprod()
    return cumulative


def main() -> None:
    st.title("Portfolio Optimization Dashboard")
    st.write(
        "Optimize asset allocations using Modern Portfolio Theory. Enter tickers, date range, and constraints to compute the efficient frontier and optimal portfolios."
    )

    # Sidebar inputs
    with st.sidebar:
        st.header("Inputs")
        tickers_text: str = st.text_input(
            "Tickers (comma-separated)", value="AAPL, MSFT, GOOG, AMZN"
        )
        today = dt.date.today()
        default_start = dt.date(2018, 1, 1)
        start_date: dt.date = st.date_input("Start Date", value=default_start)
        end_date: dt.date = st.date_input("End Date", value=today)
        risk_free_rate: float = st.number_input(
            "Risk-free rate (annual)", min_value=-1.0, max_value=1.0, value=0.02, step=0.005
        )
        max_alloc: Optional[float] = st.number_input(
            "Max allocation per asset (optional)", min_value=0.0, max_value=1.0, value=1.0, step=0.05
        )
        allow_short: bool = st.checkbox("Allow shorting", value=False)

        run_opt = st.button("Run Optimization", type="primary")

    if not run_opt:
        st.info("Enter inputs in the sidebar and click 'Run Optimization'.")
        return

    # Parse and validate tickers
    tickers: List[str] = parse_tickers(tickers_text)
    if len(tickers) == 0:
        st.error("Please provide at least one ticker.")
        return

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    # Fetch data
    try:
        prices: pd.DataFrame = fetch_prices(tickers, str(start_date), str(end_date))
    except Exception as exc:  # Defensive error handling
        st.error(f"Failed to fetch data: {exc}")
        return

    # Basic data validation
    if prices.empty or prices.shape[0] < 30:
        st.error("Insufficient price data to proceed.")
        return

    # Compute returns and covariance
    returns: pd.DataFrame = compute_returns(prices, method="log")
    if returns.isna().all().all() or returns.shape[0] < 20:
        st.error("Insufficient returns data after processing.")
        return

    cov_matrix: pd.DataFrame = compute_covariance(returns)
    mean_returns = returns.mean() * 252.0

    # Random portfolios for frontier visualization
    try:
        sampled = random_portfolios(
            num_portfolios=3000,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            allow_short=allow_short,
            max_weight=max_alloc,
        )
    except Exception as exc:
        st.warning(f"Sampling portfolios failed: {exc}")
        sampled = None

    # Optimizations
    try:
        opt_sharpe_weights, opt_sharpe_perf = optimize_max_sharpe(
            returns=returns,
            risk_free_rate=risk_free_rate,
            allow_short=allow_short,
            max_weight=max_alloc,
        )
    except Exception as exc:
        st.error(f"Max Sharpe optimization failed: {exc}")
        return

    try:
        opt_minvol_weights, opt_minvol_perf = optimize_min_volatility(
            returns=returns,
            allow_short=allow_short,
            max_weight=max_alloc,
        )
    except Exception as exc:
        st.warning(f"Min volatility optimization failed: {exc}")
        opt_minvol_weights, opt_minvol_perf = None, None

    # KPIs for Max Sharpe as primary
    er, vol, sr = opt_sharpe_perf
    _show_kpis(er, vol, sr)

    # Plots
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Efficient Frontier")
        if sampled is not None:
            fig_frontier = plot_efficient_frontier(sampled, highlight=(er, vol))
            st.plotly_chart(fig_frontier, use_container_width=True)
        else:
            st.info("Frontier sampling unavailable.")

        st.subheader("Cumulative Returns (Equal-Weight Backtest)")
        try:
            cum = _compute_cumulative_returns(returns)
            fig_cum = plot_cumulative_returns(cum)
            st.plotly_chart(fig_cum, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not plot cumulative returns: {exc}")

    with right:
        st.subheader("Optimal Allocation (Max Sharpe)")
        weights_series = pd.Series(opt_sharpe_weights)
        fig_alloc = plot_allocation_pie(weights_series)
        st.plotly_chart(fig_alloc, use_container_width=True)
        st.dataframe(weights_series.sort_values(ascending=False).to_frame("Weight"))

    # Exports
    st.divider()
    st.subheader("Export")
    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        export_dataframe(
            df=prices,
            file_basename="prices",
            as_excel=False,
            button_label="Export Prices CSV",
        )
    with col_xlsx:
        export_dataframe(
            df=prices,
            file_basename="prices",
            as_excel=True,
            button_label="Export Prices Excel",
        )


if __name__ == "__main__":
    main()


