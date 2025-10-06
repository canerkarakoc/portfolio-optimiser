from typing import List

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(show_spinner=False)
def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices for tickers between start and end dates.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols (e.g., ["AAPL", "MSFT"]).
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted closing prices indexed by date.
    """
    if len(tickers) == 0:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yfinance returns different shapes depending on number of tickers
    if isinstance(data.columns, pd.MultiIndex):
        # Select Close or Adj Close equivalent
        close_cols = [c for c in data.columns if c[1].lower() in ("close", "adj close")]
        prices = data.loc[:, close_cols]
        prices.columns = [c[0] for c in prices.columns]
    else:
        prices = data if "Close" not in data.columns else data["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

    prices = prices.sort_index().dropna(how="all")
    # Drop columns that are entirely NaN or have too few data points
    min_len = max(20, int(0.5 * len(prices)))
    prices = prices.dropna(axis=1, thresh=min_len)
    return prices


