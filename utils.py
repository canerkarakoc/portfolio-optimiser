from io import BytesIO
from typing import List

import pandas as pd
import streamlit as st


def parse_tickers(text: str) -> List[str]:
    """Parse a comma-separated ticker string into a clean list.

    Normalizes common US equities aliases (e.g., BRK.B -> BRK-B) and removes
    spaces or currency suffixes some users might include.
    """
    aliases = {
        "BRK.B": "BRK-B",
        "BRK/B": "BRK-B",
        "BRK B": "BRK-B",
        "BRK.A": "BRK-A",
        "BRK/A": "BRK-A",
        "BF.B": "BF-B",
        "BF/B": "BF-B",
        "BF A": "BF-A",
        "BF.A": "BF-A",
        # Common US index aliases
        "SPX": "^GSPC",          # S&P 500 index
        "S&P500": "^GSPC",
        "SP500": "^GSPC",
        "DJI": "^DJI",           # Dow Jones Industrial Average
        "INDU": "^DJI",
        "NASDAQ": "^IXIC",       # Nasdaq Composite
        "IXIC": "^IXIC",
        "NDX": "^NDX",           # Nasdaq 100
        "RUT": "^RUT",           # Russell 2000
        "RUA": "^RUA",           # Russell 3000
        "NYA": "^NYA",           # NYSE Composite
        "VIX": "^VIX",           # CBOE Volatility Index
        # Popular ETF synonym fallbacks
        "SP500ETF": "SPY",
        "S&P500ETF": "SPY",
        "QQQETF": "QQQ",
        "DIAETF": "DIA",
    }
    raw = [t.strip().upper() for t in text.split(",") if t.strip()]
    cleaned: List[str] = []
    for t in raw:
        t = t.replace(" ", "")
        # drop currency/place suffixes sometimes pasted, e.g., .US, USD
        for suf in [".US", ".NY", ".NQ", " USD"]:
            if t.endswith(suf):
                t = t[: -len(suf)]
        t = aliases.get(t, t)
        cleaned.append(t)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for t in cleaned:
        if t not in seen:
            result.append(t)
            seen.add(t)
    return result


def export_dataframe(
    df: pd.DataFrame,
    file_basename: str,
    as_excel: bool = False,
    button_label: str = "Export CSV",
) -> None:
    """Render a Streamlit download button for a DataFrame as CSV or Excel."""
    if df.empty:
        st.warning("No data to export.")
        return

    if as_excel:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="data")
        st.download_button(
            label=button_label,
            data=bio.getvalue(),
            file_name=f"{file_basename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label=button_label,
            data=csv,
            file_name=f"{file_basename}.csv",
            mime="text/csv",
        )


